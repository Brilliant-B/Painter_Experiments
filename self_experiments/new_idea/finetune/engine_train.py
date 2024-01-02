import torch
import math
import os
import sys
import numpy as np
import wandb
from typing import Iterable

import util.misc as misc
import util.lr_sched as lr_sched

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    loss_scale = None
    if hasattr(optimizer, 'loss_scale'):
        loss_scale = optimizer.loss_scale
    elif hasattr(optimizer, 'cur_scale'):
        loss_scale = optimizer.cur_scale
    return loss_scale, optimizer._global_grad_norm
    # return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module,
                    model_without_ddp: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int, 
                    loss_scaler,
                    log_writer=None,
                    global_rank=None,
                    args=None):
    model.train(True)
    optimizer.zero_grad()
    
    log_file = os.path.join(args.output_dir, "train_log.log")
    metric_logger = misc.MetricLogger(delimiter="  ", log_file=log_file)
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch [{}]'.format(epoch)
    accum_iter = args.accum_iter
    print_freq = accum_iter
    # if global_rank == 0 and log_writer is not None:
    #     print('tensorboard_log_dir: {}'.format(log_writer.log_dir))

    wandb_images = []
    for data_iter_step, (type, c_query, c_target, query, target, mask, valid) in enumerate(metric_logger.log_every(data_loader, print_freq, header, global_rank)):
        # per iteration lr_scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, epoch * len(data_loader) + data_iter_step + 1, args.epochs * len(data_loader), args)
        
        c_query = c_query.to(device, non_blocking=True)
        c_target = c_target.to(device, non_blocking=True)
        query = query.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss, pred, image_mask = model(type, c_query, c_target, query, target, mask, valid)
        
        loss_value = loss.item()
        loss /= accum_iter
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        if loss_scaler is None:
            print("None")        
            model.backward(loss)
            model.step()
            loss_scale_value, grad_norm = get_loss_scale_for_deepspeed(model)
        else:
            # print(next(model.parameters()).device)
            grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                                    parameters=model.parameters(),
                                    update_grad=(data_iter_step + 1) % accum_iter == 0)
            if global_rank == 0 and (math.isinf(loss_value) or math.isnan(loss_value)):
                print(f'serial {data_iter_step} error: loss is {loss_value}')
            if global_rank == 0 and grad_norm is not None and (math.isinf(grad_norm) or math.isnan(grad_norm)):
                print(f"serial {data_iter_step} error: grad_norm is {grad_norm}")
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
                if hasattr(model_without_ddp, 'c_momentum'):
                    model_without_ddp.c_blocks_momentum_update()
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(grad_norm=grad_norm)
        
        if args.output_dir and (data_iter_step + 1 == len(data_loader) or (data_iter_step + 1) % args.save_itrs == 0):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, itr=data_iter_step+1)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

            if global_rank == 0 and args.log_wandb:
                wandb.log({'train_loss': loss_value_reduce, 'lr': lr, 'train_loss_scale': loss_scale_value, 'grad_norm': grad_norm})
                if len(wandb_images) < 20:
                    y = pred[[0]].detach().cpu()
                    msk = image_mask[[0]].detach().float().cpu()
                    x = query[[0]].detach().float().cpu()
                    label = target[[0]].detach().float().cpu()
                    msk_label = label * (1 - msk)
                    frame = torch.cat((x, msk_label, y, label), dim=2)[0]
                    frame = torch.clip((frame * imagenet_std + imagenet_mean) * 255, 0, 255).int()
                    wandb_images.append(wandb.Image(frame.numpy(), caption="query; masked_label; pred; label"))

    if global_rank == 0 and args.log_wandb and len(wandb_images) > 0:
        wandb.log({"Training examples": wandb_images})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_pt(data_loader, model, device, epoch=None, global_rank=None, args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Validation:'
    # switch to evaluation mode
    model.eval()
    if args.log_wandb:  wandb_images = []
    
    for type, c_query, c_target, query, target, mask, valid in metric_logger.log_every(data_loader, 20, header):
        c_query = c_query.to(device, non_blocking=True)
        c_target = c_target.to(device, non_blocking=True)
        query = query.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss, pred, image_mask = model(type, c_query, c_target, query, target, mask, valid)

        metric_logger.update(loss=loss.item())
        if global_rank == 0 and args.log_wandb:
            y = pred[[0]].detach().cpu()
            msk = image_mask[[0]].detach().float().cpu()
            x = query[[0]].detach().float().cpu()
            label = target[[0]].detach().float().cpu()
            msk_label = label * (1 - msk)
            frame = torch.cat((x, msk_label, y, label), dim=2)[0]
            frame = torch.clip((frame * imagenet_std + imagenet_mean) * 255, 0, 255).int()
            wandb_images.append(wandb.Image(frame.numpy(), caption="query; masked_label; pred; label"))
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes() 
    out = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if global_rank == 0 and args.log_wandb:
        wandb.log({**{f'test_{k}': v for k, v in out.items()},'epoch': epoch})
        if len(wandb_images) > 0:
            wandb.log({"Testing examples": wandb_images[::2][:20]})
    if global_rank == 0:
        print('Val loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
        print(f"Val Loss of the network = {out['loss']:.3f}")
        with open(os.path.join(args.output_dir, "train_log.log"), 'a') as f:
            print(f"Val Loss of the network = {out['loss']:.3f}", file=f)

    return out
