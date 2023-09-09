import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import sys
sys.path.append(r'/root/autodl-tmp/Painter/')
import argparse
import datetime
import json
import re
import numpy as np
import time
from pathlib import Path
import timm
assert timm.__version__ == "0.3.2"  # version check
import tqdm
import deepspeed
from deepspeed import DeepSpeedConfig

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from util import ddp_utils
import util.lr_decay as lrd
import util.misc as misc
from util.misc import get_parameter_groups
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed, interpolate_rel_pos_embed_proto_mo
from data.pairdataset_variant import PairDataset
from util.ddp_utils import DatasetTest
from torch.utils.data import DataLoader
import data.pair_transforms as pair_transforms
from util.masking_generator import MaskingGenerator
from data.sampler import DistributedSamplerWrapper
import wandb

import models.proto_mo.proto_mo_2 as painter_variant
from self_experiments.proto_mo.finetune.engine_train import train_one_epoch

TRAIN_JSON_BANK = {
    "ade20k_image2semantic": "ade20k/ade20k_training_image_semantic.json",
    "coco_image2panoptic_sem_seg": "coco/pano_sem_seg/coco_train2017_image_panoptic_sem_seg.json",
    "nyuv2_image2depth": "nyu_depth_v2/nyuv2_sync_image_depth.json",
    "derain_image2derain": "derain/derain_train.json",
    "lol_image2enhance": "light_enhance/enhance_lol_train.json",
    "ssid_image2denoise": "denoise/denoise_ssid_train.json",
    "coco_pano_inst": "coco/pano_ca_inst/coco_train_image_panoptic_inst.json",
    "coco_pose": "coco_pose/coco_pose_256x192_train.json",
}

VAL_JSON_BANK = {
    "ade20k_image2semantic": "ade20k/ade20k_validation_image_semantic.json",
    "coco_image2panoptic_sem_seg": "coco/pano_sem_seg/coco_val2017_image_panoptic_sem_seg.json",
    "nyuv2_image2depth": "nyu_depth_v2/nyuv2_test_image_depth.json",
    "derain_image2derain": "derain/derain_test_rain100h.json",
    "lol_image2enhance": "light_enhance/enhance_lol_val.json",
    "ssid_image2denoise": "denoise/denoise_ssid_val.json",
    "coco_pano_inst": "coco/pano_ca_inst/coco_val_image_panoptic_inst.json",
    "coco_pose": "coco_pose/coco_pose_256x192_val.json",
}


def get_args_parser():
    parser = argparse.ArgumentParser('Painter_Variant_2 fine-tuning', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int, 
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--accum_iter', default=16, type=int, 
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    # Model parameters
    parser.add_argument('--model_name', default='painter_varient_1', type=str, help='Name of model to train')
    parser.add_argument('--model', default='painter_varient_1_patch16_win_dec64_8glb_sl1', type=str, metavar='MODEL',
                        help='Full-Name of model to train')
    parser.add_argument('--img_size', default=448, type=int, nargs='+', help='images input size')
    parser.add_argument('--mask_ratio', default=0.5, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--num_mask_patches', default=784, type=int, help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)
    parser.add_argument('--stop_grad_patch_embed', action='store_true', help='stop-grad after first conv, or patch embedding')
    parser.set_defaults(stop_grad_patch_embed=False)
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--drop_path', default=0., type=float, help='Drop path rate (default: 0.)')
    parser.add_argument('--min_random_scale', default=0.3, type=float, help='Minimal random scale for randomresizecrop (default: 0.3)')
    parser.add_argument('--last_norm_instance', action='store_true', default=False,
                        help='use instance norm to normalize each channel map before the decoder layer')
    parser.add_argument('--half_mask_ratio', default=0.1, type=float, help='ratio of using half mask during training (default: 0.1)')
    parser.add_argument('--use_checkpoint', action='store_true', default=False, help='use checkpoint to save GPU memory')


    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay (default: 0.1)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_itrs', type=int, default=160, metavar='N',
                        help='itrs to warmup LR')
    parser.add_argument('--save_itrs', type=int, default=1000,
                        help='save checkkpoints frequency iteration')
    parser.add_argument('--clip_grad', type=float, default=3.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--layer_decay', type=float, default=1.0, metavar='LRD',
                        help='Learning rate layer decay')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--json_path', default='./', nargs='+', type=str,
                        help='json path')
    parser.add_argument('--val_json_path', default='./', nargs='+',type=str,
                        help='json path')
    parser.add_argument('--base_output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=False)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--log_wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--zero_stage', default=0, type=int,
                        help='ZeRO optimizer stage (default: 0)')

    # for deepspeed
    known_args, _ = parser.parse_known_args()
    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None
    args = parser.parse_args()
    args.ds_init = ds_init
    
    return args



def freeze_match(name, f_list):
    ret = False
    for n in f_list:
        ret = ret or (re.search(n, name) is not None)
    return ret

def prepare_model(args, prints=False):
    # get model with args
    model = painter_variant.__dict__[args.model](
        seed=args.seed,
        datasets=args.datasets_weights.keys(),
        num_contexts_in=args.nci,
        num_contexts=args.nc,
        cq_depth=args.cq,
        p_depth=args.p,
        encoder_momentum_weight=args.emo,
        context_momentum_weight=args.cmo,
        query_momentum_weight=args.qmo,
        skip_query=args.skip_query,
        use_attn_mean=args.use_attn_mean,
        use_random_nc=args.use_random_nc,
        dataset_loss_weight=args.datasets_weights,
        is_infer=False,
    ).to("cuda")
    
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')['model']    
        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint)
        interpolate_rel_pos_embed_proto_mo(model, checkpoint)
        # interpolate patch embedding
        if "patch32" in args.model:
            patch_weight = checkpoint['patch_embed.proj.weight']
            new_patch_weight = F.interpolate(patch_weight, size=(32, 32), mode='bicubic', align_corners=False)
            checkpoint['patch_embed.proj.weight'] = new_patch_weight
        if prints:  print("Load pre-trained checkpoint from: %s" % args.finetune)
        state_dict = model.state_dict()
        rm_key_list = ['decoder_embed.weight', 'decoder_embed.bias',  'mask_token']
        if args.last_norm_instance:
            rm_key_list.extend(['norm.weight', 'norm.bias'])
        
        finetune_code, freeze_list = args.finetune_code, []
        if finetune_code < 3:
            freeze_list.append("decoder*")
            code2layers = [0, model.cq, model.depth]
            for l in range(model.depth):
                if l >= code2layers[finetune_code]:
                    freeze_list.append(f"blocks.{l}.*")
        # print(freeze_list)
        for k in rm_key_list:
            if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
                if prints:  print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        
        # load pre-trained model
        msg = model.load_state_dict(checkpoint, strict=False)
        if args.emo > 0.0: model.init_cm_encoder()
        if prints:  print(msg)
        
        # freeze part of the modules
        for (name, param) in model.named_parameters():
            if freeze_match(name, freeze_list): param.requires_grad = False
            if prints:  print(name, param.requires_grad)
    return model



def prepare_data(args, prints=False):
    transform_train = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop(args.img_size[1], scale=(args.min_random_scale, 1.0), interpolation=3),  # 3 is bicubic
            pair_transforms.RandomApply([pair_transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            pair_transforms.RandomHorizontalFlip(),
            pair_transforms.ToTensor(),
            pair_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_train2 = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop(args.img_size[1], scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
            pair_transforms.ToTensor(),
            pair_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_train3 = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop(args.img_size[1], scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
            pair_transforms.ToTensor(),
            pair_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_train_seccrop = pair_transforms.Compose([
            pair_transforms.RandomResizedCrop(args.img_size, scale=(args.min_random_scale, 1.0), ratio=(0.3, 0.7), interpolation=3),  # 3 is bicubic
            ])
    # transform_val = pair_transforms.Compose([
    #         pair_transforms.RandomResizedCrop(args.img_size[1], scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
    #         pair_transforms.ToTensor(),
    #         pair_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    masked_position_generator = MaskingGenerator(
        args.window_size, num_masking_patches=args.num_mask_patches,
        max_num_patches=args.max_mask_patches_per_block,
        min_num_patches=args.min_mask_patches_per_block,
    )
    dataset_train = PairDataset(args, args.json_path, transform=transform_train, transform2=transform_train2, 
                                transform3=transform_train3, transform_seccrop=transform_train_seccrop, 
                                masked_position_generator=masked_position_generator)
    # dataset_val = PairDataset(args.data_path, args.val_json_path, args=args, transform=transform_val, transform2=None, transform3=None, 
    #                           masked_position_generator=masked_position_generator, mask_ratio=1.)
    if prints:  print(dataset_train)
    return dataset_train



def init_model_queue(args, model):
    device = torch.device("cuda")
    model.use_fmo = False
    model.nci = model.nc
    for type in args.datasets_weights.keys():
        dataset = DatasetTest(args.data_path, TRAIN_JSON_BANK[dataset_name], args.img_size[0], args.nc)
        data_loader = DataLoader(dataset, batch_size=1, drop_last=False, collate_fn=ddp_utils.collate_fn, num_workers=2)
        c_query, c_target = [], []
        for data in tqdm.tqdm(data_loader):
            img, _, tgt, _, _ = data[0]
            c_query.append(img)
            c_target.append(tgt)
        c_query = torch.stack(c_query, dim=0).unsqueeze(0)
        c_target = torch.stack(c_target, dim=0).unsqueeze(0)
        valid = torch.ones_like(c_target[:, 0])
        mask = torch.ones(model.ori_window_size).unsqueeze(0)
        with torch.no_grad():
            _, _, _ = model([type], c_query.float().to(device), c_target.float().to(device), c_query[:, 0].float().to(device), 
                            c_target[:, 0].float().to(device), mask.float().to(device), valid.float().to(device))
    model.use_fmo = True
    model.nci = args.nci



def main(args, INFO):
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    mix_data = "Joint" if args.joint_datasets else "Seperate"
    output_dir = args.output_dir = os.path.join(args.base_output_dir, \
        f"{mix_data}|{args.nci}:{args.nc}:{args.cq}:{args.p}:{args.qmo}:{args.cmo}:{args.emo}|{args.finetune_code}:{args.mask_ratio}")
    train_log_dir = os.path.join(output_dir, "train_log.log")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print('output_dir: {}'.format(output_dir))
    print('job_dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    
    # define the model
    print("[Prepare Model]")
    with open(Path(train_log_dir), 'a') as f:
        print(json.dumps(INFO, sort_keys=False, indent=4), file=f)
        print(json.dumps(INFO, sort_keys=False, indent=4))
    model = prepare_model(args)
    args.patch_size = patch_size = model.patch_size
    args.window_size = (args.img_size[0] // patch_size, args.img_size[1] // patch_size)
    model.to(device)
    model_without_ddp = model.to(device)
    # print("Model = %s" % str(model_without_ddp))
    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, model.no_weight_decay()
        )
        model, optimizer, _, _ = args.ds_init(
            args=args, model=model, model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )
        # print("model.gradient_accumulation_steps() = %d" %
        #         model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.accum_iter
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu]) # find_unused_parameters=True
            model_without_ddp = model.module
        # following timm: set wd as 0 for bias and norm layers
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=args.opt_betas)
        loss_scaler = NativeScaler()
        # print(optimizer)

    # define and augment the datasets
    print("[Prepare Data]")
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    dataset_train = prepare_data(args)
    num_samples_train = len(dataset_train)
    weights_train = dataset_train.weights
    sampler_train = torch.utils.data.WeightedRandomSampler(weights_train, num_samples_train, replacement=True)
    sampler_train = DistributedSamplerWrapper(sampler_train, num_replicas=num_tasks, rank=global_rank, shuffle=args.joint_datasets)
    # print("Sampler_train = %s" % str(sampler_train))
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    '''
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    '''
    # load tools
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    if global_rank == 0 and args.log_wandb:
        experiment = args.log_dir.split('/')[-2]
        if args.resume == '':
            wandb.init(project="Painter", name=experiment, config=args)
        else:
            wandb.init(project="Painter", name=experiment, config=args, resume=True)
    misc.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    # initialize the model queue:
    print("[Initialize Model Queue]")
    init_model_queue(args, model_without_ddp)
    
    # show important hyper-parameters
    print("[Important Hyper-Parameters]")
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # start training
    print(f"[Start training for {args.epochs} epochs]")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)  
        train_stats = train_one_epoch(
            model, model_without_ddp, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            global_rank=global_rank,
            args=args,
        )
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,}
        if output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(output_dir, "epoch_log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Done! Training time {}\n'.format(total_time_str))

    if global_rank == 0 and args.log_wandb:
        wandb.finish()



if __name__ == '__main__':
    args = get_args_parser()
    cudnn.benchmark = True
    misc.init_distributed_mode(args)
    if args.ds_init is not None:
        misc.create_ds_config(args)
    
    INFO = dict()
    INFO['seed'] = args.seed = 0
    INFO['datasets_weights'] = args.datasets_weights = datasets = {
        "ade20k_image2semantic": 28,
        "coco_image2panoptic_sem_seg": 30,
        "nyuv2_image2depth": 20,
        "lol_image2enhance": 12,
        "derain_image2derain": 10,
        "ssid_image2denoise": 18,
    }
    # INFO['datasets_weights'] = args.datasets_weights = datasets = {
    #     "ade20k_image2semantic": 28,
    #     "coco_image2panoptic_sem_seg": 30,
    #     "nyuv2_image2depth": 18,
    #     "lol_image2enhance": 10,
    #     "derain_image2derain": 10,
    #     "ssid_image2denoise": 20,
    # }
    json_path, val_json_path = [], []
    for dataset_name in datasets.keys():
        json_path.append(os.path.join(args.data_path, TRAIN_JSON_BANK[dataset_name]))
        val_json_path.append(os.path.join(args.data_path, VAL_JSON_BANK[dataset_name]))
    args.json_path, args.val_json_path = json_path, val_json_path
    
    INFO['epochs'] = args.epochs = 3
    INFO['save_freq'] = args.save_itrs = 32000
    INFO['batch_size'] = args.batch_size = 2
    INFO['accum_iter'] = args.accum_iter = 64
    INFO['learning_rate'] = args.lr = 1e-4
    INFO['warmup_itrs'] = args.warmup_itrs = 2048

    INFO['joint_train'] = args.joint_datasets = True
    INFO['finetune'] = args.finetune_code = 2
    INFO['mask_ratio'] = args.mask_ratio = 0.99
    
    INFO['skip_query'] = args.skip_query = False
    INFO['use_attn_mean'] = args.use_attn_mean = True
    INFO['use_random_nc'] = args.use_random_nc = False
    INFO['encoder_momentum_weight'] = args.emo = 0.99
    INFO['context_momentum_weight'] = args.cmo = 0
    INFO['query_momentum_weight'] = args.qmo = 1
    
    INFO['num_contexts_input'] = args.nci = 1
    INFO['num_contexts_used'] = args.nc = 5
    INFO['cr_depth'] = args.cq = 15
    INFO['p_depth'] = args.p = 1
    
    main(args, INFO)
