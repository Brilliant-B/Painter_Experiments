# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import sys
import warnings

import requests
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm
import time
import random
import itertools
from thop import profile
from fvcore.nn import FlopCountAnalysis, flop_count_table

sys.path.append('.')
import matplotlib.pyplot as plt
from PIL import Image
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from util.ddp_utils import DatasetTest
from util import ddp_utils
import painter_variant_0, painter_variant_1


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
PROMPTS_BANK = ['ADE_train_00001098', 'ADE_train_00003821', 'ADE_train_00007380', 'ADE_train_00009464', 'ADE_train_00010441', 
                'ADE_train_00010565', 'ADE_train_00013301', 'ADE_train_00014165', 'ADE_train_00016499', 'ADE_train_00017885',]

def get_args_parser():
    parser = argparse.ArgumentParser('ADE20k semantic segmentation', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt', default='')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    # parser.add_argument('--prompts', type=str, help='prompt image in train set',
    #                     default='ADE_train_00014165', nargs='+')
    parser.add_argument('--img_size', type=int, default=448)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1', args=None, test_flop=False, prints=True):
    # build model
    model = getattr(painter_variant_1, arch)(num_prompts=args.num_prompts, cr_depth=args.cr_depth, xcr_depth=args.xcr_depth)    
    if test_flop:
        dummy_inputs = (torch.randn(1, 3, args.input_size * (args.num_prompts + 1), args.input_size), 
                        torch.randn(1, 3, args.input_size * (args.num_prompts + 1), args.input_size),
                        torch.randn(1, 3, args.input_size * (args.num_prompts + 1), args.input_size),)
        flops = FlopCountAnalysis(model, dummy_inputs)
        print("FLOPs: ", flops.total())
        # print(flop_count_table(model))
        # flops, params = profile(model, dummy_inputs)
        # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
        # print('flops: ', flops, 'params: ', params)  
    
    model.to("cuda")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module
    
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')['model']
    
    for i in range(args.cr_depth):
        param = checkpoint[f'blocks.{i}.attn.rel_pos_h'].permute(1, 0).unsqueeze(0)
        checkpoint[f'blocks.{i}.attn.rel_pos_h'] = F.interpolate(
            param, size=(0 * 56 + 55), mode='linear')[0].permute(1, 0)
    for i in range(args.cr_depth, args.cr_depth + args.xcr_depth):
        param = checkpoint[f'blocks.{i}.attn.rel_pos_h'].permute(1, 0).unsqueeze(0)
        checkpoint[f'blocks.{i}.attn.rel_pos_h'] = F.interpolate(
            param, size=(args.num_prompts * 56 + 55), mode='linear')[0].permute(1, 0)
    for i in range(args.cr_depth + args.xcr_depth, 24):
        param = checkpoint[f'blocks.{i}.attn.rel_pos_h'].permute(1, 0).unsqueeze(0)
        checkpoint[f'blocks.{i}.attn.rel_pos_h'] = F.interpolate(
            param, size=(1 * 56 + 55), mode='linear')[0].permute(1, 0)
        
    msg = model_without_ddp.load_state_dict(checkpoint, strict=False)
    if prints:
        print(msg)
        print("Model Loaded.")
    return model


def run_one_image(prompts, query, img_size, model, out_path, device):
    pred = model(prompts.float().to(device), query.float().to(device))
    output = pred.detach().cpu()
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    output = F.interpolate(output.unsqueeze(0).permute(0, 3, 1, 2), size=(img_size[1], img_size[0]),
        mode='bilinear').permute(0, 2, 3, 1)[0]
    assert output.shape[-1] == 3
    output = output.int()
    output = Image.fromarray(output.numpy().astype(np.uint8))
    output.save(out_path)


def prompts_test_step(args, test_flop):
    dataset_dir = "toy_datasets/"
    device = torch.device("cuda")
    ckpt_path = args.ckpt_path
    model = args.model
    img_size = args.img_size
    num_prompts = args.num_prompts
    cr_depth = args.cr_depth
    xcr_depth = args.xcr_depth
    path_splits = ckpt_path.split('/')
    ckpt_dir, ckpt_file = path_splits[-2], path_splits[-1]
    dst_dir = "self_experiments/inference_painter_variant_1/inference_multi_ade20k_{}_semseg_{}_contexts_{}_crdepth_{}_xcrdepth".format(
        img_size, num_prompts, cr_depth, xcr_depth)

    if ddp_utils.get_rank() == 0:
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        # print("output_dir: {}".format(dst_dir))

    print(f'n_contexts: {args.num_prompts}, cr-depth: {args.cr_depth}, xcr_depth: {args.xcr_depth}')
    model_painter = prepare_model(ckpt_path, model, args=args, test_flop=test_flop, prints=False)
    device = torch.device("cuda")
    model_painter.to(device)
    
    img_src_dir = dataset_dir + "ade20k/images/validation"
    # img_path_list = glob.glob(os.path.join(img_src_dir, "*.jpg"))
    dataset_val = DatasetTest(img_src_dir, img_size, ext_list=('*.jpg',))
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
    data_loader_val = DataLoader(dataset_val, batch_size=1, sampler=sampler_val,
                                drop_last=False, collate_fn=ddp_utils.collate_fn, num_workers=2)

    prompts = random.sample(PROMPTS_BANK, k = num_prompts)
    imgp_paths = [dataset_dir + "ade20k/images/training/{}.jpg".format(prompt) for prompt in prompts]
    tgtp_paths = [dataset_dir + "ade20k/annotations_with_color/training/{}.png".format(prompt) for prompt in prompts]

    # load the shared prompt image pair
    prompts = []
    for imgp_path, tgtp_path in zip(imgp_paths, tgtp_paths):
        imgp = Image.open(imgp_path).convert("RGB")
        imgp = imgp.resize((img_size, img_size))
        imgp = torch.from_numpy(np.array(imgp) / 255.)
        tgtp = Image.open(tgtp_path)
        tgtp = tgtp.resize((img_size, img_size))
        tgtp = torch.from_numpy(np.array(tgtp) / 255.)
        prompt = torch.stack([imgp, tgtp], dim=0)
        prompt = (prompt - imagenet_mean) / imagenet_std
        prompts.append(prompt)
    prompts = torch.stack(prompts, dim=1)
    assert prompts.shape == (2, num_prompts, img_size, img_size, 3)
    
    T0 = time.perf_counter()
    model_painter.eval()
    for _ in range(5):
        for data in tqdm.tqdm(data_loader_val):
            """ Load an image """
            assert len(data) == 1
            img, img_path, size = data[0]
            img_name = os.path.basename(img_path)
            out_path = os.path.join(dst_dir, img_name.replace('.jpg', '.png'))
            query = torch.from_numpy(img).unsqueeze(0)
            query = (query - imagenet_mean) / imagenet_std
            assert query.shape == (1, img_size, img_size, 3)
            # make random mask reproducible (comment out to make it change)
            torch.manual_seed(2)
            run_one_image(prompts, query, size, model_painter, out_path, device)
    T1 = time.perf_counter()
    print(f'timing = {T1 - T0} s')
    return T1 - T0


def graphing(costs, prompts_choice, cr_depth_choice, xcr_depth_choice):
    plt.figure(figsize=(50, 35), dpi=100)
    plt.style.use('fivethirtyeight')
    
    x_idx = []
    for cr_depth in cr_depth_choice:
        for xcr_depth in xcr_depth_choice[:25-cr_depth:2]:
            x_idx.append(f"{cr_depth}+{xcr_depth}")
    y_idce = []
    for p in range(len(prompts_choice)):
        y_idx = list(itertools.chain(*costs[p-1]))
        y_idx = [y_idx[i] for i in range(len(y_idx)) if y_idx[i] > 0.]
        y_idce.append(y_idx)
    y_idce = np.array(y_idce)
    for i in range(len(y_idce)):
        plt.bar(x_idx, y_idce[i] if i==0 else y_idce[i]-y_idce[i-1], 
                bottom=0 if i==0 else y_idce[i-1], 
                label=f"{i+1}-context", width=0.9)
    
    plt.legend(loc='best')
    plt.xticks(cr_depth_choice)
    plt.yticks(range(np.max(y_idce).astype(np.int64)+2))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("merge layer index", fontdict={'size': 18})
    plt.ylabel("time / s", fontdict={'size': 18})
    plt.title("Real-time Costs for Different n_prompts & cr_depth & xcr_depth (50 images)", fontdict={'size': 20})
    plt.tight_layout()
    plt.savefig("self_experiments/inference_painter_variant_1/results.jpg")


def prompts_test(test_flop=False):
    args = get_args_parser()
    args = ddp_utils.init_distributed_mode(args)
    print("WARM-UP started")
    args.xcr_depth = 4
    args.cr_depth = 12
    args.num_prompts = 3
    prompts_test_step(args, test_flop)
    print("WARM-UP finished")
    prompts_choice = range(1, 4) # range(1, 3)
    cr_depth_choice = range(6, 25, 2) # range(4, 25)
    xcr_depth_choice = range(25) # range(25)
    real_time_costs = []
    for num_prompts in prompts_choice:
        p_cost = []
        for cr_depth in cr_depth_choice:
            cr_cost = []
            for xcr_depth in xcr_depth_choice:
                if xcr_depth in xcr_depth_choice[:25-cr_depth:2]:
                    args.xcr_depth = xcr_depth
                    args.cr_depth = cr_depth
                    args.num_prompts = num_prompts
                    time = prompts_test_step(args, test_flop)
                    cr_cost.append(time)
                else:
                    cr_cost.append(0.)
            p_cost.append(cr_cost)
        real_time_costs.append(p_cost)
    real_time_costs = np.array(real_time_costs)
    np.save('self_experiments/inference_painter_variant_1/results.npy', real_time_costs)
    print(real_time_costs)
    graphing(real_time_costs, prompts_choice, cr_depth_choice, xcr_depth_choice)          


if __name__ == '__main__':
    prompts_test() # test_flop=True
    
