import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1,"
import sys
sys.path.append('.')
sys.path.insert(0, "./")

import torch
import torch.nn.functional as F
import numpy as np
try:
    np.int
except:
    np.int = np.int32
    np.float = np.float32
import argparse
import glob
import tqdm
import time
import random
import re
import json
import matplotlib.pyplot as plt
from PIL import Image

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from util.ddp_utils import DatasetTest
from util import ddp_utils

from data.pairdataset_variant import PairDataset

from data.ade20k.gen_color_ade20k_sem import define_colors_per_location_mean_sep
from util.pos_embed import interpolate_pos_embed, interpolate_rel_pos_embed

import models.painter_variant_2 as painter_variant



def eval_coco_pano_semseg(metric_results, args, verbose=False):
    from eval.coco_panoptic.COCOPanoSemSegEvaluatorCustom import SemSegEvaluatorCustom
    from data.coco_semseg.gen_color_coco_panoptic_segm import define_colors_by_mean_sep
    
    # load categories info
    panoptic_coco_categories = 'data/panoptic_coco_categories.json'
    with open(panoptic_coco_categories, 'r') as f:
        categories_list = json.load(f)
    categories = {category['id']: category for category in categories_list}
    
    # define colors (dict of cat_id to color mapper)
    PALETTE_DICT = define_colors_by_mean_sep(num_colors=len(categories))
    PALETTE = [v for k, v in PALETTE_DICT.items()]
    dataset_name = 'coco_2017_val_panoptic_with_sem_seg' # 
    evaluator = SemSegEvaluatorCustom(
        dataset_name,
        distributed=True,
        output_dir=args.output_dir,
        palette=PALETTE,
        pred_dir=args.dst_dir,
        dist_type=args.dist_type,
    )
    prediction_list = glob.glob(os.path.join(args.dst_dir, "*.png"))
    if verbose: print(f"loading predictions: {len(prediction_list)}")
    inputs, outputs = [], []
    for file_name in prediction_list:
        inputs.append({"file_name": file_name})
        outputs.append({"sem_seg": file_name})

    evaluator.reset()
    evaluator.process(inputs, outputs)
    results = evaluator.evaluate()

    if verbose: print(results)
    for key in ['mIoU', 'fwIoU', 'mACC', 'pACC']:
        metric_results[key] = results['sem_seg'][key]
    return metric_results



def eval_ade20k_semseg(metric_results, args, verbose=False):
    from eval.ade20k_semantic.ADE20kSemSegEvaluatorCustom import SemSegEvaluatorCustom
    PALETTE = define_colors_per_location_mean_sep()
    dataset_name = 'ade20k_sem_seg_val'
    evaluator = SemSegEvaluatorCustom(
        dataset_name,
        distributed=True,
        output_dir=args.output_dir,
        palette=PALETTE,
        pred_dir=args.dst_dir,
        dist_type=args.dist_type,
    )
    prediction_list = glob.glob(os.path.join(args.dst_dir, "*.png"))
    if verbose: print(f"loading predictions: {len(prediction_list)}")
    inputs, outputs = [], []
    for file_name in prediction_list:
        inputs.append({"file_name": file_name})
        outputs.append({"sem_seg": file_name})
    
    evaluator.reset()
    evaluator.process(inputs, outputs)
    results = evaluator.evaluate()

    if verbose: print(results)
    for key in ['mIoU', 'fwIoU', 'mACC', 'pACC']:
        metric_results[key] = results['sem_seg'][key]    
    return metric_results



def get_args_parser():
    parser = argparse.ArgumentParser('Multi-Dataset Test Portal', add_help=False)
    # inference parameters
    parser.add_argument('--infer', action='store_true', help='evaluate or not')
    parser.add_argument('--test_flops', action='store_true', help='test flops or not')
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt', default='')
    parser.add_argument('--model_name', type=str, help='model name', default='painter_vit_large')
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--batch_size', type=int, default=2)
    # evaluation parameters
    parser.add_argument('--eval', action='store_true', help='evaluate or not')
    parser.add_argument('--dist_type', type=str, help='color type', default='abs', choices=['abs', 'square', 'mean'])
    parser.add_argument('--output_dir', type=str, help='output_directory', default='self_experiments/painter_vit_large')
    # dataset parameters
    parser.add_argument('--dataset_root', default='datasets/', type=str, help='datasets root path')
    parser.add_argument('--dataset_name', default='ade20k_semseg', type=str, help='one dataset name')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser.parse_args()



def prepare_model(arch='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1', args=None, prints=True):
    # build model
    model = getattr(painter_variant, arch)(
        num_prompts=args.num_prompts, 
        cr_depth=args.cr_depth, 
        xcr_depth=args.xcr_depth,
        use_cr_bank=args.use_cr_bank,
    )
    model.to("cuda")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module
    
    # processing FLOPS
    if args.test_flops:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        dummy_inputs = (torch.randn(1, 2, args.num_prompts + 1, args.input_size, args.input_size, 3), 
                        torch.randn(1, args.input_size, args.input_size, 3),
                        torch.randn(1, args.input_size, args.input_size, 3),
                        torch.randn(1, *model_without_ddp.ori_window_size),
                        torch.randn(1, args.input_size, args.input_size, 3),)
        FLOPS = FlopCountAnalysis(model_without_ddp, dummy_inputs)
        flops = FLOPS.total() / 1000000.0
        if prints:
            print(f"FLOPs: {flops} M")
            print(flop_count_table(model_without_ddp))
    
    # load checkpoint and interpolate
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')['model']
    interpolate_pos_embed(model_without_ddp, checkpoint)
    interpolate_rel_pos_embed(model_without_ddp, checkpoint)
    
    msg = model_without_ddp.load_state_dict(checkpoint, strict=False)
    if prints:  
        print(msg)
        print("Model Loaded.")
    
    return model, None if not args.test_flops else flops



def run_one_batch(prompts, query, model, device):
    target = torch.zeros_like(query)
    valid = torch.ones_like(target)
    mask = torch.ones(model.module.ori_window_size).repeat(valid.shape[0], 1, 1)

    _, pred, _ = model(prompts.float().to(device), query.float().to(device), target.float().to(device), 
                       mask.float().to(device), valid.float().to(device))
    
    output= pred.detach().cpu()
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    assert output.shape[-1] == 3
    return output



def test_one_dataset(args, contexts, verbose=False, warm_up=False):
    dataset_dir = args.dataset_root
    dataset_name = args.dataset_name
    device = torch.device("cuda")
    model_name = args.model_name
    img_size = args.img_size
    batch_size = args.batch_size
    output_dir = args.output_dir
    
    num_prompts = args.num_prompts
    cr_depth = args.cr_depth
    xcr_depth = args.xcr_depth
    finetune_code = args.finetune_code
    use_cr_bank = args.use_cr_bank
    
    
    model = f"{model_name}_patch16_win_dec64_8glb_sl1"
    args.dst_dir = dst_dir = os.path.join(
        output_dir, "{}_contexts_{}_crdepth_{}_xcrdepth/{}".format(num_prompts, cr_depth, xcr_depth, args.dataset_name)
    )
    if ddp_utils.get_rank() == 0:
        if not warm_up and not os.path.exists(dst_dir): os.makedirs(dst_dir)
        if verbose: print("output_dir: {}".format(dst_dir))

    metric_results = dict()
    
    # Inference Processing
    if args.infer:
        model_painter, flops = prepare_model(model, args=args, prints=False)
        device = torch.device("cuda")
        model_painter.to(device)
        
        if dataset_dir == "toy_datasets/":  args.num_val = None
        img_src_dir = dataset_dir + VAL_PATH[dataset_name]
        dataset_val = DatasetTest(img_src_dir, img_size, args.num_val, ext_list=('*.jpg',))
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        data_loader_val = DataLoader(dataset_val, batch_size=batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=ddp_utils.collate_fn, num_workers=2)

        # load context image-pairs as prompts
        prompts = []
        for pairs in contexts:
            imgp = Image.open(os.path.join(args.dataset_root, pairs["image_path"])).convert("RGB")
            imgp = imgp.resize((img_size, img_size))
            imgp = torch.from_numpy(np.array(imgp) / 255.)
            tgtp = Image.open(os.path.join(args.dataset_root, pairs["target_path"]))
            tgtp = tgtp.resize((img_size, img_size))
            tgtp = torch.from_numpy(np.array(tgtp) / 255.)
            prompt = torch.stack([imgp, tgtp], dim=0)
            prompt = (prompt - imagenet_mean) / imagenet_std
            prompts.append(prompt)
        prompts = torch.stack(prompts, dim=1).repeat(batch_size, 1, 1, 1, 1, 1)
        assert prompts.shape == (batch_size, 2, num_prompts, img_size, img_size, 3)
        
        # start running inference for metrics
        model_painter.eval()
        T0 = time.perf_counter()
        
        for data in tqdm.tqdm(data_loader_val):
            assert batch_size == len(data)
            query, sizes, out_paths = [], [], []
            for unit_data in data:
                img, img_path, size = unit_data
                img_name = os.path.basename(img_path)
                out_path = os.path.join(dst_dir, img_name.replace('.jpg', '.png'))
                out_paths.append(out_path)
                query.append((torch.from_numpy(img).unsqueeze(0) - imagenet_mean) / imagenet_std)
                sizes.append(size)
            query = torch.cat(query, dim=0)
            assert query.shape == (batch_size, img_size, img_size, 3)
            
            output = run_one_batch(prompts, query, model_painter, device)
            
            if warm_up: continue
            for r in range(output.shape[0]):
                out = F.interpolate(output[r:r+1].permute(0, 3, 1, 2), size=sizes[r][::-1], mode='bilinear')
                out = out.permute(0, 2, 3, 1).squeeze(0).int()
                out_img = Image.fromarray(out.numpy().astype(np.uint8))
                out_img.save(out_paths[r])
        
        T1 = time.perf_counter()
        TIME = T1 - T0
        
        metric_results['time'] = TIME
        if args.test_flops: metric_results['flops'] = flops
        if not args.eval and not warm_up:
            result_file = os.path.join(dst_dir, "metrics.txt")
            with open(result_file, 'a') as f:
                if not args.use_cr_bank:
                    print("Without CR-Bank:")
                    print("Without CR-Bank:", file=f)
                print(f"[Hyper] n_context={num_prompts}, cr_depth={cr_depth}, xcr_depth={xcr_depth}, cr_bank={use_cr_bank}, finetune={finetune_code}:")
                print(f"[Hyper] n_context={num_prompts}, cr_depth={cr_depth}, xcr_depth={xcr_depth}, cr_bank={use_cr_bank}, finetune={finetune_code}:", file=f)
                print(json.dumps(metric_results, indent=4), "\n")
                print(json.dumps(metric_results, indent=4), "\n", file=f)

    # Evaluation Processing
    if args.eval and not warm_up:
        if dataset_name == "ade20k_image2semantic":
            metric_results = eval_ade20k_semseg(metric_results, args, verbose)
        elif dataset_name == "coco_image2panoptic_sem_seg":
            metric_results = eval_coco_pano_semseg(metric_results, args, verbose)
        
        result_file = os.path.join(output_dir, "metrics.txt")
        if not warm_up:
            with open(result_file, 'a') as f:
                if not args.use_cr_bank:
                    print("Without CR-Bank:")
                    print("Without CR-Bank:", file=f)
                print(f"[Hyper] n_context={num_prompts}, cr_depth={cr_depth}, xcr_depth={xcr_depth}, cr_bank={use_cr_bank}, finetune={finetune_code}:")
                print(f"[Hyper] n_context={num_prompts}, cr_depth={cr_depth}, xcr_depth={xcr_depth}, cr_bank={use_cr_bank}, finetune={finetune_code}:", file=f)
                print(json.dumps(metric_results, indent=4), "\n")
                print(json.dumps(metric_results, indent=4), "\n", file=f)

    return None if warm_up else metric_results



def get_context_set(args, dataset_names):
    random.seed(args.seed)
    context_set = PairDataset(
        root=args.dataset_root, 
        json_path_list=[os.path.join(args.dataset_root, TRAIN_JSON_BANK[name]) for name in dataset_names], 
        mask_ratio=0.0,
    )
    pairs = context_set.pairs
    typed_pairs = context_set.pair_type_dict
    contexts = {name: [pairs[idx] for idx in random.choices(typed_pairs[name], k=10)] for name in dataset_names}
    return contexts



imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

TRAIN_JSON_BANK = {
    "ade20k_image2semantic": "ade20k/ade20k_training_image_semantic.json",
    "coco_image2panoptic_sem_seg": "coco/pano_sem_seg/coco_train2017_image_panoptic_sem_seg.json",
    "denoise": "denoise/denoise_ssid_train.json",
    "derain": "derain/derain_train.json",
    "light_enhance": "light_enhance/enhance_lol_train.json",
    "nyu_depth_v2": "nyu_depth_v2/nyuv2_sync_image_depth.json",
    "coco_pano_inst": "coco/pano_ca_inst/coco_train_image_panoptic_inst.json",
    "coco_pose": "coco_pose/coco_pose_256x192_train.json",
}

VAL_PATH = {
    "ade20k_image2semantic": "ade20k/images/validation",
    "coco_image2panoptic_sem_seg": "coco/val2017",
}

if __name__ == '__main__':
    args = get_args_parser()
    args = ddp_utils.init_distributed_mode(args)
    
    args.seed = 0
    dataset_names = [
        "ade20k_image2semantic",
        "coco_image2panoptic_sem_seg",
    ]
    contexts = get_context_set(args, dataset_names)
    
    print("Anchor Test Started: Original Painter")
    args.num_prompts = 1
    args.cr_depth = 0
    args.xcr_depth = 24
    args.finetune_code = None
    args.use_cr_bank = False
    args.num_val = 10
    anchor = {name: dict() for name in dataset_names}
    for dataset_name in dataset_names:
        args.dataset_name = dataset_name
        d_context = contexts[dataset_name][:args.num_prompts]
        results = test_one_dataset(args, d_context)
        for key in results.keys():
            anchor[dataset_name][key] = results[key]
    print("Anchor Results:")
    print(json.dumps(anchor, sort_keys=False, indent=4), "\n\n")
    with open(os.path.join(args.output_dir, "metrics.txt"), 'a') as f:
        print("Anchor Results:", file=f)
        print(json.dumps(anchor, sort_keys=False, indent=4), "\n\n", file=f)
    print("Anchor Test Done!\n")
    
    print("Main Test Started:")
    args.num_prompts = 2
    args.cr_depth = 9
    args.xcr_depth = 12
    args.finetune_code = 1
    args.use_cr_bank = True
    args.num_val = 10
    args.ckpt_path = "pretrained/painter_vit_large/painter_vit_large.pth"
    # args.ckpt_path = os.path.join(
    #     f"workbench/train_{args.model_name}", \
    #     f"{args.num_prompts}_contexts_{args.cr_depth}_cr_depth_{args.xcr_depth}_xcr_depth_{args.finetune_code}_finetune_code", \
    #     "1000.pth"
    # )
    metrics = {name: dict() for name in dataset_names}
    for dataset_name in dataset_names:
        args.dataset_name = dataset_name
        d_context = contexts[dataset_name][:args.num_prompts]
        results = test_one_dataset(args, d_context)
        for key in results.keys():
            metrics[dataset_name][key] = results[key]
    
    print("Main Results:")
    print(json.dumps(anchor, sort_keys=False, indent=4), "\n\n")
    with open(os.path.join(args.output_dir, "metrics.txt"), 'a') as f:
        print("Main Results:", file=f)
        print(json.dumps(anchor, sort_keys=False, indent=4), "\n\n", file=f)
    print("Main Test Done!\n")
