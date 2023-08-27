import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1,"
import sys
sys.path.append('.')
sys.path.insert(0, "./")
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm
import time
import random
import json

try:
    np.int
except:
    np.int = np.int32
    np.float = np.float32

import matplotlib.pyplot as plt
from PIL import Image
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from util.ddp_utils import DatasetTest
from util import ddp_utils
from util.pos_embed import interpolate_pos_embed, interpolate_rel_pos_embed
from data.pairdataset_variant import PairDataset

import models.painter_variant_2 as painter_variant

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


def get_args_parser():
    # basic inference parameters
    parser = argparse.ArgumentParser('ADE20k semantic segmentation', add_help=False)
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
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser.parse_args()



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
    from data.ade20k.gen_color_ade20k_sem import define_colors_per_location_mean_sep
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
                print(f"[Hyper] n_context={num_prompts}, cr_depth={cr_depth}, xcr_depth={xcr_depth}, cr_bank={use_cr_bank}, finetune={finetune_code}:\nDataset: {dataset_name}")
                print(f"[Hyper] n_context={num_prompts}, cr_depth={cr_depth}, xcr_depth={xcr_depth}, cr_bank={use_cr_bank}, finetune={finetune_code}:\nDataset: {dataset_name}", file=f)
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
                print(f"[Hyper] n_context={num_prompts}, cr_depth={cr_depth}, xcr_depth={xcr_depth}, cr_bank={use_cr_bank}, finetune={finetune_code}:\nDataset: {dataset_name}")
                print(f"[Hyper] n_context={num_prompts}, cr_depth={cr_depth}, xcr_depth={xcr_depth}, cr_bank={use_cr_bank}, finetune={finetune_code}:\nDataset: {dataset_name}", file=f)
                print(json.dumps(metric_results, indent=4), "\n")
                print(json.dumps(metric_results, indent=4), "\n", file=f)

    return None if warm_up else metric_results



def get_context_set(args, dataset_name):
    random.seed(args.seed)
    context_set = PairDataset(
        root=args.dataset_root, 
        json_path_list=[os.path.join(args.dataset_root, TRAIN_JSON_BANK[dataset_name])], 
        mask_ratio=0.0,
    )
    pairs = context_set.pairs
    typed_pairs = context_set.pair_type_dict
    contexts = [pairs[idx] for idx in random.choices(typed_pairs[dataset_name], k=10)]
    return contexts



def graphing(anchor, metrics, output_dir, num_val):
    # graph the main metrics statistics: miou, time
    for metric in metrics.keys():
        plt.figure(figsize=(30, 20), dpi=100)
        plt.style.use('fivethirtyeight')
        adata, data = anchor[metric], metrics[metric]
        s = list(adata.keys())[0]
        x_labels = np.array(list(adata[s].keys())+list(data[s].keys()))
        x_idx = np.arange(len(x_labels))
        y_idce = np.array([list(adata[p].values())+list(data[p].values()) for p in data.keys()])
        y_min, y_max = np.min(y_idce), np.max(y_idce)
        plt.ylim(y_min-0.1*(y_max-y_min), y_max+0.1*(y_max-y_min))
        n, w= y_idce.shape[0], 0.75
        for p in range(len(y_idce)):
            bias_x_idx = x_idx + w*0.5*(2*p-n+1)/n
            plt.bar(bias_x_idx, y_idce[p], bottom=0, label=f"{p+1}-context", width=w/n)
        # post_y_idce = 0
        # for p in range(len(y_idce)):
        #     plt.bar(x_idx, y_idce[p]-post_y_idce, bottom=post_y_idce, label=f"{p}-context", width=0.9)
        #     post_y_idce = y_idce[p]
        plt.legend(loc='best', prop={'size': 30})
        plt.rcParams.update({'font.size': 30})
        plt.xticks(x_idx, labels=x_labels, fontsize=27)
        plt.yticks(np.linspace(y_min-0.1*(y_max-y_min), y_max+0.1*(y_max-y_min), 20), fontsize=22)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlabel("cr_depth ; xcr_depth", fontdict={'size': 30})
        plt.ylabel(metric, fontdict={'size': 30})
        plt.title(f"{metric.upper()} for Different n_prompts & cr_depth & xcr_depth (val {num_val})", fontdict={'size': 40})
        plt.tight_layout()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f"metrics_{metric}.jpg"))
        plt.clf()



def hyper_param_test(args):
    args.seed = 1
    args.num_val = 50
    args.batch_size = 1
    args.dataset_root = "datasets/"
    
    args.dataset_name = "ade20k_image2semantic" # "coco_image2panoptic_sem_seg"
    contexts = get_context_set(args, args.dataset_name)
    
    num_prompts_choice = [2, 3, 4]
    cr_depth_choice = [9, 12, 15, 18]
    xcr_depth_choice = [9, 12, 15, 18]
    anchor, metrics = dict(), dict()
    args.use_cr_bank = True
    if args.eval:
        anchor['mIoU'] = {pr: dict() for pr in num_prompts_choice}
        metrics['mIoU'] = {pr: dict() for pr in num_prompts_choice}
    if args.infer:
        anchor['time'] = {pr: dict() for pr in num_prompts_choice}
        metrics['time'] = {pr: dict() for pr in num_prompts_choice}
        if args.test_flops:
            anchor['flops'] = {pr: dict() for pr in num_prompts_choice}
            metrics['flops'] = {pr: dict() for pr in num_prompts_choice}
    # print("WARM-UP Started")
    # args.dataset_root = "toy_datasets/"
    # args.use_cr_bank = False
    # args.xcr_depth = 18
    # args.cr_depth = 0
    # args.num_prompts = 3
    # test_step(args, contexts[:args.num_prompts], warm_up=True)
    # print("WARM-UP Done")
    
    # print("Anchor Test: Original Painter Settings")
    # args.use_cr_bank = False
    # args.xcr_depth = 24
    # args.cr_depth = 0
    # args.num_prompts = 1
    # args.finetune_code = None
    # print(args.ckpt_path)
    # anchor_results = test_one_dataset(args, contexts[:args.num_prompts])
    # for metric in anchor.keys():
    #     anchor[metric][1]["Painter"] = anchor_results[metric]
    # for num_prompts in num_prompts_choice:
    #     if num_prompts < 2:
    #         for metric in anchor.keys():
    #             anchor[metric][num_prompts]["Painter"] = 0.
    # print("Anchor Test Done")
    
    print("Hyper-Params Test started")
    for num_prompts in num_prompts_choice:
        for cr_depth in cr_depth_choice:
            for xcr_depth in xcr_depth_choice:
                if cr_depth <= xcr_depth <= 24:
                    args.use_cr_bank = True
                    args.xcr_depth = xcr_depth
                    args.cr_depth = cr_depth
                    args.num_prompts = num_prompts
                    args.finetune_code = None
                    # print(args.ckpt_path)
                    eval_results = test_one_dataset(args, contexts[:args.num_prompts])
                    for metric in metrics.keys():
                        metrics[metric][num_prompts][f"{cr_depth};{xcr_depth}"] = eval_results[metric]
    print("Hyper-Params Test Done")
    graphing(anchor, metrics, args.output_dir, args.num_val)


'''
def finetune_test(args, info):
    args.dataset_name = dataset_name = "coco_image2panoptic_sem_seg"
    contexts = get_context_set(args, dataset_name)
    args.dataset_root = "datasets/"
    
    # original painter
    print("Anchor Test: (Painter)")
    args.use_cr_bank = False
    args.xcr_depth = 24
    args.cr_depth = 0
    args.num_prompts = 1
    args.finetune_code = None
    test_one_dataset(args, contexts[:args.num_prompts], info["num_val"])
    
    # finetuned painter variants
    print("Finetuned Model Test Started")
    for finetune_code in info["finetune_choices"]:
        for num_prompts in info["num_prompts_choices"]:
            for cr_depth in info["cr_depth_choices"]:
                for xcr_depth in info["xcr_depth_choices"]:  
                    args.use_cr_bank = True
                    args.xcr_depth = xcr_depth
                    args.cr_depth = cr_depth
                    args.num_prompts = num_prompts
                    args.finetune_code = finetune_code
                    # args.ckpt_path = f"workbench/train_painter_variant_2/{args.num_prompts}_contexts_{args.cr_depth}_cr_depth_{args.xcr_depth}_xcr_depth_{finetune_code}_finetune_code/checkpoint-0.pth"
                    test_one_dataset(args, contexts[:args.num_prompts], info["num_val"])
                    if info["test_cr_bank"]:
                        args.use_cr_bank = False
                        test_one_dataset(args, contexts[:args.num_prompts], info["num_val"])
    print("Finetuned Model Test Done")
'''


if __name__ == '__main__':
    args = get_args_parser()
    args = ddp_utils.init_distributed_mode(args)
    hyper_param_test(args)
    
    # info = {
    #     "num_prompts_choices": [1, 2],
    #     "cr_depth_choices": [9, 12],
    #     "xcr_depth_choices": [9, 12],
    #     "finetune_choices": [1],
    #     "num_val": 100,
    #     "test_cr_bank": True,
    # }
    # finetune_test(args, info)
