import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
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
from util.pos_embed import interpolate_pos_embed, interpolate_rel_pos_embed_proto_mo

import models.proto_mo.proto_mo_2_with_ct as painter_variant


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



def eval_nyu_depth_v2(metric_results, args, verbose=False):
    import fnmatch, cv2
    from eval.nyuv2_depth.eval_with_pngs import eval as NYU_Eval
    args.gt_path = "datasets/nyu_depth_v2/official_splits/test/"
    args.min_depth_eval, args.max_depth_eval = 1e-3, 10
    args.eigen_crop, args.garg_crop, args.do_kb_crop =  True, False, False
    
    gt_depths = []
    missing_ids = set()
    pred_filenames, pred_depths = [], []
    for root, _, filenames in os.walk(args.dst_dir):
        for pred_filename in fnmatch.filter(filenames, '*.png'):
            if 'cmap' in pred_filename or 'gt' in pred_filename:
                continue
            dirname = root.replace(args.dst_dir, '')
            pred_filenames.append(os.path.join(dirname, pred_filename))
    
    num_test_samples = len(pred_filenames)
    for i in range(num_test_samples):
        pred_depth_path = os.path.join(args.dst_dir, pred_filenames[i])
        pred_depth = cv2.imread(pred_depth_path, -1)
        if verbose: print(pred_depth_path)
        if pred_depth is None:
            if verbose: print('Missing: %s ' % pred_depth_path)
            missing_ids.add(i)
            continue
        pred_depths.append(pred_depth.astype(np.float32) / 1000.0)
    
    if verbose:
        print('Raw png files reading done')
        print('Evaluating {} files'.format(len(pred_depths)))
    for t_id in range(num_test_samples):
        file_dir = pred_filenames[t_id].split('.')[0]
        filename = file_dir.split('_')[-1]
        gt_depth_path = glob.glob(os.path.join(args.gt_path, '*', 'sync_depth_' + filename + '.png'))[0]
        if verbose: print(gt_depth_path, gt_depth_path[0])
        depth = cv2.imread(gt_depth_path, -1)
        if depth is None:
            print('Missing: %s ' % gt_depth_path)
            missing_ids.add(t_id)
            continue
        depth = depth.astype(np.float32) / 1000.0
        gt_depths.append(depth)

    if verbose:
        print('GT files reading done')
        print('{} GT files missing'.format(len(missing_ids)))
        print('Computing errors')
    
    info = [pred_depths, gt_depths, missing_ids]
    d1, d2, d3, abs_rel, sq_rel, rms, log_rms, silog, log10 = NYU_Eval(args, info)
    results = {'RMSE': rms, 'RMSELog': log_rms, 'SILog': silog, 'Abs.Rel': abs_rel, 'Sq.Rel': sq_rel, \
        'Log10': log10, 'd1': d1, 'd2': d2, 'd3': d3}
    for key in results.keys():
        metric_results[key] = float(results[key])
    return metric_results



def eval_low_light_enhance(metric_results, args, lol_outputs, verbose=False):
    from skimage.metrics import peak_signal_noise_ratio as psnr_loss
    from skimage.metrics import structural_similarity as ssim_loss
    
    gt_path = os.path.join(args.dataset_root, "light_enhance/eval15/high")
    num_pred = len(lol_outputs)
    if verbose: print(f"loading predictions: {num_pred}")
    
    psnr_val_rgb, ssim_val_rgb = [], []
    for rgb_pred, pred_path in lol_outputs:
        file_name = pred_path.split('/')[-1]
        rgb_gt = Image.open(os.path.join(gt_path, file_name)).convert("RGB")
        rgb_gt = np.array(rgb_gt) / 255.
        psnr = psnr_loss(rgb_pred, rgb_gt)
        ssim = ssim_loss(rgb_pred, rgb_gt, channel_axis=-1, data_range=1)
        psnr_val_rgb.append(psnr)
        ssim_val_rgb.append(ssim)
        if verbose: print("PSNR:", psnr, ", SSIM:", ssim, file_name, rgb_pred.shape)
    
    psnr_val_rgb = sum(psnr_val_rgb) / num_pred
    ssim_val_rgb = sum(ssim_val_rgb) / num_pred
    if verbose: print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb))
    
    metric_results["PSNR"] = psnr_val_rgb
    metric_results["SSIM"] = ssim_val_rgb
    return metric_results



def get_args_parser():
    parser = argparse.ArgumentParser('Multi-Dataset Test Portal', add_help=False)
    # inference parameters
    parser.add_argument('--infer', action='store_true', help='evaluate or not')
    parser.add_argument('--test_flops', action='store_true', help='test flops or not')
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt', default='')
    parser.add_argument('--model_name', type=str, help='model name', default='painter_vit_large')
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--batch_size', type=int, default=1)
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
        seed=args.seed,
        datasets=args.dataset_names,
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
        is_infer=True,
        use_cache=args.use_cache,
        is_context_tuning=args.is_context_tuning,
    ).to("cuda")
    assert model.nc % args.batch_size == 0 and model.nc >= args.batch_size, "queue coherence error"
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module
    
    # processing FLOPS
    if args.test_flops:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        dummy_inputs = (torch.randn(args.nci, 2, args.input_size, args.input_size, 3), 
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
    interpolate_rel_pos_embed_proto_mo(model_without_ddp, checkpoint)
    
    msg = model_without_ddp.load_state_dict(checkpoint, strict=False)
    if "vit" in args.ckpt_path:
        model_without_ddp.init_cm_encoder()

    # for k in checkpoint:
    #     print(k)    
    if prints:
        print(msg)
        print("Model Loaded.")
    
    return model, None if not args.test_flops else flops



def get_contexts(args, dataset_name):
    random.seed(args.seed)
    context_set = DatasetTest(args.dataset_root, TRAIN_JSON_BANK[dataset_name], args.img_size, None)
    idce = random.choices(range(len(context_set)), k=6)
    c_query, c_target = [], []
    for idx in idce:
        img, _, tgt, _, _ = context_set[idx]
        c_query.append(img)
        c_target.append(tgt)
    c_query = torch.stack(c_query, dim=0).repeat(args.batch_size, 1, 1, 1, 1)
    c_target = torch.stack(c_target, dim=0).repeat(args.batch_size, 1, 1, 1, 1)
    assert c_query.shape == c_target.shape == (args.batch_size, 6, args.img_size, args.img_size, 3)
    return c_query, c_target



def run_one_batch(type, c_query, c_target, query, model, device):
    target = torch.zeros_like(query)
    valid = torch.ones_like(target)
    mask = torch.ones(model.module.ori_window_size).repeat(valid.shape[0], 1, 1)
    pred = model(type, c_query.float().to(device), c_target.float().to(device), query.float().to(device), 
                       target.float().to(device), mask.float().to(device), valid.float().to(device))
    output = pred.detach().cpu()
    output = output * imagenet_std + imagenet_mean
    assert output.shape[-1] == 3
    return output



def show_image(image, title='test'):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    image = image.detach().cpu()
    image = torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255)
    out_image = Image.fromarray(image.int().numpy().astype(np.uint8))
    out_image.save(f'{title}.jpg')
    return



def test_one_dataset(args, INFO, verbose=False, warm_up=False):
    dataset_dir = args.dataset_root
    dataset_name = args.dataset_name
    device = torch.device("cuda")
    model_name = args.model_name
    img_size = args.img_size
    batch_size = args.batch_size
    output_dir = args.output_dir
    
    print(f"Eval.Dataset: 【{dataset_name}】")
    model = f"{model_name}_patch16_win_dec64_8glb_sl1"
    args.dst_dir = dst_dir = os.path.join(output_dir, f"{dataset_name}")
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
        dataset_val = DatasetTest(dataset_dir, VAL_JSON_BANK[dataset_name], img_size, args.num_val)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        num_val = len(dataset_val)
        data_loader_val = DataLoader(dataset_val, batch_size=batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=ddp_utils.collate_fn, num_workers=2)

        # load context image-pairs as contextst
        type = batch_size * [dataset_name]
        c_query = args.context_base[dataset_name][0][:, :args.nci]
        c_target = args.context_base[dataset_name][1][:, :args.nci]
        assert c_query.shape == c_target.shape == (batch_size, args.nci, img_size, img_size, 3)
        
        # start running inference for metrics
        model_painter.eval()
        if "lol" in dataset_name:   lol_outputs = []
        TIME = 0
        for data in tqdm.tqdm(data_loader_val):
            assert batch_size == len(data)
            query, sizes, out_paths = [], [], []
            for unit_data in data:
                img, img_path, _, _, size = unit_data
                img_name = os.path.basename(img_path)
                out_path = os.path.join(dst_dir, img_name.replace('.jpg', '.png'))
                out_paths.append(out_path)
                query.append(img.unsqueeze(0))
                sizes.append(size)
            query = torch.cat(query, dim=0)
            assert query.shape == (batch_size, img_size, img_size, 3)
            
            TIME -= time.perf_counter()
            output = run_one_batch(type, c_query, c_target, query, model_painter, device)
            TIME += time.perf_counter()
            
            if warm_up: continue
            for r in range(output.shape[0]):
                if "depth" in dataset_name:
                    out = torch.clip(output[r:r+1] * 10000, 0, 10000)
                    out = F.interpolate(out.permute(0, 3, 1, 2), size=sizes[r][::-1], mode='bilinear')
                    out = out.permute(0, 2, 3, 1).squeeze(0).mean(-1).int()
                    out_img = Image.fromarray(out.numpy())
                elif "lol" in dataset_name:
                    out = F.interpolate(output[r:r+1].permute(0, 3, 1, 2), size=sizes[r][::-1], mode='bilinear')
                    out = out.permute(0, 2, 3, 1).squeeze(0)
                    out = torch.clip(out, 0, 1)
                    lol_outputs.append([np.array(out), out_paths[r]])
                    out = out * 255
                    out_img = Image.fromarray(out.numpy().astype(np.uint8))
                else:
                    out = torch.clip(output[r:r+1] * 255, 0, 255)
                    out = F.interpolate(out.permute(0, 3, 1, 2), size=sizes[r][::-1], mode='bilinear')
                    out = out.permute(0, 2, 3, 1).squeeze(0).int()
                    out_img = Image.fromarray(out.numpy().astype(np.uint8))
                out_img.save(out_paths[r])
        
        metric_results['num_val'] = num_val
        metric_results['time'] = TIME
        if args.test_flops: metric_results['flops'] = flops
        if not args.eval and not warm_up:
            result_file = os.path.join(output_dir, "metrics.txt")
            with open(result_file, 'a') as f:
                print(json.dumps(metric_results, indent=4), "\n")
                print(json.dumps(metric_results, indent=4), "\n", file=f)

    # Evaluation Processing
    if args.eval and not warm_up:
        if dataset_name == "ade20k_image2semantic":
            metric_results = eval_ade20k_semseg(metric_results, args, verbose)
        elif dataset_name == "coco_image2panoptic_sem_seg":
            metric_results = eval_coco_pano_semseg(metric_results, args, verbose)
        elif dataset_name == "nyuv2_image2depth":
            metric_results = eval_nyu_depth_v2(metric_results, args, verbose)
        elif dataset_name == "lol_image2enhance":
            metric_results = eval_low_light_enhance(metric_results, args, lol_outputs, verbose)
        elif dataset_name == "derain_image2derain":
            print("Derain Evaluation: Please Run [eval/derain/evaluate_PSNR_SSIM.m] with MATLAB.")
        elif dataset_name == "ssid_image2denoise":
            print("Denoise Evaluation: Please Run [eval/sidd/eval_sidd.m] with MATLAB.")
        
        result_file = os.path.join(output_dir, "metrics.txt")
        if not warm_up:
            with open(result_file, 'a') as f:
                print(json.dumps(metric_results, indent=4), "\n")
                print(json.dumps(metric_results, indent=4), "\n", file=f)

    return None if warm_up else metric_results



def graphing(anchor, metrics, output_dir, num_val):
    # graph the main metrics statistics: miou, time
    for dataset in metrics.keys():
        plt.figure(figsize=(30, 20), dpi=100)
        plt.style.use('fivethirtyeight')
        adata, data = anchor[dataset], metrics[dataset]
        x_labels = np.array(list(adata.keys())+list(data.keys()))
        x_idx = np.arange(len(x_labels))
        y_idce = np.array(list(adata.values())+list(data.values()))
        y_min, y_max = np.min(y_idce), np.max(y_idce)
        # plt.ylim(y_min-0.1*(y_max-y_min), y_max+0.1*(y_max-y_min))
        n, w = y_idce.shape[0], 0.75
        for p in range(len(y_idce)):
            bias_x_idx = x_idx + w*0.5*(2*p-n+1)/n
            plt.bar(bias_x_idx, y_idce[p], bottom=0, label="baseline" if not p else "finetuned", width=w/n)
        
        plt.legend(loc='best', prop={'size': 30})
        plt.rcParams.update({'font.size': 30})
        plt.xticks(x_idx, labels=x_labels, fontsize=27)
        # plt.yticks(np.linspace(y_min-0.1*(y_max-y_min), y_max+0.1*(y_max-y_min), 20), fontsize=22)
        plt.yticks(np.linspace(0, y_max+0.1*(y_max-y_min), 20), fontsize=22)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlabel("cr_depth ; xcr_depth", fontdict={'size': 30})
        plt.ylabel(dataset, fontdict={'size': 30})
        plt.title(f"{dataset.upper()} for [2 context 16:18] val {num_val}", fontdict={'size': 40})
        plt.tight_layout()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f"metrics_{dataset}.jpg"))
        plt.clf()



imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

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



if __name__ == '__main__':
    args = get_args_parser()
    args = ddp_utils.init_distributed_mode(args)
    
    INFO = dict()
    INFO['seed'] = args.seed = 0
    INFO['use_cache'] = args.use_cache = True
    INFO['is_context_tuning'] = args.is_context_tuning = True
    dataset_names = args.dataset_names = [
        "ade20k_image2semantic",
        "coco_image2panoptic_sem_seg",
        "nyuv2_image2depth",
        "lol_image2enhance",
        # "derain_image2derain",
        # "ssid_image2denoise",
    ]
    args.context_base = {dataset_name: list(get_contexts(args, dataset_name)) for dataset_name in dataset_names}
    
    print("Main Test Started:")
    INFO['num_val'] = args.num_val = 50
    INFO['joint_train'] = args.joint_datasets = True
    INFO['train_mask_ratio'] = args.train_mask_ratio = 0.99
    INFO['train_batch_size'] = args.train_batch_size = 128
    
    INFO['skip_query'] = args.skip_query = True
    INFO['use_attn_mean'] = args.use_attn_mean = True
    INFO['use_random_nc'] = args.use_random_nc = False
    INFO['encoder_momentum_weight'] = args.emo = 0.99
    INFO['context_momentum_weight'] = args.cmo = 0
    INFO['query_momentum_weight'] = args.qmo = 1
    
    INFO['train_num_contexts'] = 5
    INFO['num_contexts_used'] = args.nc = INFO['num_contexts_input'] = args.nci = 3
    INFO['cr_depth'] = args.cq = 15
    INFO['p_depth'] = args.p = 1
    INFO['ckpt_path'] = args.ckpt_path = "workbench/train_proto_mo_3/Joint|3:3:15:1:1:0:0.99|ct:0.99/checkpoint-0-32000.pth"
    
    mix_data = "Joint" if args.joint_datasets else "Seperate"
    args.output_dir = os.path.join(args.output_dir, \
        f"{mix_data}|{args.nci}:{args.nc}:{args.cq}:{args.p}:{args.qmo}:{args.cmo}:{args.emo}|ct:{args.train_mask_ratio}")
    if ddp_utils.get_rank() == 0:   os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.txt"), 'a') as f:
        print(json.dumps(INFO, sort_keys=False, indent=4))
        print(json.dumps(INFO, sort_keys=False, indent=4), file=f)
    
    metrics = {name: dict() for name in dataset_names}
    for dataset_name in dataset_names:
        args.dataset_name = dataset_name
        results = test_one_dataset(args, INFO)
        for key in results.keys():
            metrics[dataset_name][key] = results[key]
    
    print("Main Results:")
    print(json.dumps(metrics, sort_keys=False, indent=4), "\n\n")
    with open(os.path.join(args.output_dir, "metrics.txt"), 'a') as f:
        print("Main Results:", file=f)
        print(json.dumps(metrics, sort_keys=False, indent=4), "\n\n", file=f)
    print("Main Test Done!\n")
