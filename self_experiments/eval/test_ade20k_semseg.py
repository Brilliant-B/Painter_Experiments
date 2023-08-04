# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1,"
import sys
sys.path.append('.')
sys.path.insert(0, "./")
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

try:
    np.int
except:
    np.int = np.int32
    np.float = np.float32

import matplotlib.pyplot as plt
from PIL import Image
from detectron2.evaluation import SemSegEvaluator
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from util.ddp_utils import DatasetTest
from util import ddp_utils
from util.pos_embed import interpolate_rel_pos_embed
import models.painter_variant_1 as painter_variant_1


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def get_args_parser():
    # basic inference parameters
    parser = argparse.ArgumentParser('ADE20k semantic segmentation', add_help=False)
    parser.add_argument('--infer', action='store_true', help='evaluate or not')
    parser.add_argument('--test_flops', action='store_true', help='test flops or not')
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt', default='')
    parser.add_argument('--model_name', type=str, help='model name', default='painter_vit_large')
    parser.add_argument('--img_size', type=int, default=448)    
    # evaluation parameters
    parser.add_argument('--eval', action='store_true', help='evaluate or not')
    parser.add_argument('--dist_type', type=str, help='color type', default='abs', choices=['abs', 'square', 'mean'])
    parser.add_argument('--output_dir', type=str, help='output_directory', default='self_experiments/painter_vit_large')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser.parse_args()


class SemSegEvaluatorCustom(SemSegEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        palette=None,
        pred_dir=None,
        dist_type=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
        """
        super().__init__(
            dataset_name=dataset_name,
            distributed=distributed,
            output_dir=output_dir,
        )
        # update source names
        self.input_file_to_gt_file_custom = {}
        for src_file, tgt_file in self.input_file_to_gt_file.items():
            assert os.path.basename(src_file).replace('.jpg', '.png') == os.path.basename(tgt_file)
            src_file_custom = os.path.join(pred_dir, os.path.basename(tgt_file))  # output is saved as png
            self.input_file_to_gt_file_custom[src_file_custom] = tgt_file
        color_to_idx = {}
        for cls_idx, color in enumerate(palette):
            color = tuple(color)
            # in ade20k, foreground index starts from 1
            color_to_idx[color] = cls_idx + 1
        self.color_to_idx = color_to_idx
        self.palette = torch.tensor(palette, dtype=torch.float, device="cuda")  # (num_cls, 3)
        self.pred_dir = pred_dir
        self.dist_type = dist_type

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        # print("processing")
        for input in tqdm.tqdm(inputs):
            # output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)  # chw --> hw
            output = input["file_name"]
            output = np.array(Image.open(output)) # (H, W, 3)
            pred = self.post_process_segm_output(output) # (H, W)
            # use custom input_file_to_gt_file mapping
            gt_filename = self.input_file_to_gt_file_custom[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=np.int)
            gt[gt==self._ignore_label] = self._num_classes
            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)
            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt.astype(np.uint8))
                b_pred = self._mask_to_boundary(pred.astype(np.uint8))
                self._b_conf_matrix += np.bincount(
                    (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)
            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def post_process_segm_output(self, segm):
        """
        Post-processing to turn output segm image to class index map
        Args: segm: (H, W, 3)
        Returns: class_map: (H, W)
        """
        segm = torch.from_numpy(segm).float().to(self.palette.device)  # (h, w, 3)
        # pred = torch.einsum("hwc, kc -> hwk", segm, self.palette)  # (h, w, num_cls)
        h, w, k = segm.shape[0], segm.shape[1], self.palette.shape[0]
        if self.dist_type == 'abs':
            dist = torch.abs(segm.view(h, w, 1, 3) - self.palette.view(1, 1, k, 3))  # (h, w, k)
        elif self.dist_type == 'square':
            dist = torch.pow(segm.view(h, w, 1, 3) - self.palette.view(1, 1, k, 3), 2)  # (h, w, k)
        elif self.dist_type == 'mean':
            dist_abs = torch.abs(segm.view(h, w, 1, 3) - self.palette.view(1, 1, k, 3))  # (h, w, k)
            dist_square = torch.pow(segm.view(h, w, 1, 3) - self.palette.view(1, 1, k, 3), 2)  # (h, w, k)
            dist = (dist_abs + dist_square) / 2.
        else:
            raise NotImplementedError
        dist = torch.sum(dist, dim=-1)
        pred = dist.argmin(dim=-1).cpu()  # (H, W)
        pred = np.array(pred, dtype=np.int)
        return pred


def prepare_model(arch='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1', args=None, prints=True):
    # build model
    print(f'n_contexts: {args.num_prompts}, cr-depth: {args.cr_depth}, xcr_depth: {args.xcr_depth}')
    model = getattr(painter_variant_1, arch)(num_prompts=args.num_prompts, cr_depth=args.cr_depth, xcr_depth=args.xcr_depth)    
    
    if args.test_flops:
        from thop import profile
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        dummy_inputs = (torch.randn(1, 3, args.input_size * (args.num_prompts + 1), args.input_size), 
                        torch.randn(1, 3, args.input_size * (args.num_prompts + 1), args.input_size),
                        torch.randn(1, 3, args.input_size * (args.num_prompts + 1), args.input_size),)
        FLOPS = FlopCountAnalysis(model, dummy_inputs)
        flops = FLOPS.total() / 1000000.0
        print(f"FLOPs: {flops} M")
        if prints:
            print(flop_count_table(model))
        # flops, params = profile(model, dummy_inputs)
        # print('FLOPs: %.2f M, params: %.2f M' % (flops, params / 1000000.0))
    model.to("cuda")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module
    
    # load model
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')['model']
    interpolate_rel_pos_embed(model_without_ddp, checkpoint)
        
    msg = model_without_ddp.load_state_dict(checkpoint, strict=False)
    if prints:
        print(msg)
        print("Model Loaded.")
    return model, None if not args.test_flops else flops


def run_one_image(prompts, query, img_size, model, device):
    target = torch.zeros_like(query)
    mask = torch.ones(model.module.ori_window_size).unsqueeze(0)
    valid = torch.ones_like(target)
    # valid for ade20k
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
    imagenet_std = torch.tensor([0.229, 0.224, 0.225])
    thres = torch.ones(3) * (1e-5) # ignore black
    thres = (thres - imagenet_mean) / imagenet_std
    valid[target < thres[None, None, :]] = 0
    valid = valid.float()

    _, pred, _ = model(prompts.float().to(device), query.float().to(device), target.float().to(device), 
                       mask.float().to(device), valid.float().to(device))
    output= pred.detach().cpu()
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    output = F.interpolate(output.permute(0, 3, 1, 2), size=img_size[::-1],
        mode='bilinear').permute(0, 2, 3, 1)[0]
    assert output.shape[-1] == 3
    output = output.int()
    return output


def test_step(args, contexts, num_val=2000, warm_up=False):
    dataset_dir = args.dataset_root
    device = torch.device("cuda")
    ckpt_path = args.ckpt_path
    model_name = args.model_name
    img_size = args.img_size
    output_dir = args.output_dir
    
    num_prompts = args.num_prompts
    cr_depth = args.cr_depth
    xcr_depth = args.xcr_depth
    
    path_splits = ckpt_path.split('/')
    ckpt_dir, ckpt_file = path_splits[-2], path_splits[-1]
    model = f"{model_name}_patch16_win_dec64_8glb_sl1"
    dst_dir = os.path.join(output_dir, "{}_contexts_{}_crdepth_{}_xcrdepth_ade20k_{}_semseg__inference".format(
        num_prompts, cr_depth, xcr_depth, img_size))

    if ddp_utils.get_rank() == 0:
        if not warm_up and not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        # print("output_dir: {}".format(dst_dir))

    copy_paste_results = dict()
    
    if args.infer:
        model_painter, flops = prepare_model(model, args=args, prints=False)
        device = torch.device("cuda")
        model_painter.to(device)
        
        if dataset_dir == "toy_datasets/":
            num_val = None
        img_src_dir = dataset_dir + "ade20k/images/validation"
        # img_path_list = glob.glob(os.path.join(img_src_dir, "*.jpg"))
        dataset_val = DatasetTest(img_src_dir, img_size, num_val, ext_list=('*.jpg',))
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        data_loader_val = DataLoader(dataset_val, batch_size=1, sampler=sampler_val,
                                    drop_last=False, collate_fn=ddp_utils.collate_fn, num_workers=2)

        imgp_paths = [f"datasets/ade20k/images/training/{os.path.splitext(os.path.split(context)[-1])[0]}.jpg" for context in contexts]
        tgtp_paths = [f"datasets/ade20k/annotations_with_color/training/{os.path.splitext(os.path.split(context)[-1])[0]}.png" for context in contexts]

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
        prompts = torch.stack(prompts, dim=1).unsqueeze(0)
        assert prompts.shape == (1, 2, num_prompts, img_size, img_size, 3)
        
        T0 = time.perf_counter()
        model_painter.eval()
        for data in tqdm.tqdm(data_loader_val):
            """ Load an image """
            assert len(data) == 1
            img, img_path, size = data[0]
            img_name = os.path.basename(img_path)
            out_path = os.path.join(dst_dir, img_name.replace('.jpg', '.png'))
            query = torch.from_numpy(img).unsqueeze(0)
            query = (query - imagenet_mean) / imagenet_std
            assert query.shape == (1, img_size, img_size, 3)
            output = run_one_image(prompts, query, size, model_painter, device)
            if not warm_up:
                output = Image.fromarray(output.numpy().astype(np.uint8))
                output.save(out_path)
        T1 = time.perf_counter()
        TIME = T1 - T0
        
        copy_paste_results['time'] = TIME
        if args.test_flops:
            copy_paste_results['flops'] = flops
    
    if args.eval and not warm_up:
        from data.ade20k.gen_color_ade20k_sem import define_colors_per_location_mean_sep
        PALETTE = define_colors_per_location_mean_sep()
        dataset_name = 'ade20k_sem_seg_val'
        evaluator = SemSegEvaluatorCustom(
            dataset_name,
            distributed=True,
            output_dir=output_dir,
            palette=PALETTE,
            pred_dir=dst_dir,
            dist_type=args.dist_type,
        )
        inputs, outputs = [], []
        prediction_list = glob.glob(os.path.join(dst_dir, "*.png"))
        # print(f"loading predictions: {len(prediction_list)}")
        for file_name in prediction_list:
            # keys in input: "file_name", keys in output: "sem_seg"
            inputs.append({"file_name": file_name})
            outputs.append({"sem_seg": file_name})
        
        evaluator.reset()
        evaluator.process(inputs, outputs)
        results = evaluator.evaluate()
        
        for key in ['mIoU', 'fwIoU', 'mACC', 'pACC']:
            copy_paste_results[key] = results['sem_seg'][key]
        print(copy_paste_results)
        
        result_file = os.path.join(output_dir, "metrics.txt")
        with open(result_file, 'a') as f:
            print(f"[Hyper] n_context={num_prompts}, cr_depth={cr_depth}, xcr_depth={xcr_depth}:", file=f)
            print(copy_paste_results, file=f)
            print("", file=f)

    return None if warm_up else copy_paste_results


def graphing(anchor, metrics, output_dir):
    # graph the main metrics statistics: miou, time
    for metric in metrics.keys():
        plt.figure(figsize=(50, 30), dpi=100)
        plt.style.use('fivethirtyeight')
        adata, data = anchor[metric], metrics[metric]
        x_labels = np.array(list(adata[2].keys())+list(data[2].keys()))
        x_idx = np.arange(len(x_labels))
        y_idce = np.array([list(adata[p].values())+list(data[p].values()) for p in data.keys()])
        n = y_idce.shape[0]
        for p in range(len(y_idce)):
            bias_x_idx = x_idx + 0.9*0.5*(2*p-n+1)/n
            plt.bar(bias_x_idx, y_idce[p], bottom=0, label=f"{p+1}-context", width=0.9/n)
        # post_y_idce = 0
        # for p in range(len(y_idce)):
        #     plt.bar(x_idx, y_idce[p]-post_y_idce, bottom=post_y_idce, label=f"{p}-context", width=0.9)
        #     post_y_idce = y_idce[p]
        plt.legend(loc='best')
        plt.xticks(x_idx, labels=x_labels)
        plt.yticks(np.linspace(0, np.max(y_idce).astype(np.int64)+2, 30))
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlabel("cr_depth ; xcr_depth", fontdict={'size': 25})
        plt.ylabel(metric, fontdict={'size': 25})
        plt.title(f"{metric.upper()} for Different n_prompts & cr_depth & xcr_depth", fontdict={'size': 30})
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"metrics_{metric}.jpg"))
        plt.clf()


PROMPTS_BANK = glob.glob(os.path.join("datasets/ade20k/images/training", "*.jpg"))
# ['ADE_train_00017885', 'ADE_train_00016499', 'ADE_train_00014165']


def test():
    args = get_args_parser()
    args = ddp_utils.init_distributed_mode(args) 
    
    random.seed(0)
    contexts = random.sample(PROMPTS_BANK, k = 6)
    num_prompts_choice = [1, 2, 3]
    cr_depth_choice = [4]
    xcr_depth_choice = [17, 16, 15, 14, 13, 12]
    anchor, metrics = dict(), dict()
    if args.eval:
        anchor['mIoU'] = {pr: dict() for pr in num_prompts_choice}
        metrics['mIoU'] = {pr: dict() for pr in num_prompts_choice}
    if args.infer:
        anchor['time'] = {pr: dict() for pr in num_prompts_choice}
        metrics['time'] = {pr: dict() for pr in num_prompts_choice}
        if args.test_flops:
            anchor['flops'] = {pr: dict() for pr in num_prompts_choice}
            metrics['flops'] = {pr: dict() for pr in num_prompts_choice}
    
    print("WARM-UP Started")
    args.dataset_root = "toy_datasets/"
    args.xcr_depth = 19
    args.cr_depth = 0
    args.num_prompts = 3
    test_step(args, contexts[:args.num_prompts], warm_up=True)
    print("WARM-UP Done")
    
    print("Anchor Test: Original Painter Settings")
    args.dataset_root = "datasets/"
    for num_prompts in num_prompts_choice:
        if num_prompts < 3:
            args.xcr_depth = 24
            args.cr_depth = 0
            args.num_prompts = num_prompts
            anchor_results = test_step(args, contexts[:args.num_prompts], 2000)
            for metric in anchor.keys():
                anchor[metric][num_prompts]["0;24 Ori"] = anchor_results[metric]
        else:
            for metric in anchor.keys():
                anchor[metric][num_prompts]["0;24 Ori"] = 0.
    print("Anchor Test Done")
    
    print("Main Test started")
    args.dataset_root = "datasets/"
    for num_prompts in num_prompts_choice:
        for cr_depth in cr_depth_choice:
            for xcr_depth in xcr_depth_choice:
                if cr_depth <= xcr_depth <= 24:
                    args.xcr_depth = xcr_depth
                    args.cr_depth = cr_depth
                    args.num_prompts = num_prompts
                    eval_results = test_step(args, contexts[:args.num_prompts], 2000)
                    for metric in metrics.keys():
                        metrics[metric][num_prompts][f"{cr_depth};{xcr_depth}"] = eval_results[metric]

    print("Main Test Done")
    graphing(anchor, metrics, args.output_dir)


if __name__ == '__main__':
    test()
    
