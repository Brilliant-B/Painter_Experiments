# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import os.path
import json
from typing import Any, Callable, List, Optional, Tuple
import random
from PIL import Image
import numpy as np

import torch
from torchvision.datasets.vision import VisionDataset, StandardTransform


class PairDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        json_path_list: list,
        args = None,
        transform: Optional[Callable] = None,
        transform2: Optional[Callable] = None,
        transform3: Optional[Callable] = None,
        transform_seccrop: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        masked_position_generator: Optional[Callable] = None,
        mask_ratio:float = 1.,
        seed = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.pairs = []
        self.weights = []
        type_weight_list = [0.1, 0.2, 0.15, 0.25, 0.2, 0.15, 0.05, 0.05]
        for idx, json_path in enumerate(json_path_list):
            cur_pairs = json.load(open(json_path))
            self.pairs.extend(cur_pairs)
            cur_num = len(cur_pairs)
            self.weights.extend([type_weight_list[idx] * 1./cur_num] * cur_num)
            # print(json_path, type_weight_list[idx])
        
        self.pair_type_dict = {}
        for idx, pair in enumerate(self.pairs):
            if "type" in pair:
                if pair["type"] not in self.pair_type_dict:
                    self.pair_type_dict[pair["type"]] = [idx]
                else: 
                    self.pair_type_dict[pair["type"]].append(idx)
        for t in self.pair_type_dict:
            print(t, len(self.pair_type_dict[t]))
        
        self.seed = seed
        self.transforms = PairStandardTransform(transform, target_transform) if transform is not None else None
        self.transforms2 = PairStandardTransform(transform2, target_transform) if transform2 is not None else None
        self.transforms3 = PairStandardTransform(transform3, target_transform) if transform3 is not None else None
        self.transforms_seccrop = PairStandardTransform(transform_seccrop, target_transform) if transform_seccrop is not None else None
        self.masked_position_generator = masked_position_generator
        if masked_position_generator is not None:
            self.mask_ratio = mask_ratio
            self.ori_window_size = self.masked_position_generator.get_shape()
        
        if args is not None:
            self.img_size = args.img_size
            self.num_prompts = args.num_prompts
            self.cr_depth = args.cr_depth
            self.xcr_depth = args.xcr_depth

    def _load_image(self, path: str) -> Image.Image:
        while True:
            try:
                img = Image.open(os.path.join(self.root, path))
            except OSError as e:
                print(f"Catched exception: {str(e)}. Re-trying...")
                import time
                time.sleep(1)
            else:
                break
        # process for nyuv2 depth: scale to 0~255
        if "sync_depth" in path:
            # nyuv2's depth range is 0~10m
            img = np.array(img) / 10000.
            img = img * 255
            img = Image.fromarray(img)
        img = img.convert("RGB")
        return img

    def get_k_type(self, type: str, k: int):
        # randomly sample the prompt-pairs belonging to the same type
        # if self.seed is not None:   random.seed(self.seed)
        prompt_pairs_index = random.choices(self.pair_type_dict[type], k=k)
        prompt_pairs = [self.pairs[idx] for idx in prompt_pairs_index]
        # print(prompt_pairs)
        image_prompts = [self._load_image(pair['image_path']) for pair in prompt_pairs]
        target_prompts = [self._load_image(pair['target_path']) for pair in prompt_pairs]
        return image_prompts, target_prompts
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pair = self.pairs[index]
        pair_type = pair['type']
        if "depth" in pair_type or "pose" in pair_type:
            interpolation1 = 'bicubic'
            interpolation2 = 'bicubic'
        elif "image2" in pair_type:
            interpolation1 = 'bicubic'
            interpolation2 = 'nearest'
        elif "2image" in pair_type:
            interpolation1 = 'nearest'
            interpolation2 = 'bicubic'
        else:
            interpolation1 = 'bicubic'
            interpolation2 = 'bicubic'
        # no aug for instance segmentation
        if "inst" in pair_type and self.transforms2 is not None:
            cur_transforms = self.transforms2
        elif "pose" in pair_type and self.transforms3 is not None:
            cur_transforms = self.transforms3
        else:
            cur_transforms = self.transforms
        
        image_prompts, target_prompts = self.get_k_type(pair_type, self.num_prompts)
        image_pt, target_pt = [], []
        for image_p, target_p in zip(image_prompts, target_prompts):
            transformed = cur_transforms(image_p, target_p, interpolation1, interpolation2)
            image_pt.append(transformed[0])
            target_pt.append(transformed[1])
        img_p = torch.stack(image_pt, dim=0)
        tgt_p = torch.stack(target_pt, dim=0)
        prompts = torch.stack([img_p, tgt_p], dim=0)
        prompts = torch.einsum('mnchw->mnhwc', prompts)
        assert prompts.shape == (2, self.num_prompts, *self.img_size, 3)        
        # print(pair)
        query = self._load_image(pair['image_path'])
        target = self._load_image(pair['target_path'])
        query, target = cur_transforms(query, target, interpolation1, interpolation2)
        query, target = torch.einsum('chw->hwc', query), torch.einsum('chw->hwc', target)
        assert query.shape == target.shape == (*self.img_size, 3)
        
        # get mask for target training
        if self.masked_position_generator is not None:
            mask = torch.rand(self.masked_position_generator.get_shape()) <= self.mask_ratio
            assert mask.shape == self.ori_window_size
        else:   mask = None
        
        # get valid output pixels standards for each tasks
        valid = torch.ones_like(target)
        imagenet_mean=torch.tensor([0.485, 0.456, 0.406])
        imagenet_std=torch.tensor([0.229, 0.224, 0.225])
        if "nyuv2_image2depth" in pair_type:
            thres = torch.ones(3) * (1e-3 * 0.1)
            thres = (thres - imagenet_mean) / imagenet_std
            valid[target < thres[None, None, :]] = 0
        elif "ade20k_image2semantic" in pair_type:
            thres = torch.ones(3) * (1e-5) # ignore black
            thres = (thres - imagenet_mean) / imagenet_std
            valid[target < thres[None, None, :]] = 0
        elif "coco_image2panoptic_sem_seg" in pair_type:
            thres = torch.ones(3) * (1e-5) # ignore black
            thres = (thres - imagenet_mean) / imagenet_std
            valid[target < thres[None, None, :]] = 0
        elif "image2pose" in pair_type:
            thres = torch.ones(3) * (1e-5) # ignore black
            thres = (thres - imagenet_mean) / imagenet_std
            valid[target > thres[None, None, :]] = 10.0
            fg = target > thres[None, None, :]
            if fg.sum() < 100*3:
                valid = valid * 0.
        elif "image2panoptic_inst" in pair_type:
            thres = torch.ones(3) * (1e-5) # ignore black
            thres = (thres - imagenet_mean) / imagenet_std
            fg = target > thres[None, None, :]
            if fg.sum() < 100*3:
                valid = valid * 0.
        assert valid.shape == (*self.img_size, 3)

        return prompts, query, target, mask, valid

    def __len__(self) -> int:
        return len(self.pairs)


class PairStandardTransform(StandardTransform):
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(transform=transform, target_transform=target_transform)

    def __call__(self, input: Any, target: Any, interpolation1: Any, interpolation2: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input, target = self.transform(input, target, interpolation1, interpolation2)
        return input, target
