import os
import glob
from PIL import Image
import numpy as np
import json

import torch
from torch.utils.data import Dataset
import torch.distributed as dist

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


class DatasetTest(Dataset):
    def __init__(self, dataset_root, json_path, input_size, num_val=None):
        super(DatasetTest, self).__init__()
        self.dataset_root = dataset_root
        self.json_path = os.path.join(dataset_root, json_path)
        self.input_size = input_size
        val_pairs = json.load(open(self.json_path))
        self.img_paths = list(val_pairs[:num_val])
        # self.root = os.path.commonprefix(val_pairs)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = os.path.join(self.dataset_root, self.img_paths[index]["image_path"])
        img = Image.open(img_path)
        ori_size = list(img.size)
        img = img.convert("RGB").resize((self.input_size, self.input_size))
        img = torch.from_numpy((np.array(img) / 255. - imagenet_mean) / imagenet_std)
        
        tgt_path = os.path.join(self.dataset_root, self.img_paths[index]["target_path"])
        tgt = Image.open(tgt_path)
        if "depth" in self.json_path:
            tgt = np.array(tgt) * 225 / 10000.
            tgt = Image.fromarray(tgt)
        tgt = tgt.convert("RGB").resize((self.input_size, self.input_size))
        tgt = torch.from_numpy((np.array(tgt) / 255. - imagenet_mean) / imagenet_std)
        
        assert img.shape == tgt.shape
        return img, img_path, tgt, tgt_path, ori_size


class DatasetTest_Ori(Dataset):
    """
    define dataset for ddp
    """
    def __init__(self, img_src_dir, input_size, num_val=None, ext_list=('*.png', '*.jpg'), ):
        super(DatasetTest_Ori, self).__init__()
        self.img_src_dir = img_src_dir
        self.input_size = input_size

        img_path_list = []
        for ext in ext_list:
            if num_val is not None:
                img_path_tmp = glob.glob(os.path.join(img_src_dir, ext))[:num_val]
            else:
                img_path_tmp = glob.glob(os.path.join(img_src_dir, ext))
            img_path_list.extend(img_path_tmp)
        self.img_path_list = img_path_list

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = Image.open(img_path).convert("RGB")
        size_org = img.size
        img = img.resize((self.input_size, self.input_size))
        img = np.array(img) / 255.
        return img, img_path, size_org


def collate_fn(batch):
    return batch
    # batch = list(zip(*batch))
    # return tuple(batch)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return args

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

    return args
