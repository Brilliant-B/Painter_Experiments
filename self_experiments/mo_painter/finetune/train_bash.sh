#!/bin/bash

NUM_GPUS=1
DATA_PATH=datasets
name=mo_painter_1
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=12358 \
	--use_env self_experiments/mo_painter/finetune/multi_finetune_portal.py \
    --batch_size 2 \
    --accum_iter 32 \
    --model_name $name \
    --model ${name}_patch16_win_dec64_8glb_sl1 \
    --max_mask_patches_per_block 392 \
    --epochs 1 \
    --warmup_itrs 1000 \
    --lr 5e-4 \
    --clip_grad 3 \
    --layer_decay 0.8 \
    --drop_path 0.1 \
    --img_size 448 448 \
    --data_path $DATA_PATH/ \
    --base_output_dir workbench/train_$name \
    --log_dir workbench/train_$name/tensorboard_logs \
    --finetune pretrained/painter_vit_large/painter_vit_large.pth \
