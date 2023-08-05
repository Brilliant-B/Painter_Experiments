#!/bin/bash

NUM_GPUS=1
DATA_PATH=datasets
name=painter_variant_2
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=12358 \
	--use_env self_experiments/finetune/finetune_ade20k_semseg.py \
    --batch_size 2 \
    --accum_iter 16  \
    --model painter_varient_2_patch16_win_dec64_8glb_sl1 \
    --max_mask_patches_per_block 392 \
    --epochs 2 \
    --warmup_epochs 1 \
    --lr 1e-3 \
    --clip_grad 3 \
    --layer_decay 0.8 \
    --drop_path 0.1 \
    --img_size 448 448 \
    --save_freq 1 \
    --data_path $DATA_PATH/ \
    --json_path  \
    $DATA_PATH/ade20k/ade20k_training_image_semantic.json \
    --val_json_path \
    $DATA_PATH/ade20k/ade20k_validation_image_semantic.json \
    --base_output_dir workbench/train_$name \
    --log_dir workbench/train_$name/tensorboard_logs \
    --finetune pretrained/painter_vit_large/painter_vit_large.pth \
