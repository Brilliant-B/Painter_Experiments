#!/bin/bash

NUM_GPUS=1
DATA_PATH=datasets
name=painter_variant_2
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=12358 \
	--use_env self_experiments/finetune/multi_finetune_portal.py \
    --batch_size 2 \
    --accum_iter 16 \
    --model_name $name \
    --model ${name}_patch16_win_dec64_8glb_sl1 \
    --max_mask_patches_per_block 392 \
    --epochs 15 \
    --warmup_epochs 1 \
    --lr 1e-3 \
    --clip_grad 3 \
    --layer_decay 0.8 \
    --drop_path 0.1 \
    --img_size 448 448 \
    --save_freq 1 \
    --data_path $DATA_PATH/ \
    --json_path \
    $DATA_PATH/ade20k/ade20k_training_image_semantic.json \
    --val_json_path \
    $DATA_PATH/ade20k/ade20k_validation_image_semantic.json \
    --base_output_dir workbench/train_$name \
    --log_dir workbench/train_$name/tensorboard_logs \
    --finetune pretrained/painter_vit_large/painter_vit_large.pth \


    # --json_path \
    # $DATA_PATH/nyu_depth_v2/nyuv2_sync_image_depth.json \
    # $DATA_PATH/ade20k/ade20k_training_image_semantic.json \
    # $DATA_PATH/coco/pano_ca_inst/coco_train_image_panoptic_inst.json \
    # $DATA_PATH/coco/pano_sem_seg/coco_train2017_image_panoptic_sem_seg.json \
    # $DATA_PATH/coco_pose/coco_pose_256x192_train.json \
    # $DATA_PATH/denoise/denoise_ssid_train.json \
    # $DATA_PATH/derain/derain_train.json \
    # $DATA_PATH/light_enhance/enhance_lol_train.json \
    # --val_json_path \
    # $DATA_PATH/nyu_depth_v2/nyuv2_test_image_depth.json \
    # $DATA_PATH/ade20k/ade20k_validation_image_semantic.json \
    # $DATA_PATH/coco/pano_ca_inst/coco_val_image_panoptic_inst.json \
    # $DATA_PATH/coco/pano_sem_seg/coco_val2017_image_panoptic_sem_seg.json \
    # $DATA_PATH/coco_pose/coco_pose_256x192_val.json \
    # $DATA_PATH/denoise/denoise_ssid_val.json \
    # $DATA_PATH/derain/derain_test_rain100h.json \
    # $DATA_PATH/light_enhance/enhance_lol_val.json \
