#!/bin/bash

NUM_GPUS=1
DATA_PATH=datasets
name=proto_mo_5
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=12358 \
	--use_env self_experiments/proto_mo/finetune/multi_finetune_portal.py \
    --batch_size 2 \
    --accum_iter 32 \
    --model_name $name \
    --model ${name}_patch16_win_dec64_8glb_sl1 \
    --epochs 3 \
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
