# !/bin/bash

set -x

NUM_GPUS=1

SIZE=448
MODEL_NAME="painter_variant_2"
CKPT_PATH="pretrained/painter_vit_large/painter_vit_large.pth"
OUTPUT_DIR="workbench/eval_${MODEL_NAME}"

# inference and post_evaluation
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=29504 --use_env \
  self_experiments/eval/hyper_test_portal.py --infer --eval \
  --model_name ${MODEL_NAME} --ckpt_path ${CKPT_PATH} --output_dir ${OUTPUT_DIR} --img_size ${SIZE}
