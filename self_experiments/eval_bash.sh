# !/bin/bash

set -x

NUM_GPUS=1
JOB_NAME="painter_vit_large"
CKPT_FILE="painter_vit_large.pth"

SIZE=448
MODEL_NAME="painter_varient_1"
CKPT_PATH="pretrained/${JOB_NAME}/${CKPT_FILE}"
OUTPUT_DIR="self_experiments/${MODEL_NAME}"

# inference and post_evaluation
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=29504 --use_env \
  self_experiments/inference_multi_tests.py --infer --eval \
  --model_name ${MODEL_NAME} --ckpt_path ${CKPT_PATH} --output_dir ${OUTPUT_DIR} --img_size ${SIZE}
