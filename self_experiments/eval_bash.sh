# !/bin/bash

set -x

NUM_GPUS=1
JOB_NAME="painter_vit_large"
CKPT_FILE="painter_vit_large.pth"

SIZE=448
MODEL="painter_varient_patch16_win_dec64_8glb_sl1"

CKPT_PATH="models/${JOB_NAME}/${CKPT_FILE}"
DST_DIR="models_inference/${JOB_NAME}/ade20k_semseg_inference_${CKPT_FILE}_${PROMPT}_size${SIZE}"

# inference
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=29504 --use_env \
  self_experiments/inference_multi_tests.py \
  --model ${MODEL} --ckpt_path ${CKPT_PATH} --img_size ${SIZE}

# postprocessing and evaluation
# python self_experiments/eval_multi_tests.py \
#   --pred_dir ${DST_DIR}