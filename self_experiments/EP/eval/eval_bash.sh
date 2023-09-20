# !/bin/bash
set -x

GPUS=1
MODEL_NAME="EP_0"
CKPT_PATH="pretrained/painter_vit_large/painter_vit_large.pth"
OUTPUT_DIR="workbench/eval_${MODEL_NAME}"

# inference and post_evaluation
python -m torch.distributed.launch --nproc_per_node=${GPUS} --master_port=29504 --use_env \
  self_experiments/EP/eval/multi_test_portal.py --infer --eval \
  --model_name ${MODEL_NAME} --ckpt_path ${CKPT_PATH} --output_dir ${OUTPUT_DIR}
