# !/bin/bash

set -x
DST_DIR="workbench/eval_painter_variant_2/2_contexts_16_crdepth_18_xcrdepth/nyuv2_image2depth/"

python eval/nyuv2_depth/eval_with_pngs.py \
  --pred_path ${DST_DIR} \
  --gt_path datasets/nyu_depth_v2/official_splits/test/ \
  --dataset nyu --min_depth_eval 1e-3 --max_depth_eval 10 --eigen_crop
