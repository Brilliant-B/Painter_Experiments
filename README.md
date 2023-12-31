<div align="center">
<h1>Painter Experiments @Visual In-Context Learning</h1>
Brilliant-B
</div>
<br>

This repo is the modification and experiments of **Painter**. <br>
for installation & data preparation & pretrained preparation, please refer to [Official docs](docs/Official_README.md).

## Models
The modified Painter model is in the directory `$Painter_ROOT/models/` <br>
- **Painter Variant 2**: `painter_variant_2.py` <br>
  add cr_banks and some other modifications <br>
  - `variant_3.py`: attempt to delete encoder layers after xcr_depth, in order to examine the LLM-transferability. <br>
- Painter Variant 1: `painter_variant_1.py` <br>
  add controls of num_contexts/cr_depth/xcr_depth
- Painter Variant 0: `painter_variant_0.py` <br>
  change some of the code structure
- Original Painter: `models_painter.py` <br>


## Training
The model will be trained, mostly finetuned, based on pretrained checkpoints under multiple hyper-parameters. <br>
For new model training experiments, check the directory `self_experiments/finetune` <br>
You can modify and run `train_bash.sh` <br>
- Multi-datasets Training PORTAL: `multi_finetune_portal.py` <br>
  - Modify it in the script. What's more, you can choose to use joint-dataset or seperate-dataset training. <br>
- Hyper-parameter Testing: Finetune for ADE-20K semantic segmentation: `finetune_ade20k_semseg.py` <br>

(more will be issued later) <br>


## Evaluation
For new model evaluation experiments, check the directory `$Painter_ROOT/self_experiments/eval` <br>
You can modify and run `eval_bash.sh` <br>
- Multi-datasets Evaluation PORTAL: `multi_test_portal.py` <br>
  - Modify it in the script, where datasets are evaluated one-by-one and overall metrics will be generated <br>
- Hyper-parameter Testing: Evaluation for ADE-20K semantic segmentation: `test_ade20k_semseg.py` <br>

(more will be issued later) <br>

