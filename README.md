<div align="center">
<h1>Painter Experiments @Visual In-Context Learning</h1>
Brilliant-B
</div>
<br>

This repo is the modification and experiments of **Painter**. <br>
for installation & data preparation & pretrained preparation, please refer to [Official docs](docs/Official_README.md).

## Models
The modified Painter model is in the directory `$Painter_ROOT/models/` <br>
- **Painter Variant 1**: `painter_variant_1.py`
- Painter Variant 0: `painter_variant_0.py`
- Original Painter: `models_painter.py`


## Training
The model will be trained, mostly finetuned, based on pretrained checkpoints under multiple hyper-parameters. <br>
For new model training experiments, check the directory `self_experiments/finetune` <br>
You can modify and run `train_bash.sh`
- Finetune for ADE-20K semantic segmentation: `finetune_ade20k_semseg.py` <br>

(more will be issued later) <br>


## Evaluation
For new model evaluation experiments, check the directory `$Painter_ROOT/self_experiments/eval` <br>
You can modify and run `eval_bash.sh`
- Evaluation for ADE-20K semantic segmentation: `test_ade20k_semseg.py` <br>

(more will be issued later) <br>

