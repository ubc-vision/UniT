# UniT: Unified Knowledge Transfer for Any-shot Object Detection and Segmentation

This repository contains the code for the CVPR 2021 paper titled [**"UniT: Unified Knowledge Transfer for Any-shot Object Detection and Segmentation"**](https://arxiv.org/pdf/2006.07502.pdf).

## Requirements
To setup the environment with all the required dependencies, follow the steps detailed in [INSTALL.md](https://github.com/ubc-vision/UniT/blob/main/INSTALL.md). 

## Prepare Dataset
- To obtain the data and curate the splits for PASCAL VOC
```python
python data/prepare_voc.py --DATA-ROOT "Path/to/Save/Location/"

```
**Note**: These splits are the same as the ones released by the authors of [Few-shot Object Detection via Feature Reweighting](https://github.com/bingykang/Fewshot_Detection).


## Model Training
Download ImageNet pretrained models from [this Google drive](https://drive.google.com/drive/folders/1plLDI55qKvwPa5OuT_DcGobdnAPqBfq1?usp=sharing), and place them in the `models/` folder in the root directory. The directory structure should look like this:
```
UniT
└── models
    └── resnet_50_MSRA_C4.pth
    └── resnet_101_MSRA_C4.pth
    ...
└── modeling
└── scripts
└── solver
...
```

### Base Training
- Training on VOC
```python
python scripts/train_VOC.py --config-file "configs/VOC/VOC-RCNN-101-C4-split{num}.yaml" --num-gpus 4 --resume SOLVER.IMS_PER_BATCH 8 TEST.AUG.ENABLED False SOLVER.BASE_LR 0.02

```

### Fine Tuning
- Fine Tuning on VOC
```python
python scripts/finetune_VOC.py --config-file "configs/VOC/FT/{num}_shot/VOC-RCNN-101-C4-split{num}-ft.yaml" --num-gpus 4 --resume OUTPUT_DIR "Path/for/Checkpointing" MODEL.WEIGHTS "Path/to/Base/Training/Model/Weights" 
```

### Evaluation
- Evaluation on VOC
Evaluation can be done at any stage using the `--eval-only` flag. For example, the model obtained after fine tuning can be evaluated as follows:
```python
python scripts/finetune_VOC.py --config-file "configs/VOC/VOC-RCNN-101-C4-split{num}.yaml" --num-gpus 4 --eval-only --resume OUTPUT_DIR "Path/for/Checkpointing" MODEL.WEIGHTS "Path/to/Fine/Tune/Model/Weights"
```

**Note**: The default training/testing assumes 4 GPUs. It can be modified to suit other GPU configurations, but would require changing the learning rate and batch sizes accordingly. Please look at `SOLVER.REFERENCE_WORLD_SIZE` parameter in the [detectron2 configurations](https://detectron2.readthedocs.io/en/latest/modules/config.html#config-references) for details on how this can be done automatically.

This repository is still being updated. Instructions on how to run the code on MS-COCO will be provided shortly.
