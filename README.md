## Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation
This is the official implementation of the ECCV'2022 paper "Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation".

<p align="middle">
    <img src="assets/architecture.png">
</p>

## Requirements

- Python 3.7
- PyTorch 1.5.1
- cuda 10.1
- tensorboard 1.14

Conda environment settings:

```bash
conda create -n DCAMA python=3.7
conda activate DCAMA

conda install pytorch=1.5.1 torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge tensorflow
pip install tensorboardX
```

## Prepare Datasets

Download COCO2014 train/val images and annotations: 

```bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
```

Download COCO2014 train/val annotations from Google Drive: [[train2014.zip](https://drive.google.com/file/d/1fcwqp0eQ_Ngf-8ZE73EsHKP8ZLfORdWR/view?usp=sharing)], [[val2014.zip](https://drive.google.com/file/d/16IJeYqt9oHbqnSI9m2nTXcxQWNXCfiGb/view?usp=sharing)].(and locate both train2014/ and val2014/ under annotations/ directory).

Create a directory 'datasets' and appropriately place coco to have following directory structure:

    datasets/
        └── COCO2014/           
            ├── annotations/
            │   ├── train2014/  # (dir.) training masks (from Google Drive) 
            │   ├── val2014/    # (dir.) validation masks (from Google Drive)
            │   └── ..some json files..
            ├── train2014/
            └── val2014/

## Prepare backbones

Downloading the following pre-trained backbones:

> 1. [ResNet-50](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1h-35c100f8.pth) pretrained on ImageNet-1K by [TIMM](https://github.com/rwightman/pytorch-image-models)
> 2. [ResNet-101](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth) pretrained on ImageNet-1K by [TIMM](https://github.com/rwightman/pytorch-image-models)
> 3. [Swin-B](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth) pretrained on ImageNet by [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

Create a directory 'backbones' to place the above backbones. The overall directory structure should be like this:

    ../                         # parent directory
    ├── DCAMA/                  # current (project) directory
    │   ├── common/             # (dir.) helper functions
    │   ├── data/               # (dir.) dataloaders and splits for each FSS dataset
    │   ├── model/              # (dir.) implementation of DCAMA
    │   ├── scripts/            # (dir.) Scripts for training and testing
    │   ├── README.md           # intstruction for reproduction
    │   ├── train.py            # code for training
    │   └── test.py             # code for testing
    ├── datasets/               # (dir.) Few-Shot Segmentation Datasets
    └── backbones/              # (dir.) Pre-trained backbones

## Train and Test
You can use our scripts to build your own. Training will take approx. 1.5 days until convergence (trained with four V100 GPUs). For more information, please refer to ./common/config.py

> ```bash
> sh ./scripts/train.sh
> ```
> 
> - For each experiment, a directory that logs training progress will be automatically generated under logs/ directory. 
> - From terminal, run 'tensorboard --logdir logs/' to monitor the training progress.
> - Choose the best model when the validation (mIoU) curve starts to saturate. 

For testing, you have to prepare a pretrained model. You can train one by yourself or just download our [pretrained models](https://drive.google.com/drive/folders/1vEw35yKoWkuDgkrclJPNSXeDtsTOVy_c?usp=sharing). 
> ```bash
> sh ./scripts/test.sh
> ```
> 

## BibTeX
If you are interested in our paper, please cite:
```
@inproceedings{shi2022dense,
  title={Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation},
  author={Shi, Xinyu and Wei, Dong and Zhang, Yu and Lu, Donghuan and Ning, Munan and Chen, Jiashun and Ma, Kai and Zheng, Yefeng},
  booktitle={European Conference on Computer Vision},
  pages={151--168},
  year={2022},
  organization={Springer}
}
```
