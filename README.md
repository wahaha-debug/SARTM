<div align="center"> 

## SARTM: Segment Any RGB-Thermal Model with Language-aided Distillation</div>

</div>

## 💬 Introduction

The recent Segment Anything Model (SAM) demonstrates strong instance segmentation performance across various downstream tasks. However, SAM is trained solely on RGB data, limiting its direct applicability to RGB-thermal (RGB-T) semantic segmentation. Given that RGB-T provides a robust solution for scene understanding in adverse weather and lighting conditions, such as low light and overexposure, we propose a novel framework, SARTM, which customizes the powerful SAM for RGB-T semantic segmentation.Our key idea is to unleash the potential of SAM while introduce semantic understanding modules for RGB-T data pairs.Specifically, our framework first involves fine tuning the original SAM by adding extra LoRA layers, aiming at preserving SAM's strong generalization and segmentation capabilities for downstream tasks. Secondly, we introduce language information as guidance for training our SARTM. To address cross-modal inconsistencies, we introduce a new module that effectively achieves modality adaptation while maintaining its generalization capabilities. This semantic module enables the minimization of modality gaps and alleviates semantic ambiguity, facilitating the combination of any modality under any visual conditions. Furthermore, we enhance the segmentation performance by adjusting the segmentation head of SAM and incorporating an auxiliary semantic segmentation head, which integrates multi-scale features for effective fusion. 
Extensive experiments are conducted across three multi-modal RGBT semantic segmentation benchmarks: MFNET, PST900, and FMB. Both quantitative and qualitative results consistently demonstrate that the proposed \name  significantly outperforms state-of-the-art approaches across a variety of conditions. 

## 🚀 Updates
- [x] 04/2025: init repository and release the code.
- [x] 04/2025: release SARTM model weights. Download from [**GoogleDrive**](https://drive.google.com/drive/folders/18Y0xnkEDwxTEgFzhnIkKCGemVlfprs67?dmr=1&ec=wgc-drive-globalnav-goto).

## 👁️ SARTM model

<div align="center"> 
  
![SARTM](fig/SARTM.png)

**Figure:** Overall architecture of SARTM model.

</div>

## 🔍 Environment

First, create and activate the environment using the following commands: 
```bash
conda env create -f environment.yaml
conda activate SARTM
```

## 📦 Data preparation
Download the dataset:
- [PST900](https://github.com/haqishen/MFNet-pytorch), for PST900 dataset with RGB-Infrared modalities
- [FMB](https://github.com/JinyuanLiu-CV/SegMiF), for FMB dataset with RGB-Infrared modalities.
- [MFNet](https://github.com/haqishen/MFNet-pytorch), for MFNet dataset with RGB-Infrared modalities.

Then, put the dataset under `data` directory as follows:

```
data/
├── PST900
│   ├── test
│   │   ├── thermal
│   │   ├── labels
│   │   └── rgb
│   ├── train
│   │   ├── thermal
│   │   ├── labels
│   │   └── rgb
```

## 📦 Model Zoo


### PST900
| Model-Modal      | mIoU   | weigh |
| :--------------- | :----- | :----- |
| PST900      | 89.88 | [GoogleDrive](https://drive.google.com/drive/folders/18Y0xnkEDwxTEgFzhnIkKCGemVlfprs67?dmr=1&ec=wgc-drive-globalnav-goto) |

## Training

Before training, please download pre-trained SAM, and put it in the correct directory following this structure:

```text
checkpoints
├── download_ckpts.sh
├── sam2_hiera_small.pth
├── sam2_hiera_tiny.pth
├── sam2_hiera_base_plus.pth
└── sam2_hiera_large.pth
```

To train SARTM model, please update the appropriate configuration file in `configs/` with appropriate paths and hyper-parameters. Then run as follows:

```bash
cd path/to/SARTM
conda activate SARTM
python -m tools.train_mm --cfg configs/pst_rgbt.yaml
python -m tools.train_mm --cfg configs/fmb_rgbt.yaml
```

##  Evaluation

```text
python -m tools.val_mm --cfg configs/pst_rgbt.yaml
```

