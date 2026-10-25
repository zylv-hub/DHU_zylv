# Role-Matched Capacity for Efficient LiDAR-Centric Multi-Modal 3D Object Detection

This repository contains the implementation of our method for compressing LiDAR-centric multi-modal 3D object detectors. Built upon [BEVDilation](https://github.com/gwenzhang/BEVDilation) (AAAI 2026), we reduce parameters from 80.42M to 42.99M and inference latency from 120.3ms to 67.5ms (1.78x) on A40 GPU, while maintaining competitive accuracy on nuScenes.

## Method Overview

We allocate capacity in proportion to what each stage can still change about the prediction:

1. **Image Pathway**: ResNet-50 -> ResNet-18, depth bins D: 118 -> 30 (Delta: 0.5 -> 2.0m)
2. **Voxel Stage**: Sparse encoder widths x2/3, densification threshold tau_d: 0.4 -> 0.6
3. **BEV Backbone**: DCNv3 -> LSK + PConv + tile gate + restored image guidance

Training additions: FGN loss (Focaler-GIoU-NWD) and feature distillation from the frozen baseline.

## Key Results (nuScenes val)

| Setting | mAP | NDS | Params (M) | Latency (ms) |
|---------|-----|-----|------------|--------------|
| BEVDilation (baseline) | 72.21 | 74.37 | 80.42 | 120.3 |
| **Ours (domain-pretrained)** | **73.07** | **75.02** | **42.99** | **67.5** |
| Ours (matched/ImageNet) | 68.17 | 71.80 | 42.99 | 67.5 |

## Installation

Follow the [BEVDilation installation guide](https://github.com/gwenzhang/BEVDilation). The additional modules introduced in this work have no extra dependencies beyond the BEVDilation requirements.

```bash
# Clone this repository
git clone https://github.com/zylv-hub/DHU_zylv.git
cd DHU_zylv

# Install dependencies (same as BEVDilation)
pip install -r requirements.txt
python setup.py develop

# Build DCNv3 operators
cd ops_dcnv3
python setup.py build_ext --inplace
cd ..
```

## Data Preparation

Follow the [BEVDilation data preparation](https://github.com/gwenzhang/BEVDilation) to set up the nuScenes dataset under `data/nuscenes/`.

## Training

### Domain-pretrained setting (recommended)

```bash
# Step 1: Train the baseline (or use the official BEVDilation checkpoint)
python tools/dist_train.py configs/bevdilation/bevdilation.py 4

# Step 2: Train the compressed detector with distillation
python tools/dist_train.py configs/bevdilation/role_matched.py 4 \
    --cfg-options model.teacher_ckpt=path/to/baseline.pth \
                  load_from=path/to/baseline.pth
```

### Matched (ImageNet) setting

```bash
python tools/dist_train.py configs/bevdilation/role_matched_imagenet.py 4 \
    --cfg-options model.teacher_ckpt=path/to/imagenet_baseline.pth \
                  load_from=path/to/imagenet_baseline.pth
```

## Evaluation

```bash
python tools/dist_test.py configs/bevdilation/role_matched.py \
    path/to/checkpoint.pth 4 --eval bbox
```

## New Modules

| File | Description |
|------|-------------|
| `mmdet3d/models/backbones/lsk_block.py` | Large Selective Kernel block with zero-init gate |
| `mmdet3d/models/backbones/pconv_block.py` | Partial Convolution residual block |
| `mmdet3d/models/backbones/tile_gate.py` | Tile-level foreground gate |
| `mmdet3d/models/backbones/bev_backbone_ours.py` | Role-matched BEV backbone |
| `mmdet3d/models/losses/fgn_loss.py` | Focaler-GIoU-NWD loss |
| `mmdet3d/models/losses/distillation_loss.py` | Feature distillation loss |
| `mmdet3d/models/detectors/role_matched.py` | Compressed detector class |
| `configs/bevdilation/role_matched.py` | Config (domain-pretrained) |
| `configs/bevdilation/role_matched_imagenet.py` | Config (matched/ImageNet) |

## Citation

If you find this work useful, please cite:

```bibtex
@article{lyu2026rolematched,
  title={Role-Matched Capacity for Efficient LiDAR-Centric Multi-Modal 3D Object Detection},
  author={Lyu, Zheyu and Liao, Xiaofei and Zhou, You},
  year={2026}
}
```

## Acknowledgements

This codebase is built upon [BEVDilation](https://github.com/gwenzhang/BEVDilation), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [LSKNet](https://github.com/zcablii/LSKNet), and [FasterNet](https://github.com/JierunChen/FasterNet). We thank the authors for their excellent work.

## License

This project is released under the [Apache 2.0 License](LICENSE).
