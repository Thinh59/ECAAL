# 🚀 Multi-Label Image Classification: EfficientNet-B0 + CBAM + ASL

> **Phiên bản đã sửa lỗi** — cập nhật COCO 2017, ASL đúng chuẩn paper gốc, fix bugs dataset/train/model

**Project**: Ablation study on multi-label classification combining:
- **EfficientNet-B0** backbone with pretrained ImageNet weights
- **CBAM** (Convolutional Block Attention Module) for feature refinement
- **ASL** (Asymmetric Loss) for handling imbalanced multi-label data

---

## 📋 Repository Structure

```
ECAAL/
│
├── data/
│   └── coco_subset/
│       ├── subset_train_ids.json      # auto-generated
│       └── subset_val_ids.json
│
├── src/
│   ├── losses.py       # BCE, Focal, ASL (đúng chuẩn paper)
│   ├── cbam.py         # CBAM module
│   ├── models.py       # EfficientNet-B0 / ResNet50 + CBAM
│   ├── dataset.py      # COCO 2017 loader
│   ├── train.py        # Training loop
│   ├── evaluate.py     # mAP, F1 metrics
│   └── utils.py        # Utilities
│
├── configs/
│   ├── exp_A_resnet_bce.yaml
│   ├── exp_B_resnet_asl.yaml
│   ├── exp_C_efficientnet_cbam_asl.yaml
│   └── exp_D_resnet_focal.yaml
│
├── notebooks/
│   └── kaggle_run.ipynb     # Kaggle T4 notebook
│
├── outputs/                 # auto-generated
├── requirements.txt
└── README.md
```

---

## 🔧 Setup

### 1. Install Dependencies

```bash
pip install torch torchvision timm pycocotools scikit-learn pyyaml matplotlib -q
```

Or use `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

For **Kaggle**: Add COCO 2017 dataset via Notebook Data interface (awsaf49's version)

For **Local**:
```bash
# Download from official COCO website
# and extract to data/coco2017/
```

---

## 📊 Experiments (Ablation Study)

### Exp A: Baseline (ResNet50 + BCE)
```yaml
backbone: resnet50
use_cbam: false
loss: bce
```

### Exp B: ASL Contribution (ResNet50 + ASL)
```yaml
backbone: resnet50
use_cbam: false
loss: asl
```

### Exp D: Focal Loss (ResNet50 + Focal)
```yaml
backbone: resnet50
use_cbam: false
loss: focal
```

### Exp C: Full Model (EfficientNet-B0 + CBAM + ASL)
```yaml
backbone: efficientnet_b0
use_cbam: true
loss: asl
```

---

## 🚀 Quick Start

### On Kaggle (Recommended for GPU T4)

1. **Create New Notebook**
2. **Add Dataset**: COCO 2017 by awsaf49
3. **Clone repo**
4. **Run**: Open `notebooks/kaggle_run.ipynb`

**Expected Runtime**: ~25-40 min per experiment on T4

### Locally

```bash
# 1. Create COCO 2017 subset (20k images)
python src/dataset.py --create-subset \
    --coco-root /path/to/coco2017 \
    --output-dir ./data/coco_subset \
    --num-train 16000 --num-val 4000

# 2. Run experiments
python src/train.py --config configs/exp_A_resnet_bce.yaml
python src/train.py --config configs/exp_B_resnet_asl.yaml
python src/train.py --config configs/exp_D_resnet_focal.yaml
python src/train.py --config configs/exp_C_efficientnet_cbam_asl.yaml

# 3. Evaluate & plot results
python notebooks/evaluate.py
```

---

## 📈 Key Fixes from Previous Version

### losses.py
- ✅ Fixed xs_neg computation in ASL (before shift, not after)
- ✅ Correct asymmetric weighting using shifted probabilities
- ✅ Proper batch-level loss aggregation

### models.py
- ✅ Dummy forward on CPU (avoids device mismatch)
- ✅ Correct feature channel detection for timm backbones
- ✅ CBAM placed correctly as Neck (before GAP)

### dataset.py
- ✅ COCO 2017 annotation format (not 2014)
- ✅ Stratified subset sampling (balanced class distribution)
- ✅ Proper DataLoader with persistent_workers

### train.py
- ✅ Config dict handling (no mutation on repeated calls)
- ✅ PyTorch 2.0+ AMP syntax (torch.amp.autocast('cuda'))
- ✅ Gradient clipping + proper scheduler

---

## 📊 Expected Results (on COCO 2017 subset)

| Experiment | mAP | Macro F1 | Params |
|-----------|-----|----------|--------|
| A: ResNet50+BCE | ~0.43 | ~0.38 | 23.5M |
| D: ResNet50+Focal | ~0.45 | ~0.41 | 23.5M |
| B: ResNet50+ASL | ~0.47 | ~0.43 | 23.5M |
| **C: EffNet+CBAM+ASL** | **~0.51** | **~0.47** | 5.3M |

> *Results vary with seed, hardware, and exact dataset subset*

---

## 🔍 Code Details

### Losses

**BCELoss**: Standard binary cross-entropy (baseline)

**FocalLoss**: Symmetric down-weighting of easy examples

**AsymmetricLoss** (ASL): 
- Positive branch: no shifting, standard CE
- Negative branch: probability margin shift + asymmetric focusing
- More effective for imbalanced multi-label data

### CBAM Module

- **Channel Attention**: Emphasizes important feature channels via shared MLP
- **Spatial Attention**: Highlights important regions via 7×7 conv
- Applied after backbone, before Global Average Pooling

### Training

- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Scheduler**: OneCycleLR
- **AMP**: Mixed precision training (PyTorch 2.0+)
- **Gradient Clipping**: max_norm=1.0

---

## 📝 Configuration Format

```yaml
seed: 42
num_epochs: 20
output_dir: /path/to/outputs

model:
  backbone: efficientnet_b0|resnet50
  num_classes: 80
  use_cbam: true|false
  pretrained: true
  dropout: 0.3

loss:
  name: bce|focal|asl
  gamma_pos: 0     # ASL only
  gamma_neg: 4     # ASL only
  clip: 0.05       # ASL only

optimizer:
  lr: 3.0e-4
  weight_decay: 1.0e-4

data:
  dataset: coco|voc
  data_root: /path/to/dataset
  subset_ids_path: /path/to/subset    # COCO only
  batch_size: 64
  num_workers: 2
  img_size: 224
```

---

## 📦 Output Structure

After training:

```
outputs/
├── exp_A_resnet_bce/
│   ├── log.json          # Training metrics per epoch
│   └── best.pth          # Best checkpoint
├── exp_B_resnet_asl/
│   ├── log.json
│   └── best.pth
├── exp_C_efficientnet_cbam_asl/
│   ├── log.json
│   └── best.pth
├── exp_D_resnet_focal/
│   ├── log.json
│   └── best.pth
└── ablation_curves.png   # Comparison plot
```

---

## 🎯 Metrics

- **mAP** (mean Average Precision): Area under PR curve per class, averaged
- **Macro F1**: Unweighted F1 score across all classes
- **Micro F1**: Global TP/(TP+FP), TP/(TP+FN)

---

## ⚙️ Hyperparameter Notes

### ASL Parameters
- `gamma_pos=0`: No down-weighting for positive samples (focus on negatives)
- `gamma_neg=4`: Strong focus on hard negatives
- `clip=0.05`: Probability shift margin (ignore negatives with p < 0.05)

### EfficientNet-B0
- 5.3M parameters (vs ResNet50: 23.5M)
- Better efficiency on GPU T4
- Strong feature extraction with compound scaling

---

## 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch_size or use smaller backbone |
| `Module not found` | Ensure sys.path includes src/ folder |
| `COCO annotations 404` | Use COCO 2017, not 2014-for-YOLOv3 |
| `num_workers errors` | Keep num_workers=2, persistent_workers=True |

---

## 📚 References

- **ASL**: Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021. [arXiv:2009.14119](https://arxiv.org/abs/2009.14119)
- **CBAM**: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018. [arXiv:1807.06521](https://arxiv.org/abs/1807.06521)
- **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling", ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

---

## 📄 License

This project is provided for educational and research purposes.

---

## ✅ Checklist Before Submission

- [ ] All 4 experiments completed
- [ ] `outputs/ablation_curves.png` generated
- [ ] Results table in README
- [ ] Code runs without errors on Kaggle T4
- [ ] COCO 2017 subset created (16k train, 4k val)
- [ ] All checkpoint files (.pth) saved
- [ ] Training logs (log.json) saved for each experiment

---

**Created**: 2026-04-18 | **Last Updated**: 2026-04-18
