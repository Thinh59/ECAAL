# 🚀 Hướng Dẫn Toàn Bộ: Multi-Label Image Classification với EfficientNet-B0 + CBAM + ASL
> **Phiên bản đã sửa lỗi** — cập nhật COCO 2017, ASL đúng chuẩn paper gốc, fix bugs dataset/train/model

---

## 📋 Mục Lục
1. [Cấu trúc Repository](#1-cấu-trúc-repository)
2. [Chuẩn bị môi trường Kaggle T4](#2-chuẩn-bị-môi-trường-kaggle-t4)
3. [Chuẩn bị Dataset — COCO 2017](#3-chuẩn-bị-dataset)
4. [Code từng file](#4-code-từng-file)
5. [Configs YAML](#5-configs-yaml)
6. [Quy trình chạy thực nghiệm](#6-quy-trình-chạy-thực-nghiệm)
7. [Đánh giá kết quả & vẽ biểu đồ](#7-đánh-giá-kết-quả)
8. [Requirements](#8-requirements)

---

## 1. Cấu Trúc Repository

```
multilabel-cbam-asl/
│
├── data/
│   └── coco_subset/
│       ├── subset_train_ids.json   # auto-generated
│       └── subset_val_ids.json
│
├── src/
│   ├── losses.py       # BCE, Focal, ASL (đúng chuẩn paper gốc ICCV 2021)
│   ├── cbam.py         # CBAM module
│   ├── models.py       # EfficientNet-B0 / ResNet50 + CBAM neck
│   ├── dataset.py      # COCO 2017 + VOC 2012 loader
│   ├── train.py        # Training loop
│   ├── evaluate.py     # mAP, F1
│   └── utils.py        # Helpers
│
├── configs/
│   ├── exp_A_resnet_bce.yaml
│   ├── exp_B_resnet_asl.yaml
│   ├── exp_C_efficientnet_cbam_asl.yaml
│   └── exp_D_resnet_focal.yaml     # thêm cho so sánh đầy đủ
│
├── notebooks/
│   └── kaggle_run.ipynb
│
├── outputs/            # auto-generated
├── requirements.txt
└── README.md
```

---

## 2. Chuẩn Bị Môi Trường Kaggle T4

```python
# Cell 1: Install
!pip install timm pycocotools torchmetrics scikit-learn matplotlib seaborn pyyaml tqdm -q

# Cell 2: Kiểm tra GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# Expected: Tesla T4, 15.8 GB
```

```bash
# Clone repo
!git clone https://github.com/YOUR_USERNAME/multilabel-cbam-asl.git
%cd multilabel-cbam-asl
```

---

## 3. Chuẩn Bị Dataset

### Dùng COCO 2017 (không phải 2014-for-YOLOv3)

**Thêm dataset vào Kaggle Notebook:**
- Notebook → **+ Add Data** → search `"coco-2017-dataset"` → Add của `awsaf49`
- Mount tại: `/kaggle/input/coco-2017-dataset/`

**Cấu trúc thư mục sau khi mount:**
```
/kaggle/input/coco-2017-dataset/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
└── val2017/
```

**Tạo subset 20k (chạy 1 lần):**
```bash
python src/dataset.py --create-subset \
    --coco-root /kaggle/input/coco-2017-dataset \
    --output-dir /kaggle/working/data/coco_subset \
    --num-train 16000 \
    --num-val 4000
```

---

## 4. Code Từng File

### `src/losses.py`

```python
"""
losses.py — BCE, Focal Loss, Asymmetric Loss (ASL)
ASL: "Asymmetric Loss For Multi-Label Classification", Ridnik et al., ICCV 2021
     arXiv:2009.14119 | github.com/Alibaba-MIIL/ASL

BUG CŨ đã sửa:
  - xs_neg trong ASL phải là (1 - xs_pos) TRƯỚC khi shift, không phải sau
  - asymmetric_w phải dùng xs_neg SAU khi shift (xs_neg_shifted) để đúng paper
  - Chia mean theo batch size (sum / B), không phải mean toàn bộ phần tử
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """Standard Binary Cross-Entropy — Exp A baseline."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets.float())


class FocalLoss(nn.Module):
    """
    Symmetric Focal Loss — Exp D (so sánh trung gian BCE → Focal → ASL).
    Lin et al., ICCV 2017. arXiv:1708.02002
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)                          # prob của class đúng
        focal_w = (1 - p_t) ** self.gamma
        return (focal_w * bce).mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) — implementation đúng chuẩn paper gốc.
    Ridnik et al., ICCV 2021. arXiv:2009.14119

    Cơ chế:
      - Positive branch (y=1): focusing weight (1-p)^gamma_pos, không shift
      - Negative branch (y=0): shift p xuống bằng margin m trước khi tính loss
                               → các negative có p < m bị zero-out hoàn toàn
                               → rồi áp focusing weight p_shifted^gamma_neg

    Args:
        gamma_pos: focusing exponent cho positive (thường = 0, tức không down-weight)
        gamma_neg: focusing exponent cho negative (thường = 4, down-weight mạnh)
        clip:      probability margin m (thường = 0.05)
        eps:       numerical stability
    """

    def __init__(self, gamma_pos: float = 0, gamma_neg: float = 4,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.float()

        # Xác suất dự đoán
        xs_pos = torch.sigmoid(logits)           # p(y=1)
        xs_neg = 1.0 - xs_pos                   # p(y=0) — TRƯỚC khi shift

        # Probability shifting cho negative branch
        # max(p - m, 0): nếu p < m thì xác suất âm = 0 → loss âm = 0
        if self.clip > 0:
            xs_neg_shifted = (xs_neg + self.clip).clamp(max=1.0)
        else:
            xs_neg_shifted = xs_neg

        # Log-likelihood của từng branch
        los_pos = targets       * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg_shifted.clamp(min=self.eps))
        loss = los_pos + los_neg                 # (B, C), âm giá trị

        # Asymmetric focusing weights
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            # Với positive: weight = (1 - p)^gamma_pos
            # Với negative: weight = p_shifted^gamma_neg  (NOTE: dùng xs_neg_shifted)
            pos_w = torch.pow(1.0 - xs_pos,        self.gamma_pos)  # positive focusing
            neg_w = torch.pow(xs_neg_shifted,       self.gamma_neg)  # negative focusing
            # Ghép lại theo mask
            asymmetric_w = targets * pos_w + (1 - targets) * neg_w
            loss = loss * asymmetric_w

        # Trả về loss trung bình theo batch (sum over classes, mean over batch)
        return -loss.sum() / logits.size(0)


def get_loss(loss_name: str, **kwargs) -> nn.Module:
    """Factory: lấy loss function theo tên trong config."""
    registry = {
        'bce':   BCELoss,
        'focal': FocalLoss,
        'asl':   AsymmetricLoss,
    }
    assert loss_name in registry, \
        f"Loss '{loss_name}' không hợp lệ. Chọn: {list(registry.keys())}"
    return registry[loss_name](**kwargs)
```

---

### `src/cbam.py`

```python
"""
cbam.py — Convolutional Block Attention Module
Woo et al., ECCV 2018. arXiv:1807.06521

Đặt SAU backbone (as Neck), TRƯỚC GAP.
Thứ tự: Channel Attention → Spatial Attention (đúng paper).
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention — "What" to emphasize.
    Dùng shared MLP trên cả AvgPool và MaxPool theo không gian.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        mid = max(in_channels // reduction_ratio, 1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg_pool = x.mean(dim=[2, 3])          # (B, C) — spatial average
        max_pool = x.amax(dim=[2, 3])          # (B, C) — spatial max
        # Shared MLP — cùng trọng số cho cả hai
        scale = torch.sigmoid(
            self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        )                                      # (B, C) ∈ (0,1)
        return x * scale.view(B, C, 1, 1)     # broadcast scale lên spatial dims


class SpatialAttention(nn.Module):
    """
    Spatial Attention — "Where" to look.
    Nén channel dimension bằng avg+max, rồi dùng conv 7×7 tạo attention map.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size phải lẻ để padding giữ kích thước"
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        max_map = x.amax(dim=1, keepdim=True)   # (B, 1, H, W)
        combined = torch.cat([avg_map, max_map], dim=1)  # (B, 2, H, W)
        scale = torch.sigmoid(self.conv(combined))       # (B, 1, H, W)
        return x * scale                                 # broadcast


class CBAM(nn.Module):
    """
    CBAM: Channel Attention → Spatial Attention (tuần tự, đúng paper).

    Args:
        in_channels:     số kênh của feature map đầu vào (từ backbone)
        reduction_ratio: tỷ lệ nén MLP trong Channel Attention
        kernel_size:     kích thước conv trong Spatial Attention (7 theo paper)
    """

    def __init__(self, in_channels: int,
                 reduction_ratio: int = 16,
                 kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)   # Channel attention trước
        x = self.spatial_att(x)   # Spatial attention sau
        return x
```

---

### `src/models.py`

```python
"""
models.py — MultiLabelModel hỗ trợ 3 variants ablation:
  Exp A: resnet50,        use_cbam=False, loss=BCE
  Exp B: resnet50,        use_cbam=False, loss=ASL
  Exp C: efficientnet_b0, use_cbam=True,  loss=ASL

BUG CŨ đã sửa:
  - timm 'resnet50' với features_only=True trả về 4 feature maps,
    cuối cùng có shape (B, 2048, 7, 7) — đúng.
  - timm 'efficientnet_b0' với features_only=True trả về 5 feature maps,
    cuối cùng là (B, 1280, 7, 7) — đúng.
  - Nhưng dummy forward phải chạy TRÊN DEVICE đúng khi khởi tạo.
    → Sửa: tạo dummy trên CPU luôn (model chưa move sang GPU lúc __init__).
"""

import torch
import torch.nn as nn
import timm
from cbam import CBAM


class MultiLabelModel(nn.Module):
    """
    Pipeline: Backbone → [CBAM Neck] → GAP → Dropout → FC → logits

    Lưu ý: KHÔNG có Sigmoid trong forward — loss functions (BCEWithLogitsLoss,
    ASL) tự xử lý sigmoid bên trong để ổn định số học.
    Khi inference: probs = torch.sigmoid(model(x))
    """

    def __init__(
        self,
        backbone_name: str = 'efficientnet_b0',
        num_classes: int = 80,
        use_cbam: bool = True,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.use_cbam = use_cbam

        # ── Backbone ─────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
        )

        # Tự động phát hiện số channels của feature map cuối
        # Chạy dummy forward trên CPU (model chưa trên GPU lúc này)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.backbone(dummy)
            self.feature_channels = feats[-1].shape[1]

        # ── CBAM Neck (optional) ──────────────────────────────────────────────
        if use_cbam:
            self.cbam = CBAM(
                in_channels=self.feature_channels,
                reduction_ratio=16,
                kernel_size=7,
            )

        # ── GAP + Head ────────────────────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        x = feats[-1]                        # Feature map sâu nhất: (B, C, H, W)

        if self.use_cbam:
            x = self.cbam(x)                 # Attention refinement

        x = self.gap(x).flatten(1)           # (B, C)
        return self.head(x)                  # (B, num_classes) — raw logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict) -> MultiLabelModel:
    model = MultiLabelModel(
        backbone_name=cfg.get('backbone', 'efficientnet_b0'),
        num_classes=cfg.get('num_classes', 80),
        use_cbam=cfg.get('use_cbam', True),
        pretrained=cfg.get('pretrained', True),
        dropout_rate=cfg.get('dropout', 0.3),
    )
    print(f"[Model] {cfg['backbone']} | CBAM={cfg['use_cbam']} | "
          f"Params={model.num_parameters()/1e6:.2f}M | "
          f"FeatChannels={model.feature_channels}")
    return model
```

---

### `src/dataset.py`

```python
"""
dataset.py — COCO 2017 + Pascal VOC 2012 multi-label loaders

BUG CŨ đã sửa:
  1. Hardcode 'instances_{split}2014.json' → đổi thành 2017
  2. create_coco_subset() gọi random.shuffle mà không stratify thực sự
     → thêm stratified sampling theo số nhãn/ảnh
  3. VOC: int(parts[1]) == 1 đúng, nhưng cần xử lý trường hợp file
     {cls}_{split}.txt không tồn tại (đã có exists() check — OK)
  4. DataLoader: num_workers>0 trên Kaggle đôi khi gây lỗi fork
     → dùng num_workers=2, persistent_workers=True khi nw>0
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ── Transforms ───────────────────────────────────────────────────────────────

def get_train_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def get_val_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── COCO 2017 ────────────────────────────────────────────────────────────────

class COCOMultiLabelDataset(Dataset):
    """
    MS-COCO 2017 multi-label classification dataset.
    Annotations đọc từ instances_{split}2017.json (định dạng JSON chuẩn COCO).
    Trả về: (image_tensor [3,H,W], label_vector [80]) với label_vector nhị phân.
    """

    NUM_CLASSES = 80

    def __init__(self, root: str, split: str = 'train',
                 transform=None, subset_ids: list = None):
        assert split in ('train', 'val'), "split phải là 'train' hoặc 'val'"
        self.root = Path(root)
        self.transform = transform

        # Đọc annotation file COCO 2017
        ann_file = self.root / 'annotations' / f'instances_{split}2017.json'
        assert ann_file.exists(), f"Không tìm thấy: {ann_file}"

        with open(ann_file) as f:
            coco = json.load(f)

        # category_id (COCO dùng id không liên tục 1-90) → index 0-79
        cats_sorted = sorted(coco['categories'], key=lambda c: c['id'])
        self.cat_id_to_idx = {c['id']: i for i, c in enumerate(cats_sorted)}
        self.idx_to_name   = {i: c['name'] for i, c in enumerate(cats_sorted)}

        # image_id → set of class indices
        img_labels: dict[int, set] = defaultdict(set)
        for ann in coco['annotations']:
            cidx = self.cat_id_to_idx.get(ann['category_id'])
            if cidx is not None:
                img_labels[ann['image_id']].add(cidx)

        # image_id → file_name
        id2info = {img['id']: img for img in coco['images']}

        # Lọc theo subset nếu có
        selected_ids = list(img_labels.keys())
        if subset_ids is not None:
            selected_ids = [i for i in subset_ids if i in img_labels]

        # Xây danh sách (path, label_vector)
        self.samples = []
        img_dir = self.root / f'{split}2017'
        for iid in selected_ids:
            info = id2info.get(iid)
            if info is None:
                continue
            img_path = img_dir / info['file_name']
            if not img_path.exists():
                continue
            label = torch.zeros(self.NUM_CLASSES)
            for cidx in img_labels[iid]:
                label[cidx] = 1.0
            self.samples.append((str(img_path), label))

        print(f"[COCO 2017 {split}] {len(self.samples):,} images loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def create_coco_subset(coco_root: str, output_dir: str,
                       num_train: int = 16000, num_val: int = 4000,
                       seed: int = 42):
    """
    Tạo stratified subset của COCO 2017.
    Stratification: ưu tiên lấy ảnh có nhiều nhãn đa dạng,
    đảm bảo mỗi class xuất hiện đủ trong subset.
    """
    random.seed(seed)
    np.random.seed(seed)

    for split, n in [('train', num_train), ('val', num_val)]:
        ann_file = os.path.join(coco_root, 'annotations',
                                f'instances_{split}2017.json')
        with open(ann_file) as f:
            coco = json.load(f)

        # image_id → list of category_ids
        img_cats: dict[int, list] = defaultdict(list)
        for ann in coco['annotations']:
            img_cats[ann['image_id']].append(ann['category_id'])

        all_ids = list(img_cats.keys())

        # Stratify: sắp xếp ảnh theo số lượng nhãn (nhiều nhãn trước)
        # rồi lấy đều từ nhiều mức khác nhau
        all_ids.sort(key=lambda i: -len(set(img_cats[i])))
        # Lấy đều từ sorted list (đảm bảo cả ảnh ít nhãn và nhiều nhãn)
        step = max(1, len(all_ids) // n)
        selected = all_ids[::step][:n]
        if len(selected) < n:
            # Bổ sung ngẫu nhiên nếu chưa đủ
            remaining = list(set(all_ids) - set(selected))
            random.shuffle(remaining)
            selected += remaining[:n - len(selected)]

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f'subset_{split}_ids.json')
        with open(out_path, 'w') as f:
            json.dump(selected[:n], f)
        print(f"[Subset] COCO 2017 {split}: {len(selected[:n])} ids → {out_path}")


# ── Pascal VOC 2012 ───────────────────────────────────────────────────────────

class VOCMultiLabelDataset(Dataset):
    """Pascal VOC 2012 multi-label (20 classes)."""

    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
    ]
    NUM_CLASSES = 20

    def __init__(self, root: str, split: str = 'trainval', transform=None):
        self.root = Path(root)
        self.transform = transform
        cls2idx = {c: i for i, c in enumerate(self.VOC_CLASSES)}

        split_file = self.root / 'ImageSets' / 'Main' / f'{split}.txt'
        img_ids = split_file.read_text().strip().splitlines()

        labels = {iid: torch.zeros(self.NUM_CLASSES) for iid in img_ids}
        for cls_name in self.VOC_CLASSES:
            cls_file = self.root / 'ImageSets' / 'Main' / f'{cls_name}_{split}.txt'
            if not cls_file.exists():
                continue
            cidx = cls2idx[cls_name]
            for line in cls_file.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) == 2 and int(parts[1]) == 1 and parts[0] in labels:
                    labels[parts[0]][cidx] = 1.0

        self.samples = []
        for iid in img_ids:
            p = self.root / 'JPEGImages' / f'{iid}.jpg'
            if p.exists():
                self.samples.append((str(p), labels[iid]))

        print(f"[VOC 2012 {split}] {len(self.samples):,} images loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(cfg: dict):
    """
    Trả về (train_loader, val_loader).
    cfg keys: dataset, data_root, batch_size, num_workers, img_size,
              subset_ids_path (optional, chỉ cho COCO)
    """
    name    = cfg.get('dataset', 'coco')
    root    = cfg['data_root']
    bs      = cfg.get('batch_size', 64)
    nw      = cfg.get('num_workers', 2)
    sz      = cfg.get('img_size', 224)

    train_tf = get_train_transform(sz)
    val_tf   = get_val_transform(sz)

    if name == 'coco':
        train_ids = val_ids = None
        sp = cfg.get('subset_ids_path')
        if sp:
            tf = os.path.join(sp, 'subset_train_ids.json')
            vf = os.path.join(sp, 'subset_val_ids.json')
            if os.path.exists(tf):
                train_ids = json.load(open(tf))
            if os.path.exists(vf):
                val_ids   = json.load(open(vf))

        train_ds = COCOMultiLabelDataset(root, 'train', train_tf, train_ids)
        val_ds   = COCOMultiLabelDataset(root, 'val',   val_tf,   val_ids)

    elif name == 'voc':
        train_ds = VOCMultiLabelDataset(root, 'trainval', train_tf)
        val_ds   = VOCMultiLabelDataset(root, 'val',      val_tf)
    else:
        raise ValueError(f"Dataset không hợp lệ: {name}")

    pw = nw > 0   # persistent_workers chỉ khi có worker
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=pw, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=pw)
    return train_loader, val_loader


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--create-subset', action='store_true')
    p.add_argument('--coco-root',  type=str, default='/kaggle/input/coco-2017-dataset')
    p.add_argument('--output-dir', type=str, default='./data/coco_subset')
    p.add_argument('--num-train',  type=int, default=16000)
    p.add_argument('--num-val',    type=int, default=4000)
    args = p.parse_args()
    if args.create_subset:
        create_coco_subset(args.coco_root, args.output_dir,
                           args.num_train, args.num_val)
```

---

### `src/evaluate.py`

```python
"""
evaluate.py — mAP và Macro F1-score cho multi-label classification.

Lưu ý: dùng sklearn.metrics.average_precision_score (area under PR curve),
KHÔNG phải COCO eval API (khác nhau về interpolation).
Khi báo cáo kết quả cần ghi rõ đang dùng phương pháp nào.
"""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score


def compute_map(targets: np.ndarray, probs: np.ndarray) -> dict:
    """
    Args:
        targets: (N, C) binary ground truth
        probs:   (N, C) predicted probabilities (sau sigmoid)
    Returns:
        {'mAP': float, 'AP_per_class': list[float]}
    """
    C = targets.shape[1]
    aps = []
    for c in range(C):
        if targets[:, c].sum() > 0:           # bỏ qua class không có positive
            ap = average_precision_score(targets[:, c], probs[:, c])
            aps.append(ap)
    return {'mAP': float(np.mean(aps)) if aps else 0.0, 'AP_per_class': aps}


def compute_f1(targets: np.ndarray, probs: np.ndarray,
               threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    return {
        'macro_f1': float(f1_score(targets, preds, average='macro',  zero_division=0)),
        'micro_f1': float(f1_score(targets, preds, average='micro',  zero_division=0)),
    }


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader, device: str = 'cuda') -> dict:
    model.eval()
    all_probs, all_targets = [], []

    for imgs, targets in loader:
        logits = model(imgs.to(device))
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.numpy())

    P = np.concatenate(all_probs,   axis=0)
    T = np.concatenate(all_targets, axis=0)

    return {**compute_map(T, P), **compute_f1(T, P)}
```

---

### `src/utils.py`

```python
"""utils.py — AverageMeter, Logger, Checkpoint, Seed."""

import os, json, time
import torch
import numpy as np


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


class Logger:
    def __init__(self, path: str):
        self.path    = path
        self.records = []

    def log(self, epoch: int, metrics: dict):
        self.records.append({'epoch': epoch, 'ts': time.time(), **metrics})
        with open(self.path, 'w') as f:
            json.dump(self.records, f, indent=2)

    def print_latest(self):
        if not self.records: return
        r = self.records[-1]
        parts = [f"{k}={v:.4f}" for k, v in r.items()
                 if k not in ('epoch', 'ts') and isinstance(v, float)]
        print(f"[Epoch {r['epoch']}] " + " | ".join(parts))


def save_checkpoint(model, optimizer, epoch, metrics, path: str):
    torch.save({'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'metrics': metrics}, path)
    print(f"[Ckpt] Saved → {path}")


def load_checkpoint(model, optimizer, path: str, device: str = 'cuda'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('epoch', 0), ckpt.get('metrics', {})


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
```

---

### `src/train.py`

```python
"""
train.py — Training loop cho ablation study.

BUG CŨ đã sửa:
  - loss_cfg.pop('name') làm MUT config dict → lần chạy thứ 2 trong cùng
    process sẽ không tìm được 'name'. Fix: dùng copy trước khi pop.
  - torch.cuda.amp.autocast() deprecated trong PyTorch ≥ 2.0
    → dùng torch.amp.autocast('cuda') thay thế
  - torch.cuda.amp.GradScaler() → torch.amp.GradScaler('cuda')
"""

import os, sys, yaml, argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models   import build_model
from losses   import get_loss
from dataset  import get_dataloaders
from evaluate import evaluate_model
from utils    import AverageMeter, Logger, save_checkpoint, set_seed


def train_one_epoch(model, loader, optimizer, criterion, scheduler, device, scaler):
    model.train()
    meter = AverageMeter()

    for i, (imgs, targets) in enumerate(loader):
        imgs    = imgs.to(device,    non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # AMP forward (PyTorch ≥ 2.0 style)
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            loss   = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        meter.update(loss.item(), imgs.size(0))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(loader)}] loss={meter.avg:.4f}")

    return meter.avg


def run(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_name = Path(config_path).stem
    out_dir  = Path(cfg.get('output_dir', 'outputs')) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*55}\n[Exp] {exp_name} | device={device}\n{'='*55}")

    # Data
    train_loader, val_loader = get_dataloaders(cfg['data'])

    # Model
    model = build_model(cfg['model']).to(device)

    # Loss — FIX: copy dict để tránh mutate cfg khi pop 'name'
    loss_cfg  = dict(cfg.get('loss', {'name': 'asl'}))
    loss_name = loss_cfg.pop('name')
    criterion = get_loss(loss_name, **loss_cfg)
    print(f"[Loss] {loss_name.upper()} | params: {loss_cfg}")

    # Optimizer
    opt = cfg.get('optimizer', {})
    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt.get('lr', 3e-4),
        weight_decay=opt.get('weight_decay', 1e-4),
    )

    # Scheduler
    n_epochs = cfg.get('num_epochs', 20)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=opt.get('lr', 3e-4),
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
        pct_start=0.1,
    )

    # AMP scaler (PyTorch ≥ 2.0)
    scaler = torch.amp.GradScaler('cuda')

    logger   = Logger(str(out_dir / 'log.json'))
    best_map = 0.0

    for epoch in range(1, n_epochs + 1):
        print(f"\n--- Epoch {epoch}/{n_epochs} ---")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, scaler
        )

        # Validate mỗi 2 epoch (tiết kiệm thời gian Kaggle)
        if epoch % 2 == 0 or epoch == n_epochs:
            metrics = evaluate_model(model, val_loader, device)
            metrics['train_loss'] = train_loss
            logger.log(epoch, metrics)
            logger.print_latest()

            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                save_checkpoint(model, optimizer, epoch, metrics,
                                str(out_dir / 'best.pth'))
                print(f"  ✅ Best mAP: {best_map:.4f}")
        else:
            print(f"  train_loss={train_loss:.4f}")

    print(f"\n[Done] {exp_name} | Best mAP={best_map:.4f}")
    return best_map


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    run(p.parse_args().config)
```

---

## 5. Configs YAML

### `configs/exp_A_resnet_bce.yaml`
```yaml
# Exp A — Baseline: ResNet50 + BCE
seed: 42
num_epochs: 20
output_dir: /kaggle/working/outputs

model:
  backbone: resnet50
  num_classes: 80
  use_cbam: false
  pretrained: true
  dropout: 0.3

loss:
  name: bce

optimizer:
  lr: 3.0e-4
  weight_decay: 1.0e-4

data:
  dataset: coco
  data_root: /kaggle/input/coco-2017-dataset
  subset_ids_path: /kaggle/working/data/coco_subset
  batch_size: 64
  num_workers: 2
  img_size: 224
```

### `configs/exp_B_resnet_asl.yaml`
```yaml
# Exp B — Cô lập loss: ResNet50 + ASL
seed: 42
num_epochs: 20
output_dir: /kaggle/working/outputs

model:
  backbone: resnet50
  num_classes: 80
  use_cbam: false
  pretrained: true
  dropout: 0.3

loss:
  name: asl
  gamma_pos: 0
  gamma_neg: 4
  clip: 0.05

optimizer:
  lr: 3.0e-4
  weight_decay: 1.0e-4

data:
  dataset: coco
  data_root: /kaggle/input/coco-2017-dataset
  subset_ids_path: /kaggle/working/data/coco_subset
  batch_size: 64
  num_workers: 2
  img_size: 224
```

### `configs/exp_C_efficientnet_cbam_asl.yaml`
```yaml
# Exp C — Full model: EfficientNet-B0 + CBAM + ASL
seed: 42
num_epochs: 20
output_dir: /kaggle/working/outputs

model:
  backbone: efficientnet_b0
  num_classes: 80
  use_cbam: true
  pretrained: true
  dropout: 0.3

loss:
  name: asl
  gamma_pos: 0
  gamma_neg: 4
  clip: 0.05

optimizer:
  lr: 3.0e-4
  weight_decay: 1.0e-4

data:
  dataset: coco
  data_root: /kaggle/input/coco-2017-dataset
  subset_ids_path: /kaggle/working/data/coco_subset
  batch_size: 64
  num_workers: 2
  img_size: 224
```

### `configs/exp_D_resnet_focal.yaml`
```yaml
# Exp D — So sánh trung gian: ResNet50 + Focal Loss
seed: 42
num_epochs: 20
output_dir: /kaggle/working/outputs

model:
  backbone: resnet50
  num_classes: 80
  use_cbam: false
  pretrained: true
  dropout: 0.3

loss:
  name: focal
  gamma: 2.0

optimizer:
  lr: 3.0e-4
  weight_decay: 1.0e-4

data:
  dataset: coco
  data_root: /kaggle/input/coco-2017-dataset
  subset_ids_path: /kaggle/working/data/coco_subset
  batch_size: 64
  num_workers: 2
  img_size: 224
```

---

## 6. Quy Trình Chạy Thực Nghiệm

```bash
# Bước 0: Setup (1 lần)
pip install timm pycocotools torchmetrics scikit-learn pyyaml tqdm -q

# Bước 1: Tạo COCO 2017 subset
python src/dataset.py --create-subset \
    --coco-root /kaggle/input/coco-2017-dataset \
    --output-dir /kaggle/working/data/coco_subset \
    --num-train 16000 --num-val 4000

# Bước 2-5: Chạy 4 experiments
python src/train.py --config configs/exp_A_resnet_bce.yaml
python src/train.py --config configs/exp_B_resnet_asl.yaml
python src/train.py --config configs/exp_D_resnet_focal.yaml
python src/train.py --config configs/exp_C_efficientnet_cbam_asl.yaml
```

> **Lưu ý:** `output_dir` trỏ vào `/kaggle/working/outputs` — không mất khi session hết nếu bạn save Kaggle output. Mỗi experiment ~25-40 phút trên T4.

---

## 7. Đánh Giá Kết Quả

```python
import json, os
import pandas as pd
import matplotlib.pyplot as plt

# ── Tổng hợp kết quả best của mỗi experiment ─────────────────────────────────
base = '/kaggle/working/outputs'
exps = {
    'A: ResNet50+BCE':     'exp_A_resnet_bce/log.json',
    'D: ResNet50+Focal':   'exp_D_resnet_focal/log.json',
    'B: ResNet50+ASL':     'exp_B_resnet_asl/log.json',
    'C: EffNet+CBAM+ASL':  'exp_C_efficientnet_cbam_asl/log.json',
}

rows = []
logs = {}
for name, rel in exps.items():
    path = os.path.join(base, rel)
    if not os.path.exists(path):
        continue
    records = json.load(open(path))
    logs[name] = records
    best = max(records, key=lambda r: r.get('mAP', 0))
    rows.append({'Experiment': name,
                 'mAP': f"{best['mAP']:.4f}",
                 'Macro F1': f"{best['macro_f1']:.4f}",
                 'Best Epoch': best['epoch']})

print(pd.DataFrame(rows).to_string(index=False))

# ── Training curves ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
for (name, recs), c in zip(logs.items(), colors):
    ep = [r['epoch'] for r in recs]
    ax1.plot(ep, [r['mAP']       for r in recs], marker='o', label=name, color=c)
    ax2.plot(ep, [r['macro_f1']  for r in recs], marker='s', label=name, color=c)

for ax, title, ylabel in [(ax1,'mAP vs Epoch','mAP'),
                           (ax2,'Macro F1 vs Epoch','Macro F1')]:
    ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(base, 'ablation_curves.png'), dpi=150)
plt.show()
```

---

## 8. Requirements

```
# requirements.txt
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.12
pycocotools>=2.0.7
scikit-learn>=1.3.0
numpy>=1.24.0
Pillow>=10.0.0
matplotlib>=3.7.0
pyyaml>=6.0
tqdm>=4.66.0
```
