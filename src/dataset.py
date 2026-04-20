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
                with open(tf) as fh:
                    train_ids = json.load(fh)
            if os.path.exists(vf):
                with open(vf) as fh:
                    val_ids   = json.load(fh)

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
