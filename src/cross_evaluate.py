import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Thêm đường dẫn để có thể import các module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_model
from dataset import VOCMultiLabelDataset, get_val_transform
from evaluate import compute_map, compute_f1

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_coco_to_voc_mapping():
    mapping = {}
    voc_to_coco_names = {
        'aeroplane': 'airplane',
        'motorbike': 'motorcycle',
        'tvmonitor': 'tv',
        'pottedplant': 'potted plant',
        'diningtable': 'dining table',
        'sofa': 'couch'
    }
    
    for i, voc_name in enumerate(VOC_CLASSES):
        coco_name = voc_to_coco_names.get(voc_name, voc_name)
        if coco_name in COCO_CLASSES:
            mapping[i] = COCO_CLASSES.index(coco_name)
    return mapping

@torch.no_grad()
def evaluate_cross_dataset(model, loader, device='cuda'):
    model.eval()
    mapping = get_coco_to_voc_mapping()
    all_probs, all_targets = [], []
    
    for imgs, targets in tqdm(loader, desc="Evaluating VOC", leave=False):
        logits = model(imgs.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        
        voc_probs = np.zeros((probs.shape[0], 20))
        for voc_idx, coco_idx in mapping.items():
            voc_probs[:, voc_idx] = probs[:, coco_idx]
            
        all_probs.append(voc_probs)
        all_targets.append(targets.numpy())
        
    P = np.concatenate(all_probs, axis=0)
    T = np.concatenate(all_targets, axis=0)
    
    return {**compute_map(T, P), **compute_f1(T, P)}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc-root', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, default='/kaggle/working/outputs')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    results = []
    device = args.device
    
    transform = get_val_transform(img_size=224)
    try:
        voc_ds = VOCMultiLabelDataset(args.voc_root, split='val', transform=transform)
        voc_loader = torch.utils.data.DataLoader(voc_ds, batch_size=64, num_workers=2)
    except Exception as e:
        print(f"❌ Không thể load VOC dataset: {e}")
        return

    output_path = Path(args.outputs_dir)
    if not output_path.exists():
        print(f"❌ Không tìm thấy thư mục {output_path}")
        return

    exp_dirs = [d for d in output_path.iterdir() if d.is_dir()]
    
    for exp_dir in exp_dirs:
        pth_path = exp_dir / 'best.pth'
        
        if not pth_path.exists():
            print(f"⚠️ Skip {exp_dir.name}: Không tìm thấy best.pth")
            continue
            
        print(f"\n🔍 Đang đánh giá VOC cho: {exp_dir.name}")
        try:
            backbone = 'efficientnet_b0'
            use_cbam = True
            if 'resnet' in exp_dir.name.lower():
                backbone = 'resnet50'
                use_cbam = 'cbam' in exp_dir.name.lower()
            elif 'efficientnet_asl' in exp_dir.name.lower():
                use_cbam = False
            
            model = build_model({
                'backbone': backbone,
                'use_cbam': use_cbam,
                'num_classes': 80,
                'pretrained': False
            }).to(device)
            
            state_dict = torch.load(pth_path, map_location=device)
            model.load_state_dict(state_dict)
            
            metrics = evaluate_cross_dataset(model, voc_loader, device)
            
            results.append({
                'Experiment': exp_dir.name,
                'VOC_mAP': metrics['mAP'],
                'VOC_Macro_F1': metrics['macro_f1'],
                'VOC_Micro_F1': metrics['micro_f1']
            })
            print(f"✅ {exp_dir.name}: VOC mAP = {metrics['mAP']:.4f}")
            
        except Exception as e:
            print(f"❌ Lỗi khi đánh giá {exp_dir.name}: {e}")
            continue

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path / 'voc_cross_evaluation.csv', index=False)
        print(f"\n🎉 Đã lưu kết quả vào {output_path / 'voc_cross_evaluation.csv'}")
    else:
        print("\n❌ Không có kết quả nào được ghi nhận.")

if __name__ == '__main__':
    main()
