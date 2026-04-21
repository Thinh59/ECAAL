"""
evaluate.py — mAP và Macro F1-score cho multi-label classification.

Lưu ý: dùng sklearn.metrics.average_precision_score (area under PR curve),
KHÔNG phải COCO eval API (khác nhau về interpolation).
Khi báo cáo kết quả cần ghi rõ đang dùng phương pháp nào.

FIXXXX: Bỏ torch.amp.mixed_precision trong evaluate_model — evaluate cần fp32 precision
     để sigmoid + numpy chính xác. Việc dùng chế độ tự động ép kiểu trong eval không cần thiết và
     có thể gây sai số nhỏ ảnh hưởng mAP.
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
    """
    FIX (branch_fix → master): KHÔNG dùng chế độ tự động ép kiểu trong evaluate.
    fp16 → sigmoid precision thấp hơn → mAP thấp hơn (~2-3%).
    Evaluate luôn chạy fp32 để đảm bảo kết quả chính xác.
    """
    model.eval()
    all_probs, all_targets = [], []

    for imgs, targets in loader:
        # FIX: không dùng chế độ ép kiểu tự động ở đây — fp32 cho evaluation
        logits = model(imgs.to(device))
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.numpy())

    P = np.concatenate(all_probs,   axis=0)
    T = np.concatenate(all_targets, axis=0)

    return {**compute_map(T, P), **compute_f1(T, P)}
