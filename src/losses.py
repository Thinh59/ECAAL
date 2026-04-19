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
    Asymmetric Loss (ASL) — implementation ĐÚNG CHUẨN paper gốc.
    Ridnik et al., ICCV 2021. arXiv:2009.14119

    ⚠️ BUG FIX: Theo paper, negative branch phải shift TRỰC TIẾP trên p (probability),
    không phải trên (1-p). Formula đúng:
      - p_shifted = max(p - m, 0)  ← margin TRỪ từ p
      - Khi p < m: p_shifted = 0 → loss_neg = 0 (zero-out easy negatives)

    Code cũ (SAI): xs_neg_shifted = (1-p) + m = (1.05 - p)
      → Khi p = 1.0: xs_neg_shifted = 0.05 → log(0.05) = -3.0 (RẤT ÂM) ✗
      → Khi p = 0.0: xs_neg_shifted = 1.0 → log(1.0) = 0 (zero-out) ✓
      → NGỢ CHIỀU! Down-weight dễ, up-weight khó!

    Cơ chế đúng:
      - Positive branch (y=1): focusing weight (1-p)^gamma_pos, không shift
      - Negative branch (y=0): shift p xuống bằng margin m TRƯỚC khi tính loss
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
        xs_neg = 1.0 - xs_pos                   # p(y=0) = 1 - p(y=1)

        # ⚠️ CORRECT: Probability shifting cho negative branch
        # Theo ASL paper: p_shifted = max(p - m, 0)
        # p here is xs_pos, vì đó là xác suất để model dự đoán là positive
        # Nếu p < m (threshold thấp), ta bỏ qua (zero-out)
        if self.clip > 0:
            # max(p - m, 0) → clamped minimum 0
            xs_pos_shifted = (xs_pos - self.clip).clamp(min=0)
            xs_neg_shifted = 1.0 - xs_pos_shifted  # 1 - max(p - m, 0) = min(1 - p + m, 1)
        else:
            xs_neg_shifted = xs_neg

        # Log-likelihood của từng branch
        los_pos = targets       * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg_shifted.clamp(min=self.eps))
        loss = los_pos + los_neg                 # (B, C), âm giá trị

        # Asymmetric focusing weights
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            # Với positive: weight = (1 - p)^gamma_pos
            # Với negative: weight = p_shifted^gamma_neg (dùng xs_pos_shifted)
            pos_w = torch.pow(1.0 - xs_pos,           self.gamma_pos)  # positive focusing
            neg_w = torch.pow(xs_pos_shifted,         self.gamma_neg)  # negative focusing (dùng p_shifted)
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
