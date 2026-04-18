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
