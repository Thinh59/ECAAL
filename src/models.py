"""
models.py — MultiLabelModel hỗ trợ 3 variants ablation:
  Exp A: resnet50,        use_cbam=False, loss=BCE
  Exp B: resnet50,        use_cbam=False, loss=ASL
  Exp C: efficientnet_b0, use_cbam=True,  loss=ASL
"""

import torch
import torch.nn as nn
import timm
from cbam import CBAM


class MultiLabelModel(nn.Module):
    """
    Pipeline: Backbone ->[CBAM Neck] -> GAP -> Dropout -> FC -> logits
    """

    def __init__(
        self,
        backbone_name: str = 'efficientnet_b0',
        num_classes: int = 80,
        use_cbam: bool = True,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        cbam_mask_prob: float = 0.0,
    ):
        super().__init__()
        self.use_cbam = use_cbam

        # Backbone
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

        # CBAM Neck (optional)
        if use_cbam:
            self.cbam = CBAM(
                in_channels=self.feature_channels,
                reduction_ratio=16,
                kernel_size=7,
                mask_prob=cbam_mask_prob,
            )

        # GAP + Head
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
    # Hỗ trợ cả 'dropout' (Exp A–F) và 'dropout_rate' (Exp G+) để backward compatible
    dropout_val = cfg.get('dropout_rate', cfg.get('dropout', 0.3))
    model = MultiLabelModel(
        backbone_name=cfg.get('backbone', 'efficientnet_b0'),
        num_classes=cfg.get('num_classes', 80),
        use_cbam=cfg.get('use_cbam', True),
        pretrained=cfg.get('pretrained', True),
        dropout_rate=dropout_val,
        cbam_mask_prob=cfg.get('cbam_mask_prob', 0.0),
    )
    print(f"[Model] {cfg.get('backbone', 'efficientnet_b0')} | CBAM={cfg.get('use_cbam', True)} | "
          f"Dropout={dropout_val} | Params={model.num_parameters()/1e6:.2f}M | "
          f"FeatChannels={model.feature_channels}")
    return model
