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

    def __init__(self, kernel_size: int = 7, mask_prob: float = 0.0):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size phải lẻ để padding giữ kích thước"
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.mask_prob = mask_prob
        self.last_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        max_map = x.amax(dim=1, keepdim=True)   # (B, 1, H, W)
        combined = torch.cat([avg_map, max_map], dim=1)  # (B, 2, H, W)
        scale = torch.sigmoid(self.conv(combined))       # (B, 1, H, W)
        
        self.last_scale = scale  # Lưu lại để tính Sparsity / Consistency Loss
        
        if self.training and self.mask_prob > 0.0:
            B, C, H, W = scale.shape
            flattened = scale.view(B, -1)
            k = int(H * W * self.mask_prob)
            if k > 0:
                # Lấy giá trị lớn thứ (N - k + 1) để làm ngưỡng
                thresholds = torch.kthvalue(flattened, H * W - k + 1, dim=1).values
                mask = scale < thresholds.view(B, 1, 1, 1)
                scale = scale * mask

        return x * scale                                 # broadcast


class CBAM(nn.Module):
    """
    CBAM: Channel Attention → Spatial Attention (tuần tự, đúng paper).

    Args:
        in_channels:     số kênh của feature map đầu vào (từ backbone)
        reduction_ratio: tỷ lệ nén MLP trong Channel Attention
        kernel_size:     kích thước conv trong Spatial Attention (7 theo paper)
        mask_prob:       Tỷ lệ che khuất các điểm chú ý cao nhất (dùng cho Thí nghiệm 1)
    """

    def __init__(self, in_channels: int,
                 reduction_ratio: int = 16,
                 kernel_size: int = 7,
                 mask_prob: float = 0.0):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size, mask_prob=mask_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)   # Channel attention trước
        x = self.spatial_att(x)   # Spatial attention sau
        return x
