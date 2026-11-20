"""[Owner: TV4] (TV3 sửa khi kiểm tra head với mọi backbone — xem ghi chú bên dưới)"""
import torch.nn as nn

from .i3d import load_pytorchvideo


def build_x3d(num_classes, pretrained=True):
    """X3D-S từ PyTorchVideo (mô hình nhẹ cho realtime/CPU).
    TODO(TV4): thử MoViNet-A0 làm phương án thay thế; X3D-S chuẩn dùng 13 frame, 160px.

    Head gốc dùng AvgPool3d kernel cố định (13, 5, 5) cho clip 13x160 -> lỗi với clip 16x112 (feature
    map 16x4x4). Thay bằng AdaptiveAvgPool3d(1) bên trong ProjectedPool."""
    m = load_pytorchvideo("x3d_s", pretrained)
    head = m.blocks[-1]
    feat_dim = head.proj.in_features
    head.pool.pool = nn.AdaptiveAvgPool3d(1)
    head.proj = nn.Identity()
    head.activation = nn.Identity()
    return m, feat_dim
