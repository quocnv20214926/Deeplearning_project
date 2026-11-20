"""[Owner: TV2] (TV3 sửa khi kiểm tra head với mọi backbone — xem ghi chú bên dưới)"""
import torch.nn as nn


def load_pytorchvideo(name, pretrained=True):
    """Ưu tiên gói pytorchvideo đã cài (không cần GitHub API, ổn định hơn torch.hub)."""
    try:
        from pytorchvideo.models import hub
        return getattr(hub, name)(pretrained=pretrained)
    except ImportError:
        import torch
        return torch.hub.load("facebookresearch/pytorchvideo", name, pretrained=pretrained)


def build_i3d(num_classes, pretrained=True):
    """I3D-R50 từ PyTorchVideo. Bỏ lớp proj cuối để lấy feature 2048-d.

    Head gốc dùng AvgPool3d kernel cố định (4, 7, 7) cho clip 8x224 -> lỗi với clip 16x112 của giao thức
    chung (feature map 8x4x4). Thay bằng AdaptiveAvgPool3d(1): chạy được với mọi kích thước clip,
    với 8x224 cho kết quả giống hệt pool gốc."""
    m = load_pytorchvideo("i3d_r50", pretrained)
    head = m.blocks[-1]
    feat_dim = head.proj.in_features
    head.pool = nn.AdaptiveAvgPool3d(1)
    head.proj = nn.Identity()
    return m, feat_dim
