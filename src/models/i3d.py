"""[Owner: TV2]"""
import torch
import torch.nn as nn


def build_i3d(num_classes, pretrained=True):
    """I3D-R50 từ PyTorchVideo. Bỏ lớp proj cuối để lấy feature 2048-d.
    TODO(TV2): kiểm tra tên module head trong phiên bản pytorchvideo đang dùng."""
    m = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=pretrained)
    feat_dim = m.blocks[-1].proj.in_features
    m.blocks[-1].proj = nn.Identity()
    return m, feat_dim
