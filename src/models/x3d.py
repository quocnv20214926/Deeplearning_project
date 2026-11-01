"""[Owner: TV4]"""
import torch
import torch.nn as nn


def build_x3d(num_classes, pretrained=True):
    """X3D-S từ PyTorchVideo (mô hình nhẹ cho realtime/CPU).
    TODO(TV4): thử MoViNet-A0 làm phương án thay thế; X3D-S chuẩn dùng 13 frame, 160px."""
    m = torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=pretrained)
    feat_dim = m.blocks[-1].proj.in_features
    m.blocks[-1].proj = nn.Identity()
    m.blocks[-1].activation = nn.Identity()
    return m, feat_dim
