"""[Owner: TV3]"""
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
import torch.nn as nn


def build_swin3d(num_classes, pretrained=True):
    m = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1 if pretrained else None)
    feat_dim = m.head.in_features
    m.head = nn.Identity()
    return m, feat_dim
