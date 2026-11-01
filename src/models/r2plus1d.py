"""[Owner: TV1]"""
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import torch.nn as nn


def build_r2plus1d(num_classes, pretrained=True):
    m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
    feat_dim = m.fc.in_features
    m.fc = nn.Identity()
    return m, feat_dim
