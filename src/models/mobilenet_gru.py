"""[Owner: TV1]"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetGRU(nn.Module):
    """Baseline 2D CNN + GRU: so sánh để chứng minh lợi ích của mô hình 3D."""

    def __init__(self, pretrained=True, hidden=256):
        super().__init__()
        cnn = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        self.cnn = nn.Sequential(cnn.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.gru = nn.GRU(576, hidden, batch_first=True)
        self.out_dim = hidden

    def forward(self, x):            # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        f = self.cnn(x.transpose(1, 2).reshape(B * T, C, H, W)).view(B, T, -1)
        _, h = self.gru(f)
        return h[-1]


def build_mobilenet_gru(num_classes, pretrained=True):
    m = MobileNetGRU(pretrained)
    return m, m.out_dim
