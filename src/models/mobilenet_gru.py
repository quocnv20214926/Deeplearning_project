"""[Owner: TV1] B0: MobileNetV3 (ImageNet) trích feature từng frame + GRU theo thời gian.

CNN 2D nhìn từng frame riêng lẻ, chỉ GRU mô hình hoá chuyển động -> so với các mô hình 3D để
chứng minh lợi ích của tích chập không gian–thời gian. Đây cũng là mô hình nhẹ nhất (mốc FPS cho edge).
"""
import torch
import torch.nn as nn
from torchvision.models import (MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights, mobilenet_v3_large,
                                mobilenet_v3_small)

from src.data.transforms import KINETICS_MEAN, KINETICS_STD

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_VARIANTS = {
    "large": (mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V2, 960),
    "small": (mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1, 576),
}


class MobileNetGRU(nn.Module):
    """(B, 3, T, H, W) -> feature (B, hidden): trạng thái ẩn cuối của GRU (chỉ phụ thuộc frame quá khứ,
    nên chạy được online trên luồng video).

    Clip vào đã chuẩn hoá theo Kinetics (transforms.py, dùng chung với các mô hình 3D) nhưng MobileNetV3
    được train với chuẩn hoá ImageNet -> mô hình tự đổi chuẩn hoá bên trong.
    set_freeze("backbone" | "partial") đóng băng toàn bộ CNN, chỉ train GRU; "none" train tất cả.
    """

    def __init__(self, pretrained=True, variant="large", hidden=256, num_layers=1, dropout=0.0):
        super().__init__()
        build, weights, cnn_dim = _VARIANTS[variant]
        net = build(weights=weights if pretrained else None)
        self.cnn = nn.Sequential(net.features, net.avgpool, nn.Flatten())
        self.gru = nn.GRU(cnn_dim, hidden, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.feat_dim = hidden
        self.frozen = []
        # x_kinetics -> x_imagenet = x * k_std / i_std + (k_mean - i_mean) / i_std
        k_mean, k_std = torch.tensor(KINETICS_MEAN), torch.tensor(KINETICS_STD)
        i_mean, i_std = torch.tensor(IMAGENET_MEAN), torch.tensor(IMAGENET_STD)
        self.register_buffer("scale", (k_std / i_std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("shift", ((k_mean - i_mean) / i_std).view(1, 3, 1, 1), persistent=False)

    def set_freeze(self, mode="none"):
        freeze = mode in ("backbone", "partial")
        self.cnn.requires_grad_(not freeze)
        self.frozen = [self.cnn] if freeze else []
        return self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for m in self.frozen:
            m.eval()
        return self

    def frame_features(self, x):
        """(B, 3, T, H, W) -> feature CNN từng frame (B, T, cnn_dim)."""
        b, c, t, h, w = x.shape
        frames = x.transpose(1, 2).reshape(b * t, c, h, w) * self.scale + self.shift
        return self.cnn(frames).view(b, t, -1)

    def forward(self, x):
        _, h = self.gru(self.frame_features(x))
        return h[-1]


def build_mobilenet_gru(num_classes, pretrained=True):
    m = MobileNetGRU(pretrained)
    return m, m.feat_dim
