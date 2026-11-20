"""[Owner: TV1] M1: R(2+1)D-18 pretrained Kinetics-400 (Tran et al., CVPR 2018).

Mỗi tích chập 3D được tách thành tích chập không gian 2D + tích chập thời gian 1D.
Trọng số torchvision nhận clip 16x112x112 chuẩn hoá theo Kinetics — đúng đầu ra của src/data/transforms.py.
"""
import torch.nn as nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18


class R2Plus1DBackbone(nn.Module):
    """(B, 3, T, H, W) -> feature (B, 512). Head (ClsHead/MILHead) gắn riêng.

    set_freeze("none" | "backbone" | "partial"): "partial" đóng băng stem + layer1 + layer2 và chỉ
    fine-tune layer3, layer4. BatchNorm của phần bị đóng băng giữ nguyên thống kê khi train.
    """

    feat_dim = 512

    def __init__(self, pretrained=True):
        super().__init__()
        self.net = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
        self.net.fc = nn.Identity()
        self.frozen = []

    def set_freeze(self, mode="none"):
        stages = [self.net.stem, self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]
        n = {"none": 0, "partial": 3, "backbone": 5}[mode]
        for i, stage in enumerate(stages):
            stage.requires_grad_(i >= n)
        self.frozen = stages[:n]
        return self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for stage in self.frozen:
            stage.eval()
        return self

    def forward(self, x):
        return self.net(x)


def build_r2plus1d(num_classes, pretrained=True):
    m = R2Plus1DBackbone(pretrained)
    return m, m.feat_dim
