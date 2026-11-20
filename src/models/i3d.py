"""[Owner: TV2] M2: I3D-R50 pretrained Kinetics-400 (PyTorchVideo `i3d_r50`).

Cùng giao diện với R2Plus1DBackbone của TV1: (B, 3, T, H, W) -> feature (B, 2048);
head (ClsHead/MILHead) gắn riêng trong heads.py; set_freeze("none" | "backbone" | "partial").

Khác bản gốc: head của PyTorchVideo dùng AvgPool3d kích thước CỐ ĐỊNH (cho đầu vào 224), không khớp
clip 16x112x112 theo giao thức chung -> bỏ head đó, thay bằng AdaptiveAvgPool3d(1).
CHƯA CHẠY THỬ (sandbox không có torch/pytorchvideo): chạy `pytest tests/test_i3d.py` trước khi train.
"""
import torch.nn as nn


class I3DBackbone(nn.Module):
    """stem + 4 stage (res2..res5) của I3D-R50.

    "partial": đóng băng stem + res2 + res3, chỉ fine-tune res4, res5 (giống "partial" của R(2+1)D).
    BatchNorm của phần bị đóng băng giữ nguyên thống kê khi train.
    """

    feat_dim = 2048

    def __init__(self, pretrained=True):
        super().__init__()
        try:
            from pytorchvideo.models.hub import i3d_r50
        except ImportError as e:
            raise ImportError("Cần cài pytorchvideo: pip install pytorchvideo") from e
        net = i3d_r50(pretrained=pretrained)
        blocks = list(net.blocks)
        if len(blocks) != 6:  # stem, 4 stage, head
            raise RuntimeError(f"i3d_r50 có {len(blocks)} block, kỳ vọng 6 (stem + 4 stage + head); "
                               f"phiên bản pytorchvideo khác cấu trúc -> kiểm tra lại")
        self.stages = nn.ModuleList(blocks[:5])
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.frozen = []

    def set_freeze(self, mode="none"):
        n = {"none": 0, "partial": 3, "backbone": 5}[mode]
        for i, stage in enumerate(self.stages):
            stage.requires_grad_(i >= n)
        self.frozen = list(self.stages)[:n]
        return self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for stage in self.frozen:
            stage.eval()
        return self

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return self.pool(x).flatten(1)


def build_i3d(num_classes, pretrained=True):
    m = I3DBackbone(pretrained)
    return m, m.feat_dim
