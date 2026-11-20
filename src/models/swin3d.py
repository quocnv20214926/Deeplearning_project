"""[Owner: TV3] M3: Video Swin-T pretrained Kinetics-400 (Liu et al., CVPR 2022), torchvision `swin3d_t`.

Clip 16x112x112 -> patch 2x4x4 -> lưới token 8x28x28 -> 4 stage (cửa sổ 8x7x7) -> feature 768-d.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models.video import Swin3D_T_Weights, swin3d_t

from src.data.transforms import KINETICS_MEAN, KINETICS_STD

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Swin3DBackbone(nn.Module):
    """(B, 3, T, H, W) -> feature (B, 768).

    - Clip vào chuẩn hoá Kinetics (transforms.py, dùng chung cả nhóm) nhưng trọng số Swin3D K400 của
      torchvision được train với chuẩn hoá ImageNet -> đổi chuẩn hoá bên trong (giống MobileNet-GRU của TV1).
    - set_freeze("none" | "partial" | "backbone"): "partial" đóng băng patch embedding + stage 1–2,
      chỉ fine-tune stage 3–4 (stage 3 chiếm phần lớn tham số).
    - set_grad_checkpointing(True): tính lại activation từng Swin block khi backward -> giảm VRAM
      đáng kể, đổi lại chậm hơn ~30%.
    """

    feat_dim = 768

    def __init__(self, pretrained=True):
        super().__init__()
        net = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1 if pretrained else None)
        net.head = nn.Identity()
        self.net = net
        self.frozen = []
        self.grad_checkpointing = False
        k_mean, k_std = torch.tensor(KINETICS_MEAN), torch.tensor(KINETICS_STD)
        i_mean, i_std = torch.tensor(IMAGENET_MEAN), torch.tensor(IMAGENET_STD)
        # x_imagenet = x_kinetics * k_std / i_std + (k_mean - i_mean) / i_std
        self.register_buffer("scale", (k_std / i_std).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("shift", ((k_mean - i_mean) / i_std).view(1, 3, 1, 1, 1), persistent=False)

    def stages(self):
        f = self.net.features  # [stage1, merge, stage2, merge, stage3, merge, stage4]
        return [self.net.patch_embed, f[0], f[1], f[2], f[3], f[4], f[5], f[6], self.net.norm]

    def set_freeze(self, mode="none"):
        stages = self.stages()
        n = {"none": 0, "partial": 4, "backbone": len(stages)}[mode]
        for i, stage in enumerate(stages):
            stage.requires_grad_(i >= n)
        self.frozen = stages[:n]
        return self.train(self.training)

    def set_grad_checkpointing(self, enabled=True):
        self.grad_checkpointing = enabled
        return self

    def train(self, mode=True):
        super().train(mode)
        for stage in self.frozen:
            stage.eval()
        return self

    def forward(self, x):
        x = x * self.scale + self.shift
        n = self.net
        x = n.pos_drop(n.patch_embed(x))  # B, T', H', W', C
        for layer in n.features:
            if self.grad_checkpointing and self.training and isinstance(layer, nn.Sequential):
                for block in layer:
                    x = checkpoint(block, x, use_reentrant=False) if x.requires_grad else block(x)
            else:
                x = layer(x)
        x = n.norm(x).permute(0, 4, 1, 2, 3)
        return torch.flatten(n.avgpool(x), 1)


def build_swin3d(num_classes, pretrained=True):
    m = Swin3DBackbone(pretrained)
    return m, m.feat_dim
