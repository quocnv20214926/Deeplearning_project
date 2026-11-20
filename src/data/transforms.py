"""[Owner: TV1] Tiền xử lý dùng chung cho train, eval VÀ app (phải giống hệt nhau).

Đầu vào: các frame uint8 (T, H, W, 3) theo thứ tự màu **RGB** (OpenCV đọc ra BGR -> cần
`frame[..., ::-1]` trước). Đầu ra: tensor float32 (3, T, size, size) chuẩn hoá theo Kinetics.

Mô hình học trên ảnh Kaggle 64x64, nên frame lớn hơn (video thật trong app) được thu nhỏ
về 64x64 trước rồi mới phóng lên `size` — giống hệt đường đi của dữ liệu lúc train.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .constants import SOURCE_SIZE

KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


class ClipTransform:
    """train=True: random resized crop (60–100% diện tích) + lật ngang p=0,5, áp dụng giống
    nhau cho mọi frame trong clip (giữ tính liên tục của chuyển động)."""

    def __init__(self, size=112, train=False, source_size=SOURCE_SIZE, crop_scale=(0.6, 1.0), flip_p=0.5):
        self.size, self.train, self.source_size = size, train, source_size
        self.crop_scale, self.flip_p = crop_scale, flip_p
        self.mean = torch.tensor(KINETICS_MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(KINETICS_STD).view(3, 1, 1, 1)

    def __call__(self, frames):
        x = torch.as_tensor(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float().div_(255)  # (T, 3, H, W)
        if self.source_size and x.shape[-2:] != (self.source_size, self.source_size):
            x = F.interpolate(x, size=(self.source_size, self.source_size), mode="area")
        if self.train:
            x = self._augment(x)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return (x.permute(1, 0, 2, 3) - self.mean) / self.std  # (3, T, size, size)

    def _augment(self, x):
        _, _, h, w = x.shape
        scale = np.random.uniform(*self.crop_scale)
        ch, cw = max(1, int(h * scale ** 0.5)), max(1, int(w * scale ** 0.5))
        top, left = np.random.randint(0, h - ch + 1), np.random.randint(0, w - cw + 1)
        x = x[:, :, top:top + ch, left:left + cw]
        if np.random.rand() < self.flip_p:
            x = x.flip(-1)
        return x


def build_train_transform(cfg):
    return ClipTransform(cfg["data"]["input_size"], train=True)


def build_eval_transform(cfg):
    return ClipTransform(cfg["data"]["input_size"], train=False)
