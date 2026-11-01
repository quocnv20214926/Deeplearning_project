"""[Owner: TV1] Dataset clip cho chế độ phân loại (cls).

TODO(TV1):
- Quét Train/<Class>/*.png, nhóm ảnh theo video (tên file: <Video>_<idx>.png), sort theo idx.
- Tạo split train/val (90/10 theo video, cố định seed) -> data/processed/splits/*.txt
- __getitem__: lấy clip_len ảnh liên tiếp (stride clip_stride), resize input_size,
  chuẩn hoá theo mean/std Kinetics, trả tensor (C, T, H, W), label lớp, label nhị phân.
- Cân bằng Normal/Anomaly bằng WeightedRandomSampler.
"""
import torch
from torch.utils.data import Dataset


class UCFCrimeClips(Dataset):
    def __init__(self, cfg, split="train", transform=None):
        self.cfg, self.split, self.transform = cfg, split, transform
        self.items = []  # list[(video, start_idx, class_id)]
        raise NotImplementedError("TV1: implement indexing")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        raise NotImplementedError
