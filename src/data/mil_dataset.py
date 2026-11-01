"""[Owner: TV3] Dataset cho MIL: mỗi item = 1 video -> (num_segments, feat_dim) feature.

Feature đọc từ data/features/<model>/<split>/<video>.npy (do extract_features.py tạo).
Batch gồm nửa video Normal, nửa video Anomaly.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def to_segments(feat, num_segments=32):
    """Gộp T snippet feature thành num_segments đoạn bằng trung bình."""
    T = feat.shape[0]
    edges = np.linspace(0, T, num_segments + 1).astype(int)
    return np.stack([feat[a:max(b, a + 1)].mean(0) for a, b in zip(edges[:-1], edges[1:])])


class MILFeatureDataset(Dataset):
    def __init__(self, feature_dir, video_list, labels, num_segments=32):
        self.dir, self.videos, self.labels, self.k = feature_dir, video_list, labels, num_segments

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        f = np.load(f"{self.dir}/{self.videos[i]}.npy")
        return torch.from_numpy(to_segments(f, self.k)).float(), self.labels[i]
