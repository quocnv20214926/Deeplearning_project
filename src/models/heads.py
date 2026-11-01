"""[Owner: TV3] Head dùng chung cho mọi backbone."""
import torch.nn as nn


class ClsHead(nn.Module):
    """Chế độ (a): phân loại 14 lớp. score bất thường = 1 - softmax[normal]."""

    def __init__(self, feat_dim, num_classes=14, p=0.5):
        super().__init__()
        self.fc = nn.Sequential(nn.Dropout(p), nn.Linear(feat_dim, num_classes))

    def forward(self, f):
        return self.fc(f)


class MILHead(nn.Module):
    """Chế độ (b): chấm điểm bất thường cho từng segment (Sultani et al. 2018)."""

    def __init__(self, feat_dim, p=0.6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(), nn.Dropout(p),
            nn.Linear(512, 32), nn.Dropout(p),
            nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):            # (B, S, D) -> (B, S)
        return self.net(x).squeeze(-1)
