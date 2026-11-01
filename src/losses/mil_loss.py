"""[Owner: TV3] MIL ranking loss (Sultani et al., CVPR 2018)."""
import torch


def mil_ranking_loss(s_anom, s_norm, lambda_smooth=8e-5, lambda_sparse=8e-5):
    """s_anom, s_norm: (B, S) điểm segment của video bất thường / bình thường."""
    rank = torch.relu(1.0 - s_anom.max(1).values + s_norm.max(1).values).mean()
    smooth = ((s_anom[:, 1:] - s_anom[:, :-1]) ** 2).sum(1).mean()
    sparse = s_anom.sum(1).mean()
    return rank + lambda_smooth * smooth + lambda_sparse * sparse
