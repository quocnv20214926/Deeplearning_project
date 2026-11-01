import numpy as np
from src.utils.metrics import frame_auc, false_positive_rate, snippet_to_frame
from src.losses.mil_loss import mil_ranking_loss
import torch


def test_frame_auc_perfect():
    labels = {"a": np.array([0, 0, 1, 1]), "b": np.array([0, 0, 0, 0])}
    scores = {"a": np.array([.1, .2, .9, .8]), "b": np.array([.1, .1, .2, .3])}
    assert frame_auc(scores, labels) == 1.0
    assert false_positive_rate(scores, labels, 0.5) == 0.0


def test_snippet_to_frame():
    assert len(snippet_to_frame([0.1, 0.9], 10)) == 10


def test_mil_loss_positive():
    assert mil_ranking_loss(torch.rand(2, 32), torch.rand(2, 32)).item() >= 0
