import numpy as np
import pytest

from src.utils import metrics as M


def _toy(perfect=True, seed=0):
    rng = np.random.default_rng(seed)
    labels, scores, cls = {}, {}, {}
    for i, c in enumerate(["Abuse", "Arson", "Normal", "Normal"]):
        v = f"v{i}"
        y = np.zeros(100, int)
        if c != "Normal":
            y[30:60] = 1
        labels[v] = y
        scores[v] = (y * 0.8 + 0.1) if perfect else rng.random(100)
        cls[v] = c
    return scores, labels, cls


def test_perfect_auc():
    s, l, _ = _toy()
    assert M.frame_auc(s, l) == 1.0


def test_inverted_auc():
    s, l, _ = _toy()
    assert M.frame_auc({k: 1 - v for k, v in s.items()}, l) == 0.0


def test_random_auc_near_half():
    s, l, _ = _toy(perfect=False)
    assert abs(M.frame_auc(s, l) - 0.5) < 0.1


def test_length_mismatch_raises():
    s, l, _ = _toy()
    s["v0"] = s["v0"][:-1]
    with pytest.raises(ValueError):
        M.frame_auc(s, l)


def test_per_class_excludes_normal():
    s, l, c = _toy()
    out = M.per_class_auc(s, l, c)
    assert set(out) == {"Abuse", "Arson"} and all(v == 1.0 for v in out.values())


def test_per_class_ignores_classes_not_in_labels():
    s, l, c = _toy()
    c["extra_train_video"] = "Robbery"  # không có trong labels
    assert "Robbery" not in M.per_class_auc(s, l, c)


def test_fpr_and_fpr_at_tpr_perfect():
    s, l, c = _toy()
    assert M.false_positive_rate(s, l, 0.5) == 0.0
    assert M.normal_video_fpr(s, l, c, 0.5) == 0.0
    assert M.fpr_at_tpr(s, l, 0.9) == 0.0


def test_fpr_all_alarm():
    s, l, _ = _toy()
    assert M.false_positive_rate(s, l, 0.0) == 1.0


def test_threshold_at_fpr_respects_target():
    s, l, _ = _toy(perfect=False)
    t = M.threshold_at_fpr(s, l, 0.05)
    assert M.false_positive_rate(s, l, t) <= 0.05


def test_bootstrap_with_repeated_videos():
    s, l, _ = _toy()
    assert M.frame_auc(s, l, vids=["v0", "v0", "v2", "v3"]) == 1.0


def test_multiclass_accuracy():
    assert M.multiclass_accuracy([0, 1, 2, 2], [0, 1, 1, 2]) == 0.75


def test_snippet_to_frame_length_and_values():
    out = M.snippet_to_frame([0.1, 0.9], 10)
    assert len(out) == 10 and out[0] == 0.1 and out[-1] == 0.9


def test_clips_to_frame_length_and_overlap():
    out = M.clips_to_frame([0.0, 1.0], num_frames=24, clip_len=16, stride=8)
    assert len(out) == 24
    assert out[0] == 0.0 and out[12] == 0.5 and out[23] == 1.0


def test_clips_to_frame_tail_filled():
    out = M.clips_to_frame([0.2], num_frames=20, clip_len=16, stride=8)
    assert len(out) == 20 and np.all(out == 0.2)
