import json
import tempfile
from pathlib import Path

import numpy as np

from src import evaluate as E
from src.benchmark import summarize_latencies
from src.utils import metrics as M


def _ref_clip_starts(n_frames, clip_len, stride):          # bản sao clip_starts của TV1
    last = max(n_frames - clip_len, 0)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def test_assemble_matches_ucfcrimeclips_items_layout():
    n_frames = {"A_x264": 40, "N_x264": 20}
    items, cs, cp = [], [], []
    for v, cid in (("A_x264", 3), ("N_x264", 0)):
        for st in _ref_clip_starts(n_frames[v], 16, 16):
            items.append((v, st, cid))
            cs.append(0.9 if cid else 0.1)
            cp.append([0.1, 0.0, 0.0, 0.9] if cid else [0.9, 0.0, 0.0, 0.1])
    scores, true_id, pred_id = E.assemble(items, np.array(cs), np.array(cp), n_frames, 16)
    assert {v: len(s) for v, s in scores.items()} == n_frames
    assert np.allclose(scores["A_x264"], 0.9) and np.allclose(scores["N_x264"], 0.1)
    assert true_id == {"A_x264": 3, "N_x264": 0} and pred_id == {"A_x264": 3, "N_x264": 0}


def test_assemble_interleaved_order_is_handled():
    n_frames = {"a": 16, "b": 16}
    items = [("a", 0, 1), ("b", 0, 0)]
    scores, _, _ = E.assemble(items, np.array([0.8, 0.2]), np.array([[0.2, 0.8], [0.8, 0.2]]), n_frames, 16)
    assert np.allclose(scores["a"], 0.8) and np.allclose(scores["b"], 0.2)


def test_evaluate_scores_dummy_has_all_keys_and_sane_values():
    s, l, vc, ti, pi = E.make_dummy()
    r = E.evaluate_scores(s, l, vc, 0.5, "NormalVideos", ti, pi)
    for k in ("frame_auc", "per_class_auc", "fpr@0.5", "normal_video_fpr@0.5", "fpr@tpr90", "roc", "acc_14cls"):
        assert k in r
    assert 0.8 < r["frame_auc"] <= 1.0
    assert "NormalVideos" not in r["per_class_auc"]
    assert len(r["roc"]["fpr"]) == len(r["roc"]["tpr"]) == 201


def test_write_outputs_roundtrip_and_nan_is_json_safe():
    s, l, vc, ti, pi = E.make_dummy()
    r = E.evaluate_scores(s, l, vc, 0.5, "NormalVideos", ti, pi)
    r["per_class_auc"]["X"] = float("nan")
    with tempfile.TemporaryDirectory() as d:
        p = E.write_outputs(r, s, "t", "cls", d)
        assert json.loads(Path(p).read_text())["per_class_auc"]["X"] is None
        assert set(np.load(Path(d) / "predictions" / "t_cls.npz").files) == set(s)


def test_check_alignment_errors_are_descriptive():
    labels = {"a": np.zeros(3), "b": np.zeros(3)}
    for n in ({"a": 3, "b": 2}, {"a": 3, "c": 3}):
        try:
            E.check_alignment(n, labels)
        except ValueError:
            continue
        raise AssertionError("phải báo lỗi")
    E.check_alignment({"a": 3, "b": 3}, labels)


def test_latency_summary():
    r = summarize_latencies([10.0] * 100, clip_len=16)
    assert r["latency_ms_mean"] == 10.0 and r["clips_per_s"] == 100.0 and r["fps"] == 1600.0


def test_assemble_mil_has_no_class_predictions():
    n_frames = {"a": 20}
    items = [("a", 0, 2), ("a", 4, 2)]
    scores, true_id, pred_id = E.assemble(items, np.array([0.2, 0.6]), None, n_frames, 16)
    assert len(scores["a"]) == 20 and true_id == {"a": 2} and pred_id == {}
    assert scores["a"][0] == 0.2 and abs(scores["a"][10] - 0.4) < 1e-9 and scores["a"][19] == 0.6
