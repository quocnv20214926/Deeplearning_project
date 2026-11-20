import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np

from src import bootstrap as B
from src import evaluate as E
from src import summary as S
from src import visualize as V

ROOT = Path(__file__).resolve().parents[1]


def _model_scores(labels, noise, seed):
    rng = np.random.default_rng(seed)
    return {v: np.clip(y * 0.5 + 0.25 + rng.normal(0, noise, len(y)), 0, 1) for v, y in labels.items()}


def _write_eval(d, tag, mode, scores, labels, vclass, true, pred):
    r = E.evaluate_scores(scores, labels, vclass, 0.5, "NormalVideos", true, pred)
    r.update({"model": tag, "mode": mode, "amp": False})
    E.write_outputs(r, scores, tag, mode, d)


def _bench(tag, mode, prec, device, fps, vram):
    return {"model": tag, "mode": mode, "precision": prec, "device": device, "gpu": "Fake GPU",
            "params_m": 31.0, "file_mb": 120.0, "vram_peak_mb": vram, "ram_mb": 900.0,
            "latency_ms_mean": 16000 / fps, "latency_ms_p95": 17000 / fps, "fps": fps}


def _fake_results(d):
    _, labels, vclass, true, pred = E.make_dummy()
    _write_eval(d, "m1", "cls", _model_scores(labels, 0.15, 1), labels, vclass, true, pred)
    _write_eval(d, "m2", "cls", _model_scores(labels, 0.45, 2), labels, vclass, true, None)
    t = Path(d) / "tables"
    for tag, mode, prec, dev, fps, vram in [("m1", "cls", "fp32", "cuda", 200, 800), ("m1", "cls", "fp16", "cuda", 350, 500),
                                            ("m2", "cls", "fp32", "cuda", 90, 1500), ("m2", "cls", "fp32", "cpu", 4, None),
                                            ("ghost", "cls", "fp32", "cuda", 50, 600)]:   # không có evaluate JSON
        (t / f"{tag}_{mode}_{prec}_{dev}_bench.json").write_text(json.dumps(_bench(tag, mode, prec, dev, fps, vram)))
    return labels


def test_summary_joins_eval_and_bench():
    with tempfile.TemporaryDirectory() as d:
        _fake_results(d)
        rows = S.build_summary(Path(d) / "tables", Path(d) / "tables" / "summary.csv")
        assert len(rows) == 5
        r = next(r for r in rows if r["model"] == "m1" and r["precision"] == "fp16")
        assert 0.8 < r["frame_auc"] <= 1.0 and r["fpr"] is not None and r["normal_video_fpr"] is not None
        assert r["fpr_at_tpr90"] is not None and r["acc_14cls"] is not None and r["fps"] == 350
        ghost = next(r for r in rows if r["model"] == "ghost")
        assert ghost["frame_auc"] is None
        header = (Path(d) / "tables" / "summary.csv").read_text().splitlines()[0].split(",")
        assert header == S.COLUMNS


def test_visualize_writes_all_figures():
    with tempfile.TemporaryDirectory() as d:
        _fake_results(d)
        S.build_summary(Path(d) / "tables", Path(d) / "tables" / "summary.csv")
        evals = V.load_eval_tables(Path(d) / "tables")
        assert set(evals) == {"m1_cls", "m2_cls"}                       # không lẫn file *_bench.json
        V.plot_roc(evals, Path(d) / "roc.png")
        V.plot_per_class(evals, Path(d) / "pc.png")
        rows = V.read_summary(Path(d) / "tables" / "summary.csv")
        assert V.plot_pareto(rows, Path(d) / "p_gpu.png", "fp32", "gpu")
        for f in ("roc.png", "pc.png", "p_gpu.png"):
            assert (Path(d) / f).stat().st_size > 1000


def test_pareto_cpu_row_and_missing_vram_are_handled():
    with tempfile.TemporaryDirectory() as d:
        _fake_results(d)
        S.build_summary(Path(d) / "tables", Path(d) / "tables" / "summary.csv")
        rows = V.read_summary(Path(d) / "tables" / "summary.csv")
        assert V.plot_pareto(rows, Path(d) / "cpu.png", "fp32", "cpu")      # VRAM None -> kích thước mặc định
        assert not V.plot_pareto(rows, Path(d) / "none.png", "onnx", "gpu")  # không có dữ liệu -> False
        for f in ("cpu.png",):
            assert (Path(d) / f).stat().st_size > 1000


def test_bootstrap_ci_and_paired_comparison():
    _, labels, *_ = E.make_dummy()
    a, b = _model_scores(labels, 0.15, 1), _model_scores(labels, 0.45, 2)
    rows = {r["name"]: r for r in B.bootstrap_report({"a": a, "b": b, "a2": a}, labels, n_boot=200, ref="b")}
    assert rows["a"]["ci_low"] <= rows["a"]["auc"] <= rows["a"]["ci_high"]
    assert rows["a"]["diff"] > 0.05 and rows["a"]["p_value"] < 0.05          # a tốt hơn b rõ rệt
    assert abs(rows["a"]["diff"] - rows["a2"]["diff"]) < 1e-12                # cùng điểm -> cùng kết quả
    same = {r["name"]: r for r in B.bootstrap_report({"a": a, "a2": a}, labels, n_boot=200, ref="a")}
    assert same["a2"]["diff"] == 0 and same["a2"]["p_value"] == 1.0


def test_bootstrap_rejects_length_mismatch():
    _, labels, *_ = E.make_dummy()
    bad = _model_scores(labels, 0.2, 3)
    k = sorted(bad)[0]
    bad[k] = bad[k][:-1]
    try:
        B.bootstrap_report({"x": bad}, labels, n_boot=5)
    except ValueError as e:
        assert k in str(e)
    else:
        raise AssertionError("phải báo lỗi")


def test_ablation_variant_parser():
    spec = importlib.util.spec_from_file_location("run_ablation", ROOT / "scripts" / "run_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.parse_variants(["full=train.freeze=none", "x=train.lr=5e-5,train.freeze=partial"]) == [
        ("full", ["train.freeze=none"]), ("x", ["train.lr=5e-5", "train.freeze=partial"])]
    for bad in ("nofield", "name="):
        try:
            mod.parse_variants([bad])
        except ValueError:
            continue
        raise AssertionError("phải báo lỗi")
