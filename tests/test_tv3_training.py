"""[Owner: TV3] Test framework train / MIL / checkpoint trên dữ liệu giả (CPU, không cần dữ liệu Kaggle, ~1 phút).

    pytest tests/test_tv3_training.py
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models import build_model
from src.models.heads import ClsHead, MILHead
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
HAS_PTV = importlib.util.find_spec("pytorchvideo") is not None
DIMS = {"r2plus1d_18": 512, "mobilenet_gru": 256, "i3d_r50": 2048, "swin3d_t": 768, "x3d_s": 2048}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_config_overrides(monkeypatch):
    monkeypatch.setenv("CFG_OVERRIDES", "train.batch_size=4;data.frames_root=/kaggle/input/x")
    cfg = load_config(ROOT / "configs/m3_swin3d.yaml", ["train.lr=5e-5", "model.grad_checkpointing=true"])
    assert cfg["train"]["batch_size"] == 4 and cfg["data"]["frames_root"] == "/kaggle/input/x"
    assert cfg["train"]["lr"] == 5e-5 and cfg["model"]["grad_checkpointing"] is True
    assert cfg["model"]["name"] == "swin3d_t" and cfg["data"]["clip_len"] == 16  # vẫn kế thừa base.yaml


# --------------------------------------------------------------------------- #
# Head với mọi backbone (task tuần 1)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(DIMS))
def test_every_backbone_with_heads(name):
    if name in ("i3d_r50", "x3d_s") and not HAS_PTV:
        pytest.skip("cần pip install pytorchvideo")
    backbone, feat_dim = build_model(name, num_classes=14, pretrained=False)
    assert feat_dim == DIMS[name]
    with torch.no_grad():
        f = backbone.eval()(torch.randn(2, 3, 16, 112, 112))  # giao thức chung: 16 frame, 112px
        assert f.shape == (2, feat_dim)
        assert ClsHead(feat_dim, 14).eval()(f).shape == (2, 14)
        s = MILHead(feat_dim).eval()(torch.randn(2, 32, feat_dim))
        assert s.shape == (2, 32) and ((s >= 0) & (s <= 1)).all()


def test_swin_renormalisation_and_forward_match_torchvision():
    from src.data.transforms import KINETICS_MEAN, KINETICS_STD
    from src.models.swin3d import IMAGENET_MEAN, IMAGENET_STD, Swin3DBackbone
    m = Swin3DBackbone(pretrained=False).eval()
    pixels = torch.rand(1, 3, 16, 64, 64)
    view = lambda v: torch.tensor(v).view(1, 3, 1, 1, 1)  # noqa: E731
    x_kin = (pixels - view(KINETICS_MEAN)) / view(KINETICS_STD)
    x_img = (pixels - view(IMAGENET_MEAN)) / view(IMAGENET_STD)
    with torch.no_grad():
        assert torch.allclose(m(x_kin), m.net(x_img), atol=1e-4)  # forward tự viết == forward torchvision


def test_swin_freeze_and_grad_checkpointing():
    from src.models.swin3d import Swin3DBackbone
    m = Swin3DBackbone(pretrained=False)
    m.set_freeze("partial")
    assert not any(p.requires_grad for p in m.net.patch_embed.parameters())
    assert all(p.requires_grad for p in m.net.features[4].parameters())
    m.set_freeze("none").train()
    x = torch.randn(1, 3, 16, 64, 64)
    torch.manual_seed(0)
    m.net.features.apply(lambda mod: mod.eval() if isinstance(mod, torch.nn.Dropout) else None)
    m.eval()
    with torch.no_grad():
        ref = m(x)
    m.set_grad_checkpointing(True).train()
    for mod in m.modules():  # tắt dropout/stochastic depth để so sánh
        if mod.__class__.__name__ in ("Dropout", "StochasticDepth"):
            mod.eval()
    out = m(x)
    out.sum().backward()
    assert torch.allclose(out, ref, atol=1e-4)
    assert m.net.features[0][0].attn.qkv.weight.grad is not None


# --------------------------------------------------------------------------- #
# Pipeline đầu-cuối trên dữ liệu giả: train cls -> extract -> train MIL -> evaluate.py (TV2)
# --------------------------------------------------------------------------- #
CLASSES_USED = ["NormalVideos", "Abuse", "Fighting"]


def make_fake_dataset(root, rng):
    """Frame packed (.npy) cho train/val/test + PNG cho test (evaluate.py đọc PNG) + split + nhãn frame."""
    from PIL import Image
    packed, raw, splits = root / "frames", root / "raw", root / "splits"
    splits.mkdir(parents=True)
    lines = {"train": [], "val": [], "test": []}
    labels = {}
    for split, folder, per_class in [("train", "Train", 3), ("val", "Train", 2), ("test", "Test", 2)]:
        (packed / folder).mkdir(parents=True, exist_ok=True)
        for c in CLASSES_USED:
            for j in range(per_class):
                v = f"{c}{split}{j:02d}_x264"
                n = int(rng.integers(20, 60))
                frames = rng.integers(0, 60, (n, 64, 64, 3), dtype=np.uint8)
                y = np.zeros(n, int)
                if c != "NormalVideos":  # đoạn bất thường sáng hơn -> model học được
                    a = n // 3
                    frames[a:a + n // 3] += 150
                    y[a:a + n // 3] = 1
                np.save(packed / folder / f"{v}.npy", frames)
                lines[split].append(f"{v} {c} {n}")
                if split == "test":
                    labels[v] = y
                    d = raw / "Test" / c
                    d.mkdir(parents=True, exist_ok=True)
                    for i, fr in enumerate(frames):
                        Image.fromarray(fr).save(d / f"{v}_{i * 10}.png")
    for s, ls in lines.items():
        (splits / f"{s}.txt").write_text("\n".join(ls) + "\n", encoding="utf-8")
    np.savez_compressed(splits / "test_frame_labels.npz", **labels)
    return [f"data.packed_dir={packed}", f"data.split_dir={splits}", f"data.frames_root={raw}"]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tv3")
    overrides = make_fake_dataset(tmp, np.random.default_rng(0)) + [
        "data.input_size=32", "train.batch_size=4", "train.num_workers=0", "train.epochs=2",
        "train.warmup_epochs=0", "train.lr=1e-3", "mil.batch_size=4", "mil.num_workers=0", "mil.eval_every=10"]
    return tmp, overrides


def test_pipeline_end_to_end(pipeline):
    from src.extract_features import extract_split
    from src.train import build_loaders, fit
    from src.train_mil import train_mil
    from src.utils.model_io import apply_freeze, build_video_model, load_backbone, load_model

    tmp, overrides = pipeline
    cfg = load_config(ROOT / "configs/b0_mobilenet_gru.yaml", overrides)
    torch.manual_seed(0)
    model, _ = build_video_model(cfg, "cls", pretrained=False)
    apply_freeze(model.backbone, "none")
    tl, vl, vids = build_loaders(cfg)
    best_path, best = fit(cfg, model, tl, vl, vids, "cpu", out_dir=tmp / "weights", use_wandb=False,
                          log_dir=tmp / "logs")
    assert best_path.exists() and np.isfinite(best)
    assert (tmp / "weights" / "mobilenet_gru_cls_last.pt").exists()
    assert len((tmp / "logs" / "mobilenet_gru_cls.csv").read_text().splitlines()) == 3  # header + 2 epoch

    # resume từ last checkpoint: epoch 3
    cfg3 = load_config(ROOT / "configs/b0_mobilenet_gru.yaml", overrides + ["train.epochs=3"])
    model3, _ = build_video_model(cfg3, "cls", pretrained=False)
    fit(cfg3, model3, tl, vl, vids, "cpu", out_dir=tmp / "weights", use_wandb=False, log_dir=tmp / "logs3",
        resume=tmp / "weights" / "mobilenet_gru_cls_last.pt")
    assert len((tmp / "logs3" / "mobilenet_gru_cls.csv").read_text().splitlines()) == 2  # chỉ epoch 3

    # checkpoint cls đọc lại đúng
    m2 = load_model(cfg, "cls", best_path)
    x = torch.randn(2, 3, 16, 32, 32)
    assert m2(x).shape == (2, 14)

    # trích feature (mọi split) -> đúng số snippet
    backbone, feat_dim = load_backbone(cfg, best_path)
    fdir = tmp / "features" / "mobilenet_gru"
    for split in ["train", "val", "test"]:
        extract_split(cfg, backbone, split, fdir, "cpu", batch_size=5, num_workers=0)
    (fdir / "meta.json").write_text(json.dumps({"feat_dim": feat_dim, "ckpt": str(best_path)}))
    f = np.load(next((fdir / "test").glob("*.npy")))
    assert f.dtype == np.float16 and f.shape[1] == 256
    assert extract_split(cfg, backbone, "test", fdir, "cpu", num_workers=0) == 0  # đã có -> bỏ qua

    # MIL trên feature + chấm test
    ckpt, best_mil, res = train_mil(cfg, fdir, "cpu", out_dir=tmp / "weights", results_dir=tmp / "results",
                                    iterations=30)
    assert ckpt.exists() and res["n_videos"] == 6 and 0 <= res["frame_auc"] <= 1
    assert (tmp / "results" / "tables" / "mobilenet_gru_mil.json").exists()
    mil = load_model(cfg, "mil", ckpt)
    out = mil(x)
    assert out.shape == (2,) and torch.isfinite(out).all()


@pytest.mark.parametrize("mode", ["cls", "mil"])
def test_evaluate_cli_loads_tv3_checkpoints(pipeline, mode):
    """Giao kèo với TV2: src.evaluate đọc được checkpoint do train.py / train_mil.py tạo ra."""
    tmp, overrides = pipeline
    ckpt = tmp / "weights" / f"mobilenet_gru_{mode}.pt"
    if not ckpt.exists():
        pytest.skip("cần chạy test_pipeline_end_to_end trước")
    env = {**os.environ, "CFG_OVERRIDES": ";".join(overrides)}
    r = subprocess.run([sys.executable, "-m", "src.evaluate", "--config", "configs/b0_mobilenet_gru.yaml",
                        "--ckpt", str(ckpt), "--mode", mode, "--out_dir", str(tmp / "eval"), "--device", "cpu"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((tmp / "eval" / "tables" / f"b0_mobilenet_gru_{mode}.json").read_text(encoding="utf-8"))
    assert res["n_videos"] == 6
