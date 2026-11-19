"""[Owner: TV2] Đánh giá trên 290 video test:
frame AUC, AUC theo lớp, FPR@0.5, FPR@TPR90, accuracy 14 lớp (cls), ROC curve.
Kết quả ghi results/tables/<tag>_<mode>.json, điểm từng frame -> results/predictions/<tag>_<mode>.npz.

    python -m src.evaluate --config configs/m1_r2plus1d.yaml --ckpt weights/r2plus1d_18_cls.pt --mode cls
    python -m src.evaluate --dummy          # chạy thử pipeline không cần dữ liệu/model/torch

cls: cửa sổ trượt (clip_len, stride) -> clips_to_frame.
mil: num_segments clip rải đều -> snippet_to_frame.
torch/cv2 chỉ import khi thật sự chạy model, nên phần tính metric test được độc lập.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils import metrics as M
from src.utils.model_io import get_class_names, load_cfg, normal_index

TPR_TARGET = 0.9
ROC_GRID = np.linspace(0, 1, 201)


# --------------------------------------------------------------------------- #
# Chỉ số clip (thuần numpy)
# --------------------------------------------------------------------------- #
def sliding_clip_starts(n, clip_len=16, stride=8):
    """Clip k bắt đầu tại k*stride (khớp metrics.clips_to_frame). Đuôi ngắn hơn stride
    được điền bằng điểm clip gần nhất."""
    return list(range(0, max(n - clip_len, 0) + 1, stride))


def segment_starts(n, num_segments=32, clip_len=16):
    return np.linspace(0, max(n - clip_len, 0), num_segments).round().astype(int).tolist()


def clip_frame_idx(start, clip_len, n):
    """Chỉ số frame của 1 clip; video ngắn hơn clip_len -> lặp frame cuối."""
    return np.clip(start + np.arange(clip_len), 0, n - 1)


# --------------------------------------------------------------------------- #
# Tính toàn bộ metric (thuần numpy, test được)
# --------------------------------------------------------------------------- #
def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def evaluate_scores(scores, labels, video_class, thr=0.5, normal="Normal",
                    true_cls_idx=None, pred_cls_idx=None):
    """scores/labels: {video: array}; video_class: {video: tên lớp}."""
    fpr, tpr, _ = M.roc_points(scores, labels)
    res = {
        "n_videos": len(labels),
        "n_frames": int(sum(len(v) for v in labels.values())),
        "frame_auc": M.frame_auc(scores, labels),
        "per_class_auc": M.per_class_auc(scores, labels, video_class, normal),
        f"fpr@{thr}": M.false_positive_rate(scores, labels, thr),
        f"normal_video_fpr@{thr}": M.normal_video_fpr(scores, labels, video_class, thr, normal),
        f"fpr@tpr{int(TPR_TARGET * 100)}": M.fpr_at_tpr(scores, labels, TPR_TARGET),
        "roc": {"fpr": ROC_GRID.tolist(), "tpr": np.interp(ROC_GRID, fpr, tpr).tolist()},
    }
    if true_cls_idx is not None and pred_cls_idx is not None:
        res["acc_14cls"] = M.multiclass_accuracy(true_cls_idx, pred_cls_idx)
    return res


def write_outputs(res, scores, tag, mode, out_dir="results"):
    out = Path(out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    jp = out / "tables" / f"{tag}_{mode}.json"
    jp.write_text(json.dumps(_clean(res), indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / "predictions" / f"{tag}_{mode}.npz", **scores)
    return jp


# --------------------------------------------------------------------------- #
# Dữ liệu test (CẦN KHỚP với TV1: tên file ảnh, vị trí npz nhãn, tiền xử lý)
# --------------------------------------------------------------------------- #
def list_test_videos(cfg):
    """{video: (tên lớp, [đường dẫn ảnh theo thứ tự])}. Giả định tên ảnh '<Video>_<idx>.png'
    (Kaggle: Abuse028_x264_10.png). Thay bằng hàm của TV1 khi có."""
    root = Path(cfg["data"]["frames_root"]) / "Test"
    out = {}
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        groups = defaultdict(list)
        for f in cdir.glob("*.png"):
            stem, _, idx = f.stem.rpartition("_")
            groups[stem].append((int(idx), f))
        for v, fr in groups.items():
            out[v] = (cdir.name, [p for _, p in sorted(fr)])
    return out


def load_labels(path):
    d = np.load(path, allow_pickle=True)
    return {k: np.asarray(d[k]).astype(int) for k in d.files}


def check_alignment(videos, labels):
    a, b = set(videos), set(labels)
    if a != b:
        raise ValueError(f"Video ảnh vs nhãn không khớp: thiếu nhãn={sorted(a - b)[:5]}, "
                         f"thừa nhãn={sorted(b - a)[:5]}")
    bad = {v: (len(videos[v][1]), len(labels[v])) for v in a if len(videos[v][1]) != len(labels[v])}
    if bad:
        raise ValueError(f"Số frame ≠ số nhãn (frame, nhãn): {dict(list(bad.items())[:5])}")


def load_frames_u8(paths, size):
    """Ảnh -> uint8 (N,H,W,3). TODO: dùng ĐÚNG transform eval của TV1 để khớp lúc train."""
    import cv2
    fr = []
    for p in paths:
        im = cv2.imread(str(p))[:, :, ::-1]
        h, w = im.shape[:2]
        s = min(h, w)
        im = im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        fr.append(cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR))
    return np.stack(fr)


MEAN = (0.43216, 0.394666, 0.37645)   # Kinetics (torchvision video)
STD = (0.22803, 0.22145, 0.216989)


# --------------------------------------------------------------------------- #
# Chạy model (cần torch)
# --------------------------------------------------------------------------- #
def score_video(model, frames_u8, starts, clip_len, mode, normal_idx, device,
                batch_size=16, amp=True, mil_output="logit"):
    """Trả (điểm bất thường mỗi clip [n_clip], xác suất lớp mỗi clip [n_clip,K] hoặc None)."""
    import torch

    f = torch.from_numpy(frames_u8)                       # N,H,W,3 uint8
    mean = torch.tensor(MEAN, device=device).view(3, 1, 1, 1)
    std = torch.tensor(STD, device=device).view(3, 1, 1, 1)
    n = f.shape[0]
    scores, probs = [], []
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            clips = []
            for s in starts[i:i + batch_size]:
                c = f[torch.from_numpy(clip_frame_idx(s, clip_len, n))]    # T,H,W,3
                clips.append(c.permute(3, 0, 1, 2))                         # 3,T,H,W
            x = torch.stack(clips).to(device).float().div_(255)
            x = (x - mean) / std
            with torch.autocast(device_type=device.split(":")[0], enabled=amp and device != "cpu"):
                out = model(x)
            out = out.float()
            if mode == "cls":
                p = out.softmax(-1)
                scores.append((1 - p[:, normal_idx]).cpu())
                probs.append(p.cpu())
            else:
                o = out.reshape(len(clips), -1)[:, 0]
                scores.append((o if mil_output == "prob" else o.sigmoid()).cpu())
    s = torch.cat(scores).numpy()
    return s, (torch.cat(probs).numpy() if probs else None)


def run_model_eval(cfg, args):
    import torch

    from src.utils.model_io import load_model

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = cfg["data"]
    class_names = get_class_names(cfg)
    nidx = normal_index(class_names, d["normal_class"])
    videos = list_test_videos(cfg)
    labels = load_labels(args.labels or Path(d["split_dir"]) / "test_frame_labels.npz")
    check_alignment(videos, labels)

    model = load_model(cfg, args.mode, args.ckpt, device)
    scores, video_class, true_idx, pred_idx = {}, {}, [], []
    for i, v in enumerate(sorted(videos)):
        cls, paths = videos[v]
        frames = load_frames_u8(paths, d["input_size"])
        n = len(frames)
        if args.mode == "cls":
            starts = sliding_clip_starts(n, d["clip_len"], d["clip_stride"])
        else:
            starts = segment_starts(n, cfg["mil"]["num_segments"], d["clip_len"])
        s, p = score_video(model, frames, starts, d["clip_len"], args.mode, nidx, device,
                           args.batch_size, not args.no_amp, args.mil_output)
        scores[v] = (M.clips_to_frame(s, n, d["clip_len"], d["clip_stride"]) if args.mode == "cls"
                     else M.snippet_to_frame(s, n))
        video_class[v] = cls
        if p is not None:
            true_idx.append(class_names.index(cls))
            pred_idx.append(int(p.mean(0).argmax()))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(videos)} videos", flush=True)
    return scores, labels, video_class, (true_idx or None), (pred_idx or None)


# --------------------------------------------------------------------------- #
# Chế độ chạy thử (không cần torch/dữ liệu)
# --------------------------------------------------------------------------- #
def make_dummy(seed=0, noise=0.25):
    rng = np.random.default_rng(seed)
    classes = ["Abuse", "Arson", "Burglary", "Shoplifting", "NormalVideos"]
    scores, labels, vclass = {}, {}, {}
    for ci, c in enumerate(classes):
        for j in range(6):
            v, n = f"{c}{j:03d}_x264", int(rng.integers(120, 300))
            y = np.zeros(n, int)
            if c != "NormalVideos":
                a = int(rng.integers(10, n // 2))
                y[a:a + int(rng.integers(20, n // 2))] = 1
            labels[v] = y
            scores[v] = np.clip(y * 0.5 + 0.25 + rng.normal(0, noise, n), 0, 1)
            vclass[v] = c
    true = [classes.index(vclass[v]) for v in sorted(vclass)]
    pred = [t if rng.random() < 0.7 else int(rng.integers(len(classes))) for t in true]
    return scores, labels, vclass, true, pred


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt")
    ap.add_argument("--mode", choices=["cls", "mil"], default="cls")
    ap.add_argument("--tag", help="mặc định: tên file config")
    ap.add_argument("--labels", help="mặc định: <split_dir>/test_frame_labels.npz")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--device")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--mil_output", choices=["logit", "prob"], default="logit")
    ap.add_argument("--dummy", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config) if Path(args.config).exists() else None
    if args.dummy:
        cfg = cfg or {"eval": {"fpr_threshold": 0.5}, "data": {"normal_class": "NormalVideos"}}
        tag = args.tag or "dummy"
        scores, labels, vclass, ti, pi = make_dummy()
    else:
        if not args.ckpt:
            ap.error("--ckpt là bắt buộc (hoặc dùng --dummy)")
        tag = args.tag or Path(args.config).stem
        t0 = time.time()
        scores, labels, vclass, ti, pi = run_model_eval(cfg, args)
        print(f"Inference xong sau {time.time() - t0:.0f}s")

    res = evaluate_scores(scores, labels, vclass, cfg["eval"]["fpr_threshold"],
                          cfg["data"]["normal_class"], ti, pi)
    res.update({"model": tag, "mode": args.mode, "ckpt": args.ckpt})
    path = write_outputs(res, scores, tag, args.mode, args.out_dir)
    print(f"frame AUC = {res['frame_auc']:.4f} | acc14 = {res.get('acc_14cls')} -> {path}")


if __name__ == "__main__":
    main()
