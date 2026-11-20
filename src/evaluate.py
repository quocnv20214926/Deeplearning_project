"""[Owner: TV2] Đánh giá trên 290 video test (dùng đúng pipeline dữ liệu của TV1):
frame AUC, AUC theo lớp, FPR@0.5, FPR@TPR90, FPR trên video Normal, accuracy 14 lớp, ROC curve.
Kết quả: results/tables/<tag>_<mode>.json, điểm từng frame: results/predictions/<tag>_<mode>.npz.

    python -m src.evaluate --config configs/m1_r2plus1d.yaml --ckpt weights/r2plus1d_18_cls.pt
    python -m src.evaluate --config configs/m2_i3d.yaml --ckpt weights/i3d_r50_partial_cls.pt --tag i3d_r50_partial
    python -m src.evaluate --config configs/base.yaml --dummy      # thử pipeline, không cần torch/dữ liệu

Cách chấm: UCFCrimeClips(split="test") sinh clip theo `test_clip_stride` (mặc định = clip_len, có thêm
clip căn theo cuối video); điểm frame = trung bình điểm các clip phủ frame đó.
  cls: điểm clip = 1 - P(Normal) (model trả logits 14 lớp).
  mil: điểm clip = sigmoid(logit) (VideoModel mil trả logit của MILHead; mỗi clip là 1 snippet).
Nhãn frame: data/processed/splits/test_frame_labels.npz (TV1), khoá = tên video.
Model nạp bằng src.utils.model_io.load_model (TV3): config phải cùng `model.name` với checkpoint.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils import metrics as M
from src.utils.config import load_config

TPR_TARGET = 0.9
ROC_GRID = np.linspace(0, 1, 201)


# --------------------------------------------------------------------------- #
# Phần thuần numpy (test được không cần torch)
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


def check_alignment(n_frames: dict, labels: dict):
    """n_frames: {video: số ảnh} theo split của TV1; labels: {video: nhãn từng ảnh}."""
    a, b = set(n_frames), set(labels)
    if a != b:
        raise ValueError(f"Video split vs file nhãn không khớp: thiếu nhãn={sorted(a - b)[:5]}, "
                         f"thừa nhãn={sorted(b - a)[:5]} (khoá npz mẫu: {sorted(b)[:3]})")
    bad = {v: (n_frames[v], len(labels[v])) for v in a if n_frames[v] != len(labels[v])}
    if bad:
        raise ValueError(f"Số ảnh ≠ số nhãn (ảnh, nhãn): {dict(list(bad.items())[:5])}")


def assemble(items, clip_score, clip_prob, n_frames, clip_len):
    """Ghép điểm clip về từng frame.
    items[i] = (video, start, class_id) của clip i (UCFCrimeClips.items, đúng thứ tự loader);
    clip_score [n_clip], clip_prob [n_clip, K] hoặc None (chế độ mil: không có phân loại lớp).
    Trả (scores, class_id thật, class_id dự đoán) theo video; pred_id rỗng nếu clip_prob=None."""
    by = defaultdict(list)
    for i, (v, _, _) in enumerate(items):
        by[v].append(i)
    scores, true_id, pred_id = {}, {}, {}
    for v, idx in by.items():
        scores[v] = M.clips_to_frame(clip_score[idx], n_frames[v], clip_len, starts=[items[i][1] for i in idx])
        true_id[v] = items[idx[0]][2]
        if clip_prob is not None:
            pred_id[v] = int(np.asarray(clip_prob)[idx].mean(0).argmax())
    return scores, true_id, pred_id


def evaluate_scores(scores, labels, video_class, thr=0.5, normal="Normal", true_cls_idx=None, pred_cls_idx=None):
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
# Chạy model thật (cần torch + code của TV1/TV3)
# --------------------------------------------------------------------------- #
def run_model_eval(cfg, args):
    import torch
    from torch.utils.data import DataLoader

    from src.data.constants import CLASS_TO_ID, NORMAL                       # TV1
    from src.data.transforms import build_eval_transform                      # TV1
    from src.data.ucf_crime import UCFCrimeClips, build_store, data_paths, read_split  # TV1
    from src.utils.model_io import get_class_names, load_model, normal_index   # TV3

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.startswith("cuda")
    id2name = {i: n for n, i in CLASS_TO_ID.items()}
    class_names = get_class_names(cfg)                     # thứ tự = chỉ số output của ClsHead
    if [id2name[i] for i in range(len(id2name))] != class_names:
        raise RuntimeError("CLASS_TO_ID (TV1) khác thứ tự CLASSES (TV3): điểm/accuracy sẽ sai")
    normal_id = normal_index(class_names, NORMAL)

    n_frames = {v: n for v, _, n in read_split(cfg, "test")}
    lab = np.load(data_paths(cfg)["test_labels"], allow_pickle=True)
    labels = {k: np.asarray(lab[k]).astype(int) for k in lab.files}
    check_alignment(n_frames, labels)

    ds = UCFCrimeClips(cfg, "test", build_eval_transform(cfg), store=build_store(cfg), stride=args.stride)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=cfg["train"].get("num_workers", 4), pin_memory=device.startswith("cuda"))
    model = load_model(cfg, args.mode, args.ckpt, device)
    print(f"{len(n_frames)} video, {len(ds)} clip test | thiết bị {device} | AMP {amp}", flush=True)

    probs, mil_scores = [], []
    with torch.no_grad():
        for i, (x, _, _) in enumerate(loader):
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=amp):
                logits = model(x.to(device, non_blocking=True))
            if args.mode == "cls":
                probs.append(logits.float().softmax(-1).cpu())
            else:
                mil_scores.append(logits.float().reshape(-1).sigmoid().cpu())
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(loader)} batch", flush=True)
    if args.mode == "cls":
        probs = torch.cat(probs).numpy()
        clip_score = 1.0 - probs[:, normal_id]
    else:
        probs, clip_score = None, torch.cat(mil_scores).numpy()

    scores, true_id, pred_id = assemble(ds.items, clip_score, probs, n_frames, cfg["data"]["clip_len"])
    vids = sorted(scores)
    video_class = {v: id2name[true_id[v]] for v in vids}
    acc = ([true_id[v] for v in vids], [pred_id[v] for v in vids]) if probs is not None else (None, None)
    return (scores, labels, video_class, acc[0], acc[1], NORMAL)


# --------------------------------------------------------------------------- #
# Chế độ chạy thử
# --------------------------------------------------------------------------- #
def make_dummy(seed=0, noise=0.25):
    rng = np.random.default_rng(seed)
    classes = ["Abuse", "Arson", "Burglary", "Shoplifting", "NormalVideos"]
    scores, labels, vclass = {}, {}, {}
    for c in classes:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KHOA=GIA_TRI", help="ghi đè config như train.py")
    ap.add_argument("--ckpt")
    ap.add_argument("--mode", choices=["cls", "mil"], default="cls")
    ap.add_argument("--tag", help="mặc định: tên file config")
    ap.add_argument("--stride", type=int, help="ghi đè stride clip test (mặc định: data.test_clip_stride)")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--device")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--dummy", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    tag = args.tag or ("dummy" if args.dummy else Path(args.config).stem)
    if args.dummy:
        scores, labels, vclass, ti, pi = make_dummy()
        normal = "NormalVideos"
    else:
        if not args.ckpt:
            ap.error("--ckpt là bắt buộc (hoặc dùng --dummy)")
        t0 = time.time()
        scores, labels, vclass, ti, pi, normal = run_model_eval(cfg, args)
        print(f"Inference xong sau {time.time() - t0:.0f}s")

    res = evaluate_scores(scores, labels, vclass, cfg["eval"]["fpr_threshold"], normal, ti, pi)
    res.update({"model": tag, "mode": args.mode, "ckpt": args.ckpt, "stride": args.stride,
                "amp": (not args.no_amp) and not args.dummy})
    path = write_outputs(res, scores, tag, args.mode, args.out_dir)
    print(f"frame AUC = {res['frame_auc']:.4f} | acc14 = {res.get('acc_14cls')} -> {path}")


if __name__ == "__main__":
    main()
