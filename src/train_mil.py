"""[Owner: TV3] Train MILHead trên feature (chế độ mil, Sultani et al. 2018).
Mỗi bước: batch = mil.batch_size/2 video Normal + mil.batch_size/2 video Anomaly, mỗi video gộp thành
mil.num_segments segment; loss = MIL ranking + smooth + sparse (src/losses/mil_loss.py).

    python -m src.extract_features --config configs/m1_r2plus1d.yaml --ckpt weights/r2plus1d_18_cls.pt
    python -m src.train_mil --config configs/m1_r2plus1d.yaml

Chọn best theo val_video_auc (val không có nhãn frame; điểm video = max segment).
Cuối cùng chấm luôn 290 video test từ feature test (nếu đã trích): điểm từng snippet (16 frame) -> điểm từng frame,
cùng bộ metric với src/evaluate.py -> results/tables/<tag>_mil.json + results/predictions/<tag>_mil.npz.
Lưu: weights/<model>_mil.pt (dùng được với `python -m src.evaluate --mode mil --ckpt ...`).

Khoá cấu hình (mục `mil`, đều có mặc định): num_segments 32, lambda_smooth/lambda_sparse 8e-5, batch_size 60,
iterations 3000, lr 1e-3, weight_decay 1e-3, optimizer adagrad|adamw, eval_every 100, dropout 0.6.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data.constants import NORMAL
from src.data.mil_dataset import MILFeatureDataset, to_segments
from src.data.ucf_crime import ROOT, clip_starts, read_split
from src.extract_features import feature_dir
from src.losses.mil_loss import mil_ranking_loss
from src.models.heads import MILHead
from src.utils.config import load_config
from src.utils.model_io import save_mil_checkpoint
from src.utils.seed import set_seed


def forever(loader):
    while True:
        yield from loader


def load_feat(path):
    return np.load(path).astype(np.float32)


@torch.no_grad()
def video_level_auc(head, fdir, videos, num_segments, device):
    """AUC mức video trên val: điểm video = max điểm segment."""
    head.eval()
    y, s = [], []
    for v, c, _ in videos:
        seg = torch.from_numpy(to_segments(load_feat(fdir / f"{v}.npy"), num_segments)).to(device)
        s.append(head(seg[None])[0].max().item())
        y.append(int(c != NORMAL))
    head.train()
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def snippets_to_frames(snippet_scores, n_frames, clip_len):
    """Snippet i phủ frame [start_i, start_i + clip_len) với start theo clip_starts (stride = clip_len);
    frame bị 2 snippet phủ (đoạn cuối) lấy trung bình."""
    acc, cnt = np.zeros(n_frames), np.zeros(n_frames)
    starts = clip_starts(n_frames, clip_len, clip_len)
    if len(starts) != len(snippet_scores):
        raise ValueError(f"{len(snippet_scores)} snippet nhưng video {n_frames} frame cần {len(starts)}")
    for st, sc in zip(starts, snippet_scores):
        acc[st:st + clip_len] += sc
        cnt[st:st + clip_len] += 1
    return acc / np.maximum(cnt, 1)


@torch.no_grad()
def score_test(head, fdir, cfg, device):
    """{video: điểm từng frame} cho các video test có feature."""
    head.eval()
    clip_len = cfg["data"]["clip_len"]
    scores, vclass = {}, {}
    for v, c, n in read_split(cfg, "test"):
        f = torch.from_numpy(load_feat(fdir / f"{v}.npy")).to(device)
        s = head(f[None])[0].cpu().numpy()  # MILHead chấm từng snippet
        scores[v] = snippets_to_frames(s, n, clip_len)
        vclass[v] = c
    return scores, vclass


def evaluate_test(head, fdir, cfg, device, tag, out_dir):
    from src.evaluate import evaluate_scores, load_labels, write_outputs
    labels = load_labels(Path(ROOT / cfg["data"]["split_dir"]) / "test_frame_labels.npz")
    scores, vclass = score_test(head, fdir, cfg, device)
    for v in labels:
        if len(labels[v]) != len(scores[v]):
            raise ValueError(f"{v}: {len(scores[v])} điểm vs {len(labels[v])} nhãn")
    res = evaluate_scores(scores, labels, vclass, cfg["eval"]["fpr_threshold"], cfg["data"]["normal_class"])
    res.update({"model": tag, "mode": "mil", "source": "features"})
    return res, write_outputs(res, scores, tag, "mil", out_dir)


def train_mil(cfg, fdir, device="cpu", tag=None, out_dir=None, results_dir=None, iterations=None):
    m = cfg.get("mil", {})
    fdir = Path(fdir)
    meta = json.loads((fdir / "meta.json").read_text(encoding="utf-8"))
    tag = tag or cfg["model"]["name"]
    k = m.get("num_segments", 32)
    train = read_split(cfg, "train")
    missing = [v for v, _, _ in train if not (fdir / "train" / f"{v}.npy").exists()]
    if missing:
        raise SystemExit(f"Thiếu feature train của {len(missing)} video (vd {missing[:3]}): chạy extract_features trước")
    half = m.get("batch_size", 60) // 2
    loaders = {}
    for name, vids in [("normal", [v for v, c, _ in train if c == NORMAL]),
                       ("anomaly", [v for v, c, _ in train if c != NORMAL])]:
        ds = MILFeatureDataset(str(fdir / "train"), vids, [int(name == "anomaly")] * len(vids), k)
        loaders[name] = forever(DataLoader(ds, batch_size=half, shuffle=True, drop_last=len(ds) >= half,
                                           num_workers=m.get("num_workers", 2)))
    head = MILHead(meta["feat_dim"], p=m.get("dropout", 0.6)).to(device)
    lr, wd = float(m.get("lr", 1e-3)), float(m.get("weight_decay", 1e-3))
    opt = (torch.optim.Adagrad(head.parameters(), lr=lr, weight_decay=wd) if m.get("optimizer", "adagrad") == "adagrad"
           else torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd))
    val = read_split(cfg, "val")
    has_val = all((fdir / "val" / f"{v}.npy").exists() for v, _, _ in val)
    iterations = iterations or m.get("iterations", 3000)
    eval_every = m.get("eval_every", 100)
    ckpt_path = Path(out_dir or ROOT / "weights") / f"{tag}_mil.pt"
    log_path = Path(results_dir or ROOT / "results") / "logs" / f"{tag}_mil.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows, best, run_loss, t0 = [], -1.0, 0.0, time.time()
    print(f"MIL {tag}: feature {meta['feat_dim']}-d từ ckpt={meta['ckpt']} | {iterations} bước | batch {half}+{half}")
    for it in range(1, iterations + 1):
        xa, _ = next(loaders["anomaly"])
        xn, _ = next(loaders["normal"])
        b = min(len(xa), len(xn))
        s = head(torch.cat([xa[:b], xn[:b]]).to(device))
        loss = mil_ranking_loss(s[:b], s[b:], m.get("lambda_smooth", 8e-5), m.get("lambda_sparse", 8e-5))
        opt.zero_grad()
        loss.backward()
        opt.step()
        run_loss += loss.item()
        if it % eval_every == 0 or it == iterations:
            auc = video_level_auc(head, fdir / "val", val, k, device) if has_val else float("nan")
            row = {"iter": it, "loss": run_loss / eval_every, "val_video_auc": auc}
            rows.append(row)
            run_loss = 0.0
            better = (not np.isnan(auc) and auc > best) or (not has_val)
            if better:
                best = auc if has_val else best
                save_mil_checkpoint(ckpt_path, head, cfg, meta["feat_dim"], meta["ckpt"], iteration=it, metrics=row)
            print(f"  iter {it} | loss {row['loss']:.4f} | val video AUC {auc:.4f}{' *' if better else ''} | "
                  f"{time.time() - t0:.0f}s", flush=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("iter,loss,val_video_auc\n" + "".join(f"{r['iter']},{r['loss']},{r['val_video_auc']}\n" for r in rows))
    head.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["head_state_dict"])
    test = read_split(cfg, "test")
    if all((fdir / "test" / f"{v}.npy").exists() for v, _, _ in test):
        res, path = evaluate_test(head, fdir / "test", cfg, device, tag, results_dir or ROOT / "results")
        fpr_key = "fpr@{}".format(cfg["eval"]["fpr_threshold"])
        print(f"TEST frame AUC = {res['frame_auc']:.4f} | {fpr_key} = {res[fpr_key]:.4f} -> {path}")
    else:
        res = None
        print("Chưa có feature test -> bỏ qua chấm test")
    print(f"Best val video AUC = {best:.4f} -> {ckpt_path}")
    return ckpt_path, best, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KHOA=GIA_TRI")
    ap.add_argument("--features", help="thư mục feature (mặc định data/features/<model>)")
    ap.add_argument("--tag")
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--device")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    cfg["model"]["mode"] = "mil"
    set_seed(cfg.get("seed", 42))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_mil(cfg, args.features or feature_dir(cfg), device, args.tag, iterations=args.iterations)


if __name__ == "__main__":
    main()
