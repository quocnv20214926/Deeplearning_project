"""[Owner: TV3 — framework; mỗi TV chạy cho model của mình]
Fine-tune backbone + ClsHead (chế độ cls) trên clip UCF-Crime, nhãn clip = nhãn video (14 lớp).

    python -m src.train --config configs/m1_r2plus1d.yaml
    python -m src.train --config configs/m3_swin3d.yaml --set train.batch_size=8 train.grad_accum=2
    python -m src.train --config configs/m1_r2plus1d.yaml --limit_videos 3 --epochs 1 --no_wandb   # chạy thử
    python -m src.train --config configs/m1_r2plus1d.yaml --resume weights/r2plus1d_18_cls_last.pt

Quy trình: AdamW + warmup tuyến tính + cosine (theo bước), AMP (fp16) trên GPU, tích luỹ gradient,
đóng băng theo train.freeze, sampler cân bằng Normal/Anomaly (TV1).
Val mỗi epoch — tập val KHÔNG có nhãn frame (tách từ Train), nên báo:
    val_clip_auc   AUC nhị phân mức clip, score = 1 - softmax[Normal], nhãn clip = nhãn video  (mặc định để chọn best)
    val_video_auc  AUC mức video, điểm video = trung bình top-3 clip
    val_acc14      accuracy 14 lớp mức video (trung bình xác suất các clip)
Lưu: weights/<model>_cls.pt (best theo train.monitor), weights/<model>_cls_last.pt (để --resume),
log CSV results/logs/<tag>_cls.csv (+ wandb nếu bật).

Khoá cấu hình thêm (đều có mặc định, ghi đè trong config của model hoặc --set):
    train.grad_accum (1)  train.warmup_epochs (1)  train.patience (5)  train.monitor (val_clip_auc)
    train.samples_per_epoch (= số clip train)  train.val_max_clips (16 clip/video)  train.label_smoothing (0)
    train.grad_clip (1.0)  model.grad_checkpointing (false)  model.dropout (0.5)
"""
import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data.transforms import build_eval_transform, build_train_transform
from src.data.ucf_crime import ROOT, UCFCrimeClips, build_store, make_balanced_sampler, read_split
from src.utils.model_io import apply_freeze, build_video_model, save_checkpoint
from src.utils.seed import set_seed
from src.utils.config import load_config


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #
def limit_per_class(videos, n):
    """Giữ tối đa n video mỗi lớp (chạy thử nhanh)."""
    if not n:
        return videos
    seen, out = defaultdict(int), []
    for v in videos:
        if seen[v[1]] < n:
            out.append(v)
            seen[v[1]] += 1
    return out


def make_scheduler(optimizer, total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def safe_auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


class Logger:
    """CSV luôn ghi; wandb nếu được bật và cài đặt (lỗi wandb không làm dừng train)."""

    def __init__(self, path, cfg, tag, use_wandb):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows, self.run = [], None
        if use_wandb:
            try:
                import wandb
                self.run = wandb.init(project=cfg["logging"]["wandb_project"], name=tag, config=cfg,
                                      job_type="cls")
            except Exception as e:  # chưa login / không có mạng
                print(f"[wandb] tắt: {e}")

    def log(self, row):
        self.rows.append(row)
        keys = list(dict.fromkeys(k for r in self.rows for k in r))
        with self.path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(self.rows)
        if self.run:
            self.run.log(row, step=row.get("epoch"))

    def finish(self):
        if self.run:
            self.run.finish()


# --------------------------------------------------------------------------- #
# Train / val một epoch
# --------------------------------------------------------------------------- #
def autocast(device, enabled):
    return torch.autocast(device_type=device.split(":")[0], dtype=torch.float16,
                          enabled=enabled and device.startswith("cuda"))


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device, amp=True, grad_accum=1,
                    grad_clip=1.0, max_steps=None, log_every=50):
    model.train()
    total, n, correct, t0 = 0.0, 0, 0, time.time()
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader) if max_steps is None else min(len(loader), max_steps * grad_accum)
    for i, (x, y, _) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, amp):
            logits = model(x)
            loss = criterion(logits.float(), y)
        scaler.scale(loss / grad_accum).backward()
        if (i + 1) % grad_accum == 0 or i + 1 == n_batches:
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        total += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
        if log_every and (i + 1) % log_every == 0:
            print(f"    [{i + 1}/{n_batches}] loss {total / n:.4f} acc {correct / n:.3f} "
                  f"lr {scheduler.get_last_lr()[0]:.2e} {n / (time.time() - t0):.0f} clip/s", flush=True)
    return {"train_loss": total / max(n, 1), "train_acc": correct / max(n, 1)}


@torch.no_grad()
def validate(model, loader, video_ids, device, amp=True, normal_idx=0):
    """video_ids[i] = video của clip i theo đúng thứ tự loader (shuffle=False)."""
    model.eval()
    probs, cls = [], []
    for x, y, _ in loader:
        with autocast(device, amp):
            logits = model(x.to(device, non_blocking=True))
        probs.append(logits.float().softmax(-1).cpu())
        cls.append(y)
    probs, cls = torch.cat(probs).numpy(), torch.cat(cls).numpy()
    score = 1.0 - probs[:, normal_idx]
    binary = (cls != normal_idx).astype(int)
    by_video = defaultdict(list)
    for i, v in enumerate(video_ids[:len(score)]):
        by_video[v].append(i)
    v_score, v_label, v_true, v_pred = [], [], [], []
    for v, idx in by_video.items():
        s = np.sort(score[idx])
        v_score.append(s[-min(3, len(s)):].mean())
        v_label.append(binary[idx[0]])
        v_true.append(cls[idx[0]])
        v_pred.append(probs[idx].mean(0).argmax())
    return {
        "val_clip_auc": safe_auc(binary, score),
        "val_video_auc": safe_auc(v_label, v_score),
        "val_acc14": float(np.mean(np.array(v_true) == np.array(v_pred))),
        "val_clip_acc14": float((probs.argmax(1) == cls).mean()),
    }


# --------------------------------------------------------------------------- #
def build_loaders(cfg, limit_videos=0, samples_per_epoch=None):
    t = cfg["train"]
    store = build_store(cfg)
    train_videos = limit_per_class(read_split(cfg, "train"), limit_videos)
    val_videos = limit_per_class(read_split(cfg, "val"), limit_videos)
    train_ds = UCFCrimeClips(cfg, "train", build_train_transform(cfg), store=store, videos=train_videos)
    val_ds = UCFCrimeClips(cfg, "val", build_eval_transform(cfg), store=store, videos=val_videos,
                           max_clips=t.get("val_max_clips", 16), jitter=False)
    nw = t.get("num_workers", 4)
    kw = dict(num_workers=nw, pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0)
    samples = samples_per_epoch or t.get("samples_per_epoch") or len(train_ds)
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], drop_last=True,
                              sampler=make_balanced_sampler(train_ds, samples), **kw)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"] * 2, shuffle=False, **kw)
    return train_loader, val_loader, [v for v, _, _ in val_ds.items]


def fit(cfg, model, train_loader, val_loader, val_video_ids, device, out_dir="weights", tag=None, epochs=None,
        max_steps=None, use_wandb=True, resume=None, log_dir=None):
    t = cfg["train"]
    name = cfg["model"]["name"]
    tag = tag or name
    epochs = epochs or t["epochs"]
    amp = t.get("amp", True) and device.startswith("cuda")
    grad_accum = t.get("grad_accum", 1)
    monitor = t.get("monitor", "val_clip_auc")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
    steps_per_epoch = math.ceil((len(train_loader) if max_steps is None
                                 else min(len(train_loader), max_steps * grad_accum)) / grad_accum)
    scheduler = make_scheduler(optimizer, epochs * steps_per_epoch,
                               int(t.get("warmup_epochs", 1) * steps_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=t.get("label_smoothing", 0.0))
    best_path = Path(out_dir) / f"{tag}_cls.pt"
    last_path = Path(out_dir) / f"{tag}_cls_last.pt"
    start, best, bad = 1, -float("inf"), 0
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, best, bad = state["epoch"] + 1, state["best"], state["bad_epochs"]
        print(f"Tiếp tục từ epoch {state['epoch']} (best {monitor} = {best:.4f})")
    logger = Logger(Path(log_dir or ROOT / cfg["logging"]["out_dir"]) / f"{tag}_cls.csv", cfg, tag, use_wandb)
    print(f"{name}: {sum(p.numel() for p in params) / 1e6:.1f}M tham số train | {steps_per_epoch} bước/epoch | "
          f"batch {t['batch_size']}x{grad_accum} | AMP {amp} | thiết bị {device}", flush=True)
    for epoch in range(start, epochs + 1):
        t0 = time.time()
        row = {"epoch": epoch}
        row.update(train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, device, amp,
                                   grad_accum, t.get("grad_clip", 1.0), max_steps))
        row.update(validate(model, val_loader, val_video_ids, device, amp))
        row.update({"lr": scheduler.get_last_lr()[0], "epoch_min": (time.time() - t0) / 60})
        if device.startswith("cuda"):
            row["vram_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        score = row[monitor]
        improved = not math.isnan(score) and score > best
        if improved:
            best, bad = score, 0
            save_checkpoint(best_path, model, cfg, "cls", epoch=epoch, metrics=row)
        else:
            bad += 1
        row["best"] = best
        logger.log(row)
        torch.save({"state_dict": getattr(model, "_orig_mod", model).state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "best": best, "bad_epochs": bad, "cfg": cfg}, last_path)
        print(f"Epoch {epoch}/{epochs} | loss {row['train_loss']:.4f} | clip AUC {row['val_clip_auc']:.4f} | "
              f"video AUC {row['val_video_auc']:.4f} | acc14 {row['val_acc14']:.3f} | {row['epoch_min']:.1f} phút"
              f"{' | * lưu best' if improved else ''}", flush=True)
        if bad >= t.get("patience", 5):
            print(f"Early stopping: {monitor} không cải thiện {bad} epoch")
            break
    logger.finish()
    print(f"Xong. Best {monitor} = {best:.4f} -> {best_path}")
    return best_path, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KHOA=GIA_TRI", help="ghi đè config, vd train.lr=5e-5")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit_videos", type=int, default=0, help="chỉ dùng N video mỗi lớp (chạy thử)")
    ap.add_argument("--max_steps", type=int, help="giới hạn số bước optimizer mỗi epoch (chạy thử)")
    ap.add_argument("--samples_per_epoch", type=int, help="số clip mỗi epoch (mặc định: toàn bộ clip train)")
    ap.add_argument("--tag", help="tên file đầu ra (mặc định: tên model)")
    ap.add_argument("--out_dir", default=str(ROOT / "weights"))
    ap.add_argument("--device")
    ap.add_argument("--resume")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    cfg["model"]["mode"] = "cls"
    set_seed(cfg.get("seed", 42))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model, _ = build_video_model(cfg, "cls", pretrained=False if args.no_pretrained else None)
    apply_freeze(model.backbone, cfg["train"].get("freeze", "none"))
    if cfg["model"].get("grad_checkpointing") and hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(True)
    model.to(device)

    train_loader, val_loader, val_ids = build_loaders(cfg, args.limit_videos, args.samples_per_epoch)
    print(f"Train {len(train_loader.dataset)} clip, val {len(val_loader.dataset)} clip "
          f"({len(set(val_ids))} video)", flush=True)
    fit(cfg, model, train_loader, val_loader, val_ids, device, args.out_dir, args.tag, args.epochs, args.max_steps,
        not args.no_wandb, args.resume)


if __name__ == "__main__":
    main()
