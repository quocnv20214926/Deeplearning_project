"""[Owner: TV3] Trích feature snippet bằng backbone -> data/features/<model>/<split>/<video>.npy

    python -m src.extract_features --config configs/m1_r2plus1d.yaml --ckpt weights/r2plus1d_18_cls.pt
    python -m src.extract_features --config configs/m1_r2plus1d.yaml            # backbone pretrained Kinetics
    python -m src.extract_features --config configs/m3_swin3d.yaml --ckpt weights/swin3d_t_cls.pt --splits test

Mỗi video -> mảng float16 (n_snippet, D): snippet = clip `clip_len` frame KHÔNG chồng lấn, theo thứ tự thời gian
(snippet i bắt đầu ở frame i*clip_len; snippet cuối căn theo cuối video — giống `UCFCrimeClips` chế độ test).
Video đã có file được bỏ qua (chạy tiếp được khi phiên Kaggle/Colab bị ngắt); dùng --overwrite để làm lại.
Ghi kèm meta.json (ckpt, feat_dim, clip_len, stride) cho train_mil.py.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.transforms import build_eval_transform
from src.data.ucf_crime import ROOT, UCFCrimeClips, build_store, read_split
from src.train import limit_per_class
from src.utils.config import load_config
from src.utils.model_io import load_backbone


def feature_dir(cfg, root=None):
    return Path(root or ROOT / "data" / "features") / cfg["model"]["name"]


@torch.no_grad()
def extract_split(cfg, backbone, split, out_dir, device, batch_size=64, num_workers=4, overwrite=False,
                  limit_videos=0, store=None, amp=True):
    out_dir = Path(out_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = limit_per_class(read_split(cfg, split), limit_videos)
    todo = [v for v in videos if overwrite or not (out_dir / f"{v[0]}.npy").exists()]
    print(f"[{split}] {len(videos)} video, cần trích {len(todo)}", flush=True)
    if not todo:
        return 0
    clip_len = cfg["data"]["clip_len"]
    ds = UCFCrimeClips(cfg, split, build_eval_transform(cfg), store=store, videos=todo, stride=clip_len,
                       max_clips=0, jitter=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.startswith("cuda"))
    video_of = [v for v, _, _ in ds.items]
    # số clip của từng video (items xếp liền nhau theo video)
    remaining = {}
    for v in video_of:
        remaining[v] = remaining.get(v, 0) + 1
    buf, pos, done, t0 = {}, 0, 0, time.time()
    use_amp = amp and device.startswith("cuda")
    for x, _, _ in loader:
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=use_amp):
            f = backbone(x.to(device, non_blocking=True))
        f = f.float().cpu().numpy().astype(np.float16)
        for row in f:
            v = video_of[pos]
            pos += 1
            buf.setdefault(v, []).append(row)
            if len(buf[v]) == remaining[v]:
                np.save(out_dir / f"{v}.npy", np.stack(buf.pop(v)))
                done += 1
                if done % 50 == 0 or done == len(todo):
                    print(f"  [{split}] {done}/{len(todo)} video | {pos} snippet | "
                          f"{pos / (time.time() - t0):.0f} snippet/s", flush=True)
    assert not buf, f"Còn video chưa ghi: {list(buf)[:3]}"
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", help="checkpoint chế độ cls; bỏ trống = backbone pretrained Kinetics")
    ap.add_argument("--set", nargs="*", default=[], metavar="KHOA=GIA_TRI")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out_root", help="mặc định data/features")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int)
    ap.add_argument("--limit_videos", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = feature_dir(cfg, args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    ckpt = str(Path(args.ckpt).resolve()) if args.ckpt else None
    if meta_path.exists() and not args.overwrite:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("ckpt") != ckpt:
            raise SystemExit(f"{out} đang chứa feature từ ckpt={old.get('ckpt')}. Dùng --overwrite hoặc --out_root khác.")
    backbone, feat_dim = load_backbone(cfg, args.ckpt, device)
    meta = {"model": cfg["model"]["name"], "ckpt": ckpt, "feat_dim": feat_dim, "clip_len": cfg["data"]["clip_len"],
            "stride": cfg["data"]["clip_len"], "input_size": cfg["data"]["input_size"], "dtype": "float16"}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    store = build_store(cfg)
    nw = cfg["train"].get("num_workers", 4) if args.num_workers is None else args.num_workers
    for split in args.splits:
        extract_split(cfg, backbone, split, out, device, args.batch_size, nw, args.overwrite, args.limit_videos, store)
    print(f"Xong -> {out}")


if __name__ == "__main__":
    main()
