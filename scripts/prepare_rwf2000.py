"""[Owner: TV2] Chuẩn bị RWF-2000 cho đánh giá cross-dataset.

Đọc video .avi, lấy mẫu 1/frame_step frame (mặc định 10, giống bản Kaggle của UCF-Crime),
resize về size x size (mặc định 64, giống ảnh Kaggle) và lưu PNG theo bố cục giống Kaggle:
    <dst>/frames/Test/<Fight|NonFight>/<clip>_<chỉ số frame gốc>.png
rồi ghi <dst>/test_list.csv (video, label, n_frames, fps, ...), Fight=1, NonFight=0.

    python scripts/prepare_rwf2000.py --src data/external/rwf2000_raw --dst data/external/rwf2000

Mặc định chỉ xử lý tập held-out (thư mục 'val' hoặc 'test' của RWF-2000, giữ nguyên chia gốc).
"""
import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

CLASSES = {"Fight": 1, "NonFight": 0}
VIDEO_EXT = {".avi", ".mp4", ".mkv", ".mov"}


def find_split_dirs(src: Path, names=("val", "test")):
    """Tìm các thư mục <split>/<Fight|NonFight> (không phân biệt hoa thường)."""
    found = {}
    for d in src.rglob("*"):
        if d.is_dir() and d.name.lower() in names and any((d / c).is_dir() for c in CLASSES):
            found[d.name.lower()] = d
    return found


def safe_name(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "-", stem).strip("-")   # bỏ '_' để '_<idx>' tách được


def extract_clip(path: Path, out_dir: Path, name: str, step: int, size: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    kept, i = 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)   # bóp méo như Kaggle (giả định)
            cv2.imwrite(str(out_dir / f"{name}_{i}.png"), fr)
            kept += 1
        i += 1
    cap.release()
    return {"n_src_frames": i, "n_frames": kept, "fps": round(float(fps), 2)} if kept else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="thư mục đã giải nén RWF-2000")
    ap.add_argument("--dst", default="data/external/rwf2000")
    ap.add_argument("--splits", nargs="+", default=["val", "test"],
                    help="thư mục split cần xử lý (mặc định: val/test = tập held-out)")
    ap.add_argument("--frame_step", type=int, default=10)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--clip_len", type=int, default=16, help="chỉ để cảnh báo clip quá ngắn")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    splits = find_split_dirs(src, tuple(s.lower() for s in args.splits))
    if not splits:
        raise SystemExit(f"Không thấy <{'|'.join(args.splits)}>/Fight|NonFight trong {src}")
    print("Dùng split:", {k: str(v) for k, v in splits.items()})

    rows, bad = [], []
    for split, sdir in splits.items():
        for cname, label in CLASSES.items():
            vids = sorted(p for p in (sdir / cname).glob("*") if p.suffix.lower() in VIDEO_EXT)
            out_dir = dst / "frames" / "Test" / cname
            out_dir.mkdir(parents=True, exist_ok=True)
            for n, p in enumerate(vids, 1):
                name = safe_name(p.stem)
                info = extract_clip(p, out_dir, name, args.frame_step, args.size)
                if info is None:
                    bad.append(str(p))
                    continue
                rows.append({"video": name, "label": label, "class": cname, "split": split,
                             "src": str(p), **info})
                if n % 50 == 0:
                    print(f"  {split}/{cname}: {n}/{len(vids)}", flush=True)

    names = [r["video"] for r in rows]
    dup = {x for x in names if names.count(x) > 1}
    if dup:
        raise SystemExit(f"Tên clip trùng sau khi làm sạch: {sorted(dup)[:5]}")

    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "test_list.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    nf = np.array([r["n_frames"] for r in rows])
    print(f"\nXong: {len(rows)} clip ({sum(r['label'] for r in rows)} Fight) -> {dst/'test_list.csv'}")
    print(f"Số frame/clip sau lấy mẫu: min={nf.min()} median={int(np.median(nf))} max={nf.max()}")
    short = int((nf < args.clip_len).sum())
    if short:
        print(f"CẢNH BÁO: {short} clip có < {args.clip_len} frame (clip_len) -> evaluate sẽ lặp frame cuối; "
              f"thống nhất cách xử lý với TV1/TV3.")
    if bad:
        print(f"Không đọc được {len(bad)} file:", *bad[:5], sep="\n  ")


if __name__ == "__main__":
    main()
