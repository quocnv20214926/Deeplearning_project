"""[Owner: TV1] Chuẩn bị dữ liệu UCF-Crime (bản Kaggle odins0n, ảnh 64x64, 1/10 frame).

1. Tải dữ liệu vào data/raw/: file zip (không cần giải nén) hoặc thư mục giải nén data/raw/ucf_crime/
       kaggle datasets download -d odins0n/ucf-crime-dataset -p data/raw
2. Chạy:
       python scripts/prepare_data.py                      # annotations, index, pack, splits
       python scripts/prepare_data.py --no-pack            # không tạo ~16 GB .npy (vd. trên Kaggle)
       python scripts/prepare_data.py --steps splits eda   # chạy lại một số bước
   annotations: tải nhãn thời gian của tác giả nếu chưa có -> data/annotations/
   index:  bảng frame/video, kiểm tra thiếu/lỗi -> data/processed/{frames_index,videos}.csv
   pack:   mỗi video thành một file .npy -> data/processed/frames/ (đọc clip nhanh gấp ~2 lần PNG)
   splits: train/val (val_ratio theo video, phân tầng theo lớp)/test -> data/processed/splits/*.txt,
           nhãn 0/1 từng ảnh test -> data/processed/test_frame_labels.npz
   eda:    bảng + hình -> results/tables/eda/, results/figures/eda/
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.prepare import step_annotations, step_index, step_pack, step_splits
from src.data.ucf_crime import find_raw
from src.utils.config import load_config

STEPS = ["annotations", "index", "pack", "splits", "eda"]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console Windows (cp1252) không in được tiếng Việt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--raw", default=None, help="zip hoặc thư mục Kaggle (mặc định: data.frames_root hoặc data/raw/*.zip)")
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS[:4])
    ap.add_argument("--no-pack", action="store_true", help="bỏ bước pack (dataset sẽ đọc PNG trực tiếp)")
    ap.add_argument("--force-index", action="store_true", help="lập lại chỉ mục kể cả khi đã có")
    ap.add_argument("--workers", default=8, type=int, help="số luồng giải mã khi pack")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw = Path(args.raw) if args.raw else None
    for step in args.steps:
        if step == "pack" and args.no_pack:
            continue
        print(f"\n=== {step}", flush=True)
        t = time.time()
        if step == "annotations":
            step_annotations(cfg)
        elif step == "index":
            step_index(cfg, raw or find_raw(cfg), force=args.force_index)
        elif step == "pack":
            step_pack(cfg, raw or find_raw(cfg), workers=args.workers)
        elif step == "splits":
            step_splits(cfg)
        elif step == "eda":
            from src.data.eda import run_all
            summary, _ = run_all(cfg)
            print(summary)
        print(f"=== {step} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
