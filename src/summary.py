"""[Owner: TV2] Gộp results/tables/*.json -> bảng tổng hợp (summary.csv).
Mỗi dòng = (model, mode, precision): chỉ số chính xác (từ evaluate) + chỉ số hệ thống (từ benchmark).
AUC/FPR lấy từ lần chạy evaluate.py (mặc định có AMP fp16 trên GPU; xem cột `amp`), KHÔNG tính lại cho từng
định dạng ONNX/FP16; cột precision chỉ nói về tốc độ/bộ nhớ.
"""
import csv
import json
from pathlib import Path

COLUMNS = ["model", "mode", "precision", "device", "gpu", "frame_auc", "fpr", "normal_video_fpr",
           "fpr_at_tpr90", "acc_14cls", "amp", "params_m", "file_mb", "vram_peak_mb", "ram_mb",
           "latency_ms_mean", "latency_ms_p95", "fps"]


def _first(d, prefix, exclude=None):
    for k, v in d.items():
        if k.startswith(prefix) and not (exclude and k.startswith(exclude)):
            return v
    return None


def collect_rows(tables_dir="results/tables"):
    tables = Path(tables_dir)
    rows = []
    for bp in sorted(tables.glob("*_bench.json")):
        b = json.loads(bp.read_text(encoding="utf-8"))
        ep = tables / f"{b['model']}_{b['mode']}.json"
        e = json.loads(ep.read_text(encoding="utf-8")) if ep.exists() else {}
        row = {c: b.get(c) for c in COLUMNS}
        row.update({
            "frame_auc": e.get("frame_auc"),
            "fpr": _first(e, "fpr@", exclude="fpr@tpr"),
            "normal_video_fpr": _first(e, "normal_video_fpr@"),
            "fpr_at_tpr90": e.get("fpr@tpr90"),
            "acc_14cls": e.get("acc_14cls"),
            "amp": e.get("amp"),
        })
        rows.append(row)
    return sorted(rows, key=lambda r: (r["mode"], r["model"], r["precision"]))


def build_summary(tables_dir="results/tables", out_csv="results/tables/summary.csv"):
    rows = collect_rows(tables_dir)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows
