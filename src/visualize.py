"""[Owner: TV2] Hình cho báo cáo từ results/tables/*.json và summary.csv:
  roc.png              ROC của các model (chỉ chế độ/tên được chọn)
  per_class_auc.png    AUC theo từng lớp bất thường
  pareto_<gpu|cpu>_<precision>.png   AUC vs FPS (kích thước điểm = VRAM đỉnh; hình tròn = cls, tam giác = mil)

    python -m src.visualize                       # tất cả hình, đọc results/tables, ghi results/figures
    python -m src.visualize --only roc pareto --names r2plus1d_18_cls i3d_r50_full_cls
Chạy `python scripts/run_all_benchmarks.py --summary_only` trước để có summary.csv.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_eval_tables(tables_dir="results/tables", names=None):
    """{name: json} cho các file <tag>_<cls|mil>.json (bỏ *_bench.json)."""
    out = {}
    for p in sorted(Path(tables_dir).glob("*.json")):
        if p.stem.endswith("_bench") or not (p.stem.endswith("_cls") or p.stem.endswith("_mil")):
            continue
        if names and p.stem not in names:
            continue
        out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


def read_summary(path):
    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return math.nan
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("frame_auc", "fps", "vram_peak_mb", "latency_ms_mean", "params_m", "file_mb"):
                r[k] = num(r.get(k))
            rows.append(r)
    return rows


def plot_roc(evals, out):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, e in evals.items():
        ax.plot(e["roc"]["fpr"], e["roc"]["tpr"], label=f"{name} (AUC {e['frame_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Frame-level ROC (UCF-Crime test)",
           xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_per_class(evals, out):
    classes = sorted({c for e in evals.values() for c in e["per_class_auc"]})
    n = len(evals)
    w = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(classes)), 4.5))
    x = np.arange(len(classes))
    for i, (name, e) in enumerate(evals.items()):
        vals = [e["per_class_auc"].get(c) for c in classes]
        ax.bar(x + (i - (n - 1) / 2) * w, [np.nan if v is None else v for v in vals], w, label=name)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xticks(x, classes, rotation=45, ha="right")
    ax.set(ylabel="Frame AUC", ylim=(0, 1), title="AUC theo lớp (lớp đó + toàn bộ video Normal)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def device_kind(dev):
    return "gpu" if "cuda" in (dev or "").lower() else "cpu"


def plot_pareto(rows, out, precision="fp32", kind="gpu"):
    sel = [r for r in rows if r["precision"] == precision and device_kind(r["device"]) == kind
           and not math.isnan(r["frame_auc"]) and not math.isnan(r["fps"])]
    if not sel:
        return False
    vram = np.array([r["vram_peak_mb"] for r in sel], float)
    vmax = np.nanmax(vram) if np.isfinite(vram).any() else math.nan
    models = sorted({r["model"] for r in sel})
    colors = {m: plt.get_cmap("tab10")(i % 10) for i, m in enumerate(models)}
    fig, ax = plt.subplots(figsize=(7, 5))
    for r, v in zip(sel, vram):
        size = 120 if math.isnan(vmax) or math.isnan(v) else 60 + 500 * v / vmax
        ax.scatter(r["fps"], r["frame_auc"], s=size, c=[colors[r["model"]]], alpha=0.75,
                   marker="o" if r["mode"] == "cls" else "^", edgecolors="k", linewidths=0.6)
        ax.annotate(f"{r['model']}/{r['mode']}", (r["fps"], r["frame_auc"]), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.margins(x=0.25, y=0.15)
    ax.set(xlabel="FPS (log)", ylabel="Frame AUC",
           title=f"AUC vs FPS — {kind.upper()}, {precision} (kích thước ∝ VRAM đỉnh; ○ cls, △ mil)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--summary", default=None, help="mặc định: <tables>/summary.csv")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--names", nargs="*", help="chỉ vẽ các model này cho roc/per_class (vd. r2plus1d_18_cls)")
    ap.add_argument("--only", nargs="*", choices=["roc", "per_class", "pareto"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(args.only or ["roc", "per_class", "pareto"])

    evals = load_eval_tables(args.tables, args.names)
    if "roc" in only and evals:
        plot_roc(evals, out / "roc.png")
        print("->", out / "roc.png")
    if "per_class" in only and evals:
        plot_per_class(evals, out / "per_class_auc.png")
        print("->", out / "per_class_auc.png")
    if "pareto" in only:
        sp = Path(args.summary or Path(args.tables) / "summary.csv")
        if not sp.exists():
            print(f"Chưa có {sp}: chạy `python scripts/run_all_benchmarks.py --summary_only` trước")
        else:
            rows = read_summary(sp)
            for kind in ("gpu", "cpu"):
                for prec in sorted({r["precision"] for r in rows}):
                    if plot_pareto(rows, out / f"pareto_{kind}_{prec}.png", prec, kind):
                        print("->", out / f"pareto_{kind}_{prec}.png")


if __name__ == "__main__":
    main()
