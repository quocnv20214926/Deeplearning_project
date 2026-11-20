"""[Owner: TV1] Phân tích khám phá dữ liệu (EDA): bảng thống kê + hình (nhãn tiếng Anh, dùng cho báo cáo).

    from src.data.eda import run_all
    run_all(cfg)            # -> results/figures/eda/*.png, results/tables/eda/*.csv + stats.md

Dùng trong notebooks/01_eda.ipynb. Cần đã chạy scripts/prepare_data.py (bước index + splits);
đo nhiễu (`noise_stats`) cần frame đã đóng gói (.npy).
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from tqdm import tqdm

from .annotations import load_temporal_annotations
from .constants import CLASSES, NORMAL
from .ucf_crime import ROOT, build_store, clip_starts, data_paths, split_clip_params

FIG_DIR = ROOT / "results" / "figures" / "eda"
TAB_DIR = ROOT / "results" / "tables" / "eda"
SUBSETS = {"train": "Train", "val": "Validation", "test": "Test"}
COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
FPS = 30  # video gốc

plt.rcParams.update({
    "font.family": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def load_videos(cfg):
    videos = pd.read_csv(data_paths(cfg)["videos"])
    videos["seconds"] = videos["n_frames"] * cfg["data"]["frame_step"] / FPS
    return videos


def load_test_labels(cfg):
    with np.load(data_paths(cfg)["test_labels"]) as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- bảng
def class_table(videos):
    """Số video và số frame theo lớp × subset."""
    t = videos.pivot_table(index="label", columns="subset", values="n_frames", aggfunc=["count", "sum"], fill_value=0)
    t.columns = [f"{'videos' if a == 'count' else 'frames'}_{s}" for a, s in t.columns]
    t = t.reindex(CLASSES)[[f"{k}_{s}" for k in ("videos", "frames") for s in SUBSETS]]
    t.loc["TOTAL"] = t.sum()
    return t.astype(int)


def duration_table(videos):
    return videos.groupby("subset")["seconds"].describe()[["min", "50%", "mean", "max"]].reindex(list(SUBSETS)).round(1)


def test_anomaly_table(cfg, videos):
    labels = load_test_labels(cfg)
    test = videos[videos["subset"] == "test"]
    rows = {c: [0, 0] for c in CLASSES}
    for v in test.itertuples():
        rows[v.label][0] += len(labels[v.video])
        rows[v.label][1] += int(labels[v.video].sum())
    t = pd.DataFrame(rows, index=["frames", "anomalous"]).T
    t.loc["TOTAL"] = t.sum()
    t["pct"] = (100 * t["anomalous"] / t["frames"]).round(2)
    return t


def clip_table(cfg, videos):
    rows = []
    for subset in SUBSETS:
        sub = videos[videos["subset"] == subset]
        stride, normal_stride, max_clips = split_clip_params(cfg, subset)
        n = [len(clip_starts(v.n_frames, cfg["data"]["clip_len"], normal_stride if v.label == NORMAL else stride, max_clips))
             for v in sub.itertuples()]
        anom = sum(k for k, v in zip(n, sub.itertuples()) if v.class_id > 0)
        rows.append((subset, len(sub), int((sub["class_id"] == 0).sum()), int((sub["class_id"] > 0).sum()),
                     int(sub["n_frames"].sum()), sum(n), anom, round(100 * anom / sum(n), 1)))
    return pd.DataFrame(rows, columns=["subset", "videos", "normal_videos", "anomaly_videos", "frames", "clips",
                                       "anomaly_clips", "anomaly_clip_pct"]).set_index("subset")


def pixel_stats(videos, store, n_videos=200, seed=0):
    """Trung bình / độ lệch chuẩn theo kênh RGB (thang [0, 1]) trên mẫu video huấn luyện."""
    train = videos[videos["subset"] == "train"]
    sample = train.sample(min(n_videos, len(train)), random_state=seed)
    total, total_sq, n = np.zeros(3), np.zeros(3), 0
    for v in sample.itertuples():
        length = store.length(v.split, v.video)
        x = store.frames(v.split, v.video, np.arange(0, length, max(1, length // 50))).reshape(-1, 3) / 255.0
        total, total_sq, n = total + x.sum(0), total_sq + (x ** 2).sum(0), n + len(x)
    mean = total / n
    return mean.round(4).tolist(), np.sqrt(total_sq / n - mean ** 2).round(4).tolist()


BLACK_MEAN, BLANK_STD, BORDER_MEAN, BORDER_PX = 10, 6, 12, 4


def _noise_flags(frames):
    x = frames.astype(np.float32)
    gray = x.mean(axis=3)
    letterbox = (gray[:, :BORDER_PX].mean((1, 2)) < BORDER_MEAN) & (gray[:, -BORDER_PX:].mean((1, 2)) < BORDER_MEAN)
    pillarbox = (gray[:, :, :BORDER_PX].mean((1, 2)) < BORDER_MEAN) & (gray[:, :, -BORDER_PX:].mean((1, 2)) < BORDER_MEAN)
    blank = x.std(axis=(1, 2, 3)) < BLANK_STD
    return x.mean(axis=(1, 2, 3)) < BLACK_MEAN, blank, (letterbox | pillarbox) & ~blank


def noise_stats(cfg, videos, chunk=8192):
    """Tỉ lệ frame gần đen (độ sáng TB < 10), frame trống (std < 6) và video có viền đen
    (đa số frame không trống có 4 hàng trên+dưới hoặc 4 cột trái+phải tối). Cần file .npy."""
    packed = data_paths(cfg)["packed_dir"]
    rows = []
    for v in tqdm(videos.itertuples(), total=len(videos), desc="noise scan"):
        arr = np.load(packed / v.split / f"{v.video}.npy", mmap_mode="r")
        black = blank = border = 0
        for s in range(0, len(arr), chunk):
            b, k, d = _noise_flags(np.asarray(arr[s:s + chunk]))
            black, blank, border = black + b.sum(), blank + k.sum(), border + d.sum()
        rows.append((v.label, len(arr), black, blank, border / max(len(arr) - blank, 1) > 0.5))
    per = pd.DataFrame(rows, columns=["label", "frames", "black", "blank", "bordered"])
    g = per.groupby("label")
    t = pd.DataFrame({"videos": g.size(), "frames": g["frames"].sum(),
                      "black_pct": 100 * g["black"].sum() / g["frames"].sum(),
                      "blank_pct": 100 * g["blank"].sum() / g["frames"].sum(),
                      "bordered_videos_pct": 100 * g["bordered"].mean()}).reindex(CLASSES)
    tot = per.sum(numeric_only=True)
    t.loc["TOTAL"] = [len(per), tot["frames"], 100 * tot["black"] / tot["frames"], 100 * tot["blank"] / tot["frames"],
                      100 * per["bordered"].mean()]
    return t.round(2)


# ----------------------------------------------------------------------------- hình
def fig_class_distribution(videos, out):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    x, width = np.arange(len(CLASSES)), 0.27
    for ax, (agg, ylabel) in zip(axes, [("count", "Videos"), ("sum", "Frames")]):
        t = videos.pivot_table(index="label", columns="subset", values="n_frames", aggfunc=agg, fill_value=0).reindex(CLASSES)
        for k, s in enumerate(SUBSETS):
            ax.bar(x + (k - 1) * width, t[s], width, label=SUBSETS[s], color=COLORS[s])
        ax.set_yscale("log")
        ax.set_ylabel(ylabel + " (log scale)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    axes[0].set_title("(a) Videos per class", fontsize=11, loc="left")
    axes[1].set_title("(b) Frames per class", fontsize=11, loc="left")
    axes[1].set_xticks(x, CLASSES, rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_duration(videos, out):
    s = videos["seconds"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.logspace(np.log10(s.min()), np.log10(s.max()), 40)
    ax.hist(s[videos["class_id"] == 0], bins=bins, alpha=0.6, label="Normal videos", color="#4C72B0")
    ax.hist(s[videos["class_id"] > 0], bins=bins, alpha=0.6, label="Anomalous videos", color="#C44E52")
    ax.set_xscale("log")
    ax.set_xlabel("Video duration (seconds, log scale)")
    ax.set_ylabel("Videos")
    ax.legend(frameon=False)
    fig.savefig(out)
    plt.close(fig)


def fig_sample_frames(videos, store, out, n_frames=5):
    """Một video huấn luyện mỗi lớp, n_frames frame trải đều theo thời gian; 2 cột lớp."""
    train = videos[videos["subset"] == "train"]
    half = (len(CLASSES) + 1) // 2
    fig, axes = plt.subplots(half, 2 * n_frames + 1, figsize=(12.5, 0.95 * half * 1.15),
                             gridspec_kw={"width_ratios": [1] * n_frames + [1.35] + [1] * n_frames})
    for r in range(half):
        axes[r, n_frames].axis("off")
        for g in range(2):
            idx = r + g * half
            cols = list(range(g * (n_frames + 1), g * (n_frames + 1) + n_frames))
            if idx >= len(CLASSES):
                for c in cols:
                    axes[r, c].axis("off")
                continue
            v = train[train["label"] == CLASSES[idx]].sort_values("video").iloc[0]
            pos = np.linspace(0, store.length(v["split"], v["video"]) - 1, n_frames).astype(int)
            for c, img in zip(cols, store.frames(v["split"], v["video"], pos)):
                axes[r, c].imshow(img)
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
                for sp in axes[r, c].spines.values():
                    sp.set_visible(True)
                    sp.set_linewidth(0.5)
            axes[r, cols[0]].set_ylabel(CLASSES[idx], rotation=0, ha="right", va="center", fontsize=10)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    fig.savefig(out)
    plt.close(fig)


def fig_label_example(cfg, videos, store, out, video="Arrest001_x264"):
    """Nhãn mức khung hình và cách chia clip kiểm thử trên một video."""
    step, clip_len = cfg["data"]["frame_step"], cfg["data"]["clip_len"]
    v = videos[videos["video"] == video].iloc[0]
    (start, end), *_ = load_temporal_annotations(data_paths(cfg)["temporal_ann"])[video]
    n = int(v["n_frames"])
    frame_idx = np.arange(n) * step
    labels = (frame_idx >= start) & (frame_idx <= end)
    inside = np.flatnonzero(labels)
    pos = np.unique(np.concatenate([np.linspace(0, inside[0] - 4, 3), np.linspace(inside[0] + 2, inside[-1] - 2, 3),
                                    np.linspace(inside[-1] + 4, n - 1, 2)]).astype(int))
    fig = plt.figure(figsize=(10, 3.0))
    for i, (p, img) in enumerate(zip(pos, store.frames(v["split"], video, pos))):
        ax = fig.add_axes([0.02 + i * 0.122, 0.42, 0.11, 0.5])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        color = "#C44E52" if labels[p] else "#555555"
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(color)
            sp.set_linewidth(2.5 if labels[p] else 0.8)
        ax.set_title(f"frame {frame_idx[p]}", fontsize=9, color=color)
    ax = fig.add_axes([0.04, 0.12, 0.93, 0.22])
    ax.add_patch(Rectangle((start, 0.55), end - start, 0.4, color="#C44E52", alpha=0.75))
    ax.text((start + end) / 2, 1.05, f"anomalous segment {start}–{end}", ha="center", va="bottom", fontsize=9, color="#C44E52")
    test_stride = split_clip_params(cfg, "test")[0]
    for k, s in enumerate(clip_starts(n, clip_len, test_stride)):
        ax.add_patch(Rectangle((s * step, 0.05), min(clip_len, n - s) * step, 0.35,
                               fc="#DCE9F7" if k % 2 == 0 else "#B9D3EE", ec="#4C72B0", lw=0.6))
    ax.text(0, 0.47, f"test clips ({clip_len} frames each)", fontsize=9, va="bottom", color="#4C72B0")
    ax.set_xlim(0, n * step)
    ax.set_ylim(0, 1.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Frame index in the original video ({FPS} fps)")
    fig.savefig(out)
    plt.close(fig)


def _box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=color, ec="#333333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5, linespacing=1.35)


def _arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.2, color="#333333"))


def fig_pipeline(videos, out):
    """Sơ đồ quy trình tiền xử lý (scripts/prepare_data.py)."""
    n_frames, n_test = int(videos["n_frames"].sum()), int((videos["subset"] == "test").sum())
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")
    src, step, outc = "#E8E8E8", "#DCE9F7", "#DDF0DD"
    _box(ax, 0.1, 4.7, 3.3, 1.9, f"Kaggle data\n{n_frames:,} PNG frames\n64×64", src)
    _box(ax, 0.1, 0.5, 3.3, 1.7, f"Temporal annotations\n(Sultani et al.)\n{n_test} test videos", src)
    _box(ax, 4.3, 4.7, 3.6, 1.9, "index\nframes_index.csv\nvideos.csv", step)
    _box(ax, 4.3, 0.5, 3.6, 1.7, "splits\nsplit + frame labels\nsplits/*.txt, *.npz", step)
    _box(ax, 8.6, 4.7, 3.4, 1.9, f"pack\n{len(videos):,} .npy files\n(uint8)", step)
    _box(ax, 8.6, 0.5, 3.4, 1.7, "eda\nstatistics, figures,\nnoise measurement", step)
    _box(ax, 12.7, 2.6, 3.2, 1.9, "UCFCrimeClips\n+ transforms\ntensor 3×16×112×112", outc)
    _arrow(ax, (3.4, 5.65), (4.3, 5.65))
    _arrow(ax, (3.4, 1.35), (4.3, 1.35))
    _arrow(ax, (7.9, 5.65), (8.6, 5.65))
    _arrow(ax, (6.1, 4.7), (6.1, 2.2))
    _arrow(ax, (7.9, 1.35), (8.6, 1.35))
    _arrow(ax, (12.0, 5.3), (12.9, 4.5))
    _arrow(ax, (7.9, 2.1), (12.7, 3.3))
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------- tất cả
def run_all(cfg, noise=True):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    videos = load_videos(cfg)
    store = build_store(cfg)
    tables = {
        "class_distribution": class_table(videos),
        "duration_seconds": duration_table(videos),
        "test_anomaly": test_anomaly_table(cfg, videos),
        "clips": clip_table(cfg, videos),
    }
    if noise and any(data_paths(cfg)["packed_dir"].glob("*/*.npy")):
        tables["noise"] = noise_stats(cfg, videos)
    elif (TAB_DIR / "noise.csv").exists():  # dùng lại kết quả đo nhiễu lần trước (đo mất ~5 phút)
        tables["noise"] = pd.read_csv(TAB_DIR / "noise.csv", index_col=0)
    for name, t in tables.items():
        t.to_csv(TAB_DIR / f"{name}.csv")
    mean, std = pixel_stats(videos, store)
    summary = {"videos": len(videos), "frames": int(videos["n_frames"].sum()),
               "hours": round(float(videos["seconds"].sum()) / 3600, 1),
               "normal_frame_pct": round(100 * float(videos.loc[videos["class_id"] == 0, "n_frames"].sum() / videos["n_frames"].sum()), 1),
               "longest_video": videos.loc[videos["n_frames"].idxmax(), "video"],
               "longest_hours": round(float(videos["seconds"].max()) / 3600, 1),
               "pixel_mean": mean, "pixel_std": std}
    (TAB_DIR / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    fig_pipeline(videos, FIG_DIR / "pipeline.png")
    fig_class_distribution(videos, FIG_DIR / "class_distribution.png")
    fig_duration(videos, FIG_DIR / "duration.png")
    fig_sample_frames(videos, store, FIG_DIR / "sample_frames.png")
    fig_label_example(cfg, videos, store, FIG_DIR / "label_example.png")

    md = [f"# EDA\n\n```\n{json.dumps(summary, indent=1, ensure_ascii=False)}\n```\n"]
    for name, t in tables.items():
        try:
            md.append(f"## {name}\n\n{t.to_markdown()}\n")
        except ImportError:  # tabulate chưa cài
            md.append(f"## {name}\n\n```\n{t.to_string()}\n```\n")
    (TAB_DIR / "stats.md").write_text("\n".join(md), encoding="utf-8")
    return summary, tables
