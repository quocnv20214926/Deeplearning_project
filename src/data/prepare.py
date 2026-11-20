"""[Owner: TV1] Các bước chuẩn bị dữ liệu, gọi từ scripts/prepare_data.py.

annotations -> index -> pack -> splits. Mỗi bước đọc đầu ra của bước trước, ghi ra data/processed/:
    frames_index.csv      mỗi frame một dòng (split, label, class_id, video, frame_idx, pos, path)
    videos.csv            mỗi video một dòng (n_frames, frame đầu/cuối, số frame thiếu, subset)
    frames/<Train|Test>/<video>.npy    uint8 (N, 64, 64, 3) — tuỳ chọn, ~16 GB
    splits/{train,val,test}.txt        "<video> <class> <n_frames>" mỗi dòng
    test_frame_labels.npz              {video: nhãn 0/1 cho từng ảnh test}
"""
import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .annotations import frame_labels, load_temporal_annotations
from .constants import NORMAL
from .source import FrameSource, build_frame_index
from .ucf_crime import clip_starts, data_paths, split_clip_params

ANNOTATION_URL = ("https://raw.githubusercontent.com/WaqasSultani/AnomalyDetectionCVPR2018/"
                  "master/Temporal_Anomaly_Annotation.txt")


def step_annotations(cfg):
    """Tải file nhãn thời gian của tác giả (Sultani et al.) nếu chưa có."""
    path = data_paths(cfg)["temporal_ann"]
    if not path.exists():
        print(f"downloading temporal annotations -> {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(ANNOTATION_URL, timeout=60) as r:
            path.write_bytes(r.read())
    n = len(load_temporal_annotations(path))
    print(f"temporal annotations: {n} videos")
    return path


def step_index(cfg, raw, force=False):
    p = data_paths(cfg)
    source = FrameSource(raw)
    if p["frame_index"].exists() and not force:
        frames = pd.read_csv(p["frame_index"])
        n_png = len(source.names())
        if len(frames) != n_png:
            raise RuntimeError(f"{p['frame_index']} có {len(frames)} dòng nhưng dữ liệu có {n_png} ảnh; chạy lại với --force-index")
        print(f"frame index up to date ({len(frames):,} frames)")
    else:
        frames, skipped = build_frame_index(source)
        if skipped:
            print(f"WARNING: {len(skipped)} file png sai mẫu tên, vd {skipped[:3]}")
        p["processed"].mkdir(parents=True, exist_ok=True)
        frames.to_csv(p["frame_index"], index=False)

    step = cfg["data"]["frame_step"]
    g = frames.groupby(["split", "label", "class_id", "video"])["frame_idx"]
    videos = g.agg(n_frames="count", first_frame="min", last_frame="max").reset_index()
    videos["n_missing"] = (videos["last_frame"] - videos["first_frame"]) // step + 1 - videos["n_frames"]
    old = pd.read_csv(p["videos"]) if p["videos"].exists() else None
    if old is not None and "subset" in old:  # giữ split đã có
        videos = videos.merge(old[["split", "video", "subset"]], on=["split", "video"], how="left")
    videos.to_csv(p["videos"], index=False)

    # nhãn khung hình dùng ánh xạ ảnh k <-> frame gốc k*frame_step, nên frame phải bắt đầu từ 0 và liên tục
    bad = videos[(videos["n_missing"] != 0) | (videos["first_frame"] != 0)]
    dup = videos[videos.duplicated("video", keep=False)]
    print(f"{len(frames):,} frames, {len(videos):,} videos "
          f"(Train {int((videos['split'] == 'Train').sum())}, Test {int((videos['split'] == 'Test').sum())})")
    if len(bad) or len(dup):
        raise RuntimeError(f"{len(bad)} video thiếu frame / không bắt đầu từ 0, {len(dup)} tên video trùng")
    return videos


def _pack_video(source, split, video, paths, out_dir, size):
    out = out_dir / split / f"{video}.npy"
    if out.exists():
        return video, 0
    frames = np.zeros((len(paths), size, size, 3), dtype=np.uint8)
    n_bad = 0
    for i, path in enumerate(paths):
        try:
            img = Image.open(io.BytesIO(source.read(path))).convert("RGB")
            if img.size != (size, size):
                img = img.resize((size, size), Image.BILINEAR)
            frames[i] = np.asarray(img)
        except Exception:
            n_bad += 1  # ảnh lỗi: lặp lại frame trước để giữ đúng vị trí
            if i > 0:
                frames[i] = frames[i - 1]
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, frames)
    tmp.replace(out)  # ghi nguyên tử: bị ngắt giữa chừng cũng không để lại file hỏng
    return video, n_bad


def step_pack(cfg, raw, workers=8, size=64):
    """Đóng gói mỗi video thành một file .npy (đọc 1 clip = 1 lát cắt thay vì 16 file PNG)."""
    p = data_paths(cfg)
    source = FrameSource(raw)
    index = pd.read_csv(p["frame_index"])
    for split in index["split"].unique():
        (p["packed_dir"] / split).mkdir(parents=True, exist_ok=True)
    jobs = [(s, v, g["path"].tolist()) for (s, v), g in index.groupby(["split", "video"], sort=False)]
    bad = {}
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(_pack_video, source, s, v, paths, p["packed_dir"], size) for s, v, paths in jobs]
        for fut in tqdm(futures, desc="packing videos"):
            video, n_bad = fut.result()
            if n_bad:
                bad[video] = n_bad
    videos = pd.read_csv(p["videos"])
    wrong = [v.video for v in videos.itertuples()
             if np.load(p["packed_dir"] / v.split / f"{v.video}.npy", mmap_mode="r").shape[0] != v.n_frames]
    print(f"packed {len(jobs):,} videos into {p['packed_dir']}; unreadable frames: {sum(bad.values())}; "
          f"length mismatches: {len(wrong)}")
    if bad or wrong:
        raise RuntimeError(f"ảnh lỗi: {bad}; sai số frame: {wrong[:5]}")


def assign_subsets(videos, val_ratio, seed):
    """Tách val_ratio số video Train của MỖI lớp làm val (theo video -> không rò rỉ frame)."""
    rng = np.random.default_rng(seed)
    videos["subset"] = np.where(videos["split"] == "Test", "test", "train")
    for _, g in videos[videos["split"] == "Train"].groupby("label"):
        n_val = max(1, round(len(g) * val_ratio))
        videos.loc[rng.permutation(g.index)[:n_val], "subset"] = "val"
    return videos


def step_splits(cfg):
    p = data_paths(cfg)
    d = cfg["data"]
    videos = pd.read_csv(p["videos"])
    videos = assign_subsets(videos.drop(columns="subset", errors="ignore"), d.get("val_ratio", 0.1), cfg.get("seed", 42))
    videos.to_csv(p["videos"], index=False)

    p["split_dir"].mkdir(parents=True, exist_ok=True)
    for subset in ["train", "val", "test"]:
        sub = videos[videos["subset"] == subset].sort_values(["label", "video"])
        lines = [f"{v.video} {v.label} {v.n_frames}" for v in sub.itertuples()]
        (p["split_dir"] / f"{subset}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ann = load_temporal_annotations(p["temporal_ann"])
    test = videos[videos["subset"] == "test"]
    labels = {v.video: frame_labels(v.video, v.n_frames, ann, d["frame_step"]).astype(np.uint8) for v in test.itertuples()}
    np.savez_compressed(p["test_labels"], **labels)

    missing = sorted(set(test["video"]) - ann.keys())
    unused = sorted(ann.keys() - set(test["video"]))
    y = np.concatenate(list(labels.values()))
    print(f"test: {len(test)} videos, {len(test) - len(missing)} có nhãn thời gian; "
          f"{y.sum():,}/{len(y):,} frame bất thường ({100 * y.mean():.2f}%)")
    if missing or unused:
        raise RuntimeError(f"video test không có nhãn: {missing[:5]}; nhãn không khớp video: {unused[:5]}")
    for subset in ["train", "val", "test"]:
        sub = videos[videos["subset"] == subset]
        stride, normal_stride, max_clips = split_clip_params(cfg, subset)
        n_clips = [len(clip_starts(v.n_frames, d["clip_len"], normal_stride if v.label == NORMAL else stride, max_clips))
                   for v in sub.itertuples()]
        n_anom = sum(n for n, v in zip(n_clips, sub.itertuples()) if v.class_id > 0)
        print(f"{subset:5s}: {len(sub):5d} videos ({int((sub['class_id'] == 0).sum())} normal), "
              f"{int(sub['n_frames'].sum()):>9,} frames, {sum(n_clips):>7,} clips ({100 * n_anom / sum(n_clips):.1f}% anomaly)")
    return videos
