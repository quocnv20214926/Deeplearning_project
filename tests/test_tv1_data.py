"""[Owner: TV1] Test pipeline dữ liệu trên một bộ dữ liệu giả nhỏ (không cần dữ liệu thật)."""
import numpy as np
import pytest
import torch
from PIL import Image

from src.data.annotations import load_temporal_annotations
from src.data.prepare import step_index, step_pack, step_splits
from src.data.source import VideoStore
from src.data.transforms import ClipTransform
from src.data.ucf_crime import UCFCrimeClips, clip_starts, data_paths, load_video, make_balanced_sampler

# (split, class, video, số ảnh)
VIDEOS = [
    ("Train", "NormalVideos", "Normal_Videos001_x264", 40), ("Train", "NormalVideos", "Normal_Videos002_x264", 30),
    ("Train", "Abuse", "Abuse001_x264", 25), ("Train", "Abuse", "Abuse002_x264", 9),
    ("Test", "NormalVideos", "Normal_Videos_003_x264", 20), ("Test", "Abuse", "Abuse028_x264", 35),
]
ANNOTATION = "Abuse028_x264.mp4  Abuse  165  240  -1  -1\nNormal_Videos_003_x264.mp4  Normal  -1  -1  -1  -1\n"


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("ucf")
    rng = np.random.default_rng(0)
    for split, label, video, n in VIDEOS:
        d = root / "raw" / split / label
        d.mkdir(parents=True, exist_ok=True)
        for k in range(n):  # tên file theo số frame gốc (bội số của 10), giống bản Kaggle
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(d / f"{video}_{k * 10}.png")
    (root / "ann.txt").write_text(ANNOTATION)
    cfg = {"seed": 42, "data": {
        "frames_root": str(root / "raw"), "temporal_ann": str(root / "ann.txt"),
        "split_dir": str(root / "processed" / "splits"), "packed_dir": str(root / "processed" / "frames"),
        "val_ratio": 0.1, "frame_step": 10, "clip_len": 16, "clip_stride": 8, "normal_clip_stride": 16,
        "max_clips_per_video": 2, "test_clip_stride": 16, "input_size": 112}}
    step_index(cfg, root / "raw")
    step_pack(cfg, root / "raw", workers=2)
    step_splits(cfg)
    return cfg


def test_splits_by_video(cfg):
    p = data_paths(cfg)
    split = {s: [l.split()[0] for l in (p["split_dir"] / f"{s}.txt").read_text().split("\n") if l] for s in ("train", "val", "test")}
    assert len(split["val"]) == 2  # 1 video mỗi lớp
    assert len(split["train"]) + len(split["val"]) == 4 and len(split["test"]) == 2
    assert not set(split["train"]) & set(split["val"])


def test_frame_labels(cfg):
    labels = np.load(data_paths(cfg)["test_labels"])
    y = labels["Abuse028_x264"]
    assert len(y) == 35 and np.flatnonzero(y).tolist() == list(range(17, 25))  # frame 170..240
    assert labels["Normal_Videos_003_x264"].sum() == 0


def test_clip_starts():
    assert clip_starts(40, 16, 8) == [0, 8, 16, 24]
    assert clip_starts(30, 16, 16) == [0, 14]  # clip cuối căn theo cuối video
    assert clip_starts(9, 16, 8) == [0]  # video ngắn: 1 clip (lặp frame cuối)
    assert len(clip_starts(1000, 16, 8, max_clips=64)) == 64


def test_test_clips_cover_every_frame(cfg):
    ds = UCFCrimeClips(cfg, "test")
    for video, _, n in [(v, c, n) for _, c, v, n in VIDEOS if v in {i[0] for i in ds.items}]:
        covered = np.zeros(n, bool)
        for i, (v, start, _) in enumerate(ds.items):
            if v == video:
                covered[ds.frame_positions(i)] = True
        assert covered.all()


def test_dataset_item(cfg):
    ds = UCFCrimeClips(cfg, "train", ClipTransform(112, train=True))
    clip, class_id, binary = ds[0]
    assert clip.shape == (3, 16, 112, 112) and clip.dtype == torch.float32
    assert binary == int(class_id > 0)
    short = [i for i, (v, _, _) in enumerate(ds.items) if v == "Abuse002_x264"]
    if short:  # video 9 ảnh -> clip 16 frame lặp frame cuối
        assert ds[short[0]][0].shape == (3, 16, 112, 112)


def test_packed_equals_raw_png(cfg):
    p = data_paths(cfg)
    packed = UCFCrimeClips(cfg, "test", ClipTransform(112), store=VideoStore(p["packed_dir"]))
    raw = UCFCrimeClips(cfg, "test", ClipTransform(112),
                        store=VideoStore(raw=p["frames_root"], frame_index=p["frame_index"]))
    for i in range(len(packed)):
        assert torch.equal(packed[i][0], raw[i][0])


def test_feature_extraction_clips(cfg):
    """Trích đặc trưng MIL: mọi clip không chồng lấn, không giới hạn, không dịch ngẫu nhiên."""
    ds = UCFCrimeClips(cfg, "train", ClipTransform(112), stride=16, max_clips=0, jitter=False)
    assert ds.max_shift == 0
    n_frames = {v: n for _, _, v, n in VIDEOS}
    for video in {v for v, _, _ in ds.items}:
        starts = [s for v, s, _ in ds.items if v == video]
        assert starts == clip_starts(n_frames[video], 16, 16)  # cùng một bước cho mọi lớp, không giới hạn
    assert torch.equal(ds[0][0], ds[0][0])  # tất định


def test_load_video(cfg):
    frames = load_video(cfg, "test", "Abuse028_x264")
    assert frames.shape == (35, 64, 64, 3) and frames.dtype == np.uint8


def test_balanced_sampler(cfg):
    ds = UCFCrimeClips(cfg, "train")
    idx = list(make_balanced_sampler(ds, num_samples=4000))
    share = ds.binary_labels[idx].mean()
    assert 0.45 < share < 0.55


def test_transform_app_frames():
    """Frame video thật (to, BGR->RGB) đi qua cùng đường với ảnh 64x64 lúc train."""
    frames = np.random.default_rng(0).integers(0, 255, (16, 240, 320, 3), dtype=np.uint8)
    x = ClipTransform(112)(frames[..., ::-1])
    assert x.shape == (3, 16, 112, 112)
    assert torch.equal(x, ClipTransform(112)(frames[..., ::-1]))  # eval: tất định


def test_annotation_parser(cfg):
    ann = load_temporal_annotations(data_paths(cfg)["temporal_ann"])
    assert ann == {"Abuse028_x264": [(165, 240)], "Normal_Videos_003_x264": []}
