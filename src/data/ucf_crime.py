"""[Owner: TV1] Dataset clip cho chế độ phân loại (cls) + các tiện ích dùng chung về đường dẫn và clip.

    ds = UCFCrimeClips(cfg, "train", build_train_transform(cfg))
    clip, class_id, binary = ds[0]          # clip: (3, T, H, W) float32
    loader = DataLoader(ds, batch_size=16, sampler=make_balanced_sampler(ds), num_workers=4)

Cách sinh clip (cấu hình trong `configs/base.yaml`, mục `data`):
- train/val: cửa sổ `clip_len` frame, bước `clip_stride` với video bất thường và `normal_clip_stride`
  với video Normal; tối đa `max_clips_per_video` clip/video, chọn đều theo thời gian (vài video
  Normal dài tới 9 giờ). Nhãn clip = nhãn video (nhãn yếu).
- test: bước `test_clip_stride` (mặc định = clip_len), không giới hạn số clip -> phủ mọi frame.
- Luôn có một clip căn theo cuối video; video ngắn hơn clip_len được lặp frame cuối.
`ds.items[i] = (video, start, class_id)` cho biết clip i lấy từ đâu (để ghép điểm về từng frame).
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .constants import CLASS_TO_ID, NORMAL
from .source import VideoStore

ROOT = Path(__file__).resolve().parents[2]
FOLDER = {"train": "Train", "val": "Train", "test": "Test"}  # thư mục Kaggle chứa mỗi split


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def data_paths(cfg):
    d = cfg["data"]
    split_dir = resolve(d["split_dir"])
    processed = split_dir.parent
    return {
        "frames_root": resolve(d["frames_root"]),
        "temporal_ann": resolve(d["temporal_ann"]),
        "split_dir": split_dir,
        "processed": processed,
        "packed_dir": resolve(d.get("packed_dir", processed / "frames")),
        "frame_index": processed / "frames_index.csv",
        "videos": processed / "videos.csv",
        "test_labels": split_dir / "test_frame_labels.npz",  # src/evaluate.py (TV2) đọc ở đây
    }


def find_raw(cfg):
    """frames_root nếu có, nếu không thì file zip Kaggle trong data/raw/ (không cần giải nén)."""
    root = data_paths(cfg)["frames_root"]
    if root.exists():
        return root
    zips = sorted(root.parent.glob("*.zip"))
    if zips:
        return zips[0]
    raise FileNotFoundError(f"Không thấy dữ liệu Kaggle ở {root} hoặc {root.parent}/*.zip")


def read_split(cfg, split):
    """splits/<split>.txt -> list[(video, class_name, n_frames)]."""
    lines = (data_paths(cfg)["split_dir"] / f"{split}.txt").read_text(encoding="utf-8").split("\n")
    return [(v, c, int(n)) for v, c, n in (line.split() for line in lines if line.strip())]


def build_store(cfg):
    """Dùng frame đã đóng gói (.npy) nếu có, nếu không thì đọc PNG trực tiếp."""
    p = data_paths(cfg)
    if any(p["packed_dir"].glob("*/*.npy")):
        return VideoStore(p["packed_dir"])
    return VideoStore(raw=find_raw(cfg), frame_index=p["frame_index"])


def clip_starts(n_frames, clip_len, stride, max_clips=None):
    """Vị trí bắt đầu các clip trong một video có n_frames frame."""
    last = max(n_frames - clip_len, 0)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)  # luôn phủ tới cuối video
    if max_clips and len(starts) > max_clips:
        starts = [starts[i] for i in np.linspace(0, len(starts) - 1, max_clips).round().astype(int)]
    return starts


def split_clip_params(cfg, split):
    """(stride bất thường, stride Normal, số clip tối đa/video) của một split."""
    d = cfg["data"]
    if split == "test":
        s = d.get("test_clip_stride", d["clip_len"])
        return s, s, None
    return d["clip_stride"], d.get("normal_clip_stride", d["clip_stride"]), d.get("max_clips_per_video")


def load_video(cfg, split, video, store=None):
    """Toàn bộ frame của một video: uint8 (N, 64, 64, 3) RGB. Đưa từng clip qua
    transforms.build_eval_transform(cfg) để tiền xử lý giống hệt lúc train."""
    store = store or build_store(cfg)
    folder = FOLDER[split]
    return store.frames(folder, video, np.arange(store.length(folder, video)))


class UCFCrimeClips(Dataset):
    """Tham số ghi đè (mặc định lấy theo configs/base.yaml):
    - videos: chỉ dùng các video này, list[(video, class_name, n_frames)] như read_split
    - stride: một bước trượt cho mọi video; max_clips: giới hạn clip/video (0 = không giới hạn)
    - jitter: dịch ngẫu nhiên vị trí clip (mặc định chỉ bật cho split "train")
    Trích đặc trưng cho MIL: UCFCrimeClips(cfg, split, build_eval_transform(cfg), stride=clip_len,
    max_clips=0, jitter=False) -> mọi clip không chồng lấn của mỗi video, theo thứ tự thời gian.
    """

    def __init__(self, cfg, split="train", transform=None, store=None, videos=None, stride=None, max_clips=None,
                 jitter=None):
        self.cfg, self.split, self.transform = cfg, split, transform
        self.clip_len = cfg["data"]["clip_len"]
        self.folder = FOLDER[split]
        self.store = store or build_store(cfg)
        # dịch ngẫu nhiên vị trí bắt đầu khi train (tăng cường theo thời gian)
        jitter = split == "train" if jitter is None else jitter
        self.max_shift = cfg["data"]["clip_stride"] // 2 if jitter else 0
        anomaly_stride, normal_stride, cap = split_clip_params(cfg, split)
        if stride:
            anomaly_stride = normal_stride = stride
        max_clips = cap if max_clips is None else max_clips
        self.items = []  # list[(video, start, class_id)]
        for video, label, n in videos or read_split(cfg, split):
            s = normal_stride if label == NORMAL else anomaly_stride
            self.items += [(video, st, CLASS_TO_ID[label]) for st in clip_starts(n, self.clip_len, s, max_clips)]

    def __len__(self):
        return len(self.items)

    def frame_positions(self, i, shift=0):
        """Vị trí các frame của clip i (đã kẹp trong video: video ngắn lặp frame cuối)."""
        video, start, _ = self.items[i]
        n = self.store.length(self.folder, video)
        return np.clip(np.arange(start + shift, start + shift + self.clip_len), 0, n - 1)

    def __getitem__(self, i):
        video, _, class_id = self.items[i]
        shift = np.random.randint(-self.max_shift, self.max_shift + 1) if self.max_shift else 0
        frames = self.store.frames(self.folder, video, self.frame_positions(i, shift))  # (T, H, W, 3) uint8
        clip = self.transform(frames) if self.transform else torch.from_numpy(frames)
        return clip, class_id, int(class_id > 0)

    @property
    def binary_labels(self):
        return np.array([int(c > 0) for _, _, c in self.items])


def make_balanced_sampler(dataset, num_samples=None):
    """WeightedRandomSampler lấy clip Normal và Anomaly với xác suất bằng nhau (thay cho shuffle)."""
    y = dataset.binary_labels
    weights = 1.0 / np.bincount(y, minlength=2)[y]
    return WeightedRandomSampler(torch.tensor(weights), num_samples or len(y), replacement=True)
