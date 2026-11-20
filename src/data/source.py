"""[Owner: TV1] Đọc ảnh Kaggle từ file zip hoặc thư mục đã giải nén, và kho frame theo video.

- FrameSource: đọc file PNG theo đường dẫn bên trong zip/thư mục (không cần giải nén 1,38 triệu file).
- build_frame_index: bảng mỗi frame một dòng (split, label, class_id, video, frame_idx, pos, path).
- VideoStore: lấy frame của video theo vị trí -> uint8 (n, H, W, 3); đọc file .npy đã đóng gói
  (nhanh, cần ~16 GB) hoặc giải mã PNG trực tiếp (không tốn thêm dung lượng, dùng trên Kaggle).
"""
import io
import re
import threading
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .constants import CLASS_TO_ID

# vd: Train/Abuse/Abuse001_x264_120.png -> video "Abuse001_x264", frame gốc 120
_FRAME_RE = re.compile(r"(?P<split>Train|Test)/(?P<label>[^/]+)/(?P<video>[^/]+)_(?P<frame_idx>\d+)\.png$")
# mở zip phải đọc ~1,4 triệu mục (~20 s, ~0,7 GB RAM): các luồng dùng chung một handle
_ZIP_LOCK = threading.Lock()


class FrameSource:
    """Đọc file từ zip hoặc thư mục. Zip được mở lazy và không bị pickle, nên đối tượng
    gửi được sang các worker của DataLoader (mỗi worker tự mở handle riêng)."""

    def __init__(self, root):
        self.root = Path(root)
        self.is_zip = self.root.is_file()
        self._zip = None

    def __getstate__(self):
        return {**self.__dict__, "_zip": None}

    def _open_zip(self):
        if self._zip is None:
            with _ZIP_LOCK:
                if self._zip is None:
                    self._zip = zipfile.ZipFile(self.root)
        return self._zip

    def names(self):
        if self.is_zip:
            return [n for n in self._open_zip().namelist() if n.endswith(".png")]
        return [p.relative_to(self.root).as_posix() for p in self.root.rglob("*.png")]

    def read(self, name):
        if self.is_zip:
            return self._open_zip().read(name)  # zipfile kiểm tra CRC-32 khi đọc hết file
        return (self.root / name).read_bytes()

    def read_image(self, name):
        return np.asarray(Image.open(io.BytesIO(self.read(name))).convert("RGB"))


def build_frame_index(source):
    """Mỗi frame một dòng. `frame_idx` là số thứ tự frame trong video gốc (bội số của 10);
    `pos` là vị trí của frame trong video sau khi sắp xếp (= hàng trong file .npy).

    Sắp xếp theo frame_idx dạng SỐ: trong zip thứ tự là _0, _10, _100, _1000... (theo chuỗi).
    """
    rows, skipped = [], []
    for name in source.names():
        m = _FRAME_RE.search(name)
        if m is None or m["label"] not in CLASS_TO_ID:
            skipped.append(name)
            continue
        rows.append((m["split"], m["label"], m["video"], int(m["frame_idx"]), name))
    df = pd.DataFrame(rows, columns=["split", "label", "video", "frame_idx", "path"])
    df["class_id"] = df["label"].map(CLASS_TO_ID)
    df = df.sort_values(["split", "label", "video", "frame_idx"], ignore_index=True)
    df["pos"] = df.groupby(["split", "video"]).cumcount()
    return df[["split", "label", "class_id", "video", "frame_idx", "pos", "path"]], skipped


@lru_cache(maxsize=512)
def _open_packed(path):
    return np.load(path, mmap_mode="r")


class VideoStore:
    """Frame của một video theo vị trí, dạng uint8 (n, H, W, 3) RGB.

    Mặc định đọc `packed_dir/<Train|Test>/<video>.npy` (memory-map: chỉ đọc đúng các frame cần).
    Truyền `raw` (zip/thư mục Kaggle) và `frame_index` thì giải mã PNG trực tiếp.
    """

    def __init__(self, packed_dir=None, raw=None, frame_index=None):
        self.packed_dir = Path(packed_dir) if packed_dir else None
        self.source = None
        if raw is not None:
            self.source = FrameSource(raw)
            index = pd.read_csv(frame_index, usecols=["split", "video", "frame_idx", "pos", "path"])
            self._videos = {}
            for (split, video), g in index.groupby(["split", "video"], sort=False):
                g = g.sort_values("pos")
                first = g.iloc[0]
                prefix = first["path"][: -len(f"{video}_{first['frame_idx']}.png")]
                # bản ghi gọn cho mỗi video: tiền tố đường dẫn + số frame (int32)
                self._videos[(split, video)] = (prefix, g["frame_idx"].to_numpy(np.int32))

    def _packed(self, split, video):
        return _open_packed(str(self.packed_dir / split / f"{video}.npy"))

    def length(self, split, video):
        if self.source is None:
            return len(self._packed(split, video))
        return len(self._videos[(split, video)][1])

    def frames(self, split, video, pos):
        if self.source is None:
            return np.ascontiguousarray(self._packed(split, video)[pos])
        prefix, frame_idx = self._videos[(split, video)]
        return np.stack([self.source.read_image(f"{prefix}{video}_{frame_idx[p]}.png") for p in pos])
