"""[Owner: TV1] Đọc nhãn thời gian test UCF-Crime -> nhãn 0/1 cho từng frame.

Định dạng mỗi dòng: <video>.mp4  <Class>  s1 e1 s2 e2   (-1 nếu không có đoạn thứ 2)
"""
from pathlib import Path
import numpy as np


def load_temporal_annotations(path):
    ann = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        name = Path(parts[0]).stem          # vd: Abuse028_x264
        s1, e1, s2, e2 = map(int, parts[2:6])
        ann[name] = [(s, e) for s, e in ((s1, e1), (s2, e2)) if s >= 0]
    return ann


def frame_labels(video, num_frames, ann, frame_step=10):
    """Nhãn cho các ảnh đã lấy mẫu: ảnh k <-> frame gốc k*frame_step."""
    idx = np.arange(num_frames) * frame_step
    y = np.zeros(num_frames, dtype=np.int64)
    for s, e in ann.get(video, []):
        y[(idx >= s) & (idx <= e)] = 1
    return y
