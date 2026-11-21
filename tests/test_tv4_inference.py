import numpy as np
import torch
from torch import nn

from app.inference import SlidingWindowInference, export_video_scores
from src.data.constants import CLASSES


class FixedLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_shape = None

    def forward(self, clip):
        self.seen_shape = tuple(clip.shape)
        logits = torch.zeros((clip.shape[0], len(CLASSES)), device=clip.device)
        logits[:, 1] = 4.0
        return logits


class SequentialLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, clip):
        logits = torch.zeros((clip.shape[0], len(CLASSES)), device=clip.device)
        logits[:, 0 if self.calls == 0 else 1] = 4.0
        self.calls += 1
        return logits


def test_sliding_inference_samples_and_scores_windows():
    cfg = {"data": {"frame_step": 10, "clip_len": 16, "clip_stride": 8, "input_size": 8}}
    model = FixedLogits()
    inference = SlidingWindowInference(cfg, checkpoint=None, model=model)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    rows = []
    for frame_index in range(240):
        row = inference.process_frame(frame, frame_index, frame_index / 30)
        if row is not None:
            rows.append(row)

    assert len(rows) == 2
    assert [row["frame_index"] for row in rows] == [70, 150]
    assert [row["timestamp_seconds"] for row in rows] == [70 / 30, 150 / 30]
    assert all(row["predicted_class"] == "Abuse" for row in rows)
    assert all(row["raw_score"] > 0.9 for row in rows)
    assert model.seen_shape == (1, 3, 16, 8, 8)


def test_sliding_inference_smooths_scores():
    cfg = {"data": {"frame_step": 1, "clip_len": 2, "clip_stride": 1, "input_size": 8}}
    model = SequentialLogits()
    inference = SlidingWindowInference(
        cfg, checkpoint=None, model=model, smooth_window=2
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    assert inference.process_frame(frame, 0, 0.0) is None
    first = inference.process_frame(frame, 1, 1.0)
    second = inference.process_frame(frame, 2, 2.0)
    assert first is not None and second is not None
    assert second["score"] == (first["raw_score"] + second["raw_score"]) / 2


def test_video_pipeline_writes_timestamped_csv(monkeypatch, tmp_path):
    class FakeCapture:
        def __init__(self, _path):
            self.frames = iter([np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)])

        def isOpened(self):
            return True

        def get(self, _property):
            return 2.0

        def read(self):
            try:
                return True, next(self.frames)
            except StopIteration:
                return False, None

        def release(self):
            pass

    import app.inference as inference_module

    monkeypatch.setattr(inference_module.cv2, "VideoCapture", FakeCapture)
    cfg = {"data": {"frame_step": 1, "clip_len": 2, "clip_stride": 2, "input_size": 8}}
    inference = SlidingWindowInference(cfg, checkpoint=None, model=FixedLogits())
    output = tmp_path / "scores.csv"

    assert export_video_scores("fake.mp4", inference, output) == 2
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "timestamp_seconds,frame_index,score,raw_score,predicted_class,class_confidence"
    assert [line.split(",")[1] for line in lines[1:]] == ["0", "2"]
    assert [line.split(",")[0] for line in lines[1:]] == ["0.0", "1.0"]
