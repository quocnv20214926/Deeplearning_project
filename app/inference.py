"""[Owner: TV4] Bộ đệm trượt + chạy model realtime.
- Lấy 1/10 frame (khớp frame_step lúc train), resize theo transforms.build_eval_transform (TV1).
- Đủ clip_len frame -> chạy model (ONNX Runtime/PyTorch FP16) -> (score bất thường, lớp top-1).
- Làm mượt score (moving average).

Sliding-window video inference and CSV export.

Run from the repository root:
    python -m app.inference path/to/video.mp4 --config configs/m1_r2plus1d.yaml \
        --ckpt weights/r2plus1d_18_cls.pt
"""
import argparse
import csv
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.constants import CLASSES
from src.data.transforms import build_eval_transform
from src.utils.config import load_config
from src.utils.model_io import load_model

ROOT = Path(__file__).resolve().parents[1]
CSV_FIELDS = (
    "timestamp_seconds",
    "frame_index",
    "score",
    "raw_score",
    "predicted_class",
    "class_confidence",
)


class SlidingWindowInference:
    """Sample source frames and infer overlapping, transformed clips."""

    def __init__(
        self,
        cfg,
        checkpoint,
        mode="cls",
        device=None,
        smooth_window=1,
        model=None,
    ):
        self.cfg = cfg
        if mode not in {"cls", "mil"}:
            raise ValueError(f"Unsupported inference mode: {mode}")
        self.mode = mode
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        data_cfg = cfg["data"]
        self.frame_step = int(data_cfg.get("frame_step", 10))
        self.clip_len = int(data_cfg["clip_len"])
        self.stride = int(data_cfg.get("clip_stride", self.clip_len))
        if self.frame_step < 1 or self.clip_len < 1 or self.stride < 1:
            raise ValueError("frame_step, clip_len, and stride must be positive")
        if smooth_window < 1:
            raise ValueError("smooth_window must be at least 1")

        self.transform = build_eval_transform(cfg)
        self.model = model if model is not None else load_model(
            cfg, mode, checkpoint, self.device
        )
        self.model = self.model.to(self.device).eval()
        self.frames = deque(maxlen=self.clip_len)
        self.frame_indices = deque(maxlen=self.clip_len)
        self.timestamps = deque(maxlen=self.clip_len)
        self.sampled_count = 0
        self.next_inference_at = self.clip_len
        self.recent_scores = deque(maxlen=smooth_window)

    @torch.inference_mode()
    def process_frame(self, bgr_frame, frame_index, timestamp_seconds):
        """Return a score row when a full window is ready, otherwise None."""
        if frame_index % self.frame_step:
            return None

        rgb_frame = np.ascontiguousarray(bgr_frame[..., ::-1])
        self.frames.append(rgb_frame)
        self.frame_indices.append(int(frame_index))
        self.timestamps.append(float(timestamp_seconds))
        self.sampled_count += 1

        if self.sampled_count < self.next_inference_at:
            return None
        self.next_inference_at += self.stride

        clip = self.transform(np.stack(self.frames, axis=0)).unsqueeze(0).to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            output = self.model(clip).float()

        if self.mode == "cls":
            probabilities = output.softmax(dim=-1)[0]
            raw_score = 1.0 - float(probabilities[0].item())
            predicted_index = int(probabilities.argmax().item())
            predicted_class = (
                CLASSES[predicted_index] if predicted_index < len(CLASSES) else str(predicted_index)
            )
            confidence = float(probabilities[predicted_index].item())
        else:
            raw_score = float(output.reshape(-1)[0].sigmoid().item())
            predicted_class = ""
            confidence = ""

        self.recent_scores.append(raw_score)
        center = (self.clip_len - 1) // 2
        return {
            "timestamp_seconds": self.timestamps[center],
            "frame_index": self.frame_indices[center],
            "score": sum(self.recent_scores) / len(self.recent_scores),
            "raw_score": raw_score,
            "predicted_class": predicted_class,
            "class_confidence": confidence,
        }


def iter_video_scores(video_path, inference):
    """Yield score rows from a file, using source FPS for timestamps."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"Video has invalid FPS ({fps}): {video_path}")

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            row = inference.process_frame(frame, frame_index, frame_index / fps)
            if row is not None:
                yield row
            frame_index += 1
    finally:
        capture.release()


def export_video_scores(video_path, inference, output_path):
    """Run inference and write a time-indexed CSV; return the number of snippets."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in iter_video_scores(video_path, inference):
            writer.writerow(row)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Score video clips and export anomaly scores to CSV.")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--config", default="configs/m1_r2plus1d.yaml")
    parser.add_argument("--ckpt", required=True, help="Trained cls or MIL checkpoint")
    parser.add_argument("--mode", choices=("cls", "mil"), default="cls")
    parser.add_argument("--output", help="CSV path (default: results/predictions/<video>_scores.csv)")
    parser.add_argument("--device", help="Inference device, e.g. cpu or cuda")
    parser.add_argument("--smooth-window", type=int, default=1, help="Moving average window for scores")
    args = parser.parse_args()

    video_path = Path(args.video)
    config_path = Path(args.config)
    checkpoint_path = Path(args.ckpt)
    if not video_path.is_file():
        parser.error(f"Video file does not exist: {video_path}")
    if not config_path.is_file():
        parser.error(f"Config file does not exist: {config_path}")
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint_path}")

    cfg = load_config(config_path)
    inference = SlidingWindowInference(
        cfg,
        checkpoint_path,
        mode=args.mode,
        device=args.device,
        smooth_window=args.smooth_window,
    )
    output_path = Path(args.output) if args.output else (
        ROOT / "results" / "predictions" / f"{video_path.stem}_scores.csv"
    )
    count = export_video_scores(video_path, inference, output_path)
    print(f"Wrote {count} snippet scores to {output_path}")
    if count == 0:
        print(
            f"No complete clip was available "
            f"(requires {cfg['data']['clip_len']} sampled frames at 1/{inference.frame_step} rate)."
        )


if __name__ == "__main__":
    main()
