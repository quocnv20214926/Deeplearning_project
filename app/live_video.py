"""Low-latency live preview with inference performed on a worker thread."""

import queue
import threading

import av
import numpy as np

from app.inference import SlidingWindowInference
from app.video_source import VideoSource


class LiveVideoProcessor:
    def __init__(self, source: VideoSource, inference: SlidingWindowInference):
        self.source = source
        self.inference = inference
        self.score_queue = queue.Queue(maxsize=1000)
        self._inference_frames = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._stopped_event = threading.Event()
        self._error_lock = threading.Lock()
        self._error = None
        self._last_frame = None
        self._last_frame_index = -1
        self._worker = threading.Thread(
            target=self._run_inference,
            name="live-video-inference",
            daemon=True,
        )
        self._worker.start()

    def next_frame(self, pts, time_base):
        success, frame, frame_index, timestamp = self.source.read_with_metadata(
            timeout=0
        )
        if success:
            if frame_index is None:
                frame_index = self._last_frame_index + 1
            if timestamp is None:
                timestamp = frame_index / self.source.fps
            self._last_frame_index = frame_index
            self._last_frame = frame.copy()

            if (
                frame_index % self.inference.frame_step == 0
                and self.get_error() is None
                and not self._stop_event.is_set()
            ):
                self._enqueue_inference_frame(frame, frame_index, timestamp)
        elif self._last_frame is not None:
            frame = self._last_frame
        else:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)

        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

    def _enqueue_inference_frame(self, frame, frame_index, timestamp):
        item = (frame.copy(), frame_index, timestamp)
        try:
            self._inference_frames.put_nowait(item)
        except queue.Full:
            try:
                self._inference_frames.get_nowait()
            except queue.Empty:
                pass
            self._inference_frames.put_nowait(item)

    def _run_inference(self):
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._inference_frames.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    return

                frame, frame_index, timestamp = item

                try:
                    row = self.inference.process_frame(
                        frame, frame_index, timestamp
                    )
                except (RuntimeError, ValueError) as exc:
                    with self._error_lock:
                        self._error = str(exc)
                    return

                if row is not None:
                    self._put_latest_score(row)
        finally:
            self._stopped_event.set()

    def _put_latest_score(self, row):
        try:
            self.score_queue.put_nowait(row)
        except queue.Full:
            try:
                self.score_queue.get_nowait()
            except queue.Empty:
                pass
            self.score_queue.put_nowait(row)

    def drain_scores(self):
        rows = []
        while True:
            try:
                rows.append(self.score_queue.get_nowait())
            except queue.Empty:
                return rows

    def get_error(self):
        with self._error_lock:
            return self._error

    def stop(self):
        self._stop_event.set()
        while True:
            try:
                self._inference_frames.get_nowait()
            except queue.Empty:
                break
        try:
            self._inference_frames.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=0.25)
