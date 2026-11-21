import queue
import threading
from fractions import Fraction

import numpy as np

from app.live_video import LiveVideoProcessor


class FakeVideoSource:
    fps = 25.0

    def __init__(self):
        self.frames = queue.Queue()

    def add_frame(self, frame, frame_index, timestamp):
        self.frames.put((frame, frame_index, timestamp))

    def read_with_metadata(self, timeout=0):
        assert timeout == 0
        try:
            frame, frame_index, timestamp = self.frames.get_nowait()
        except queue.Empty:
            return False, None, None, None
        return True, frame, frame_index, timestamp


class BlockingInference:
    frame_step = 1

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.processed = []
        self.processed_latest = threading.Event()

    def process_frame(self, _frame, frame_index, _timestamp):
        if frame_index == 0:
            self.started.set()
            self.release.wait()
        self.processed.append(frame_index)
        if frame_index == 2:
            self.processed_latest.set()
        return None


def test_slow_inference_does_not_block_preview_and_stale_work_is_dropped():
    source = FakeVideoSource()
    inference = BlockingInference()
    processor = LiveVideoProcessor(source, inference)
    time_base = Fraction(1, 90000)
    frame_results = []

    try:
        source.add_frame(np.zeros((2, 2, 3), dtype=np.uint8), 0, 0.0)
        callback = threading.Thread(
            target=lambda: frame_results.append(processor.next_frame(0, time_base))
        )
        callback.start()

        assert inference.started.wait(timeout=1)
        callback.join(timeout=1)
        assert not callback.is_alive()
        np.testing.assert_array_equal(
            frame_results[0].to_ndarray(format="bgr24"),
            np.zeros((2, 2, 3), dtype=np.uint8),
        )

        source.add_frame(np.ones((2, 2, 3), dtype=np.uint8), 1, 0.04)
        processor.next_frame(3000, time_base)
        source.add_frame(np.full((2, 2, 3), 2, dtype=np.uint8), 2, 0.08)
        processor.next_frame(6000, time_base)

        inference.release.set()
        assert inference.processed_latest.wait(timeout=1)
        assert inference.processed == [0, 2]
    finally:
        inference.release.set()
        processor.stop()
