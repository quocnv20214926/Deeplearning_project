import numpy as np

from app.video_source import VideoSource


class FakeCapture:
    def __init__(self, _source):
        self.frames = [
            np.full((2, 2, 3), index, dtype=np.uint8)
            for index in range(3)
        ]

    def isOpened(self):
        return True

    def get(self, _property):
        return 25.0

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        pass


def test_video_source_read_metadata_preserves_source_timing(monkeypatch):
    monkeypatch.setattr("app.video_source.cv2.VideoCapture", FakeCapture)
    source = VideoSource("test.mp4").start()

    try:
        assert source.fps == 25.0
        first = source.read_with_metadata(timeout=1)
        second = source.read_with_metadata(timeout=1)
    finally:
        source.stop()

    assert first[0] and second[0]
    np.testing.assert_array_equal(first[1], np.zeros((2, 2, 3), dtype=np.uint8))
    np.testing.assert_array_equal(second[1], np.ones((2, 2, 3), dtype=np.uint8))
    assert first[2:] == (0, 0.0)
    assert second[2:] == (1, 1 / 25.0)


def test_video_source_read_keeps_two_value_api(monkeypatch):
    monkeypatch.setattr("app.video_source.cv2.VideoCapture", FakeCapture)
    source = VideoSource("test.mp4").start()

    try:
        ok, frame = source.read(timeout=1)
    finally:
        source.stop()

    assert ok
    np.testing.assert_array_equal(frame, np.zeros((2, 2, 3), dtype=np.uint8))
