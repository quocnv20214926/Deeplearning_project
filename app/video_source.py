"""[Owner: TV4] Đọc frame từ file / webcam / RTSP bằng OpenCV trong thread riêng (queue)."""

import math
import queue
import threading
import cv2


class VideoSource:
    def __init__(self, source):
        self.source = source
        if isinstance(source, int) or (
            isinstance(source, str) and source.startswith("rtsp://")
        ):
            self.live_source = True
            self.frames = queue.Queue(maxsize=1)
        else:
            self.live_source = False
            self.frames = queue.Queue(maxsize=32)

        self.running = False
        self.opened = False
        self.ended = False
        self.error = None
        self.fps = 30.0
        self.thread = None
        self.cap = None
        self.ready = threading.Event()

    def start(self, timeout=5):
        if self.thread is not None:
            raise RuntimeError("VideoSource has already been started")
        self.running = True
        self.thread = threading.Thread(
            target=self._read_frames, daemon=True
        )
        self.thread.start()
        if not self.ready.wait(timeout):
            self.running = False
            raise TimeoutError("Timed out while opening video source")
        if not self.opened:
            raise RuntimeError(self.error or "Cannot open video source")
        return self

    def _read_frames(self):
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                self.error = "Cannot open video source"
                return

            source_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            if math.isfinite(source_fps) and source_fps > 0:
                self.fps = source_fps
            self.opened = True
            self.ready.set()

            frame_index = 0
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    if not self.live_source:
                        self.ended = True
                    else:
                        self.error = "Cannot read frame from live source"
                    break

                packet = (frame, frame_index, frame_index / self.fps)
                frame_index += 1
                if self.live_source:
                    if self.frames.full():
                        try:
                            self.frames.get_nowait()
                        except queue.Empty:
                            pass
                    try:
                        self.frames.put_nowait(packet)
                    except queue.Full:
                        pass
                else:
                    while self.running:
                        try:
                            self.frames.put(packet, timeout=0.1)
                            break
                        except queue.Full:
                            continue
        except Exception as exc:
            self.error = str(exc)

        finally:
            if self.cap is not None:
                self.cap.release()
            self.running = False
            self.ready.set()

    def read(self, timeout=1.0):
        """Return a frame, preserving the original two-value reader API."""
        ok, frame, _, _ = self.read_with_metadata(timeout)
        return ok, frame

    def read_with_metadata(self, timeout=1.0):
        """Return (success, frame, source frame index, timestamp in seconds)."""
        try:
            packet = self.frames.get(timeout=timeout)
        except queue.Empty:
            return False, None, None, None

        if isinstance(packet, tuple) and len(packet) == 3:
            return True, packet[0], packet[1], packet[2]
        return True, packet, None, None

    def is_opened(self):
        return self.opened

    def is_ended(self):
        return self.ended

    def stop(self, timeout=3):
        self.running = False
        if self.thread is None:
            return True
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()
