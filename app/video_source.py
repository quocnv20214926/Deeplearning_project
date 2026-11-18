"""[Owner: TV4] Đọc frame từ file / webcam / RTSP bằng OpenCV trong thread riêng (queue)."""


import cv2
import threading
import queue


class VideoSource:

    def __init__(self, source):

        self.source = source

        # Webcam and RTSP are live sources.
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

        self.thread = None
        self.cap = None

        # Signals when VideoCapture initialization finishes.
        self.ready = threading.Event()

    def start(self, timeout=5):

        if self.thread is not None:
            raise RuntimeError("VideoSource has already been started")

        self.running = True

        self.thread = threading.Thread(
            target=self._read_frames,
            daemon=True
        )

        self.thread.start()

        # Wait until OpenCV finishes opening the source.
        if not self.ready.wait(timeout):
            self.running = False
            raise TimeoutError("Timed out while opening video source")

        if not self.opened:
            raise RuntimeError(
                self.error or "Cannot open video source"
            )

        return self

    def _read_frames(self):

        try:
            self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                self.error = "Cannot open video source"
                return

            self.opened = True

            # Initialization has finished successfully.
            self.ready.set()

            while self.running:

                ret, frame = self.cap.read()

                if not ret:

                    if not self.live_source:
                        self.ended = True
                    else:
                        self.error = "Cannot read frame from live source"

                    break

                if self.live_source:

                    # Keep only the newest frame.
                    if self.frames.full():
                        try:
                            self.frames.get_nowait()
                        except queue.Empty:
                            pass

                    try:
                        self.frames.put_nowait(frame)
                    except queue.Full:
                        pass

                else:

                    # Preserve file frame order.
                    while self.running:

                        try:
                            self.frames.put(frame, timeout=0.1)
                            break

                        except queue.Full:
                            continue

        except Exception as e:
            self.error = str(e)

        finally:

            if self.cap is not None:
                self.cap.release()

            self.running = False

            # Ensure start() never waits forever.
            self.ready.set()

    def read(self, timeout=1.0):

        try:
            frame = self.frames.get(timeout=timeout)
            return True, frame

        except queue.Empty:
            return False, None

    def is_opened(self):
        return self.opened

    def is_ended(self):
        return self.ended

    def stop(self, timeout=3):

        self.running = False

        if self.thread is None:
            return True

        self.thread.join(timeout=timeout)

        # False means the reader thread is still running.
        return not self.thread.is_alive()
