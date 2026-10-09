"""Frame source: the Pi Camera on the Raspberry Pi, a webcam elsewhere."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class PiCamera:
    def __init__(self, cfg):
        from picamera2 import Picamera2

        self._cam = Picamera2()
        # picamera2's "RGB888" is BGR in memory, which is what OpenCV expects.
        self._cam.configure(
            self._cam.create_preview_configuration(
                main={"size": (cfg.width, cfg.height), "format": "RGB888"}
            )
        )
        self._cam.start()

    def read(self):
        return self._cam.capture_array()

    def close(self) -> None:
        self._cam.stop()
        self._cam.close()


class OpenCVCamera:
    def __init__(self, cfg):
        import cv2

        self._cap = cv2.VideoCapture(cfg.index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cfg.index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        """Newest frame as a BGR array, or None if the camera gave nothing."""
        # Inference is slow, so the driver's buffered frame is stale: skip it.
        self._cap.grab()
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()


def open_camera(cfg):
    if cfg.backend == "opencv":
        return OpenCVCamera(cfg)
    try:
        return PiCamera(cfg)
    except Exception as exc:
        if cfg.backend != "auto":
            raise
        log.info("Pi Camera not available (%s); using OpenCV camera %d.", exc, cfg.index)
        return OpenCVCamera(cfg)
