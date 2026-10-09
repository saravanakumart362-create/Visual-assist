"""YOLOv5 object detection, from one model or several combined."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .navigator import EDGE_PX, position_of

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2) in pixels
    frame_width: int
    frame_height: int = 0  # 0 = unknown

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def position(self) -> str:
        """left / center / right third of the frame."""
        return position_of(self.cx, self.frame_width)

    @property
    def spans_center(self) -> bool:
        """True if the camera axis (where the LiDAR points) falls on the object."""
        return self.box[0] <= self.frame_width / 2.0 <= self.box[2]

    @property
    def fills_frame(self) -> bool:
        """True if the object runs off all four sides of the picture.

        Its size can then no longer be judged, but one thing is certain:
        the camera is right in front of it.
        """
        if not self.frame_height:
            return False
        x1, y1, x2, y2 = self.box
        return (x1 <= EDGE_PX and y1 <= EDGE_PX
                and x2 >= self.frame_width - EDGE_PX and y2 >= self.frame_height - EDGE_PX)


def center_detection(detections) -> Optional[Detection]:
    """The detection the LiDAR beam is most likely hitting, if any."""
    spanning = [d for d in detections if d.spans_center]
    if not spanning:
        return None
    return min(spanning, key=lambda d: abs(d.cx - d.frame_width / 2.0))


class Detector:
    def __init__(self, weights, confidence: float = 0.35, imgsz: int = 640,
                 aliases: Optional[dict] = None, classes: Optional[Iterable] = None):
        from ultralytics import YOLO

        self._model = YOLO(str(weights))
        self._confidence = confidence
        self._imgsz = imgsz
        aliases = aliases or {}
        self._names = {i: aliases.get(n, n) for i, n in self._model.names.items()}
        # `classes` limits the model to the listed names (after aliasing).
        self._keep = None
        if classes:
            self._keep = sorted(i for i, n in self._names.items() if n in set(classes))
            self._names = {i: self._names[i] for i in self._keep}
        self.labels = sorted(set(self._names.values()))

    def detect(self, frame) -> list:
        result = self._model.predict(
            frame, conf=self._confidence, imgsz=self._imgsz, classes=self._keep,
            verbose=False,
        )[0]
        height, width = frame.shape[:2]
        return [
            Detection(
                label=self._names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                box=tuple(float(v) for v in box.xyxy[0]),
                frame_width=width,
                frame_height=height,
            )
            for box in result.boxes
        ]


class MultiDetector:
    """Runs several models on each frame and pools what they find."""

    def __init__(self, detectors: list):
        self._detectors = detectors
        self.labels = sorted({label for d in detectors for label in d.labels})

    def detect(self, frame) -> list:
        return [det for detector in self._detectors for det in detector.detect(frame)]


def open_detector(cfg, resolve=Path):
    """Build the detector described by a DetectorConfig.

    `resolve` turns a path from the config into an absolute one. A missing
    extra model is skipped with a warning, so the rest still works.
    """
    detectors = [Detector(resolve(cfg.weights), cfg.confidence, cfg.imgsz, cfg.aliases)]
    for extra in cfg.extra or []:
        weights = resolve(extra["weights"])
        if not Path(weights).exists():
            log.warning("Extra model %s not found; its objects will not be detected.",
                        weights)
            continue
        detectors.append(Detector(
            weights, extra.get("confidence", cfg.confidence), extra.get("imgsz", cfg.imgsz),
            {**cfg.aliases, **extra.get("aliases", {})}, extra.get("classes"),
        ))
    return detectors[0] if len(detectors) == 1 else MultiDetector(detectors)
