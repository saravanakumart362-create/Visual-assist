"""Face recognition and registration.

Each registered person is stored as one or more 128-number face encodings
in <faces dir>/faces.json. Only the encodings are kept, never the photos,
and nothing leaves the device.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Face:
    name: Optional[str]  # None = not a registered person
    confidence: float  # 1 - face distance; 0 for unknown faces
    box: tuple  # (top, right, bottom, left) in pixels


class FaceBook:
    def __init__(self, directory, tolerance: float = 0.5):
        import face_recognition
        import numpy

        self._fr = face_recognition
        self._np = numpy
        self._tolerance = tolerance
        self._file = Path(directory) / "faces.json"
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._known = {}
        if self._file.exists():
            self._known = json.loads(self._file.read_text(encoding="utf-8"))

    @property
    def names(self) -> list:
        return sorted(self._known)

    def _encode(self, frame):
        rgb = self._np.ascontiguousarray(frame[:, :, ::-1])
        boxes = self._fr.face_locations(rgb)
        return boxes, self._fr.face_encodings(rgb, boxes)

    def identify(self, frame) -> list:
        """Every face in the frame, named where it matches a registered person."""
        names, encodings = [], []
        for name, stored in self._known.items():
            names.extend([name] * len(stored))
            encodings.extend(stored)

        faces = []
        for box, encoding in zip(*self._encode(frame)):
            name, confidence = None, 0.0
            if encodings:
                distances = self._fr.face_distance(self._np.array(encodings), encoding)
                best = int(distances.argmin())
                if distances[best] <= self._tolerance:
                    name, confidence = names[best], 1.0 - float(distances[best])
            faces.append(Face(name, confidence, tuple(box)))
        return faces

    def register(self, name: str, frame) -> bool:
        """Store the largest face in the frame under `name`. False if no face."""
        boxes, encodings = self._encode(frame)
        if not encodings:
            return False
        # (top, right, bottom, left): the biggest face is the person being introduced.
        largest = max(
            range(len(boxes)),
            key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][1] - boxes[i][3]),
        )
        self._known.setdefault(name, []).append([float(v) for v in encodings[largest]])
        self._file.write_text(json.dumps(self._known), encoding="utf-8")
        return True


def open_faces(cfg, resolve=Path) -> Optional[FaceBook]:
    if not cfg.enabled:
        return None
    try:
        return FaceBook(resolve(cfg.dir), cfg.tolerance)
    except ImportError as exc:
        log.warning("Face recognition disabled (%s).", exc)
        return None
