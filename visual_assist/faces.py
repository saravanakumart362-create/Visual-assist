"""Face recognition and registration.

Two small models from OpenCV's model zoo do the work, so nothing has to be
compiled on the device:

  YuNet  finds the faces in a picture, with five landmarks on each
         (the eyes, the nose, the corners of the mouth).
  SFace  turns one face, straightened by its landmarks, into 128 numbers:
         its "feature". Pictures of one person give features that point
         nearly the same way; `similarity` says how nearly, from -1 to 1.

Each registered person is stored as one or more features in
<faces dir>/faces.json. Only the features are kept, never the photos, and
nothing leaves the device.
"""

from __future__ import annotations

import json
import logging
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

MODEL = "sface_2021dec"  # which model the stored features came from
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
# Where each model file can be had, if it is not on the disk yet.
SOURCES = {
    "face_detection_yunet_2023mar.onnx": ZOO + "face_detection_yunet/",
    "face_recognition_sface_2021dec.onnx": ZOO + "face_recognition_sface/",
}
DETECTION_SCORE = 0.8  # how sure the detector must be that something is a face
MAX_FACES = 5  # faces looked at in one picture, the largest first
MAX_VIEWS = 8  # features kept for one person; the oldest is dropped after that
REGISTER_SCALE = 2.0  # a face being registered must be this much larger than `min_face_px`


@dataclass(frozen=True)
class Face:
    name: Optional[str]  # None = not a registered person
    confidence: float  # likeness to the nearest registered face; 0 when nobody is registered
    box: tuple  # (x1, y1, x2, y2) in pixels


def similarity(a, b) -> float:
    """How nearly two features point the same way: 1 = the same, 0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    size = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / size if size else 0.0


class People:
    """The registered people and their features, kept in a file.

    With no file they are kept in memory only, and forgotten at the end.
    """

    def __init__(self, file=None, least: float = 0.45):
        self._file = Path(file) if file else None
        self._least = least  # similarity below this is not a match
        self._known = {}  # name -> list of features
        if self._file and self._file.exists():
            try:
                saved = json.loads(self._file.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                log.warning("Ignoring unreadable face file %s (%s)", self._file, exc)
                return
            if isinstance(saved, dict) and saved.get("model") == MODEL:
                self._known = saved["people"]
            else:
                # Features from another model cannot be compared with these.
                log.warning("%s was made by a different face model; "
                            "its people must be registered again.", self._file)

    @property
    def names(self) -> list:
        return sorted(self._known)

    def nearest(self, feature):
        """(name, similarity) of the registered person a feature is most like.

        The name is None if nobody is like enough, or nobody is registered.
        """
        best_name, best = None, 0.0
        for name, features in self._known.items():
            for stored in features:
                score = similarity(feature, stored)
                if score > best:
                    best_name, best = name, score
        return (best_name if best >= self._least else None), best

    def add(self, name: str, feature) -> None:
        """Remember one more view of a person, and save."""
        views = self._known.setdefault(name, [])
        views.append([round(float(value), 5) for value in feature])
        del views[:-MAX_VIEWS]
        if self._file is None:
            return
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(json.dumps({"model": MODEL, "people": self._known}),
                              encoding="utf-8")


class FaceModels:
    """YuNet and SFace: the faces in a picture, each with its feature."""

    def __init__(self, detector, recogniser, min_face_px: int = 40):
        import cv2

        logging_ = getattr(getattr(cv2, "utils", None), "logging", None)
        level = logging_.getLogLevel() if logging_ else None
        if logging_:
            # Loading prints warnings that mean nothing to the user.
            logging_.setLogLevel(logging_.LOG_LEVEL_ERROR)
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(detector), "", (320, 320), DETECTION_SCORE, 0.3, 5000)
            self._recogniser = cv2.FaceRecognizerSF.create(str(recogniser), "")
        finally:
            if logging_:
                logging_.setLogLevel(level)
        self._min_px = min_face_px
        self._size = None

    def faces(self, frame, min_px: Optional[float] = None) -> list:
        """[(box, feature)] for the faces in a BGR picture, the largest first.

        Faces narrower than `min_px` are left out: too few pixels to tell
        one person from another.
        """
        height, width = frame.shape[:2]
        if self._size != (width, height):
            self._size = (width, height)
            self._detector.setInputSize(self._size)
        found = self._detector.detect(frame)[1]
        rows = [] if found is None else list(found)
        least = self._min_px if min_px is None else min_px
        rows = sorted((row for row in rows if row[2] >= least),
                      key=lambda row: -row[2] * row[3])[:MAX_FACES]
        faces = []
        for row in rows:
            x, y, w, h = (float(value) for value in row[:4])
            feature = self._recogniser.feature(self._recogniser.alignCrop(frame, row))
            faces.append(((x, y, x + w, y + h), feature.ravel().tolist()))
        return faces

    @property
    def min_face_px(self) -> int:
        return self._min_px


class FaceBook:
    def __init__(self, people: People, models):
        self._people = people
        self._models = models

    @property
    def names(self) -> list:
        return self._people.names

    def identify(self, frame) -> list:
        """Every face in the frame, named where it matches a registered person."""
        faces = []
        for box, feature in self._models.faces(frame):
            name, score = self._people.nearest(feature)
            faces.append(Face(name, round(score, 3), box))
        return faces

    def register(self, name: str, frame) -> bool:
        """Store the largest face in the frame under `name`.

        False if there is no face, or none close enough to the camera to
        make a good likeness. Registering someone again adds another view
        of them, which helps them be recognised from more angles.
        """
        close = self._models.min_face_px * REGISTER_SCALE
        faces = self._models.faces(frame, min_px=close)
        if not faces:
            return False
        self._people.add(name, faces[0][1])
        return True


def fetch(file: Path) -> bool:
    """Make sure a model file is on the disk, downloading it if need be."""
    file = Path(file)
    if file.exists():
        return True
    source = SOURCES.get(file.name)
    if source is None:
        log.warning("Face model %s not found.", file)
        return False
    log.info("Downloading %s (once only) ...", file.name)
    partial = file.with_suffix(file.suffix + ".part")
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(source + file.name, timeout=30) as response, \
                partial.open("wb") as out:
            while chunk := response.read(1 << 16):
                out.write(chunk)
        partial.replace(file)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        log.warning("Could not download %s (%s).", file.name, exc)
        return False
    return True


def open_faces(cfg, resolve=Path) -> Optional[FaceBook]:
    """The face recogniser described by a FacesConfig, or None if it cannot be had."""
    if not cfg.enabled:
        return None
    detector, recogniser = Path(resolve(cfg.detector)), Path(resolve(cfg.recogniser))
    if not (fetch(detector) and fetch(recogniser)):
        log.warning("Face recognition disabled: its model files are missing.")
        return None
    try:
        models = FaceModels(detector, recogniser, cfg.min_face_px)
    except Exception as exc:  # no OpenCV, or one too old for these models
        log.warning("Face recognition disabled (%s).", exc)
        return None
    return FaceBook(People(Path(resolve(cfg.dir)) / "faces.json", cfg.similarity), models)
