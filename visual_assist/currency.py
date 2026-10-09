"""Recognising Indian banknotes held up to the camera.

The model is an image classifier trained by training/train.py on the
classes 10_new, 10_old, 20, 50_new, 50_old, 100_new, 100_old, 200, 500,
2000 and "none" (no note in view). Old and new series look different, so
they are separate classes; only the value in front of the underscore is
spoken.
"""

from __future__ import annotations

from typing import Optional

NO_NOTE = "none"


def denomination(class_name: str) -> Optional[int]:
    """The rupee value of a class name such as '100_new'; None for 'none'."""
    value = class_name.split("_")[0]
    return int(value) if value.isdigit() else None


def decide(scores: dict):
    """Pick the answer from {class name: probability}.

    Returns (value, confidence). Series of the same value are added up
    first: 55% "100_new" plus 35% "100_old" is a confident 100.
    """
    totals = {}
    for name, score in scores.items():
        value = denomination(name)
        totals[value] = totals.get(value, 0.0) + score
    value = max(totals, key=totals.get)
    return value, totals[value]


class CurrencyReader:
    def __init__(self, weights):
        from ultralytics import YOLO

        self._model = YOLO(str(weights), task="classify")

    def read(self, frames: list):
        """(rupee value or None, confidence) for a note shown in `frames`.

        Several frames are averaged so one blurred frame cannot decide it.
        """
        names = self._model.names
        scores = {name: 0.0 for name in names.values()}
        for frame in frames:
            probabilities = self._model.predict(frame, verbose=False)[0].probs.data.tolist()
            for index, probability in enumerate(probabilities):
                scores[names[index]] += probability / len(frames)
        return decide(scores)


def open_currency(cfg, resolve) -> Optional[CurrencyReader]:
    if not cfg.weights:
        return None
    weights = resolve(cfg.weights)
    if not weights.exists():
        return None
    return CurrencyReader(weights)
