"""The assistant on a live camera inside a web page.

A browser has no LiDAR, no turning head and no PyTorch. The demo page's
JavaScript therefore runs the converted models on the visitor's camera and
hands the results to this module, which presents them to the Assistant as
the camera, detector and banknote reader it expects. What the assistant
then decides and says is the same code that runs on the device.

Nothing here touches a browser API, so it is tested like everything else.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

from . import commands, config
from .adaptation import UserProfile
from .assistant import Assistant
from .currency import decide
from .detector import Detection
from .lidar import NullLidar
from .servo import NullPanTilt


class LiveFrame:
    """One camera picture, reduced to what the assistant needs of it."""

    def __init__(self, width: int, height: int, thumbnail=None):
        self.width = width
        self.height = height
        self.thumbnail = list(thumbnail or [])  # a tiny grey copy, 0-255

    def motion_since(self, previous):
        """(degrees turned, share of the picture that changed) since `previous`.

        Only the change is measured; it is enough to tell holding still
        from moving. See adaptation.frame_motion.
        """
        earlier = getattr(previous, "thumbnail", None)
        if not earlier or len(earlier) != len(self.thumbnail):
            return None, None
        change = sum(abs(a - b) for a, b in zip(self.thumbnail, earlier))
        return None, change / (255.0 * len(self.thumbnail))


class LiveCamera:
    def __init__(self):
        self.frame: Optional[LiveFrame] = None  # None while the camera is off

    def read(self):
        return self.frame

    def close(self) -> None:
        self.frame = None


class LiveDetector:
    def __init__(self, labels):
        self.labels = sorted(set(labels))
        self.latest = []  # what the page's models found in the current frame

    def detect(self, frame) -> list:
        return list(self.latest) if frame is not None else []


class LiveCurrency:
    def __init__(self):
        self.scores: Optional[dict] = None  # class name -> probability

    def read(self, frames):
        """(rupee value or None, confidence), from the scores the page supplied."""
        scores, self.scores = self.scores, None  # each reading is used once
        if not scores:
            return None, 0.0
        return decide(scores)


class LiveSpeaker:
    """Passes each sentence to the page, which shows it and speaks it."""

    def __init__(self, say: Callable, is_busy: Callable, language: str, rate: int):
        self._say = say
        self._is_busy = is_busy
        self.language = language
        self.rate = rate
        self.last = ""

    @property
    def busy(self) -> bool:
        return bool(self._is_busy())

    def say(self, text: str, wait: bool = False, urgent: bool = False) -> None:
        self.last = text
        self._say(text, self.language, bool(urgent))

    def configure(self, language: str = None, rate: int = None) -> None:
        self.language = language or self.language
        self.rate = rate or self.rate

    def wait(self) -> None:
        pass

    def close(self) -> None:
        pass


class LiveAssistant:
    """What the page talks to.

    `say(text, language, urgent)` is called for every sentence, and
    `is_busy()` must tell whether the page is still speaking. `labels` are
    the object names the page's models can report.
    """

    def __init__(self, labels, say: Callable, is_busy: Callable = lambda: False,
                 language: str = "en", settings: Optional[dict] = None, clock=None):
        self.cfg = config.from_dict(settings or {})
        self._fov = (self.cfg.camera.hfov_deg, self.cfg.camera.vfov_deg)
        self.camera = LiveCamera()
        self.detector = LiveDetector(labels)
        self.currency = LiveCurrency()
        self.speaker = LiveSpeaker(say, is_busy, language, self.cfg.speech.rate)
        self.cfg.speech.language = language
        extras = {"clock": clock} if clock else {}
        self.assistant = Assistant(
            self.cfg, self.camera, self.detector, NullLidar(), NullPanTilt(), self.speaker,
            currency=self.currency,
            profile=UserProfile(None, self.cfg.adaptation.walk_speed_mps), **extras)
        self.assistant._running = True

    def greet(self) -> None:
        self.speaker.say(self.assistant.phrases.t("ready"))

    def see(self, picture: str) -> None:
        """A camera picture has been analysed; let the assistant act on it.

        `picture` is JSON: {"w", "h", "detections": [[label, confidence,
        x1, y1, x2, y2], ...], "thumb": [grey values]}.
        """
        data = json.loads(picture)
        width, height = int(data["w"]), int(data["h"])
        # A phone held upright sees more top to bottom than side to side.
        wide, narrow = max(self._fov), min(self._fov)
        camera = self.cfg.camera
        camera.hfov_deg, camera.vfov_deg = (wide, narrow) if width >= height else (narrow, wide)
        self.camera.frame = LiveFrame(width, height, data.get("thumb"))
        self.detector.latest = [
            Detection(label, float(confidence), (float(x1), float(y1), float(x2), float(y2)),
                      width, height)
            for label, confidence, x1, y1, x2, y2 in data["detections"]
        ]
        self.assistant.step()

    def hear(self, text: str) -> None:
        """A question was typed or spoken; it is answered on the next picture."""
        self.assistant.submit(text)

    def intent(self, text: str) -> Optional[str]:
        """What kind of request this is ("money", "find", ...), or None."""
        command = commands.parse(text, self.detector.labels)
        return command.intent if command else None

    def note(self, scores: str) -> None:
        """The banknote model's answer for the picture: JSON {class: probability}."""
        self.currency.scores = json.loads(scores)

    def set_language(self, code: str) -> bool:
        return self.assistant.set_language(code)

    @property
    def language(self) -> str:
        return self.assistant.phrases.language

    @property
    def guiding(self) -> Optional[str]:
        """The object the user is being guided to, if any."""
        guide = self.assistant._guide
        return guide.target if guide else None

    def camera_off(self) -> None:
        self.camera.frame = None
        self.detector.latest = []
        self.assistant._guide = None
