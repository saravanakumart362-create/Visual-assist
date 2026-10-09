"""Feedback and adaptation: fitting the assistant to its user.

The device has no motion sensor, so the user's movement is read from what
it does have:

  walking speed   how fast the LiDAR distance straight ahead shrinks
  turning         how far the camera picture slides sideways between frames
  standing still  neither of the above changing

From that, and from a small profile kept between sessions, the assistant
adjusts how early it warns, how much it says, and whom it announces.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

MOVING_MPS = 0.15  # slower than this is standing still
MAX_WALK_MPS = 2.5  # a faster "approach" means the beam jumped to another object
TURNING_DEG = 8.0  # picture shift between frames that counts as turning
STILL_CHANGE = 0.03  # picture change (0-1) below which the view is static
SPEED_SMOOTHING = 0.5
VERBOSITY = ("brief", "normal", "detailed")
RATE_STEP = 25
RATE_RANGE = (100, 260)
WARN_STEP_M = 0.25
WARN_OFFSET_RANGE = (-0.5, 1.5)


class UserProfile:
    """What the assistant has learned about its user, saved between sessions."""

    def __init__(self, path: Optional[Path] = None, walk_speed: float = 0.8):
        self._path = Path(path) if path else None
        self.language: Optional[str] = None  # None = use the configured one
        self.verbosity: Optional[str] = None  # None = decide automatically
        self.speech_rate: Optional[int] = None
        self.warn_offset_m = 0.0  # added to the obstacle warning distance
        self.walk_speed = walk_speed  # typical walking speed, m/s
        self.targets = {}  # label -> times guided there
        self.people = {}  # name -> {"count": n, "last_seen": unix time}
        if self._path and self._path.exists():
            try:
                saved = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                log.warning("Ignoring unreadable profile %s (%s)", self._path, exc)
            else:
                for key, value in saved.items():
                    if key in self._fields():
                        setattr(self, key, value)

    def _fields(self) -> dict:
        return {key: value for key, value in vars(self).items() if not key.startswith("_")}

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._fields(), indent=2, ensure_ascii=False),
                              encoding="utf-8")

    def guided(self, label: str) -> int:
        """How many times the user has been guided to this kind of object."""
        return self.targets.get(label, 0)

    def note_guided(self, label: str) -> None:
        self.targets[label] = self.guided(label) + 1
        self.save()

    def note_person(self, name: str, when: Optional[float] = None):
        """Record meeting someone. Returns (earlier meetings, days since the last)."""
        when = time.time() if when is None else when
        record = self.people.get(name, {"count": 0, "last_seen": None})
        last = record["last_seen"]
        days = int((when - last) // 86400) if last else 0
        self.people[name] = {"count": record["count"] + 1, "last_seen": when}
        self.save()
        return record["count"], days

    def meetings(self, name: str) -> int:
        return self.people.get(name, {}).get("count", 0)

    def change_rate(self, faster: bool, default: int) -> int:
        rate = (self.speech_rate or default) + (RATE_STEP if faster else -RATE_STEP)
        self.speech_rate = max(RATE_RANGE[0], min(RATE_RANGE[1], rate))
        self.save()
        return self.speech_rate

    def change_warning(self, earlier: bool) -> None:
        offset = self.warn_offset_m + (WARN_STEP_M if earlier else -WARN_STEP_M)
        self.warn_offset_m = max(WARN_OFFSET_RANGE[0], min(WARN_OFFSET_RANGE[1], offset))
        self.save()


def frame_motion(previous, current, hfov_deg: float):
    """Compare two camera frames: (degrees turned, fraction of picture changed).

    Either value is None when it cannot be told, including when the frames
    are not images at all (as in tests without a camera).
    """
    try:
        import cv2
        import numpy as np

        def small(frame):
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.resize(grey, (160, 120)).astype(np.float32)

        a, b = small(previous), small(current)
        (dx, _dy), response = cv2.phaseCorrelate(a, b)
        change = float(np.mean(np.abs(a - b))) / 255.0
    except Exception:
        return None, None
    # A weak match means the two views share too little to measure a shift.
    turn = abs(dx) / a.shape[1] * hfov_deg if response > 0.15 else None
    return turn, change


class MotionTracker:
    """Follows whether the user is walking, turning or standing still."""

    def __init__(self, cfg):
        self._cfg = cfg
        self.speed = 0.0  # current walking speed towards what is ahead, m/s
        self.turning = False
        self._time: Optional[float] = None
        self._ahead: Optional[float] = None
        self._still_since: Optional[float] = None
        self._has_moved = False
        self._pauses = deque()

    def update(self, now: float, ahead: Optional[float], turn: Optional[float] = None,
               change: Optional[float] = None) -> bool:
        """Feed one observation; returns True at the moment a pause begins.

        `ahead` is the LiDAR distance straight ahead, `turn` and `change`
        come from frame_motion(). Any of them may be None.
        """
        moving = None  # unknown until something says otherwise
        self.turning = turn is not None and turn > TURNING_DEG
        elapsed = None if self._time is None else now - self._time

        usable = (ahead is not None and self._ahead is not None and not self.turning
                  and elapsed is not None and 0.05 <= elapsed <= 5.0)
        if usable:
            closing = (self._ahead - ahead) / elapsed
            if abs(closing) <= MAX_WALK_MPS:
                self.speed += SPEED_SMOOTHING * (max(closing, 0.0) - self.speed)
                moving = abs(closing) > MOVING_MPS
        if self.turning:
            moving = True
        elif moving is None and change is not None:
            moving = change > STILL_CHANGE
        self._time, self._ahead = now, ahead

        if moving is None:
            return False
        if moving:
            self._has_moved = True
            self._still_since = None
            return False
        self.speed *= SPEED_SMOOTHING
        if self._still_since is None:
            self._still_since = now
        if self._has_moved and now - self._still_since >= self._cfg.pause_s:
            # One pause per stop: wait for movement before counting another.
            self._has_moved = False
            self._pauses.append(now)
            return True
        return False

    def pauses(self, now: float) -> int:
        """Pauses within the recent window: a sign of hesitation."""
        while self._pauses and now - self._pauses[0] > self._cfg.pause_window_s:
            self._pauses.popleft()
        return len(self._pauses)


def warn_distance(nav, cfg, profile: UserProfile, speed: float) -> float:
    """How far ahead to warn: further when walking faster.

    The base distance (plus the user's own preference) is extended by the
    ground covered in the reaction time, so a quick walker hears about an
    obstacle with the same time to spare as a slow one.
    """
    distance = nav.obstacle_warn_m + profile.warn_offset_m
    if cfg.enabled:
        distance += speed * cfg.reaction_s
    return max(0.5, min(distance, cfg.max_warn_m))


def warn_cooldown(nav, cfg, speed: float, crowded: bool) -> float:
    """Minimum time between warnings: shorter when moving fast or in a crowd."""
    cooldown = nav.warn_cooldown_s
    if cfg.enabled:
        cooldown /= 1.0 + speed
        if crowded:
            cooldown /= 2.0
    return cooldown


def is_urgent(distance: float, speed: float, cfg) -> bool:
    """True when the user would reach the obstacle within the reaction time."""
    return cfg.enabled and speed > MOVING_MPS and distance / speed < cfg.reaction_s


def is_crowded(detections, cfg) -> bool:
    people = sum(1 for det in detections if det.label == "person")
    return cfg.enabled and people >= cfg.crowded_people


def verbosity(profile: UserProfile, tracker: MotionTracker, cfg, now: float,
              target: Optional[str] = None) -> str:
    """How much to say: brief, normal or detailed.

    The user's stated preference wins. Otherwise repeated pauses call for
    more description, and a destination walked to many times for less.
    """
    if profile.verbosity in VERBOSITY:
        return profile.verbosity
    if not cfg.enabled:
        return "normal"
    if tracker.pauses(now) >= cfg.hesitant_pauses:
        return "detailed"
    if target and profile.guided(target) >= cfg.familiar_after:
        return "brief"
    return "normal"
