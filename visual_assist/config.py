"""Configuration: defaults live here, config.yaml overrides them."""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CameraConfig:
    backend: str = "auto"  # auto | picamera2 | opencv
    index: int = 0  # webcam index for the opencv backend
    width: int = 640
    height: int = 480
    hfov_deg: float = 62.2  # field of view (Pi Camera Module V2)
    vfov_deg: float = 48.8


@dataclass
class DetectorConfig:
    weights: str = "weights/yolov5nu.pt"
    confidence: float = 0.35
    imgsz: int = 640
    # Model class name -> name used in speech and voice commands.
    aliases: dict = field(
        default_factory=lambda: {
            "couch": "sofa",
            "dining table": "table",
            "tv": "television",
            "potted plant": "plant",
            "cell phone": "phone",
        }
    )
    # More models whose detections are added to the main one's. Each entry:
    # weights, and optionally classes (keep only these), confidence, imgsz.
    extra: list = field(
        default_factory=lambda: [
            {"weights": "weights/doors.pt", "classes": ["door"], "confidence": 0.4,
             "imgsz": 512},
        ]
    )


@dataclass
class LidarConfig:
    backend: str = "auto"  # auto | tfluna | none
    port: str = "/dev/serial0"
    baudrate: int = 115200


@dataclass
class ServoConfig:
    backend: str = "auto"  # auto | esp8266 | gpio | none
    port: str = "/dev/ttyUSB0"  # ESP8266 USB serial port
    baudrate: int = 115200
    pan_pin: int = 12  # BCM pins, gpio backend only
    tilt_pin: int = 13
    pan_center: int = 90
    tilt_center: int = 90
    pan_invert: bool = False  # set true if "right" makes the head turn left
    scan_angles: list = field(default_factory=lambda: [-60, -30, 0, 30, 60])
    settle_s: float = 0.4  # wait after a move before using the camera


@dataclass
class SpeechConfig:
    backend: str = "auto"  # auto | offline | gtts | console
    language: str = "en"  # en | ml | hi | ta
    rate: int = 160  # words per minute (offline voice)


@dataclass
class ListenerConfig:
    backend: str = "auto"  # auto | vosk | google | keyboard | none
    vosk_model: str = "models/vosk-model-small-en-us-0.15"
    language: str = "en-IN"  # google backend
    wake_word: str = ""  # e.g. "assistant"; empty = every phrase is a command
    timeout_s: float = 6.0


@dataclass
class FacesConfig:
    enabled: bool = True
    dir: str = "data/faces"
    # OpenCV's face models; downloaded on first use if they are not there.
    detector: str = "weights/face_detection_yunet_2023mar.onnx"
    recogniser: str = "weights/face_recognition_sface_2021dec.onnx"
    # Least likeness (0-1) for a face to be given a registered name. Higher
    # is stricter: fewer wrong names, more "someone I do not know".
    similarity: float = 0.45
    min_face_px: int = 40  # narrower faces are too far away to tell apart
    every_n_frames: int = 3
    announce_cooldown_s: float = 20.0


@dataclass
class NavigationConfig:
    step_length_m: float = 0.6
    obstacle_warn_m: float = 1.0
    arrive_m: float = 0.6  # stop this far from the target (arm's reach)
    warn_cooldown_s: float = 4.0
    # Within this angle the target counts as "straight ahead". It must be at
    # least half of a "slight" turn (about 30 degrees) or guidance oscillates.
    center_deadband_deg: float = 15.0
    lost_frames: int = 3  # looks without the target before guidance stops


@dataclass
class PlannerConfig:
    enabled: bool = True
    sweep_half_angle_deg: float = 72.0  # LiDAR sweep to each side
    sweep_step_deg: float = 6.0
    sweep_settle_s: float = 0.05
    clearance_m: float = 0.35  # half the width a walking person needs
    resolution_m: float = 0.1  # map cell size
    max_range_m: float = 6.0  # LiDAR hits beyond this are ignored


@dataclass
class AdaptationConfig:
    enabled: bool = True
    profile: str = "data/profile.json"  # what has been learned about the user
    reaction_s: float = 1.5  # warn this many seconds of walking ahead
    max_warn_m: float = 3.0
    walk_speed_mps: float = 0.8  # starting guess, refined while walking
    pause_s: float = 6.0  # standing still this long counts as a pause
    pause_window_s: float = 120.0
    hesitant_pauses: int = 2  # pauses in the window before describing more
    describe_cooldown_s: float = 30.0
    crowded_people: int = 3  # people in view that make a place "crowded"
    familiar_after: int = 3  # guided trips before a target gets brief guidance


@dataclass
class CurrencyConfig:
    weights: Optional[str] = "weights/currency.pt"
    # Below this the answer is given as a guess and the note asked for again.
    # Set high on purpose: a confident mistake about money is the costly kind.
    confidence: float = 0.8
    minimum: float = 0.5  # a match weaker than this is not a note at all


@dataclass
class Config:
    # Folder that relative paths are measured from; load() sets it to the
    # folder holding the config file.
    root: str = "."
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    servo: ServoConfig = field(default_factory=ServoConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    listener: ListenerConfig = field(default_factory=ListenerConfig)
    faces: FacesConfig = field(default_factory=FacesConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    currency: CurrencyConfig = field(default_factory=CurrencyConfig)

    def path(self, relative) -> Path:
        """Absolute form of a path written in the config file."""
        path = Path(relative)
        return path if path.is_absolute() else Path(self.root) / path


def _merge(obj, data: dict, path: str = "") -> None:
    for key, value in (data or {}).items():
        if not hasattr(obj, key) or callable(getattr(obj, key)):
            raise ValueError(f"Unknown config key: {path}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            _merge(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)


def from_dict(data: Optional[dict]) -> Config:
    cfg = Config()
    _merge(cfg, data or {})
    return cfg


def load(path) -> Config:
    """Load a YAML config file; a missing file means all defaults."""
    path = Path(path)
    data = {}
    if path.exists():
        import yaml

        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    cfg = from_dict(data)
    if "root" not in data:
        cfg.root = str(path.resolve().parent)
    return cfg
