"""Tests for the hardware-free building blocks."""

from pathlib import Path

import pytest

from visual_assist import config
from visual_assist.commands import Command, parse
from visual_assist.currency import decide, denomination
from visual_assist.detector import Detection, MultiDetector, center_detection
from visual_assist.lidar import TFLuna, encode_frame, find_frame, parse_frame
from visual_assist.navigator import (
    Instruction, Sighting, after_move, best_sightings, direction_of, fuse_distance,
    pixel_bearing, position_of, steps_for, turn_degrees, turn_for, visual_distance,
)
from visual_assist.servo import Esp8266PanTilt
from visual_assist.speech import Speaker

LABELS = ["bottle", "chair", "cup", "door", "person", "sofa", "table"]
NAV = config.NavigationConfig()
ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ config

def test_config_overrides_and_keeps_defaults():
    cfg = config.from_dict({"navigation": {"arrive_m": 0.5}})
    assert cfg.navigation.arrive_m == 0.5
    assert cfg.navigation.step_length_m == 0.6


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="lidar.prot"):
        config.from_dict({"lidar": {"prot": "/dev/ttyS0"}})
    with pytest.raises(ValueError, match="path"):
        config.from_dict({"path": "x"})


def test_shipped_config_file_is_valid_and_paths_resolve_from_its_folder():
    cfg = config.load(ROOT / "config.yaml")
    assert cfg.path(cfg.detector.weights) == ROOT / "weights" / "yolov5nu.pt"
    assert cfg.path(ROOT / "x.pt") == ROOT / "x.pt"


def test_deadband_is_wide_enough_for_guidance_to_settle():
    # A "slight" turn is about 30 degrees; a narrower deadband would have the
    # user turned back and forth across the target for ever.
    assert NAV.center_deadband_deg >= abs(turn_degrees("slight_left")) / 2


# ---------------------------------------------------------------- commands

@pytest.mark.parametrize("text, expected", [
    ("Where is the door?", Command("find", "door")),
    ("Guide me to the table.", Command("guide", "table")),
    ("take me to the desk", Command("guide", "table")),
    ("find my mug", Command("find", "cup")),
    ("is there a chair", Command("find", "chair")),
    ("where is the kitchen", Command("find", "kitchen")),
    ("how far is the sofa", Command("find", "sofa")),
    ("how far", Command("distance")),
    ("what is around me", Command("scan")),
    ("what is in front of me", Command("scan")),
    ("who is in front of me", Command("who")),
    ("register face", Command("register")),
    ("remember this person as shafeek", Command("register", "Shafeek")),
    ("what note is this", Command("money")),
    ("stop", Command("stop")),
    ("help me find the door", Command("find", "door")),
    ("help", Command("help")),
    ("shut down", Command("exit")),
    ("say that again", Command("repeat")),
    ("speak in Malayalam", Command("language", "ml")),
    ("change language to hindi", Command("language", "hi")),
    ("tamil", Command("language", "ta")),
    ("speak less", Command("verbosity", "brief")),
    ("give me more detail", Command("verbosity", "detailed")),
    ("normal detail", Command("verbosity", "normal")),
    ("speak faster", Command("rate", "faster")),
    ("slow down", Command("rate", "slower")),
    ("warn me earlier", Command("warn", "earlier")),
    ("warn me later", Command("warn", "later")),
    ("nice weather today", None),
    ("", None),
])
def test_parse(text, expected):
    assert parse(text, LABELS) == expected


# ------------------------------------------------------------------- lidar

def frame(distance_cm, strength=500):
    body = bytes([0x59, 0x59, distance_cm & 0xFF, distance_cm >> 8,
                  strength & 0xFF, strength >> 8, 0, 0])
    return body + bytes([sum(body) & 0xFF])


def test_lidar_frame_gives_metres():
    assert parse_frame(frame(240)) == pytest.approx(2.4)


def test_lidar_rejects_weak_signal_and_out_of_range():
    assert parse_frame(frame(240, strength=50)) is None
    assert parse_frame(frame(5)) is None
    assert parse_frame(frame(1200)) is None


def test_lidar_finds_latest_frame_in_noisy_stream():
    stream = b"\x01\x59" + frame(100) + frame(130) + b"\x59\x59\x10"
    assert parse_frame(find_frame(stream)) == pytest.approx(1.3)


def test_lidar_ignores_bad_checksum():
    bad = bytearray(frame(100))
    bad[8] ^= 0xFF
    assert find_frame(bytes(bad)) is None
    assert find_frame(b"") is None


def test_encode_frame_is_parse_frame_in_reverse():
    assert encode_frame(2.4)[:4] == frame(240)[:4]
    for metres in (0.2, 2.26, 8.0):
        sent = encode_frame(metres)
        assert find_frame(sent) == sent and parse_frame(sent) == pytest.approx(metres)
    nothing = encode_frame(None)
    assert find_frame(nothing) == nothing and parse_frame(nothing) is None


class Line:
    """A serial line that answers as it is told to and keeps what was written."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.written = b""

    def reset_input_buffer(self):
        pass

    def read(self, count):
        return self.answers.pop(0)

    def readline(self):
        return self.answers.pop(0)

    def write(self, data):
        self.written += data


def test_lidar_driver_reads_a_frame_that_arrives_in_two_parts():
    whole = encode_frame(1.5)
    assert TFLuna("", connection=Line(whole)).read() == pytest.approx(1.5)
    assert TFLuna("", connection=Line(b"\x08" + whole[:8], whole[8:] + whole[:8])).read() == \
        pytest.approx(1.5)
    assert TFLuna("", connection=Line(b"", b"")).read() is None


def test_servo_driver_sends_the_lines_the_firmware_expects():
    line = Line(b"OK\n", b"OK\n", b"OK\n", b"ERR\n")
    head = Esp8266PanTilt(config.from_dict({"servo": {"settle_s": 0}}).servo, connection=line)
    head.look(-60)          # left of straight ahead is a larger servo angle
    head.look(400, tilt=20)  # more than the servo can turn
    assert line.written == b"PAN 150\nTILT 90\nPAN 0\nTILT 110\n"


# --------------------------------------------------------------- navigator

def test_position_thirds():
    assert [position_of(x, 640) for x in (50, 320, 600)] == ["left", "center", "right"]


def test_pixel_bearing():
    assert pixel_bearing(320, 640, 62.2) == pytest.approx(0.0)
    assert pixel_bearing(640, 640, 62.2) == pytest.approx(31.1)
    assert pixel_bearing(0, 640, 62.2) == pytest.approx(-31.1)


def test_directions_and_turns():
    assert [direction_of(b, 15) for b in (4, -20, 60, -170)] == \
        ["ahead", "slight_left", "right", "behind_left"]
    # A turn is worded "slightly" up to 60 degrees: midway between the 30 a
    # person turns for "slightly" and the 90 they turn otherwise.
    assert [turn_for(b, 15) for b in (4, -20, 55, 70, -170)] == \
        [None, "slight_left", "slight_right", "right", "around"]
    assert [turn_degrees(t) for t in (None, "slight_left", "right", "around")] == \
        [0, -30, 90, 180]


def test_steps():
    assert steps_for(1.9, 0.6, stop_short=0.7) == 2
    assert steps_for(0.1, 0.6) == 1


def test_visual_distance_judges_by_height_or_failing_that_width():
    def judge(label, box):
        return visual_distance(label, box, 640, 480, 62.2, 48.8)

    # A 2 m door that looks 212 px tall on a 480 px, 48.8 degree frame is 5 m away.
    assert judge("door", (300, 134, 340, 346)) == pytest.approx(5.0, abs=0.05)
    # Taller than the picture: its 0.9 m width, 318 px here, says 1.5 m.
    assert judge("door", (161, 0, 479, 480)) == pytest.approx(1.5, abs=0.05)
    assert judge("door", (0, 0, 640, 480)) is None  # fills the view: just "very near"
    assert judge("table", (100, 100, 300, 300)) is None  # tables come in any size


def test_fuse_distance_prefers_lidar_unless_it_hit_something_else():
    assert fuse_distance(2.4, None) == (2.4, False)
    assert fuse_distance(None, 3.0) == (3.0, True)
    assert fuse_distance(2.4, 2.9) == (2.4, False)  # they agree: LiDAR is exact
    assert fuse_distance(1.0, 5.0) == (5.0, True)  # beam stopped by something nearer
    assert fuse_distance(None, None) == (None, False)


def test_best_sighting_prefers_measured_then_central():
    seen = [
        Sighting("chair", -20, None, offset=25),
        Sighting("chair", -22, 1.5, offset=28),
        Sighting("table", 10, None, offset=10),
        Sighting("table", 12, None, offset=3),
        Sighting("door", 0, 3.0, approx=True),
        Sighting("door", 1, 3.2, offset=9),
    ]
    best = {s.label: s for s in best_sightings(seen)}
    assert best["chair"].distance == 1.5
    assert best["table"].offset == 3
    assert best["door"].distance == 3.2  # a LiDAR reading beats a camera estimate


def test_after_move_predicts_where_the_target_will_be():
    # Target 3 m ahead; after 2 steps of 0.6 m it is 1.8 m ahead.
    bearing, distance = after_move(0, 3.0, Instruction("move", steps=2), 0.6)
    assert (bearing, distance) == (pytest.approx(0), pytest.approx(1.8))
    # Told to turn right: a target that was 60 degrees right ends up 30 left.
    bearing, _ = after_move(60, 3.0, Instruction("turn", turn="right"), 0.6)
    assert bearing == pytest.approx(-30)


# ---------------------------------------------------------------- currency

def test_denomination():
    assert [denomination(n) for n in ("100_new", "20", "none")] == [100, 20, None]


def test_decide_adds_old_and_new_series_of_the_same_note():
    assert decide({"100_new": 0.5, "100_old": 0.3, "500": 0.2}) == (100, pytest.approx(0.8))
    assert decide({"none": 0.7, "10_new": 0.3})[0] is None


# ---------------------------------------------------------------- detector

class _Fixed:
    def __init__(self, *labels):
        self.labels = list(labels)

    def detect(self, frame):
        return [Detection(label, 0.9, (0, 0, 10, 10), 640, 480) for label in self.labels]


def test_multi_detector_pools_models():
    both = MultiDetector([_Fixed("chair", "table"), _Fixed("door")])
    assert both.labels == ["chair", "door", "table"]
    assert [d.label for d in both.detect(None)] == ["chair", "table", "door"]


def test_center_detection_is_the_one_under_the_lidar():
    left = Detection("chair", 0.9, (0, 0, 100, 100), 640)
    wide = Detection("table", 0.9, (100, 0, 600, 100), 640)
    middle = Detection("cup", 0.9, (300, 0, 340, 100), 640)
    assert center_detection([left]) is None
    assert center_detection([left, wide, middle]) is middle


# ------------------------------------------------------------------ speech

def test_speaker_without_audio_still_tracks_what_was_said(capsys):
    speaker = Speaker(config.SpeechConfig(backend="console"))
    speaker.say("first")
    speaker.say("വാതിൽ", urgent=True)  # non-Latin text must not break printing
    speaker.close()
    assert speaker.last == "വാതിൽ"
    assert speaker.voices == []
    assert "[TTS]: first" in capsys.readouterr().out
