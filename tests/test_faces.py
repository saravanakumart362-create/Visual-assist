"""Face registration and recognition: the matching, the file, and the assistant's part.

The two OpenCV models are stood in for here, so these tests need neither the
model files nor a photo of anyone. The last test loads the real models if
they are on the disk.
"""

import json
import logging
import math
from pathlib import Path

import pytest

from visual_assist import config
from visual_assist.faces import (
    MAX_VIEWS, MODEL, REGISTER_SCALE, SOURCES, FaceBook, People, fetch, open_faces, similarity,
)
from visual_assist.simulation import make

ROOT = Path(__file__).resolve().parent.parent


def feature(angle_deg: float) -> list:
    """A made-up feature: a direction in a plane, so likeness is easy to reckon."""
    angle = math.radians(angle_deg)
    return [math.cos(angle), math.sin(angle)] + [0.0] * 6


class Models:
    """Stands in for YuNet and SFace: "pictures" are lists of (width, angle)."""

    min_face_px = 40

    def faces(self, frame, min_px=None):
        least = self.min_face_px if min_px is None else min_px
        wide = sorted((face for face in frame if face[0] >= least), reverse=True)
        return [((0, 0, width, width), feature(angle)) for width, angle in wide]


# ---------------------------------------------------------------- matching

def test_similarity_is_the_cosine_of_the_angle_between_features():
    assert similarity(feature(10), feature(10)) == pytest.approx(1.0)
    assert similarity(feature(0), feature(60)) == pytest.approx(0.5)
    assert similarity(feature(0), feature(90)) == pytest.approx(0.0)
    assert similarity([3.0, 0.0], [0.5, 0.0]) == pytest.approx(1.0)  # length does not matter
    assert similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_nearest_names_only_a_close_enough_match(tmp_path):
    people = People(tmp_path / "faces.json", least=0.45)
    assert people.nearest(feature(0)) == (None, 0.0)  # nobody registered yet

    people.add("Ashna", feature(0))
    people.add("Shafeek", feature(90))
    assert people.names == ["Ashna", "Shafeek"]
    name, score = people.nearest(feature(20))
    assert name == "Ashna" and score == pytest.approx(math.cos(math.radians(20)), abs=1e-4)
    assert people.nearest(feature(80))[0] == "Shafeek"
    # Half way between the two, like neither enough: 0.38 is under 0.45.
    name, score = people.nearest(feature(-67.5))
    assert name is None and score == pytest.approx(math.cos(math.radians(67.5)), abs=1e-4)


def test_more_views_of_a_person_widen_the_match(tmp_path):
    people = People(tmp_path / "faces.json", least=0.9)
    people.add("Ashna", feature(0))
    assert people.nearest(feature(50))[0] is None
    people.add("Ashna", feature(45))  # registered again, from another side
    assert people.nearest(feature(50))[0] == "Ashna"

    for index in range(MAX_VIEWS + 3):
        people.add("Ashna", feature(index))
    saved = json.loads((tmp_path / "faces.json").read_text(encoding="utf-8"))
    assert len(saved["people"]["Ashna"]) == MAX_VIEWS  # the oldest views are dropped


def test_registered_people_are_remembered_between_sessions(tmp_path):
    file = tmp_path / "deep" / "faces.json"  # the folder is made when needed
    People(file).add("Ashna George", feature(30))
    saved = json.loads(file.read_text(encoding="utf-8"))
    assert saved["model"] == MODEL and list(saved["people"]) == ["Ashna George"]
    assert People(file).nearest(feature(31))[0] == "Ashna George"


def test_a_file_from_another_model_or_a_damaged_one_is_ignored(tmp_path, caplog):
    old = tmp_path / "faces.json"
    old.write_text(json.dumps({"Ashna": [[0.1] * 128]}), encoding="utf-8")  # the old layout
    with caplog.at_level(logging.WARNING):
        assert People(old).names == []
    assert "registered again" in caplog.text

    old.write_text("{ not json", encoding="utf-8")
    assert People(old).names == []


# ------------------------------------------------------------ the face book

def test_identify_names_known_faces_and_leaves_the_rest_unnamed(tmp_path):
    people = People(tmp_path / "faces.json", least=0.45)
    people.add("Ashna", feature(0))
    book = FaceBook(people, Models())
    faces = book.identify([(120, 5), (90, 100), (20, 0)])  # the last is too far away
    assert [(face.name, face.box[2]) for face in faces] == [("Ashna", 120), (None, 90)]
    assert faces[0].confidence == pytest.approx(0.996, abs=0.001)
    assert book.identify([]) == []


def test_register_takes_the_largest_face_and_wants_it_close(tmp_path):
    book = FaceBook(People(tmp_path / "faces.json"), Models())
    close = Models.min_face_px * REGISTER_SCALE
    assert not book.register("Ashna", [])                    # no face
    assert not book.register("Ashna", [(close - 1, 0)])      # too far away for a likeness
    assert book.names == []
    assert book.register("Ashna", [(close, 0), (close + 60, 90)])
    assert book.names == ["Ashna"]
    # It was the larger face, the one at 90 degrees, that was stored.
    assert [face.name for face in book.identify([(200, 90), (200, 0)])] == ["Ashna", None]


def test_open_faces_does_without_when_switched_off_or_a_model_is_missing(tmp_path, caplog):
    cfg = config.from_dict({"faces": {"enabled": False}})
    assert open_faces(cfg.faces) is None

    # A file name the model zoo does not have is never downloaded.
    missing = tmp_path / "my_own_detector.onnx"
    assert missing.name not in SOURCES and not fetch(missing)
    cfg = config.from_dict({"faces": {"detector": str(missing), "dir": str(tmp_path)}})
    with caplog.at_level(logging.WARNING):
        assert open_faces(cfg.faces) is None
    assert "model files are missing" in caplog.text


# ------------------------------------------------- the assistant's commands

class Shy:
    """A face book that sees no face in its first few pictures."""

    def __init__(self, blank: int):
        self.blank = blank
        self.names = []

    def identify(self, frame):
        return []

    def register(self, name, frame):
        self.blank -= 1
        if self.blank >= 0:
            return False
        self.names.append(name)
        return True


def test_registering_tries_a_few_pictures_before_giving_up():
    faces = Shy(blank=3)
    assistant, _, speaker, _ = make({}, faces=faces, settings={"faces": {"every_n_frames": 99}})
    assistant.submit("remember this person as ashna")
    assistant.step()
    assert speaker.said[-1] == "Ashna has been registered." and faces.names == ["Ashna"]

    faces = Shy(blank=99)
    assistant, _, speaker, _ = make({}, faces=faces, settings={"faces": {"every_n_frames": 99}})
    assistant.submit("remember this person as ashna")
    assistant.step()
    assert speaker.said[-1] == "I could not see a face. Please try again."
    assert faces.blank == 99 - 5  # five pictures were tried


def test_without_face_recognition_the_assistant_says_so():
    assistant, _, speaker, _ = make({}, settings={"faces": {"every_n_frames": 99}})
    assistant.submit("who is this")
    assistant.submit("register face")
    assistant.step()
    assert speaker.said == ["Face recognition is not available."] * 2


# -------------------------------------------------------- the real models

def test_the_real_models_load_and_find_no_face_in_a_blank_picture(tmp_path):
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    cfg = config.load(ROOT / "config.yaml")
    if not (cfg.path(cfg.faces.detector).exists() and cfg.path(cfg.faces.recogniser).exists()):
        pytest.skip("the face models have not been downloaded")
    if not hasattr(cv2, "FaceRecognizerSF"):
        pytest.skip("this OpenCV has no face recogniser")
    cfg.faces.dir = str(tmp_path)
    book = open_faces(cfg.faces, cfg.path)
    assert book is not None and book.names == []
    blank = numpy.full((cfg.camera.height, cfg.camera.width, 3), 90, numpy.uint8)
    assert book.identify(blank) == []
    assert not book.register("Nobody", blank)
    assert not (tmp_path / "faces.json").exists()  # nothing is written for no face
