"""The assistant on a live camera in the browser: fed by the page, not by hardware."""

import json
import math

import pytest

from visual_assist.live import LiveAssistant, LiveFrame
from visual_assist.simulation import Clock

LABELS = ["bottle", "chair", "cup", "door", "person", "table"]


class Page:
    """Stands in for the demo page: collects what it is told to say."""

    def __init__(self, language="en"):
        self.said = []
        self.clock = Clock()
        self.live = LiveAssistant(LABELS, self.say, language=language, clock=self.clock)

    def say(self, text, language, urgent):
        self.said.append((text, language))

    def show(self, *detections, width=640, height=480, thumb=None, faces=None):
        """One analysed camera picture: detections are (label, confidence, x1, y1, x2, y2).

        `faces`, if the page looked for any, are (width in pixels, angle):
        see face() below.
        """
        self.clock.advance(1.0)
        picture = {"w": width, "h": height, "thumb": thumb or [],
                   "detections": [list(d) for d in detections]}
        if faces is not None:
            picture["faces"] = [face(*found) for found in faces]
        self.live.see(json.dumps(picture))

    def ask(self, text, *detections, **frame):
        self.said.clear()
        self.live.hear(text)
        self.show(*detections, **frame)
        return [text for text, _ in self.said]


def face(width: float, angle_deg: float) -> list:
    """A face as the page reports it: its box, then its feature.

    The feature is made up: a direction in a plane, so that two faces are as
    alike as the cosine of the angle between them.
    """
    angle = math.radians(angle_deg)
    return [100, 100, 100 + width, 100 + width * 1.3, [math.cos(angle), math.sin(angle), 0.0, 0.0]]


# A person 300 px tall on a 480 px picture is about 2.9 m away (1.65 m tall).
PERSON = ("person", 0.9, 270, 90, 370, 390)
# A door 240 px tall is about 4.4 m away; this one is left of centre.
DOOR_LEFT = ("door", 0.8, 80, 100, 190, 340)


def test_describes_what_the_camera_shows():
    page = Page()
    assert page.ask("what is around me", PERSON, DOOR_LEFT) == [
        "I see a door slightly to your left at about 4.5 meters, "
        "a person straight ahead at about 3 meters."]
    assert page.ask("what is around me") == ["I do not see any objects I know."]


def test_finds_an_object_and_estimates_its_distance():
    page = Page()
    assert page.ask("where is the door", DOOR_LEFT) == [
        "The door is slightly to your left, about 4.5 meters away."]
    assert page.ask("where is the chair", DOOR_LEFT) == ["I could not find the chair."]
    assert page.ask("where is the kitchen", DOOR_LEFT) == [
        "I have not been trained to recognise kitchen."]
    assert page.ask("how far", PERSON) == ["Person straight ahead, about 3 meters away."]


def test_reports_what_it_made_of_a_question_and_of_the_picture():
    """The page shows these beside its diagram of the device."""
    page = Page()
    made = lambda text: json.loads(page.live.command(text))  # noqa: E731
    assert made("Where is the door?") == {"intent": "find", "target": "door"}
    assert made("what is around me") == {"intent": "scan", "target": None}
    assert made("what note is this")["intent"] == "money"
    assert made("lovely weather") is None

    page.ask("where is the door", PERSON, DOOR_LEFT, ("table", 0.6, 500, 300, 630, 440))
    view = json.loads(page.live.view())
    assert view["hfov"] == 62.2
    assert view["objects"] == [
        {"label": "person", "confidence": 0.9, "position": "center", "bearing": 0, "looks": 2.91},
        {"label": "door", "confidence": 0.8, "position": "left", "bearing": -19, "looks": 4.41},
        # Tables come in every size, so this one's says nothing of its distance.
        {"label": "table", "confidence": 0.6, "position": "right", "bearing": 25, "looks": None},
    ]
    # An upright phone sees a narrower angle from side to side.
    page.show(("door", 0.8, 400, 200, 478, 440), width=480, height=640)
    view = json.loads(page.live.view())
    assert view["hfov"] == 48.8 and view["objects"][0]["bearing"] == 21


def test_faces_found_by_the_page_are_registered_and_named():
    page = Page()
    # Until the page has loaded its face models there is no face recognition.
    assert page.ask("who is this", faces=[(200, 0)]) == ["Face recognition is not available."]

    page.live.use_faces(True)
    assert page.ask("who is this", faces=[(200, 0)]) == ["I see one person I do not know."]
    assert page.ask("remember this person as ashna", faces=[(200, 0)]) == [
        "Registering Ashna. Please face the camera.", "Ashna has been registered."]
    # Another picture of her (10 degrees off is still very like), beside a stranger.
    assert page.ask("who is this", faces=[(120, 90), (200, 10)]) == [
        "I see Ashna and one person I do not know."]
    people = json.loads(page.live.people())
    assert people["registered"] == ["Ashna"] and people["least"] == 0.45
    assert [(found["name"], found["likeness"]) for found in people["faces"]] == [
        ("Ashna", 0.985), (None, 0.0)]

    # A face too far away is left out; one a little nearer is not close enough to register.
    assert page.ask("who is this", faces=[(30, 0)]) == ["I do not see anyone."]
    assert page.ask("remember this person as shafeek", faces=[(60, 90)])[:2] == [
        "Registering Shafeek. Please face the camera.",
        "I could not see a face. Please try again."]
    # A picture the page did not look for faces in has none.
    assert page.ask("who is this") == ["I do not see anyone."]


def test_registered_people_are_announced_as_the_camera_meets_them():
    page = Page()
    page.live.use_faces(True)
    page.ask("remember this person as ashna", faces=[(200, 0)])
    page.said.clear()
    for _ in range(6):  # faces are looked at every third picture
        page.show(faces=[(200, 5)])
    assert page.said == []  # she has only just been registered: no need to say so
    page.clock.advance(30.0)
    for _ in range(6):
        page.show(faces=[(200, 5)])
    assert [text for text, _ in page.said] == ["Ashna is in front of you."]


def test_a_name_asked_for_is_taken_from_the_next_phrase():
    """The device waits for the answer; a page must not, or it would freeze."""
    page = Page()
    page.live.use_faces(True)
    assert json.loads(page.live.hear("register face")) == {"intent": "register", "target": None}
    assert page.said[-1][0] == "What is the person's name?"
    assert page.live.intent("ashna george") == "register"  # so the page looks for a face
    assert json.loads(page.live.hear("ashna george")) == {
        "intent": "register", "target": "Ashna George"}
    page.show(faces=[(200, 0)])
    assert page.said[-1][0] == "Ashna George has been registered."
    assert page.live.intent("ashna george") is None  # an ordinary phrase again

    page.live.hear("register face")
    page.live.hear("stop")
    assert page.said[-1][0] == "Registration cancelled."
    page.live.hear("register face")
    page.live.hear("42")  # not a name
    assert page.said[-1][0] == "Registration cancelled."

    # Without face models the question is not even asked.
    page.live.use_faces(False)
    assert page.ask("register face", faces=[(200, 0)]) == ["Face recognition is not available."]


def test_an_upright_phone_picture_uses_the_right_field_of_view():
    page = Page()
    # A phone held upright: the picture is taller than wide, so its wide
    # angle runs top to bottom. A door at the right-hand edge is then only
    # about 20 degrees off, where in a landscape picture it would be 30.
    page.ask("where is the door", ("door", 0.8, 400, 200, 478, 440), width=480, height=640)
    assert page.said[0][0] == "The door is slightly to your right, about 4.5 meters away."
    assert page.live.cfg.camera.vfov_deg > page.live.cfg.camera.hfov_deg
    page.ask("where is the door", DOOR_LEFT)  # and back to landscape
    assert page.live.cfg.camera.hfov_deg > page.live.cfg.camera.vfov_deg


def test_reads_a_banknote_from_the_scores_the_page_supplies():
    page = Page()
    assert page.live.intent("what note is this") == "money"
    assert page.live.intent("where is the door") == "find"
    assert page.live.intent("lovely weather") is None

    page.live.note(json.dumps({"500": 0.93, "100_new": 0.04, "none": 0.03}))
    assert page.ask("what note is this") == ["This is a 500 rupee note."]
    # The scores belonged to that picture: they are not used a second time.
    assert page.ask("what note is this") == [
        "I do not see a currency note. Hold it closer to the camera."]
    page.live.note(json.dumps({"none": 0.9, "10_new": 0.1}))
    assert page.ask("what note is this") == [
        "I do not see a currency note. Hold it closer to the camera."]


def test_guides_by_camera_until_arrival():
    page = Page()
    chair_right = ("chair", 0.9, 470, 200, 560, 380)  # 180 px tall: about 2.6 m
    said = page.ask("guide me to the chair", chair_right)
    assert said == ["The chair is slightly to your right, about 2.5 meters away.",
                    "Turn slightly right."]
    assert page.live.guiding == "chair"

    page.said.clear()
    page.show(("chair", 0.9, 275, 200, 365, 380))  # now straight ahead
    assert page.said[-1][0].startswith("Move forward ")

    page.said.clear()
    page.clock.advance(10)
    # Close up the chair is taller than the picture; its width (0.5 m across
    # 320 px) still says it is about 0.8 m away.
    page.show(("chair", 0.9, 160, 0, 480, 480))
    assert page.said[-1][0].startswith("You have reached the chair.")
    assert page.live.guiding is None


def test_a_target_filling_the_whole_picture_counts_as_reached():
    page = Page()
    page.ask("guide me to the door", ("door", 0.8, 270, 100, 370, 340))
    page.said.clear()
    page.clock.advance(10)
    page.show(("door", 0.8, 0, 0, 640, 480))  # nothing but door: no size to judge by
    assert page.said[-1][0].startswith("You have reached the door.")


def test_guidance_ends_when_the_target_leaves_the_picture():
    page = Page()
    page.ask("guide me to the door", ("door", 0.8, 270, 100, 370, 340))
    page.said.clear()
    for _ in range(3):
        page.show()
    assert page.said[-1][0] == "I lost sight of the door. Guidance stopped."


def test_answers_in_the_chosen_language():
    page = Page(language="hi")
    assert page.ask("where is the door", DOOR_LEFT) == ["दरवाज़ा थोड़ा बाईं ओर है, लगभग 4.5 मीटर दूर।"]
    assert page.said[0][1] == "hi"
    assert page.ask("speak in tamil") == ["இனி நான் தமிழில் பேசுகிறேன்."]
    assert page.live.language == "ta" and page.said[0][1] == "ta"
    assert page.live.set_language("xx") is False


def test_turning_the_camera_off_ends_guidance():
    page = Page()
    page.ask("guide me to the door", ("door", 0.8, 270, 100, 370, 340))
    page.live.camera_off()
    assert page.live.guiding is None
    assert page.live.camera.read() is None


def test_a_frame_measures_how_much_the_picture_changed():
    still = LiveFrame(640, 480, [100] * 12)
    same = LiveFrame(640, 480, [100] * 12)
    moved = LiveFrame(640, 480, [100] * 6 + [200] * 6)
    assert same.motion_since(still) == (None, 0.0)
    turn, change = moved.motion_since(still)
    assert turn is None and change == pytest.approx(50 / 255)
    assert moved.motion_since(None) == (None, None)
    assert moved.motion_since(LiveFrame(640, 480, [1, 2, 3])) == (None, None)
