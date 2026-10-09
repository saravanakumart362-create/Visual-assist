"""The assistant end to end, in a simulated room (see visual_assist/simulation.py)."""

import math

from visual_assist.simulation import FixedPanTilt, follow_guidance, make, wall
from visual_assist.adaptation import UserProfile
from visual_assist.faces import Face


def at(bearing: float, distance: float, radius: float):
    """A disc whose near edge is `distance` away in direction `bearing`."""
    centre = distance + radius
    angle = math.radians(bearing)
    return centre * math.sin(angle), centre * math.cos(angle), radius


def distance_to(world, label: str) -> float:
    x, y, radius = world.things[label]
    return math.hypot(x - world.x, y - world.y) - radius


# ------------------------------------------------------------ find and scan

def test_find_scans_then_ranges_the_object():
    assistant, world, speaker, _ = make({"door": at(50, 2.4, 0.45), "table": at(0, 3.0, 0.5)})
    assistant.submit("Where is the door?")
    assistant.step()
    assert speaker.said == ["Scanning.", "The door is to your right, 2.4 meters away."]
    assert world.pan == 0.0  # head returned to centre


def test_find_reports_missing_and_untrained_objects():
    assistant, _, speaker, _ = make({"table": at(0, 3.0, 0.5)})
    assistant.submit("where is the chair")
    assistant.submit("where is the kitchen")
    assistant.submit("find")
    assistant.step()
    assert "I could not find the chair." in speaker.said
    assert "I have not been trained to recognise kitchen." in speaker.said
    assert speaker.said[-1] == "Which object should I look for?"


def test_the_most_confident_sighting_wins_when_an_object_is_reported_twice():
    from visual_assist.assistant import Assistant
    from visual_assist.navigator import Sighting

    shaky = Sighting("door", -40.0, 1.5, confidence=0.42)  # perhaps a cupboard
    sure = Sighting("door", 30.0, None, confidence=0.88, offset=20.0)
    sure_again = Sighting("door", 32.0, 3.1, confidence=0.80, offset=2.0)  # next view of it
    picked = Assistant._pick([shaky, sure, sure_again], "door")
    assert picked is sure_again  # the confident place, in its best-measured view
    assert Assistant._pick([shaky], "chair") is None


def test_scan_describes_the_room_from_left_to_right():
    assistant, _, speaker, _ = make({"chair": at(-60, 1.5, 0.3), "table": at(0, 3.0, 0.5)})
    assistant.submit("what is around me")
    assistant.step()
    assert speaker.said[-1] == \
        "I see a chair to your left at 1.5 meters, a table straight ahead at 3 meters."


def test_distance_to_whatever_is_ahead():
    assistant, _, speaker, _ = make({"table": at(0, 2.0, 0.5)})
    assistant.submit("how far")
    assistant.step()
    assert speaker.said == ["Table straight ahead, 2 meters away."]

    assistant, _, speaker, _ = make({}, blocks=[at(0, 1.5, 0.5)])
    assistant.submit("how far")
    assistant.step()
    assert speaker.said[0] == "The nearest object straight ahead is 1.5 meters away."

    assistant, _, speaker, _ = make({})
    assistant.submit("how far")
    assistant.step()
    assert speaker.said == ["I cannot measure the distance right now."]


def test_camera_estimates_distance_when_the_lidar_cannot():
    assistant, _, speaker, _ = make({"door": at(-40, 3.2, 0.45)}, pantilt=FixedPanTilt)
    assistant.submit("where is the door")
    assistant.step()
    assert speaker.said == ["I could not find the door."]  # a fixed head cannot look there
    assistant, _, speaker, _ = make({"door": at(-20, 3.2, 0.45)}, pantilt=FixedPanTilt)
    assistant.submit("where is the door")
    assistant.step()
    assert speaker.said == ["The door is slightly to your left, about 3 meters away."]


# ---------------------------------------------------------------- guidance

def test_guidance_on_open_floor_turns_then_walks_then_arrives():
    assistant, world, speaker, clock = make({"table": at(20, 2.5, 0.5)})
    assistant.submit("guide me to the table")
    follow_guidance(assistant, world, clock)
    assert speaker.said[:3] == [
        "Scanning.",
        "The table is slightly to your right, 2.5 meters away.",
        "Turn slightly right.",
    ]
    assert speaker.said[3].startswith("Move forward ")
    assert speaker.said[-1].startswith("You have reached the table.")
    assert 0.25 < distance_to(world, "table") <= 0.9
    assert assistant.profile.guided("table") == 1


def test_guidance_goes_around_an_obstacle_without_touching_it():
    assistant, world, speaker, clock = make(
        {"door": at(0, 5.0, 0.45), "chair": at(0, 2.2, 0.25)})
    assistant.submit("take me to the door")
    cycles = follow_guidance(assistant, world, clock)
    assert "Chair in the way. Turn slightly" in " ".join(speaker.said)
    assert speaker.said[-1].startswith("You have reached the door.")
    assert 0.25 < distance_to(world, "door") <= 0.9  # within reach, not walked into
    assert world.closest_obstacle > 0.25  # kept clear of the chair
    assert cycles < 12


def test_guidance_finds_the_opening_in_a_wall():
    opening = [disc for disc in wall(-3.0, 3.0, 2.5) if not 0.5 < disc[0] < 1.9]
    assistant, world, speaker, clock = make({"door": (1.2, 5.0, 0.45)}, blocks=opening)
    assistant.submit("guide me to the door")
    follow_guidance(assistant, world, clock)
    assert speaker.said[-1].startswith("You have reached the door.")
    assert world.closest_obstacle > 0.25  # passed through the middle of the gap
    assert world.y > 3.0  # really went through


def test_guidance_gives_up_when_there_is_no_way_through():
    ring = [(0.9 * math.sin(a / 10), 0.9 * math.cos(a / 10), 0.15) for a in range(0, 63, 2)]
    assistant, world, speaker, clock = make({"door": at(0, 4.0, 0.45)}, blocks=ring)
    assistant.submit("guide me to the door")
    follow_guidance(assistant, world, clock)
    assert speaker.said[1] == "The door is straight ahead, about 4 meters away."
    assert speaker.said[-1] == "I cannot find a clear way to the door. Guidance stopped."
    assert world.closest > 0.15


def test_guidance_stops_when_the_target_disappears():
    assistant, world, speaker, clock = make({"door": at(0, 4.0, 0.45)})
    assistant.submit("guide me to the door")
    assistant.step()
    del world.things["door"]
    follow_guidance(assistant, world, clock)
    assert "Looking for the door." in speaker.said
    assert speaker.said[-1] == "I lost sight of the door. Guidance stopped."


def test_stop_ends_guidance():
    assistant, world, speaker, clock = make({"door": at(0, 4.0, 0.45)})
    assistant.submit("guide me to the door")
    assistant.step()
    assistant.submit("stop")
    assistant.step()
    assert speaker.said[-1] == "Guidance stopped." and assistant._guide is None


def test_already_at_the_target():
    assistant, _, speaker, _ = make({"table": at(0, 0.5, 0.5)})
    assistant.submit("guide me to the table")
    assistant.step()
    assert speaker.said[-1] == "You are already at the table."


def test_a_familiar_destination_gets_brief_guidance():
    profile = UserProfile()
    for _ in range(3):
        profile.note_guided("table")
    assistant, world, speaker, clock = make({"table": at(20, 2.5, 0.5)}, profile=profile)
    assistant.submit("guide me to the table")
    assistant.step()
    assert speaker.said[1:] == ["Table, slightly to your right, 2.5 meters.",
                                "Slightly right."]


def test_guidance_without_a_turning_head_steers_by_camera():
    assistant, world, speaker, clock = make({"chair": at(20, 2.5, 0.25)}, pantilt=FixedPanTilt)
    assistant.submit("guide me to the chair")
    follow_guidance(assistant, world, clock)
    # The LiDAR cannot be aimed at it, so the distance is the camera's estimate.
    assert speaker.said[:2] == ["The chair is slightly to your right, about 2.5 meters away.",
                                "Turn slightly right."]
    assert speaker.said[-1].startswith("You have reached the chair.")
    assert 0.2 < distance_to(world, "chair") <= 0.9

    assistant, world, speaker, clock = make({"door": at(0, 4.0, 0.45)}, pantilt=FixedPanTilt)
    assistant.submit("guide me to the door")
    assistant.step()
    del world.things["door"]
    for _ in range(3):
        assistant.step()
    assert speaker.said[-1] == "I lost sight of the door. Guidance stopped."


# ------------------------------------------------------- people and money

class FaceBook:
    def __init__(self, *names):
        self.names = list(names)
        self.registered = []

    def identify(self, frame):
        return [Face(name, 0.9 if name else 0.0, (0, 0, 0, 0)) for name in self.names]

    def register(self, name, frame):
        self.registered.append(name)
        return True


def test_who_and_register():
    faces = FaceBook("Shafeek", None)
    assistant, _, speaker, _ = make({}, faces=faces, settings={"faces": {"every_n_frames": 99}})
    assistant.submit("who is this")
    assistant.submit("remember this person as sooraj")
    assistant.step()
    assert speaker.said == ["I see Shafeek and one person I do not know.",
                            "Registering Sooraj. Please face the camera.",
                            "Sooraj has been registered."]
    assert faces.registered == ["Sooraj"]


def test_register_asks_for_a_name_when_none_was_given():
    faces = FaceBook()
    assistant, _, speaker, _ = make({}, faces=faces, settings={"faces": {"every_n_frames": 99}})
    assistant.submit("register face")
    assistant.submit("ashna george")
    assistant.step()
    assert speaker.said[0] == "What is the person's name?"
    assert faces.registered == ["Ashna George"]


def test_money():
    class Reader:
        def __init__(self, answer):
            self.answer = answer

        def read(self, frames):
            assert len(frames) == 3
            return self.answer

    no_note = "I do not see a currency note. Hold it closer to the camera."
    for answer, expected in [((500, 0.97), "This is a 500 rupee note."),
                             ((100, 0.65), "This looks like a 100 rupee note, but I am not "
                                           "sure. Please show it again."),
                             ((10, 0.35), no_note),  # too weak a match to be a note
                             ((None, 0.9), no_note)]:
        assistant, _, speaker, _ = make({}, currency=Reader(answer))
        assistant.submit("what note is this")
        assistant.step()
        assert speaker.said == [expected]

    assistant, _, speaker, _ = make({})
    assistant.submit("what note is this")
    assistant.step()
    assert speaker.said == ["Currency recognition is not set up."]


# ------------------------------------------------------------ conversation

def test_configured_language_is_spoken():
    assistant, _, speaker, _ = make({"door": at(0, 2.0, 0.45)},
                                    settings={"speech": {"language": "ml"}})
    assert speaker.language == "ml"
    assistant.submit("where is the door")
    assistant.step()
    assert speaker.said == ["ചുറ്റും നോക്കുന്നു.", "വാതിൽ നേരെ മുന്നിൽ ഉണ്ട്, 2 മീറ്റർ അകലെ."]


def test_repeat():
    assistant, _, speaker, _ = make({"table": at(0, 2.0, 0.5)})
    assistant.submit("repeat")
    assistant.submit("how far")
    assistant.submit("say that again")
    assistant.step()
    assert speaker.said == ["I have not said anything yet.",
                            "Table straight ahead, 2 meters away.",
                            "Table straight ahead, 2 meters away."]


def test_wake_word_filters_other_speech():
    assistant, _, speaker, _ = make({"table": at(0, 3.0, 0.5)},
                                    settings={"listener": {"wake_word": "assistant"}})
    assistant.submit("where is the table")
    assistant.submit("assistant where is the table")
    assistant.submit("assistant blah blah")
    assistant.step()
    assert speaker.said == ["Scanning.", "The table is straight ahead, 3 meters away.",
                            "Sorry, I did not understand. Say help for options."]


def test_exit_stops_the_loop():
    assistant, _, speaker, _ = make({})
    assistant.submit("shut down")
    assistant.step()
    assert assistant._running is False
    assert speaker.said == ["Shutting down."]
