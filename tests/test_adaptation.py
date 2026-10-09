"""Feedback and adaptation: speed, pauses, crowds, familiarity, preferences."""

import pytest

from visual_assist.simulation import make
from visual_assist import config
from visual_assist.adaptation import (
    MotionTracker, UserProfile, is_urgent, verbosity, warn_cooldown, warn_distance,
)
from visual_assist.detector import Detection
from visual_assist.faces import Face

NAV = config.NavigationConfig()
CFG = config.AdaptationConfig()


def walk(tracker, start, speed, seconds, t0=0.0, tick=0.5):
    """Feed the tracker LiDAR readings of someone approaching a wall."""
    events, t = [], t0
    while t <= t0 + seconds:
        events.append(tracker.update(t, start - speed * (t - t0)))
        t += tick
    return t, events


# ---------------------------------------------------------- motion tracker

def test_walking_speed_is_read_from_the_shrinking_lidar_distance():
    tracker = MotionTracker(CFG)
    walk(tracker, start=6.0, speed=1.0, seconds=4)
    assert tracker.speed == pytest.approx(1.0, abs=0.1)


def test_a_jump_to_another_object_is_not_mistaken_for_speed():
    tracker = MotionTracker(CFG)
    tracker.update(0.0, 5.0)
    tracker.update(0.5, 1.0)  # the beam swung onto something near: not 8 m/s
    assert tracker.speed == 0.0


def test_a_pause_is_reported_once_after_standing_still():
    tracker = MotionTracker(CFG)
    t, _ = walk(tracker, 6.0, 1.0, seconds=3)
    events = [tracker.update(t + i, 3.0) for i in range(1, 12)]
    assert events.count(True) == 1
    assert events.index(True) + 1 >= CFG.pause_s  # only after standing long enough
    assert tracker.pauses(t + 12) == 1
    assert tracker.pauses(t + 12 + CFG.pause_window_s + 1) == 0  # and it is forgotten


def test_standing_still_from_the_start_is_not_a_pause():
    tracker = MotionTracker(CFG)
    assert not any(tracker.update(float(t), 2.0) for t in range(30))


def test_turning_counts_as_moving_and_hides_speed():
    tracker = MotionTracker(CFG)
    tracker.update(0.0, 4.0)
    tracker.update(1.0, 2.0, turn=25.0)  # turning: the distance change means nothing
    assert tracker.turning and tracker.speed == 0.0


def test_camera_alone_can_tell_moving_from_still():
    tracker = MotionTracker(CFG)
    tracker.update(0.0, None, change=0.2)
    events = [tracker.update(float(t), None, change=0.0) for t in range(1, 10)]
    assert events.count(True) == 1


# ---------------------------------------------------------------- policies

def test_warning_distance_grows_with_walking_speed_up_to_a_limit():
    profile = UserProfile()
    assert warn_distance(NAV, CFG, profile, 0.0) == 1.0
    assert warn_distance(NAV, CFG, profile, 1.0) == 2.5
    assert warn_distance(NAV, CFG, profile, 5.0) == CFG.max_warn_m
    profile.change_warning(earlier=True)
    assert warn_distance(NAV, CFG, profile, 0.0) == 1.25
    off = config.AdaptationConfig(enabled=False)
    assert warn_distance(NAV, off, UserProfile(), 1.0) == 1.0


def test_warnings_come_faster_when_walking_or_in_a_crowd():
    still = warn_cooldown(NAV, CFG, 0.0, crowded=False)
    assert warn_cooldown(NAV, CFG, 1.0, crowded=False) == still / 2
    assert warn_cooldown(NAV, CFG, 0.0, crowded=True) == still / 2


def test_urgent_means_reaching_the_obstacle_within_the_reaction_time():
    assert is_urgent(1.0, 1.0, CFG)  # one second away
    assert not is_urgent(2.5, 1.0, CFG)
    assert not is_urgent(0.5, 0.0, CFG)  # standing still is never urgent


def test_verbosity_follows_preference_then_hesitation_then_familiarity():
    profile, tracker = UserProfile(), MotionTracker(CFG)
    assert verbosity(profile, tracker, CFG, 0.0) == "normal"
    for _ in range(CFG.familiar_after):
        profile.note_guided("door")
    assert verbosity(profile, tracker, CFG, 0.0, "door") == "brief"
    assert verbosity(profile, tracker, CFG, 0.0, "table") == "normal"
    tracker._pauses.extend([1.0, 2.0])
    assert verbosity(profile, tracker, CFG, 3.0, "door") == "detailed"
    profile.verbosity = "brief"
    assert verbosity(profile, tracker, CFG, 3.0) == "brief"


# ----------------------------------------------------------------- profile

def test_profile_is_remembered_between_sessions(tmp_path):
    path = tmp_path / "data" / "profile.json"
    profile = UserProfile(path)
    profile.language = "ml"
    profile.note_guided("door")
    profile.change_rate(faster=True, default=160)
    assert profile.note_person("Ashna", when=1000.0) == (0, 0)
    assert profile.note_person("Ashna", when=1000.0 + 3 * 86400 + 5) == (1, 3)

    again = UserProfile(path)
    assert (again.language, again.guided("door"), again.speech_rate) == ("ml", 1, 185)
    assert again.meetings("Ashna") == 2


def test_a_damaged_profile_is_ignored(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("{not json", encoding="utf-8")
    assert UserProfile(path).language is None


# ------------------------------------------------- the assistant, adapting

def approach(assistant, world, clock, speed, seconds, tick=0.5):
    for _ in range(int(seconds / tick)):
        world.walk(speed * tick)
        clock.advance(tick)
        assistant.step()


def test_a_fast_walker_is_warned_from_further_away():
    def first_warning_distance(speed):
        assistant, world, speaker, clock = make({"table": (0, 6.0, 0.5)})
        while not speaker.said and world.y < 5.0:
            approach(assistant, world, clock, speed, seconds=0.5)
        return 5.5 - world.y  # distance to the table's near edge

    slow, fast = first_warning_distance(0.2), first_warning_distance(1.2)
    assert slow <= 1.4
    assert fast > slow + 0.8


def test_about_to_walk_into_something_gets_an_urgent_stop():
    assistant, world, speaker, clock = make({"table": (0, 4.0, 0.5)})
    approach(assistant, world, clock, speed=1.2, seconds=2.5)
    assert speaker.urgent and speaker.urgent[0].startswith("Stop, table ahead")


def test_standing_still_is_not_nagged_about_the_same_obstacle():
    assistant, world, speaker, clock = make({"chair": (0, 0.9, 0.3)})
    for _ in range(10):
        clock.advance(1)
        assistant.step()
    assert speaker.said == ["Chair detected at 0.6 meters."]
    world.walk(0.4)  # closer: that is news
    clock.advance(5)
    assistant.step()
    assert speaker.said[-1] == "Chair detected at 0.2 meters."


def test_a_hesitant_user_who_stops_is_told_what_is_ahead():
    things = {"door": (1.0, 9.0, 0.45), "chair": (-1.0, 9.0, 0.3)}
    # A far wall for the LiDAR to measure the approach against.
    assistant, world, speaker, clock = make(things, blocks=[(0, 7.5, 0.5)])

    def walk_then_stand():
        approach(assistant, world, clock, speed=0.8, seconds=2)
        for _ in range(8):
            clock.advance(1.0)
            assistant.step()

    walk_then_stand()
    assert speaker.said == []  # one pause is just a pause
    walk_then_stand()
    assert len(speaker.said) == 1
    assert speaker.said[0].startswith("You have stopped. I see a chair")
    assert "a door" in speaker.said[0]


def test_a_crowd_is_announced_and_strangers_are_not():
    class Stranger:
        def identify(self, frame):
            return [Face(None, 0.0, (0, 0, 0, 0))]

    class Crowd:
        labels = ["person"]

        def detect(self, frame):
            return [Detection("person", 0.9, (x, 100, x + 60, 400), 640, 480)
                    for x in (50, 290, 500)]

    assistant, world, speaker, clock = make(
        {"person": (0, 4.0, 0.25)}, faces=Stranger(),
        settings={"faces": {"every_n_frames": 1}})
    assistant.step()
    assert speaker.said == [
        "There is a person I do not know. Say register face to remember them."]

    assistant.detector = Crowd()  # three people in view
    speaker.said.clear()
    clock.advance(60)  # long enough that the stranger would be mentioned again
    assistant.step()
    assert speaker.said == ["This area is crowded. I will focus on obstacle warnings."]


def test_familiar_faces_come_first_and_get_shorter_greetings(tmp_path):
    class Faces:
        def identify(self, frame):
            return [Face("Newcomer", 0.9, (0, 0, 0, 0)), Face("Shafeek", 0.9, (0, 0, 0, 0))]

    profile = UserProfile(tmp_path / "profile.json")
    for day in range(4):
        profile.note_person("Shafeek", when=1000.0 + day)
    assistant, world, speaker, clock = make(
        {}, faces=Faces(), profile=profile, settings={"faces": {"every_n_frames": 1}})
    assistant.step()
    assert speaker.said[0].startswith("Shafeek")  # met most often: announced first
    assert speaker.said[1] == "Newcomer is in front of you."


def test_voice_preferences_change_behaviour_and_are_saved(tmp_path):
    profile = UserProfile(tmp_path / "profile.json")
    assistant, world, speaker, clock = make({"door": (2.0, 3.0, 0.45)}, profile=profile)
    for phrase in ("speak less", "speak faster", "warn me earlier", "speak in Hindi"):
        assistant.submit(phrase)
    assistant.step()
    assert speaker.said[:3] == ["I will keep it brief.", "Speaking faster.",
                                "I will warn you earlier."]
    assert speaker.said[3] == "अब से हिंदी में बात होगी।"
    assert (speaker.language, speaker.rate) == ("hi", 185)

    saved = UserProfile(tmp_path / "profile.json")
    assert (saved.verbosity, saved.language, saved.warn_offset_m) == ("brief", "hi", 0.25)

    speaker.said.clear()
    assistant.submit("where is the door")
    assistant.step()
    assert speaker.said[-1].startswith("दरवाज़ा, ")  # brief, and in Hindi
