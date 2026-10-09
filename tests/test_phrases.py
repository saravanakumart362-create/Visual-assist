"""The spoken sentences: correct in English, complete in every language."""

import re

import pytest

from visual_assist.navigator import Instruction, Sighting
from visual_assist.phrases import LANGUAGES, Phrases, _load, available

ENGLISH = _load("en")
OTHERS = [language for language in available() if language != "en"]
EN = Phrases("en")


def placeholders(template) -> set:
    """Placeholders a phrase uses, whichever variant is picked."""
    texts = template.values() if isinstance(template, dict) else [template]
    names = {name for text in texts for name in re.findall(r"\{(\w+)\}", text)}
    # {label_obl} is the same slot as {label}; {a} is the English article.
    return {"label" if name == "label_obl" else name for name in names} - {"a"}


def test_every_spoken_language_has_a_locale_file():
    assert set(LANGUAGES.values()) == set(available())


@pytest.mark.parametrize("language", OTHERS)
def test_locale_is_complete_and_uses_the_same_placeholders(language):
    locale = _load(language)
    assert set(locale["phrases"]) == set(ENGLISH["phrases"])
    for key, english in ENGLISH["phrases"].items():
        assert placeholders(locale["phrases"][key]) == placeholders(english), key
        if isinstance(english, dict) and "other" in english:
            assert set(locale["phrases"][key]) == {"one", "other"}, key
        elif isinstance(locale["phrases"][key], dict):
            assert "default" in locale["phrases"][key], key


@pytest.mark.parametrize("language", OTHERS)
def test_locales_translate_the_same_objects(language):
    assert set(_load(language)["labels"]) == set(_load(OTHERS[0])["labels"])
    assert {"door", "chair", "table", "sofa", "bottle", "cup", "person", "obstacle"} \
        <= set(_load(language)["labels"])


def test_unknown_language_falls_back_to_english():
    assert Phrases("xx").language == "en"
    assert Phrases("ml").label("spaceship") == "spaceship"


def test_distances():
    assert EN.distance(0.42) == "0.4 meters"
    assert EN.distance(1.04) == "1 meter"
    assert EN.distance(3.0) == "3 meters"
    assert EN.distance(2.3, approx=True) == "about 2.5 meters"
    assert EN.distance(0.1, approx=True) == "about 0.5 meters"


def test_locate():
    assert EN.locate(Sighting("door", 50, 2.44), 15) == \
        "The door is to your right, 2.4 meters away."
    assert EN.locate(Sighting("door", 0), 15) == "The door is straight ahead."
    assert EN.locate(Sighting("door", -30, 2.2, approx=True), 15) == \
        "The door is slightly to your left, about 2 meters away."
    assert EN.locate(Sighting("door", 50, 2.44), 15, brief=True) == \
        "Door, to your right, 2.4 meters."


def test_describe_lists_left_to_right_with_articles():
    seen = [Sighting("table", 0, 1.2), Sighting("chair", -60), Sighting("umbrella", 100)]
    assert EN.describe(seen, 15) == ("I see a chair to your left, a table straight ahead "
                                     "at 1.2 meters, an umbrella to your right.")
    assert EN.describe([], 15) == "I do not see any objects I know."


def test_instructions():
    assert EN.instruction(Instruction("turn", turn="slight_left")) == "Turn slightly left."
    assert EN.instruction(Instruction("move", steps=1)) == "Move forward 1 step."
    assert EN.instruction(Instruction("move", steps=3, then="right")) == \
        "Move forward 3 steps, then turn right."
    assert EN.instruction(Instruction("move")) == "Move forward slowly."
    assert EN.instruction(Instruction("move", steps=3, then="right"), brief=True) == "3 steps."
    assert EN.instruction(Instruction("turn", turn="around"), brief=True) == "Turn around."


def test_obstacles():
    assert EN.obstacle("bottle", 0.4) == "Bottle detected at 0.4 meters."
    assert EN.obstacle("chair", 1.0, brief=True) == "Chair, 1 meter."
    assert EN.obstacle("obstacle", 0.6, urgent=True) == "Stop, obstacle ahead, 0.6 meters."


def test_people_and_faces():
    assert EN.people([], 0) == "I do not see anyone."
    assert EN.people(["Shafeek"], 0) == "I see Shafeek."
    assert EN.people(["Shafeek", "Sooraj"], 2) == \
        "I see Shafeek, Sooraj and 2 people I do not know."
    assert EN.face("Ashna") == "Ashna is in front of you."
    assert EN.face("Ashna", familiar=True) == "Ashna."
    assert EN.face("Ashna", days=1) == "Ashna is in front of you. You last met yesterday."
    assert EN.face("Ashna", days=4) == "Ashna is in front of you. You last met 4 days ago."


def test_grammatical_forms_are_chosen_by_the_object():
    hindi, tamil = Phrases("hi"), Phrases("ta")
    # Hindi verbs agree with the object's gender; postpositions change its ending.
    assert hindi.t("not_found", label="door") == "दरवाज़ा नहीं मिला।"
    assert hindi.t("not_found", label="chair") == "कुर्सी नहीं मिली।"
    assert hindi.t("reached", label="door") == "आप दरवाज़े के पास पहुँच गए हैं।"
    # Tamil uses a different verb for people than for things.
    assert tamil.t("locate", label="door", direction="x").endswith("உள்ளது.")
    assert tamil.t("locate", label="person", direction="x").endswith("இருக்கிறார்.")


@pytest.mark.parametrize("language", available())
def test_every_sentence_can_be_built_in_every_language(language):
    phrases = Phrases(language)
    values = dict(label="door", word="kitchen", name="Ashna", value=500, n=2,
                  distance="2 m", direction="x", items="x", names="x", text="x",
                  first="x", second="y")
    for key in ENGLISH["phrases"]:
        assert phrases.t(key, **values) or key in ("list_sep", "and")
    for sighting in (Sighting("door", 50, 2.4), Sighting("person", -20, 1.0, approx=True),
                     Sighting("kettle", 170)):
        for brief in (False, True):
            assert phrases.locate(sighting, 15, brief)
    assert phrases.describe([Sighting("chair", 0, 1.0), Sighting("person", 60)], 15)
    for instruction in (Instruction("turn", turn="around"), Instruction("move", steps=1),
                        Instruction("move", steps=4, then="slight_left"),
                        Instruction("move")):
        for brief in (False, True):
            assert phrases.instruction(instruction, brief)
    assert phrases.obstacle("obstacle", 0.5, urgent=True)
    assert phrases.people(["A", "B"], 1) and phrases.face("A", days=3)
