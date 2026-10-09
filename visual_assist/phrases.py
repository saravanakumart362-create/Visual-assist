"""Everything the assistant says, in every language it speaks.

The sentences live in locales/<language>.json so they can be corrected or
translated without touching code. To add a language, copy en.json, translate
the right-hand sides and keep the {placeholders} as they are.

A phrase is normally one string. Two kinds of phrase have variants:
  {"one": ..., "other": ...}   chosen by a count (1 step / 3 steps)
  {"default": ..., "<form>": ...}   chosen by the grammatical form of the
      object named in the sentence (Hindi gender, Tamil person/thing). A
      label declares its form as {"text": ..., "form": ..., "obl": ...};
      "obl" is the form used before a postposition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from .navigator import Instruction, Sighting, best_sightings, direction_of

LOCALES = Path(__file__).resolve().parent / "locales"
DEFAULT = "en"
# Spoken language name -> code, for "speak in Hindi".
LANGUAGES = {"english": "en", "malayalam": "ml", "hindi": "hi", "tamil": "ta"}


def available() -> list:
    return sorted(path.stem for path in LOCALES.glob("*.json"))


def _load(language: str) -> dict:
    return json.loads((LOCALES / f"{language}.json").read_text(encoding="utf-8"))


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:]


def _number(value: float) -> str:
    return f"{value:g}"


class Phrases:
    def __init__(self, language: str = DEFAULT):
        self._english = _load(DEFAULT)
        self.language = language if language in available() else DEFAULT
        self._locale = self._english if self.language == DEFAULT else _load(self.language)

    # ---------------------------------------------------------- building blocks

    def _label(self, name: str) -> dict:
        entry = self._locale["labels"].get(name, name)
        if isinstance(entry, str):
            entry = {"text": entry}
        return {"text": entry["text"], "obl": entry.get("obl", entry["text"]),
                "form": entry.get("form", "default")}

    def label(self, name: str) -> str:
        """An object's name in the current language (English if unknown)."""
        return self._label(name)["text"]

    def t(self, key: str, n: Optional[int] = None, label: Optional[str] = None,
          **values) -> str:
        """The phrase `key` with its placeholders filled in."""
        template = self._locale["phrases"].get(key) or self._english["phrases"][key]
        form = "default"
        if label is not None:
            entry = self._label(label)
            form = entry["form"]
            article = ""
            if self.language == DEFAULT:
                article = "an" if entry["text"][:1].lower() in "aeiou" else "a"
            values.update(label=entry["text"], label_obl=entry["obl"], a=article)
        if isinstance(template, dict):
            if "other" in template:
                template = template["one" if n == 1 else "other"]
            else:
                template = template.get(form) or template["default"]
        if n is not None:
            values["n"] = n
        return template.format(**values).strip()

    def distance(self, metres: float, approx: bool = False) -> str:
        """'2.4 meters', or 'about 2.5 meters' for a camera estimate."""
        value = round(metres * 2) / 2 if approx else round(metres, 1)
        if approx:
            value = max(value, 0.5)
        text = self.t("meters", n=1 if value == 1 else 0, value=_number(value))
        return self.t("about", distance=text) if approx else text

    def direction(self, bearing: float, deadband: float) -> str:
        return self.t("dir." + direction_of(bearing, deadband))

    # ---------------------------------------------------------------- sentences

    def locate(self, sighting: Sighting, deadband: float, brief: bool = False) -> str:
        """'The door is to your right, 2.4 meters away.'"""
        key = "locate"
        values = {"direction": self.direction(sighting.bearing, deadband)}
        if sighting.distance is not None:
            key += "_dist"
            values["distance"] = self.distance(sighting.distance, sighting.approx)
        if brief:
            key += "_brief"
        return _capitalise(self.t(key, label=sighting.label, **values))

    def describe(self, sightings: Iterable[Sighting], deadband: float) -> str:
        """The surroundings, listed from left to right."""
        items = []
        for sighting in best_sightings(sightings):
            direction = self.direction(sighting.bearing, deadband)
            if sighting.distance is None:
                items.append(self.t("see_item", label=sighting.label, direction=direction))
            else:
                items.append(self.t(
                    "see_item_dist", label=sighting.label, direction=direction,
                    distance=self.distance(sighting.distance, sighting.approx)))
        if not items:
            return self.t("see_nothing")
        return self.t("see", items=self.t_raw("list_sep").join(items))

    def t_raw(self, key: str) -> str:
        """A phrase exactly as written, keeping spaces at its ends."""
        return self._locale["phrases"].get(key) or self._english["phrases"][key]

    def instruction(self, instruction: Instruction, brief: bool = False) -> str:
        """'Turn slightly left.' / 'Move forward 3 steps, then turn right.'"""
        prefix = "brief." if brief else ""
        if instruction.kind == "turn":
            text = self.t(f"{prefix}turn.{instruction.turn}")
        elif instruction.steps:
            text = self.t(f"{prefix}move", n=instruction.steps)
        else:
            text = self.t(f"{prefix}move_slowly")
        if instruction.kind == "move" and instruction.then and not brief:
            text = self.t("then", first=text, second=self.t(f"turn.{instruction.then}"))
        return _capitalise(self.t("sentence", text=text))

    def obstacle(self, label: str, metres: float, urgent: bool = False,
                 brief: bool = False) -> str:
        key = "obstacle_stop" if urgent else "obstacle_brief" if brief else "obstacle"
        return _capitalise(self.t(key, label=label, distance=self.distance(metres)))

    def people(self, names: list, unknown: int) -> str:
        """Who is in view: 'I see Shafeek and one person I do not know.'"""
        parts = list(names)
        if unknown:
            parts.append(self.t("unknown_people", n=unknown))
        if not parts:
            return self.t("see_no_one")
        joined = parts[0] if len(parts) == 1 else (
            self.t_raw("list_sep").join(parts[:-1]) + self.t_raw("and") + parts[-1])
        return self.t("who", names=joined)

    def face(self, name: str, familiar: bool = False, days: int = 0) -> str:
        """Announce a known person; mention how long it has been if a while."""
        if days >= 1:
            return f"{self.t('face_known', name=name)} {self.t('last_met', n=days)}"
        return self.t("face_known_brief" if familiar else "face_known", name=name)
