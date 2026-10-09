"""Maps transcribed speech to an intent the assistant can act on.

Commands are spoken in English whatever language the assistant answers in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from .phrases import LANGUAGES

# Words people say -> label the detector uses.
SYNONYMS = {
    "desk": "table",
    "mug": "cup",
    "couch": "sofa",
    "settee": "sofa",
    "seat": "chair",
    "doorway": "door",
    "man": "person",
    "woman": "person",
    "someone": "person",
}

EXIT_PHRASES = {"exit", "quit", "shut down", "shutdown", "power off", "goodbye"}


@dataclass(frozen=True)
class Command:
    # find | guide | scan | distance | who | register | money | stop | help | exit
    # | repeat | language | verbosity | rate | warn
    intent: str
    target: Optional[str] = None


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def _has(text: str, *phrases: str) -> bool:
    return any(re.search(rf"\b{re.escape(p)}\b", text) for p in phrases)


def _target(text: str, labels: Iterable[str]) -> Optional[str]:
    """Find the object the user named; known labels win over a raw guess."""
    names = {label: label for label in labels}
    names.update({word: label for word, label in SYNONYMS.items()})
    for name in sorted(names, key=len, reverse=True):
        if _has(text, name, name + "s"):
            return names[name]
    match = re.search(r"\b(?:the|a|an|my|to)\s+([a-z ]+)$", text)
    if match:
        return re.sub(r"^(?:the|a|an|my)\s+", "", match.group(1)).strip() or None
    return None


def _preference(text: str) -> Optional[Command]:
    """Commands that change how the assistant talks."""
    language = next((code for name, code in LANGUAGES.items() if _has(text, name)), None)
    if language and (_has(text, "speak", "talk", "language", "say", "switch", "change")
                     or text in LANGUAGES):
        return Command("language", language)
    if _has(text, "speak less", "talk less", "less detail", "be brief", "keep it short",
            "short answers"):
        return Command("verbosity", "brief")
    if _has(text, "more detail", "speak more", "talk more", "tell me more", "be detailed",
            "explain more"):
        return Command("verbosity", "detailed")
    if _has(text, "normal detail", "normal mode", "normal speech"):
        return Command("verbosity", "normal")
    if _has(text, "speak faster", "talk faster", "speed up", "too slow"):
        return Command("rate", "faster")
    if _has(text, "speak slower", "talk slower", "speak slowly", "slow down", "too fast"):
        return Command("rate", "slower")
    if _has(text, "warn me earlier", "warn earlier", "warn me sooner", "warn sooner"):
        return Command("warn", "earlier")
    if _has(text, "warn me later", "warn later", "fewer warnings", "less warning"):
        return Command("warn", "later")
    return None


def parse(text: str, labels: Iterable[str] = ()) -> Optional[Command]:
    """Return the Command for a spoken phrase, or None if it is not one."""
    text = _normalise(text or "")
    if not text:
        return None
    if text in EXIT_PHRASES:
        return Command("exit")
    preference = _preference(text)
    if preference:
        return preference
    if _has(text, "stop", "cancel"):
        return Command("stop")
    if _has(text, "repeat", "say that again", "say again", "pardon", "come again"):
        return Command("repeat")

    if _has(text, "register", "remember", "save this"):
        match = re.search(r"\b(?:as|named|called|name is)\s+([a-z ]+)$", text)
        return Command("register", match.group(1).strip().title() if match else None)
    if _has(text, "who"):
        return Command("who")
    if _has(text, "money", "note", "rupee", "rupees", "currency", "cash"):
        return Command("money")

    if _has(text, "guide me", "take me", "lead me", "navigate", "go to", "bring me"):
        return Command("guide", _target(text, labels))
    if _has(text, "where", "find", "locate", "is there", "look for", "search for"):
        return Command("find", _target(text, labels))
    if _has(text, "how far", "distance"):
        target = _target(text, labels)
        return Command("find", target) if target in set(labels) else Command("distance")
    if _has(
        text, "scan", "around", "describe", "surroundings", "in front",
        "what do you see", "what can you see", "what is here",
    ):
        return Command("scan")
    if _has(text, "help", "what can you do"):
        return Command("help")
    return None
