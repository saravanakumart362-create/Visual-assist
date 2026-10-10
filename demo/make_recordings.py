"""Write the recordings the demo page plays, and stamp the page.

    python demo/make_recordings.py

recordings.json  the preset rooms and one English guidance run through
                 each; the page plays these at once, before (or without)
                 loading Python in the browser.
device.json      one command followed through the device cable by cable,
                 for the "Inside the device" section.
index.html       each link to demo/*.css and demo/*.js gets "?v=" and a
                 code made from that file's contents. A browser that kept
                 an old copy of a file then sees a new address and fetches
                 the new one, instead of showing a new page in old styles.

The recordings are made by the real code. Run this again after changing
that code, the style sheet or a script; the tests fail while anything
here is out of date.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE.parent / "index.html"
sys.path.insert(0, str(HERE.parent))

from visual_assist.simulation import ROOMS, SCENES, record, trace  # noqa: E402


def build() -> dict:
    return {
        name: {"things": things, "blocks": blocks, "run": record(things, blocks, "en")}
        for name, (things, blocks) in ROOMS.items()
    }


def build_device() -> dict:
    return {name: trace(name) for name in SCENES}


OUTPUTS = {"recordings.json": build, "device.json": build_device}


def stamped(page: str) -> str:
    """The page with every demo style sheet and script link stamped afresh."""
    def stamp(link) -> str:
        # Line endings differ between Windows and GitHub; the code must not.
        text = (HERE.parent / link["file"]).read_text(encoding="utf-8").replace("\r\n", "\n")
        code = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f'{link["file"]}?v={code}"'

    return re.sub(r'(?P<file>demo/[\w.-]+\.(?:css|js))(?:\?v=\w*)?"', stamp, page)


if __name__ == "__main__":
    for name, make in OUTPUTS.items():
        path = HERE / name
        path.write_text(json.dumps(make(), separators=(",", ":")), encoding="utf-8")
        print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
    with PAGE.open(encoding="utf-8", newline="") as file:  # keep its line endings
        before = file.read()
    after = stamped(before)
    if after != before:
        with PAGE.open("w", encoding="utf-8", newline="") as file:
            file.write(after)
    print(f"{PAGE}: links {'stamped' if after != before else 'already up to date'}")
