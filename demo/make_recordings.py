"""Write demo/recordings.json for the demo page.

    python demo/make_recordings.py

The file holds the preset rooms and one English guidance run through each,
recorded from the real guidance code. The page plays these at once, before
(or without) loading Python in the browser. Run this again after changing
the guidance code; the tests fail while the file is out of date.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from visual_assist.simulation import ROOMS, record  # noqa: E402

OUTPUT = HERE / "recordings.json"


def build() -> dict:
    return {
        name: {"things": things, "blocks": blocks, "run": record(things, blocks, "en")}
        for name, (things, blocks) in ROOMS.items()
    }


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build(), separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size // 1024} KB)")
