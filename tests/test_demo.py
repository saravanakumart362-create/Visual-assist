"""The demo page: its recordings are current and its Python bundle is complete."""

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from visual_assist.phrases import available
from visual_assist.simulation import ROOMS, record

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"


def test_a_recording_tells_the_whole_story():
    things, blocks = ROOMS["chair"]
    run = record(things, blocks)
    events = run["events"]

    assert run["reached"] and run["closest"] > 0.3
    assert events[0] == {"type": "say", "who": "user", "text": "Guide me to the door"}
    assert {event["type"] for event in events} == {
        "say", "look", "photo", "ray", "route", "turn", "walk"}
    assert json.loads(json.dumps(run)) == run  # plain data, ready for the page

    said = [event["text"] for event in events if event["type"] == "say"]
    assert said[1] == "Scanning." and said[-1].startswith("You have reached the door.")

    # Replaying the turns and walks ends beside the door, as the page will draw it.
    x, y, _ = things["door"]
    last_walk = [event["to"] for event in events if event["type"] == "walk"][-1]
    assert ((last_walk[0] - x) ** 2 + (last_walk[1] - y) ** 2) ** 0.5 < 1.5
    # Each planned route starts where the user is standing at that moment.
    position = [0.0, 0.0]
    for event in events:
        if event["type"] == "route":
            assert event["points"][0] == pytest.approx(position, abs=0.01)
        elif event["type"] == "walk":
            position = event["to"]


@pytest.mark.parametrize("language", available())
def test_every_language_can_be_recorded(language):
    things, blocks = ROOMS["opening"]
    run = record(things, blocks, language)
    assert run["reached"]
    assert sum(event["type"] == "say" for event in run["events"]) >= 5


def test_shipped_recordings_match_the_current_code():
    spec = importlib.util.spec_from_file_location("make_recordings", DEMO / "make_recordings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shipped = json.loads((DEMO / "recordings.json").read_text(encoding="utf-8"))
    current = json.loads(json.dumps(module.build()))
    assert shipped == current, "run: python demo/make_recordings.py"


def test_the_page_loads_every_file_the_guidance_needs(tmp_path):
    """The page fetches a fixed list of project files to run in the browser.

    Copy exactly that list somewhere empty and run a guidance trip there with
    nothing but the standard library, as the browser's Python has.
    """
    script = (DEMO / "demo.js").read_text(encoding="utf-8")
    listed = re.search(r"const FILES = \[(.*?)\];", script, re.S).group(1)
    files = re.findall(r'"([^"]+)"', listed)
    assert "simulation.py" in files and "locales/en.json" in files

    for name in files:
        target = tmp_path / "visual_assist" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "visual_assist" / name, target)
    check = (
        "import json, sys; sys.path.insert(0, sys.argv[1]);"
        "from visual_assist import simulation;"
        "from visual_assist.live import LiveAssistant;"
        "run = simulation.record({'door': [0.5, 4.5, 0.45]}, [[0.1, 2.0, 0.3]], 'ta');"
        "said = [];"
        "live = LiveAssistant(['door'], lambda text, language, urgent: said.append(text));"
        "live.hear('where is the door');"
        "live.see(json.dumps({'w': 640, 'h': 480, 'detections': [['door', 0.8, 270, 100, 370, 340]]}));"
        "print(json.dumps([run['reached'], said]))"
    )
    done = subprocess.run([sys.executable, "-I", "-S", "-c", check, str(tmp_path)],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [
        True, ["The door is straight ahead, about 4.5 meters away."]]


def test_the_browser_models_match_the_settings_the_device_uses():
    """demo/models is made from config.yaml by demo/export_models.py."""
    from visual_assist import config

    cfg = config.load(ROOT / "config.yaml")
    manifest = json.loads((DEMO / "models" / "models.json").read_text(encoding="utf-8"))
    main, doors = manifest["detectors"]
    advice = "run: python demo/export_models.py"

    assert (main["size"], main["confidence"]) == (cfg.detector.imgsz, cfg.detector.confidence), advice
    assert main["keep"] is None and "sofa" in main["names"] and "couch" not in main["names"], advice
    extra = cfg.detector.extra[0]
    assert (doors["size"], doors["confidence"]) == (extra["imgsz"], extra["confidence"]), advice
    assert [doors["names"][index] for index in doors["keep"]] == extra["classes"], advice
    assert "none" in manifest["currency"]["names"] and "500" in manifest["currency"]["names"]
    for spec in [main, doors, manifest["currency"]]:
        assert (DEMO / "models" / spec["file"]).stat().st_size > 1_000_000, spec["file"]


def test_git_does_not_ignore_the_files_the_page_needs():
    """An ignored file never reaches GitHub, and the page then fails there."""
    if not (ROOT / ".git").exists() or shutil.which("git") is None:
        pytest.skip("not a git checkout")
    needed = ["index.html", ".nojekyll", "demo/demo.js", "demo/live.js", "demo/demo.css",
              "demo/recordings.json", "demo/models/models.json", "demo/models/doors.onnx",
              "demo/models/yolov5nu.onnx", "demo/models/currency.onnx",
              "visual_assist/__init__.py", "visual_assist/live.py"]
    ignored = subprocess.run(["git", "check-ignore", *needed], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    assert ignored == []


def test_the_page_points_at_files_that_exist():
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    for reference in re.findall(r'(?:href|src)="((?:demo|visual_assist)/[^"]+)"', page):
        assert (ROOT / reference).is_file(), reference
    # GitHub Pages hides files starting with "_" (such as __init__.py)
    # unless this marker file is present.
    assert (ROOT / ".nojekyll").exists()
