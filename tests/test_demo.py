"""The demo page: its recordings are current and its Python bundle is complete."""

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from visual_assist import config
from visual_assist.lidar import find_frame, parse_frame
from visual_assist.phrases import available
from visual_assist.simulation import ROOMS, SimulatedEsp8266, World, record, trace

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


def of_kind(steps, *kinds):
    return [step for step in steps if step["kind"] in kinds]


def test_a_trace_follows_a_command_through_every_cable():
    steps = trace("find")["steps"]
    assert [step["kind"] for step in steps] == [
        "hear", "understand", "speak", "turn", "see", "measure",
        "look", "look", "look", "look", "aim", "centre", "speak"]
    assert steps[1] == {"kind": "understand", "intent": "find", "target": "door"}

    # The scan: five views from 60 degrees left to 60 right, each acknowledged.
    views = of_kind(steps, "turn", "look")
    assert [view["sent"] for view in views] == [
        [f"PAN {angle}", "TILT 90"] for angle in (150, 120, 90, 60, 30)]
    assert [view["bearing"] for view in views] == [-60, -30, 0, 30, 60]
    assert all(view["replies"] == ["OK", "OK"] for view in views)
    assert views[0]["pulses_us"] == [2167, 1500] and views[2]["pulses_us"] == [1500, 1500]

    # Every frame shown means, to the real parser, the distance shown beside it.
    frames = of_kind(steps, "measure", "look", "aim")
    assert len(frames) == 6
    for step in frames:
        sent = bytes.fromhex(step["frame"])
        assert find_frame(sent) == sent
        assert parse_frame(sent) == (None if step["metres"] is None
                                     else pytest.approx(step["metres"], abs=0.006))

    # Aimed at the door, the LiDAR agrees with the camera, so it is believed.
    aim, = of_kind(steps, "aim")
    assert (aim["sent"][0], aim["target"], aim["metres"], aim["looks"]) == ("PAN 50", "door", 3.0, 3.0)
    assert aim["distance"] == 3.0 and not aim["approx"]
    assert steps[-1]["text"] == "The door is slightly to your right, 3 meters away."
    assert json.loads(json.dumps(steps)) == steps  # plain data, ready for the page


def test_a_trace_shows_why_the_route_goes_round_the_chair():
    steps = trace("guide")["steps"]
    # The beam aimed at the door stops at the chair; the camera's estimate stands.
    aim, = of_kind(steps, "aim")
    assert (aim["metres"], aim["looks"], aim["distance"], aim["approx"]) == (2.2, 5.0, 5.0, True)

    sweep, = of_kind(steps, "sweep")
    assert [reading["sent"] for reading in sweep["readings"]] == [
        f"PAN {angle}" for angle in range(162, 12, -6)]
    plan, = of_kind(steps, "plan")
    assert plan["rays"] == [[r["bearing"], r["metres"]] for r in sweep["readings"]]
    assert [ray for ray in plan["rays"] if ray[1] is not None] == [[0, 2.2]]
    assert plan["goal"] == [0, 5.0] and plan["blocker"] == "chair" and not plan["direct"]
    assert plan["legs"][0][0] < 0  # round the chair by the left
    assert [step["text"] for step in of_kind(steps, "speak")] == [
        "Scanning.", "The door is straight ahead, about 5 meters away.",
        "Chair in the way. Turn slightly left."]


def test_the_simulated_esp8266_does_what_the_firmware_does():
    world = World()
    world.events = []
    servo = config.from_dict({}).servo
    esp = SimulatedEsp8266(world, servo)

    def send(line):
        esp.write(line.encode("ascii") + b"\n")
        return esp.readline()

    assert send("PAN 0") == b"OK\n" and world.pan == 90   # hard right
    assert send("PAN 999") == b"OK\n" and world.pan == -90  # held to 180
    assert send("TILT 45") == b"OK\n" and world.pan == -90
    assert send("SPIN 3") == b"ERR\n" and send("PAN") == b"ERR\n"
    pulses = [event["pulse_us"] for event in world.events if event.get("link") == "pwm"]
    assert pulses == [500, 2500, 1000]


def make_recordings():
    spec = importlib.util.spec_from_file_location("make_recordings", DEMO / "make_recordings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["recordings.json", "device.json"])
def test_shipped_recordings_match_the_current_code(name):
    shipped = json.loads((DEMO / name).read_text(encoding="utf-8"))
    current = json.loads(json.dumps(make_recordings().OUTPUTS[name]()))
    assert shipped == current, "run: python demo/make_recordings.py"


def test_the_page_links_name_the_current_style_sheet_and_scripts():
    """A browser keeps old copies of files. A link that changes whenever its
    file does makes it fetch the new one, so a new page is never shown in
    old styles or run by old scripts."""
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    stamps = re.findall(r'"(demo/[\w.-]+\.(?:css|js))\?v=(\w{8})"', page)
    assert [file for file, _ in stamps] == [
        "demo/demo.css", "demo/demo.js", "demo/live.js", "demo/device.js"]
    assert make_recordings().stamped(page) == page, "run: python demo/make_recordings.py"

    # The code depends on the file's contents, but not on its line endings.
    stamp = make_recordings().stamped
    assert stamp('<link href="demo/demo.css">') == stamp('<link href="demo/demo.css?v=old">')
    assert stamp('href="README.md" src="demo/models/x.onnx"') == 'href="README.md" src="demo/models/x.onnx"'


def test_the_page_loads_every_file_the_guidance_needs(tmp_path):
    """The page fetches a fixed list of project files to run in the browser.

    Copy exactly that list somewhere empty and run a guidance trip there with
    nothing but the standard library, as the browser's Python has.
    """
    script = (DEMO / "demo.js").read_text(encoding="utf-8")
    listed = re.search(r"const FILES = \[(.*?)\];", script, re.S).group(1)
    files = re.findall(r'"([^"]+)"', listed)
    assert "simulation.py" in files and "locales/en.json" in files and "faces.py" in files

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

    # The two face models are the device's own files, with its settings.
    from visual_assist import faces

    found = manifest["faces"]
    assert (found["min_px"], found["score"], found["most"]) == (
        cfg.faces.min_face_px, faces.DETECTION_SCORE, faces.MAX_FACES), advice
    for part, setting in (("detector", cfg.faces.detector), ("recogniser", cfg.faces.recogniser)):
        assert found[part]["file"] == Path(setting).name, advice
        assert (DEMO / "models" / found[part]["file"]).stat().st_size > 100_000, advice


def test_git_does_not_ignore_the_files_the_page_needs():
    """An ignored file never reaches GitHub, and the page then fails there."""
    if not (ROOT / ".git").exists() or shutil.which("git") is None:
        pytest.skip("not a git checkout")
    needed = ["index.html", ".nojekyll", "demo/demo.js", "demo/live.js", "demo/device.js",
              "demo/demo.css", "demo/recordings.json", "demo/device.json",
              "demo/models/models.json", "demo/models/doors.onnx",
              "demo/models/yolov5nu.onnx", "demo/models/currency.onnx",
              "demo/models/face_detection_yunet_2023mar.onnx",
              "demo/models/face_recognition_sface_2021dec.onnx",
              "visual_assist/__init__.py", "visual_assist/live.py", "visual_assist/faces.py"]
    ignored = subprocess.run(["git", "check-ignore", *needed], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    assert ignored == []


def test_the_page_points_at_files_that_exist():
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    references = re.findall(r'(?:href|src)="((?:demo|visual_assist)/[^"?]+)(?:\?[^"]*)?"', page)
    assert "demo/demo.css" in references and "demo/device.js" in references
    for reference in references:
        assert (ROOT / reference).is_file(), reference
    # GitHub Pages hides files starting with "_" (such as __init__.py)
    # unless this marker file is present.
    assert (ROOT / ".nojekyll").exists()
