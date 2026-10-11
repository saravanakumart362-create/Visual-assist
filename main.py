"""Visual Assist - Navigating the Unseen.

    python main.py                 start the assistant
    python main.py --keyboard      type commands instead of speaking them
    python main.py --show          also show the camera view with detections
    python main.py check           report which devices and libraries work
    python main.py register NAME   register the face in front of the camera
    python main.py simulate        guide a simulated user round obstacles
                                   (no hardware; add --room opening or --seed 5)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from visual_assist import config

ROOT = Path(__file__).resolve().parent


def check(cfg) -> int:
    """Try to open each part of the system and say what was found."""
    from visual_assist.camera import open_camera
    from visual_assist.currency import open_currency
    from visual_assist.detector import open_detector
    from visual_assist.faces import open_faces
    from visual_assist.lidar import open_lidar
    from visual_assist.listener import open_listener
    from visual_assist.phrases import available
    from visual_assist.servo import open_pantilt
    from visual_assist.speech import Speaker

    parts = [
        ("camera", lambda: open_camera(cfg.camera)),
        ("detector", lambda: open_detector(cfg.detector, cfg.path)),
        ("currency", lambda: open_currency(cfg.currency, cfg.path)),
        ("lidar", lambda: open_lidar(cfg.lidar)),
        ("pan-tilt", lambda: open_pantilt(cfg.servo)),
        ("speech", lambda: Speaker(cfg.speech)),
        ("listener", lambda: open_listener(cfg.listener, cfg.path)),
        ("faces", lambda: open_faces(cfg.faces, cfg.path)),
    ]
    failures = 0
    for name, opener in parts:
        try:
            part = opener()
            status = type(part).__name__ if part is not None else "not available"
            if name == "detector":
                doors = "yes" if "door" in part.labels else "NO (weights/doors.pt missing)"
                status += f" ({len(part.labels)} object types; doors: {doors})"
            if name == "lidar" and part is not None:
                status += f" (reading: {part.read()} m)"
            if name == "speech":
                voices = ", ".join(part.voices) or "none: print only"
                status += f" (language {part.language}; voices: {voices})"
            if name == "faces" and part is not None:
                status += f" (registered: {', '.join(part.names) or 'nobody yet'})"
            if hasattr(part, "close"):
                part.close()
        except Exception as exc:
            failures += 1
            status = f"FAILED - {exc}"
        print(f"{name:10s} {status}")
    print(f"{'languages':10s} {', '.join(available())}")
    return 1 if failures else 0


def register(cfg, name: str) -> int:
    from visual_assist.camera import open_camera
    from visual_assist.faces import open_faces

    faces = open_faces(cfg.faces, cfg.path)
    if faces is None:
        print("Face recognition is not available (see the warning above).")
        return 1
    camera = open_camera(cfg.camera)
    try:
        # A camera's first pictures are often dark, and a face may be turned
        # away for a moment: keep looking for a second or two.
        done = False
        for _ in range(30):
            frame = camera.read()
            if frame is not None and faces.register(name, frame):
                done = True
                break
    finally:
        camera.close()
    if not done:
        print("No face found close enough to the camera. Face it from about "
              "half a metre and try again.")
        return 1
    print(f"Registered {name}. Known people: {', '.join(faces.names)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(ROOT / "config.yaml"),
                        help="path to the YAML config file")
    parser.add_argument("--keyboard", action="store_true",
                        help="type commands instead of using the microphone")
    parser.add_argument("--show", action="store_true",
                        help="show the camera view with detections")
    parser.add_argument("--language", help="language to speak: en, ml, hi or ta")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="start the assistant (default)")
    sub.add_parser("check", help="report which devices and libraries work")
    sub.add_parser("register", help="register a face").add_argument("name")
    simulate = sub.add_parser("simulate", help="guide a simulated user (no hardware)")
    simulate.add_argument("--room", choices=["chair", "opening"], default="chair",
                          help="a chair in the way, or a wall with one opening")
    simulate.add_argument("--seed", type=int, help="a random cluttered room instead")
    args = parser.parse_args(argv)

    # Spoken text is also printed; make sure any script can be printed anywhere.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = config.load(args.config)
    if args.language:
        cfg.speech.language = args.language

    if args.command == "check":
        return check(cfg)
    if args.command == "register":
        return register(cfg, args.name)
    if args.command == "simulate":
        from visual_assist.simulation import demonstrate

        logging.getLogger("visual_assist").setLevel(logging.WARNING)
        demonstrate(args.room, cfg.speech.language, args.seed)
        return 0

    from visual_assist.assistant import build

    assistant = build(cfg, keyboard=args.keyboard, show=args.show)
    # Asked for on the command line: overrides the remembered language.
    if args.language and not assistant.set_language(args.language):
        print(f"Unknown language '{args.language}'; continuing in "
              f"{assistant.phrases.language}.")
    assistant.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
