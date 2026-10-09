"""The main loop: see, measure, listen, speak.

All camera, LiDAR and servo access happens on the main thread. The
listener thread only puts transcribed text on a queue, and the speaker
has its own worker, so a slow inference never blocks speech or hearing.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Optional

from . import adaptation, commands
from .detector import center_detection
from .navigator import (
    AFTER_TURN_SLACK, EDGE_PX, TYPICAL_SIZE_M, Instruction, Sighting, after_move,
    best_sightings, direction_of, fuse_distance, has_arrived, next_instruction, normalise,
    pixel_bearing, pixel_width, steps_for, turn_for, visual_distance,
)
from .phrases import Phrases
from .planner import Obstacle, obstacle_from_sighting, obstacles_from_rays, plan, to_xy

log = logging.getLogger(__name__)

INSTRUCTION_REPEAT_S = 6.0
OBSTACLE_REPEAT_S = 15.0
OBSTACLE_CHANGE_M = 0.3
URGENT_REPEAT_S = 1.5
CROWD_NOTICE_S = 60.0
ASK_TIMEOUT_S = 15.0
MAX_PAN_DEG = 90.0
SAME_OBJECT_DEG = 15.0  # sightings this close in direction are one object
FULL_CIRCLE_DEG = 360.0
MAX_UNSEEN = 2  # looks in a row the target may be out of view behind the user
PROGRESS_M = 0.3  # getting this much nearer the target counts as progress
MAX_STALLED = 10  # instructions without progress before giving up
TURNED_DEG = 10.0  # the target swinging round this much means the user turned
MEMORY_DRIFT_M = 0.4  # target distance changing this much means the user walked
MAX_WAIT_S = 12.0
NEVER = float("-inf")


@dataclass
class _Guide:
    """Guidance in progress towards one object."""

    target: str
    bearing: float  # where the target is expected to be at the next look
    distance: Optional[float]
    plans: bool  # True: sweep the LiDAR and plan a route; False: steer by camera
    next_plan: float = 0.0
    lost: int = 0
    unseen: int = 0  # looks in a row where the target was behind the user
    blocked: int = 0
    last_text: str = ""
    last_time: float = NEVER
    reason: Optional[str] = None  # the obstacle last named as being in the way
    instruction: Optional[Instruction] = None
    seen: Optional[float] = None  # target bearing at the last turn instruction
    turned: float = 0.0  # degrees the user has turned since last walking
    nearest: float = float("inf")  # closest the target has been so far
    stalled: int = 0  # instructions since it last got closer
    memory: dict = field(default_factory=dict)  # obstacles found since last walking
    memory_range: float = 0.0  # how far the target was when they were found


def _turns_back(previous: Optional[Instruction], following: Instruction) -> bool:
    """True if `following` turns the user against the turn they were just given."""
    def side(turn: Optional[str]) -> Optional[str]:
        return turn.rsplit("_", 1)[-1] if turn and turn != "around" else None

    if previous is None or following.kind != "turn":
        return False
    before = side(previous.turn if previous.kind == "turn" else previous.then)
    after = side(following.turn)
    return before is not None and after is not None and before != after


class Assistant:
    def __init__(self, cfg, camera, detector, lidar, pantilt, speaker,
                 listener=None, faces=None, currency=None, profile=None,
                 show=False, clock=time.monotonic):
        self.cfg = cfg
        self.camera = camera
        self.detector = detector
        self.lidar = lidar
        self.pantilt = pantilt
        self.speaker = speaker
        self.listener = listener
        self.faces = faces
        self.currency = currency
        self.show = show
        self.profile = profile or adaptation.UserProfile(
            None, cfg.adaptation.walk_speed_mps)
        self.motion = adaptation.MotionTracker(cfg.adaptation)
        self.phrases = Phrases(self.profile.language or cfg.speech.language)
        if (self.phrases.language != speaker.language
                or (self.profile.speech_rate or speaker.rate) != speaker.rate):
            speaker.configure(language=self.phrases.language, rate=self.profile.speech_rate)

        self._clock = clock
        self._texts = queue.Queue()
        self._running = False
        self._frame_no = 0
        self._guide: Optional[_Guide] = None
        self.route = None  # the route planned most recently, for display
        self._context = []  # everything seen during the latest look around
        self._previous_frame = None
        self._last_warning = ("", 0.0, NEVER)  # label, distance, time
        self._last_stop = NEVER
        self._face_seen = {}  # name -> time last announced
        self._crowd_noticed = NEVER
        self._crowded = False
        self._described = NEVER

    # ------------------------------------------------------------- lifecycle

    def run(self) -> None:
        self._running = True
        self._say("ready")
        if self.listener is not None:
            threading.Thread(target=self._listen_loop, daemon=True).start()
        try:
            while self._running:
                self.step()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        self._running = False
        self.profile.save()
        self.pantilt.center()
        self.speaker.close()
        for part in (self.pantilt, self.lidar, self.camera):
            part.close()
        if self.show:
            import cv2

            cv2.destroyAllWindows()

    def submit(self, text: str) -> None:
        """Queue a phrase as if it had been heard on the microphone."""
        self._texts.put(text)

    def step(self) -> None:
        """One pass: handle pending commands, then look at the world once."""
        self._drain_commands()
        if not self._running:
            return
        guide = self._guide
        if guide and guide.plans and self._clock() >= guide.next_plan:
            self._plan_route()
            return
        frame = self.camera.read()
        if frame is None:
            time.sleep(0.1)
            return
        self._frame_no += 1
        detections = self.detector.detect(frame)
        distance = self.lidar.read()

        paused = self._observe_motion(frame, distance)
        self._crowded = self._notice_crowd(detections)
        if self._guide:
            self._guide_watch(detections, distance, paused)
        else:
            self._warn_obstacle(detections, distance)
            if paused:
                self._describe_on_pause(detections, distance)
        self._announce_faces(frame)
        if self.show:
            self._draw(frame, detections, distance)

    # ---------------------------------------------------------------- voice

    def _say(self, key: str, urgent: bool = False, **values) -> None:
        text = self.phrases.t(key, **values)
        self.speaker.say(text[:1].upper() + text[1:], urgent=urgent)

    def _listen_loop(self) -> None:
        while self._running:
            # Don't transcribe our own voice.
            self.speaker.wait()
            try:
                text = self.listener.listen()
            except EOFError:
                self._texts.put("exit")
                return
            except Exception:
                log.exception("Listener failed")
                time.sleep(1.0)
                continue
            if text:
                self._texts.put(text)

    def _drain_commands(self) -> None:
        while self._running:
            try:
                text = self._texts.get_nowait()
            except queue.Empty:
                return
            self._handle(text)

    def _handle(self, text: str) -> None:
        log.info("Heard: %s", text)
        wake = self.cfg.listener.wake_word.lower().strip()
        if wake:
            if wake not in text.lower():
                return
            text = text.lower().replace(wake, " ")
        command = commands.parse(text, self.detector.labels)
        if command is None:
            # Without a wake word any overheard speech lands here; stay quiet.
            if wake:
                self._say("not_understood")
            return
        handler = getattr(self, f"_cmd_{command.intent}")
        handler(command.target)

    def _ask(self, key: str) -> Optional[str]:
        self.speaker.say(self.phrases.t(key), wait=True)
        try:
            return self._texts.get(timeout=ASK_TIMEOUT_S)
        except queue.Empty:
            return None

    def _verbosity(self, target: Optional[str] = None) -> str:
        return adaptation.verbosity(self.profile, self.motion, self.cfg.adaptation,
                                    self._clock(), target)

    # ------------------------------------------------------------- commands

    def _cmd_help(self, _target) -> None:
        self._say("help")

    def _cmd_exit(self, _target) -> None:
        self._say("shutdown")
        self._running = False

    def _cmd_repeat(self, _target) -> None:
        if self.speaker.last:
            self.speaker.say(self.speaker.last)
        else:
            self._say("nothing_to_repeat")

    def _cmd_stop(self, _target) -> None:
        if self._guide:
            self._guide = None
            self.pantilt.center()
            self._say("guidance_stopped")

    def set_language(self, code: str) -> bool:
        """Switch the spoken language, and remember it. False if unknown."""
        if code not in commands.LANGUAGES.values():
            return False
        self.profile.language = code
        self.profile.save()
        self.phrases = Phrases(code)
        self.speaker.configure(language=code)
        return True

    def _cmd_language(self, code) -> None:
        self._say("language_set" if self.set_language(code) else "language_unknown")

    def _cmd_verbosity(self, level) -> None:
        self.profile.verbosity = None if level == "normal" else level
        self.profile.save()
        self._say(f"verbosity.{level}")

    def _cmd_rate(self, direction) -> None:
        rate = self.profile.change_rate(direction == "faster", self.cfg.speech.rate)
        self.speaker.configure(rate=rate)
        self._say(f"rate.{direction}")

    def _cmd_warn(self, direction) -> None:
        self.profile.change_warning(direction == "earlier")
        self._say(f"warn.{direction}")

    def _cmd_scan(self, _target) -> None:
        sightings = self._scan()
        self._recenter()
        self.speaker.say(self.phrases.describe(
            sightings, self.cfg.navigation.center_deadband_deg))

    def _cmd_distance(self, _target) -> None:
        frame = self.camera.read()
        detections = self.detector.detect(frame) if frame is not None else []
        distance = self.lidar.read()
        sightings = self._sightings(detections, distance, 0.0)
        on_axis = center_detection(detections)
        named = sightings[detections.index(on_axis)] if on_axis else None
        if named is not None and named.distance is not None:
            self._say("nearest_label", label=named.label,
                      distance=self.phrases.distance(named.distance, named.approx))
        elif distance is not None:
            self._say("nearest", distance=self.phrases.distance(distance))
        else:
            self._say("measure_fail")

    def _cmd_find(self, target) -> Optional[Sighting]:
        if not target:
            self._say("which_object")
            return None
        if target not in self.detector.labels:
            self._say("not_trained", word=target)
            return None
        sighting = self._locate(target)
        if sighting is None:
            self._say("not_found", label=target)
        else:
            self.speaker.say(self.phrases.locate(
                sighting, self.cfg.navigation.center_deadband_deg,
                brief=self._verbosity(target) == "brief"))
            self._already_told(sighting)
        return sighting

    def _already_told(self, sighting: Sighting) -> None:
        """Don't follow news about what is ahead with a warning about the same thing."""
        ahead = direction_of(sighting.bearing, self.cfg.navigation.center_deadband_deg)
        if ahead == "ahead" and sighting.distance is not None:
            self._last_warning = (sighting.label, sighting.distance, self._clock())

    def _cmd_guide(self, target) -> None:
        sighting = self._cmd_find(target)
        if sighting is None:
            return
        if has_arrived(sighting.distance, self.cfg.navigation):
            self._say("already_at", label=target)
            return
        can_plan = (self.cfg.planner.enabled and self.pantilt.can_move
                    and getattr(self.lidar, "available", True))
        self._guide = guide = _Guide(target, sighting.bearing, sighting.distance, plans=can_plan)
        if can_plan:
            self._plan_route(sighting)
        else:
            instruction = self._direct_instruction(sighting)
            self._note_turning(guide, sighting, instruction)
            self._instruct(sighting, instruction)

    def _cmd_who(self, _target) -> None:
        if self.faces is None:
            self._say("faces_unavailable")
            return
        frame = self.camera.read()
        faces = self.faces.identify(frame) if frame is not None else []
        names = [face.name for face in faces if face.name]
        self.speaker.say(self.phrases.people(names, len(faces) - len(names)))

    def _cmd_register(self, name) -> None:
        if self.faces is None:
            self._say("faces_unavailable")
            return
        if not name:
            name = self._ask("ask_name")
            if not name:
                self._say("registration_cancelled")
                return
            name = name.strip().title()
        self.speaker.say(self.phrases.t("registering", name=name), wait=True)
        frame = self.camera.read()
        if frame is not None and self.faces.register(name, frame):
            self._face_seen[name] = self._clock()
            self._say("registered", name=name)
        else:
            self._say("no_face")

    def _cmd_money(self, _target) -> None:
        if self.currency is None:
            self._say("currency_unavailable")
            return
        frames = [f for f in (self.camera.read() for _ in range(3)) if f is not None]
        value, confidence = self.currency.read(frames) if frames else (None, 0.0)
        if value is None or confidence < self.cfg.currency.minimum:
            self._say("no_note")
        elif confidence >= self.cfg.currency.confidence:
            self._say("note", value=value)
        else:
            self._say("note_unsure", value=value)

    # -------------------------------------------------------------- sensing

    def _visual(self, det) -> Optional[float]:
        """The camera's own estimate of how far a detection is."""
        cam = self.cfg.camera
        return visual_distance(det.label, det.box, det.frame_width, det.frame_height,
                               cam.hfov_deg, cam.vfov_deg)

    def _sightings(self, detections, lidar: Optional[float], angle: float) -> list:
        """Detections from one view as sightings relative to the user.

        The LiDAR measures along the camera axis only, so its reading can
        belong to at most the one object sitting on that axis.
        """
        cam = self.cfg.camera
        on_axis = center_detection(detections)
        sightings = []
        for det in detections:
            offset = pixel_bearing(det.cx, det.frame_width, cam.hfov_deg)
            visual = self._visual(det)
            distance, approx = fuse_distance(lidar if det is on_axis else None, visual)
            # An object cut off by the side of the picture has its middle
            # misjudged; rank such a view below any that shows it whole.
            cut_off = det.box[0] <= EDGE_PX or det.box[2] >= det.frame_width - EDGE_PX
            sightings.append(Sighting(
                label=det.label, bearing=angle + offset, distance=distance,
                confidence=det.confidence,
                offset=abs(offset) + (cam.hfov_deg / 2 if cut_off else 0.0), approx=approx,
                width=pixel_width(det.box[0], det.box[2], det.frame_width, cam.hfov_deg),
                visual=visual))
        return sightings

    def _look(self, angle: float) -> list:
        """Turn the head to `angle` and report what is there."""
        self.pantilt.look(angle)
        frame = self.camera.read()
        if frame is None:
            return []
        sightings = self._sightings(self.detector.detect(frame), self.lidar.read(), angle)
        self._context.extend(sightings)
        return sightings

    def _scan(self, announce: bool = True) -> list:
        """Sweep the head across the room and list what is seen where."""
        angles = self.cfg.servo.scan_angles if self.pantilt.can_move else [0]
        if announce and len(angles) > 1:
            self._say("scanning")
        self._context = []
        for angle in angles:
            self._look(angle)
        return list(self._context)

    def _sweep(self) -> list:
        """LiDAR readings across the space ahead: [(bearing, distance or None)]."""
        planner = self.cfg.planner
        rays = []
        bearing = -planner.sweep_half_angle_deg
        while bearing <= planner.sweep_half_angle_deg + 1e-6:
            self.pantilt.look(bearing, settle=planner.sweep_settle_s)
            rays.append((bearing, self.lidar.read()))
            bearing += planner.sweep_step_deg
        return rays

    def _recenter(self) -> None:
        self.pantilt.center()
        # The next frame is from a different view; motion can't be judged from it.
        self._previous_frame = None

    @staticmethod
    def _pick(sightings: list, label: str) -> Optional[Sighting]:
        """The sighting to trust for one kind of object.

        A detector is sometimes wrong, so if the object was reported in
        more than one place the most confident place wins. Among the views
        of that place, the one with the best distance is used.
        """
        candidates = [s for s in sightings if s.label == label]
        if not candidates:
            return None
        # Equally confident views: the one nearest the middle of its picture.
        surest = max(candidates, key=lambda s: (round(s.confidence, 2), -s.offset))
        here = [s for s in candidates
                if abs(normalise(s.bearing - surest.bearing)) <= SAME_OBJECT_DEG]
        return best_sightings(here)[0]

    def _range(self, sighting: Sighting) -> Sighting:
        """Point the LiDAR at an object whose distance is not yet measured."""
        if not (sighting.distance is None or sighting.approx) or not self.pantilt.can_move:
            return sighting
        aim = max(-MAX_PAN_DEG, min(MAX_PAN_DEG, sighting.bearing))
        self.pantilt.look(aim)
        lidar = self.lidar.read()
        if sighting.visual is None and sighting.label in TYPICAL_SIZE_M:
            # It was cut off at the edge of the view it was found in. Centred
            # now, the camera can judge its size, which is the only check
            # that the LiDAR is on it and not on something in front.
            frame = self.camera.read()
            if frame is not None:
                centred = self._pick(
                    self._sightings(self.detector.detect(frame), None, aim), sighting.label)
                if centred is not None:
                    sighting = replace(sighting, bearing=centred.bearing,
                                       visual=centred.visual)
        distance, approx = fuse_distance(lidar, sighting.visual)
        return replace(sighting, distance=distance, approx=approx)

    def _locate(self, label: str) -> Optional[Sighting]:
        """Find one object by scanning, then measure how far it is."""
        found = self._pick(self._scan(), label)
        if found is not None:
            found = self._range(found)
        self._recenter()
        return found

    # ------------------------------------------------------------- guidance

    def _just_turned(self, guide: _Guide, sighting: Sighting) -> bool:
        """True if the user has turned since the last instruction, as told.

        The target stands still, so if it has swung round since a turn was
        asked for, the user has turned.
        """
        last = guide.instruction
        if last is None:
            return False
        if last.kind == "turn":
            return (guide.seen is not None
                    and abs(normalise(sighting.bearing - guide.seen)) >= TURNED_DEG)
        return last.then is not None

    def _note_turning(self, guide: _Guide, sighting: Sighting,
                      instruction: Instruction) -> None:
        """Keep count of how far the user has turned on the spot."""
        if instruction.kind == "turn":
            if guide.seen is not None:
                guide.turned += abs(normalise(sighting.bearing - guide.seen))
            guide.seen = sighting.bearing
        else:
            guide.seen, guide.turned = None, 0.0

    def _direct_instruction(self, sighting: Sighting, after_turn: bool = False) -> Instruction:
        """Steer straight at the target, for when no route can be planned."""
        nav = self.cfg.navigation
        deadband = nav.center_deadband_deg * (AFTER_TURN_SLACK if after_turn else 1.0)
        turn = turn_for(sighting.bearing, deadband)
        if turn:
            return Instruction("turn", turn=turn)
        if sighting.distance is None:
            return Instruction("move")
        return Instruction("move", steps=steps_for(
            sighting.distance, nav.step_length_m, nav.arrive_m))

    def _reacquire(self, guide: _Guide) -> Optional[Sighting]:
        """Look where the target should be now; search wider if it is not there."""
        self._context = []
        hint = max(-MAX_PAN_DEG, min(MAX_PAN_DEG, guide.bearing))
        found = self._pick(self._look(hint), guide.target)
        if found is None:
            found = self._pick(self._scan(announce=False), guide.target)
        return self._range(found) if found is not None else None

    def _obstacles(self, guide: _Guide, sighting: Sighting) -> list:
        """Everything known to be in the way: swept now, seen now, or remembered.

        A sweep covers only what is in front. When the user is turned on the
        spot, what was found before is still there, just at another angle.
        The target does not move, so obstacles are remembered by their
        direction relative to it and put back wherever it is seen next.
        Walking changes every direction, so the memory is dropped then.
        """
        planner = self.cfg.planner
        obstacles = obstacles_from_rays(
            self._sweep(), planner.sweep_step_deg, planner.max_range_m)
        for seen in best_sightings(self._context):
            if seen.label != guide.target:
                obstacle = obstacle_from_sighting(seen)
                if obstacle:
                    obstacles.append(obstacle)

        if abs(sighting.distance - guide.memory_range) > MEMORY_DRIFT_M:
            guide.memory = {}  # the user has moved since: it no longer lines up
        obstacles += [Obstacle(*to_xy(angle + sighting.bearing, reach), radius, label)
                      for angle, reach, radius, label in guide.memory.values()]
        for obstacle in obstacles:
            angle = math.degrees(math.atan2(obstacle.x, obstacle.y)) - sighting.bearing
            reach = math.hypot(obstacle.x, obstacle.y)
            # Rounded keys stop repeat sweeps of the same wall piling up.
            guide.memory[(round(angle / 2.0), round(reach * 10.0))] = (
                angle, reach, obstacle.radius, obstacle.label)
        guide.memory_range = sighting.distance
        return obstacles

    def _plan_route(self, known: Optional[Sighting] = None) -> None:
        """One guidance cycle: find the target, sweep, plan, say the next step."""
        guide, nav, planner = self._guide, self.cfg.navigation, self.cfg.planner
        sighting = known or self._reacquire(guide)
        if sighting is not None:
            guide.unseen = 0
        elif (guide.distance is not None and guide.unseen < MAX_UNSEEN
              and abs(guide.bearing) > MAX_PAN_DEG + self.cfg.camera.hfov_deg / 2 - 10):
            # A detour can turn the user's back on the target, further round
            # than the head can look. It has not gone anywhere: carry on from
            # where it must be until the route turns back towards it.
            guide.unseen += 1
            sighting = Sighting(guide.target, guide.bearing, guide.distance, approx=True)
        if sighting is None:
            self._recenter()
            guide.lost += 1
            if guide.lost >= nav.lost_frames:
                self._guide = None
                self._say("lost", label=guide.target)
            else:
                if guide.lost == 1:
                    self._say("looking", label=guide.target)
                guide.next_plan = self._clock() + 1.0
            return
        guide.lost = 0
        if has_arrived(sighting.distance, nav):
            self._recenter()
            self._arrive(sighting)
            return

        just_turned = self._just_turned(guide, sighting)
        route = None
        if sighting.distance is None:
            instruction = self._direct_instruction(sighting, just_turned)
        else:
            obstacles = self._obstacles(guide, sighting)
            route = plan(sighting.bearing, sighting.distance, obstacles, nav, planner)
            instruction = next_instruction(route, nav, after_turn=just_turned)
            if just_turned and _turns_back(guide.instruction, instruction):
                # Told to turn one way a moment ago and now the other: two
                # routes of much the same length are taking turns at being
                # shortest. Stay with the one already faced, if it will do.
                steady = plan(sighting.bearing, sighting.distance, obstacles, nav, planner,
                              keep_straight=True)
                onward = next_instruction(steady, nav, after_turn=True)
                if onward.kind == "move":
                    route, instruction = steady, onward
        self.route = route
        self._recenter()

        if instruction.kind == "arrived":
            self._arrive(sighting)
            return
        guide.blocked = guide.blocked + 1 if instruction.kind == "blocked" else 0
        self._note_turning(guide, sighting, instruction)
        if sighting.distance is not None:
            if sighting.distance < guide.nearest - PROGRESS_M:
                guide.nearest, guide.stalled = sighting.distance, 0
            else:
                guide.stalled += 1
        if (guide.blocked >= nav.lost_frames or guide.turned >= FULL_CIRCLE_DEG
                or guide.stalled >= MAX_STALLED):
            # Blocked every time, turned all the way round without finding
            # anywhere to walk, or many instructions later no nearer than
            # before: there is no way through that can be found from here.
            self._guide = None
            self.speaker.say(f"{self.phrases.t('no_way', label=guide.target)} "
                             f"{self.phrases.t('guidance_stopped')}")
            return
        self._instruct(sighting, instruction, route)

    def _instruct(self, sighting: Sighting, instruction: Instruction, route=None) -> None:
        """Say the next step, unless it was only just said."""
        guide, nav = self._guide, self.cfg.navigation
        level = self._verbosity(guide.target)
        parts = []
        if instruction.kind == "blocked":
            parts.append(self.phrases.t("no_way", label=guide.target))
        else:
            if level == "detailed":
                parts.append(self.phrases.locate(sighting, nav.center_deadband_deg))
            reason = None if route is None or route.direct else route.blocker
            if reason is not None and reason != guide.reason and level != "brief":
                parts.append(self.phrases.t("in_way", label=reason) if reason
                             else self.phrases.t("way_blocked"))
            guide.reason = reason
            parts.append(self.phrases.instruction(instruction, brief=level == "brief"))
        text = " ".join(part[:1].upper() + part[1:] for part in parts)

        now = self._clock()
        fresh = text != guide.last_text or now - guide.last_time >= INSTRUCTION_REPEAT_S
        # Steering by camera runs every frame: never queue up behind our own voice.
        if fresh and (guide.plans or not self.speaker.busy):
            guide.last_text, guide.last_time = text, now
            self.speaker.say(text)

        guide.instruction = instruction
        if instruction.kind == "move":
            guide.memory = {}
        guide.bearing, guide.distance = after_move(
            sighting.bearing, sighting.distance, instruction, nav.step_length_m)
        guide.next_plan = now + self._time_for(instruction)

    def _time_for(self, instruction: Instruction) -> float:
        """Seconds to let the user carry out an instruction before re-planning."""
        if instruction.kind == "turn":
            return 3.0
        if instruction.kind == "blocked" or not instruction.steps:
            return 4.0
        pace = self.cfg.navigation.step_length_m / max(self.profile.walk_speed, 0.3)
        wait = 2.0 + instruction.steps * pace + (1.5 if instruction.then else 0.0)
        return min(wait, MAX_WAIT_S)

    def _arrive(self, sighting: Sighting) -> None:
        target = self._guide.target
        self._guide = None
        deadband = self.cfg.navigation.center_deadband_deg
        if direction_of(sighting.bearing, deadband) == "ahead":
            self._say("reached", label=target)
        else:
            self._say("reached_dir", label=target,
                      direction=self.phrases.direction(sighting.bearing, deadband))
        self._already_told(sighting)
        self.profile.note_guided(target)

    def _guide_watch(self, detections, distance, paused: bool) -> None:
        """Between planned steps: notice arrival and danger. Without a
        planner, this is the guidance itself, steering by what the camera sees."""
        guide, nav, cam = self._guide, self.cfg.navigation, self.cfg.camera
        matches = [d for d in detections if d.label == guide.target]
        det = max(matches, key=lambda d: d.confidence) if matches else None
        sighting = None
        if det is not None:
            on_axis = center_detection(detections) is det
            reach, approx = fuse_distance(distance if on_axis else None, self._visual(det))
            sighting = Sighting(det.label, pixel_bearing(det.cx, det.frame_width, cam.hfov_deg),
                                reach, det.confidence, approx=approx)
            # With no distance to go by, a target that has grown to fill the
            # whole picture is as near as the camera can tell.
            if on_axis and (has_arrived(reach, nav) or (reach is None and det.fills_frame)):
                self._arrive(sighting)
                return

        warned = self._warn_obstacle(detections, distance, ignore=guide.target)
        if guide.plans:
            if paused:
                # Standing still usually means "what now?": say it again.
                guide.last_time = NEVER
                guide.next_plan = self._clock()
            elif warned:
                guide.next_plan = min(guide.next_plan, self._clock() + 1.0)
            return

        if sighting is None:
            guide.lost += 1
            if guide.lost >= nav.lost_frames:
                self._guide = None
                self._say("lost", label=guide.target)
            return
        guide.lost = 0
        if paused:
            guide.last_time = NEVER
        instruction = self._direct_instruction(sighting, self._just_turned(guide, sighting))
        self._note_turning(guide, sighting, instruction)
        self._instruct(sighting, instruction)

    # ----------------------------------------------------------- adaptation

    def _observe_motion(self, frame, distance) -> bool:
        """Update what is known about the user's movement. True = just paused."""
        adapt = self.cfg.adaptation
        if not adapt.enabled:
            return False
        turn = change = None
        if self._previous_frame is not None:
            turn, change = adaptation.frame_motion(
                self._previous_frame, frame, self.cfg.camera.hfov_deg)
        self._previous_frame = frame
        paused = self.motion.update(self._clock(), distance, turn, change)
        if self.motion.speed > adaptation.MOVING_MPS:
            # Learn the user's usual pace a little at a time.
            self.profile.walk_speed += 0.05 * (self.motion.speed - self.profile.walk_speed)
        return paused

    def _notice_crowd(self, detections) -> bool:
        crowded = adaptation.is_crowded(detections, self.cfg.adaptation)
        now = self._clock()
        if crowded and not self._crowded and now - self._crowd_noticed >= CROWD_NOTICE_S:
            self._crowd_noticed = now
            self._say("crowded")
        return crowded

    def _warn_obstacle(self, detections, distance, ignore: Optional[str] = None) -> bool:
        """Speak if something is too close ahead. Returns True if it spoke."""
        nav, adapt = self.cfg.navigation, self.cfg.adaptation
        speed = self.motion.speed
        if distance is None or distance > adaptation.warn_distance(
                nav, adapt, self.profile, speed):
            return False
        on_axis = center_detection(detections)
        label = on_axis.label if on_axis else "obstacle"
        if label == ignore:
            return False

        urgent = adaptation.is_urgent(distance, speed, adapt)
        now = self._clock()
        last_label, last_distance, last_time = self._last_warning
        elapsed = now - last_time
        if urgent:
            # "Stop" cuts in whatever was said a moment ago, but not on itself.
            if now - self._last_stop < URGENT_REPEAT_S:
                return False
            self._last_stop = now
        else:
            if self.speaker.busy or elapsed < adaptation.warn_cooldown(
                    nav, adapt, speed, self._crowded):
                return False
            # Standing still in front of the same thing should not nag.
            unchanged = (label == last_label
                         and abs(distance - last_distance) < OBSTACLE_CHANGE_M)
            if unchanged and elapsed < OBSTACLE_REPEAT_S:
                return False
        self._last_warning = (label, distance, now)
        brief = self._crowded or self._verbosity() == "brief"
        self.speaker.say(self.phrases.obstacle(label, distance, urgent, brief), urgent=urgent)
        return True

    def _describe_on_pause(self, detections, distance) -> None:
        """A hesitant user who stops is told what is in front of them."""
        if self._crowded or self.speaker.busy or self._verbosity() != "detailed":
            return
        now = self._clock()
        if now - self._described < self.cfg.adaptation.describe_cooldown_s:
            return
        sightings = self._sightings(detections, distance, 0.0)
        if not sightings:
            return
        self._described = now
        self.speaker.say(f"{self.phrases.t('paused')} " + self.phrases.describe(
            sightings, self.cfg.navigation.center_deadband_deg))

    def _announce_faces(self, frame) -> None:
        if self.faces is None or self._frame_no % self.cfg.faces.every_n_frames:
            return
        now = self._clock()
        cooldown = self.cfg.faces.announce_cooldown_s
        faces = self.faces.identify(frame)

        def due(key: str) -> bool:
            if now - self._face_seen.get(key, NEVER) < cooldown:
                return False
            self._face_seen[key] = now
            return True

        # People the user meets most often are announced first.
        known = sorted({face.name for face in faces if face.name},
                       key=lambda name: -self.profile.meetings(name))
        for name in known:
            if due(name):
                before, days = self.profile.note_person(name)
                familiar = (before >= self.cfg.adaptation.familiar_after
                            and self._verbosity() != "detailed")
                self.speaker.say(self.phrases.face(name, familiar, days if before else 0))
        # Strangers come last, and are not announced at all in a crowd.
        if any(face.name is None for face in faces) and not self._crowded and due(""):
            self._say("face_unknown")

    # -------------------------------------------------------------- preview

    def _draw(self, frame, detections, distance) -> None:
        import cv2

        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, f"{det.label} {det.confidence:.2f} {det.position}",
                        (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 0), 1)
        if distance is not None:
            cv2.putText(frame, f"LiDAR {distance:.2f} m", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Visual Assist", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self._running = False


def build(cfg, keyboard: bool = False, show: bool = False) -> Assistant:
    """Open every device named in the config and wire up an Assistant."""
    from .camera import open_camera
    from .currency import open_currency
    from .detector import open_detector
    from .faces import open_faces
    from .lidar import open_lidar
    from .listener import KeyboardListener, open_listener
    from .servo import open_pantilt
    from .speech import Speaker

    profile = None
    if cfg.adaptation.enabled:
        profile = adaptation.UserProfile(
            cfg.path(cfg.adaptation.profile), cfg.adaptation.walk_speed_mps)
    return Assistant(
        cfg,
        camera=open_camera(cfg.camera),
        detector=open_detector(cfg.detector, cfg.path),
        lidar=open_lidar(cfg.lidar),
        pantilt=open_pantilt(cfg.servo),
        speaker=Speaker(cfg.speech),
        listener=KeyboardListener() if keyboard else open_listener(cfg.listener, cfg.path),
        faces=open_faces(cfg.faces, cfg.path),
        currency=open_currency(cfg.currency, cfg.path),
        profile=profile,
        show=show,
    )
