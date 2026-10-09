"""A simulated room, user and hardware.

It lets the whole assistant run with no camera, LiDAR or servos: the tests
use it, and `python main.py simulate` uses it to show guidance at work.

The floor is a plane seen from above: +y is "north", bearings are measured
clockwise from it. Everything solid is a disc. The simulated camera sees
labelled things (it ignores occlusion); the simulated LiDAR hits whatever
disc its beam meets first.
"""

from __future__ import annotations

import math

from . import config
from .assistant import Assistant
from .detector import Detection
from .navigator import TYPICAL_SIZE_M, normalise, turn_degrees

WIDTH, HEIGHT = 640, 480
HFOV, VFOV = 62.2, 48.8
FOCAL = (HEIGHT / 2) / math.tan(math.radians(VFOV / 2))
LIDAR_RANGE = 8.0


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class World:
    def __init__(self, things=None, blocks=()):
        self.things = dict(things or {})  # label -> (x, y, radius)
        self.blocks = list(blocks)  # anonymous (x, y, radius)
        self.x = self.y = 0.0
        self.heading = 0.0  # where the user faces
        self.pan = 0.0  # where the head points, relative to the user
        self.goal = None  # label of the thing being walked to, once known
        self.closest = math.inf  # nearest the user has come to anything solid
        self.closest_obstacle = math.inf  # ... to anything other than the goal
        self.trail = [(0.0, 0.0)]  # everywhere the user has been
        self.events = None  # a list while a run is being recorded (see record)

    def log(self, kind: str, **details) -> None:
        """Note something that happened, if this run is being recorded."""
        if self.events is not None:
            self.events.append({"type": kind, **details})

    def along(self, legs) -> list:
        """A planned route as points in the room, starting where the user stands."""
        x, y = self.x, self.y
        points = [[round(x, 3), round(y, 3)]]
        for leg in legs:
            angle = math.radians(self.heading + leg.bearing)
            x, y = x + leg.length * math.sin(angle), y + leg.length * math.cos(angle)
            points.append([round(x, 3), round(y, 3)])
        return points

    def discs(self, skip=None):
        return [disc for label, disc in self.things.items() if label != skip] + self.blocks

    def relative(self, x: float, y: float):
        """(bearing relative to the user's heading, distance) of a point."""
        dx, dy = x - self.x, y - self.y
        return normalise(math.degrees(math.atan2(dx, dy)) - self.heading), math.hypot(dx, dy)

    def clearance(self, skip=None) -> float:
        return min((math.hypot(cx - self.x, cy - self.y) - r
                    for cx, cy, r in self.discs(skip)), default=math.inf)

    def ray(self, bearing: float):
        """Distance to the first disc along a bearing relative to the user."""
        angle = math.radians(self.heading + bearing)
        ux, uy = math.sin(angle), math.cos(angle)
        nearest = None
        for cx, cy, r in self.discs():
            along = (cx - self.x) * ux + (cy - self.y) * uy
            across2 = (cx - self.x) ** 2 + (cy - self.y) ** 2 - along ** 2
            if across2 > r * r:
                continue
            hit = along - math.sqrt(r * r - across2)
            if hit > 0 and (nearest is None or hit < nearest):
                nearest = hit
        return nearest if nearest is not None and nearest <= LIDAR_RANGE else None

    def turn(self, degrees: float) -> None:
        if degrees:
            self.heading = normalise(self.heading + degrees)
            self.log("turn", heading=round(self.heading, 1))

    def walk(self, metres: float) -> None:
        angle = math.radians(self.heading)
        paces = int(round(metres / 0.05))
        for _ in range(paces):
            self.x += 0.05 * math.sin(angle)
            self.y += 0.05 * math.cos(angle)
            self.trail.append((self.x, self.y))
            self.closest = min(self.closest, self.clearance())
            self.closest_obstacle = min(self.closest_obstacle, self.clearance(self.goal))
        if paces:
            self.log("walk", to=[round(self.x, 3), round(self.y, 3)])

    def obey(self, instruction, step_length: float, rng=None) -> None:
        """Do what the assistant said, the way a person would.

        With `rng`, imprecisely: turns are off by up to a third and steps
        by up to 15%, as real ones are.
        """
        def roughly(value: float, spread: float) -> float:
            return value * (1 + rng.uniform(-spread, spread)) if rng else value

        if instruction.kind == "turn":
            self.turn(roughly(turn_degrees(instruction.turn), 0.33))
        elif instruction.kind == "move":
            self.walk(roughly(instruction.steps * step_length, 0.15))
            self.turn(roughly(turn_degrees(instruction.then), 0.33))

    def picture(self, cell: float = 0.25) -> str:
        """The room from above, as text: S start, * path, E end, # solid, D door."""
        xs = [x for x, _, _ in self.discs()] + [x for x, _ in self.trail]
        ys = [y for _, y, _ in self.discs()] + [y for _, y in self.trail]
        left, right = min(xs) - 1.0, max(xs) + 1.0
        bottom, top = min(ys) - 0.5, max(ys) + 0.75
        columns = int((right - left) / cell) + 1
        rows = int((top - bottom) / cell) + 1
        grid = [[" "] * columns for _ in range(rows)]

        def mark(x: float, y: float, char: str) -> None:
            column, row = int((x - left) / cell), int((top - y) / cell)
            if 0 <= row < rows and 0 <= column < columns:
                grid[row][column] = char

        for row in range(rows):
            for column in range(columns):
                x, y = left + (column + 0.5) * cell, top - (row + 0.5) * cell
                for label, (cx, cy, r) in list(self.things.items()) + [
                        (None, block) for block in self.blocks]:
                    # Thin things must still show: count any overlap with the cell.
                    if math.hypot(x - cx, y - cy) <= max(r, cell * 0.6):
                        grid[row][column] = label[0].upper() if label else "#"
        for x, y in self.trail:
            mark(x, y, "*")
        mark(*self.trail[0], "S")
        mark(*self.trail[-1], "E")
        border = "+" + "-" * columns + "+"
        return "\n".join([border] + ["|" + "".join(line) + "|" for line in grid] + [border])


def wall(x1: float, x2: float, y: float, spacing: float = 0.2) -> list:
    """A straight wall from x1 to x2 at depth y, as a row of small discs."""
    count = int(round((x2 - x1) / spacing))
    return [(x1 + i * spacing, y, 0.1) for i in range(count + 1)]


def cluttered_room(rng):
    """A door 3-6 m away with up to five obstacles scattered before it."""
    bearing, distance = rng.uniform(-50, 50), rng.uniform(3.0, 6.0)
    centre = distance + 0.45
    door = (centre * math.sin(math.radians(bearing)),
            centre * math.cos(math.radians(bearing)), 0.45)
    blocks = []
    for _ in range(rng.randint(1, 5)):
        for _attempt in range(20):
            x, y, r = rng.uniform(-3, 3), rng.uniform(0.9, distance), rng.uniform(0.15, 0.4)
            clear_of_user = math.hypot(x, y) - r > 0.7
            clear_of_door = math.hypot(x - door[0], y - door[1]) - r - 0.45 > 1.0
            if clear_of_user and clear_of_door:
                blocks.append((x, y, r))
                break
    return door, blocks


class FakeCamera:
    def __init__(self, world: World = None):
        self.world = world

    def read(self):
        if self.world is not None:
            self.world.log("photo")
        return object()

    def close(self):
        pass


class FakeDetector:
    def __init__(self, world: World, labels=("chair", "door", "person", "table")):
        self.world = world
        self.labels = sorted(labels)

    def detect(self, frame) -> list:
        def column(angle: float) -> float:
            return WIDTH / 2 * (1 + math.tan(math.radians(angle)) / math.tan(math.radians(HFOV / 2)))

        detections = []
        for label, (x, y, radius) in self.world.things.items():
            bearing, distance = self.world.relative(x, y)
            offset = normalise(bearing - self.world.pan)
            surface = distance - radius
            if abs(offset) >= HFOV / 2 or surface <= 0.05:
                continue
            # Drawn as a flat panel at its near face, the size a real one has.
            real_height, real_width = TYPICAL_SIZE_M.get(label, (1.0, 2 * radius))
            half = math.degrees(math.atan2(real_width / 2, surface))
            limit = HFOV / 2
            x1 = column(max(offset - half, -limit))
            x2 = column(min(offset + half, limit))
            tall = real_height * FOCAL / surface
            y1, y2 = max(0.0, HEIGHT / 2 - tall / 2), min(float(HEIGHT), HEIGHT / 2 + tall / 2)
            detections.append(Detection(label, 0.9, (x1, y1, x2, y2), WIDTH, HEIGHT))
        return detections


class FakeLidar:
    available = True

    def __init__(self, world: World):
        self.world = world

    def read(self):
        distance = self.world.ray(self.world.pan)
        self.world.log("ray", bearing=round(self.world.heading + self.world.pan, 1),
                       distance=None if distance is None else round(distance, 3))
        return distance

    def close(self):
        pass


class FakePanTilt:
    can_move = True

    def __init__(self, world: World):
        self.world = world

    def look(self, bearing, tilt=0.0, settle=None):
        pan = max(-90.0, min(90.0, bearing))
        if pan != self.world.pan:
            self.world.pan = pan
            self.world.log("look", pan=round(pan, 1))

    def center(self):
        self.look(0.0)

    def close(self):
        pass


class FixedPanTilt(FakePanTilt):
    """A head that cannot turn, as on a laptop."""

    can_move = False

    def look(self, bearing, tilt=0.0, settle=None):
        pass


class FakeSpeaker:
    busy = False
    language = "en"
    rate = 160

    def __init__(self, echo: bool = False):
        self.said = []
        self.urgent = []
        self.last = ""
        self.log = None  # set to World.log while a run is being recorded
        self._echo = echo

    def say(self, text, wait=False, urgent=False):
        self.said.append(text)
        self.last = text
        if urgent:
            self.urgent.append(text)
        if self.log is not None:
            self.log("say", who="assistant", text=text)
        if self._echo:
            print(f"  assistant: {text}", flush=True)

    def configure(self, language=None, rate=None):
        self.language = language or self.language
        self.rate = rate or self.rate

    def wait(self):
        pass

    def close(self):
        pass


def make(things=None, blocks=(), pantilt=FakePanTilt, settings=None, echo=False, **parts):
    """An Assistant in a simulated room: (assistant, world, speaker, clock)."""
    settings = dict(settings or {})
    settings.setdefault("servo", {})["settle_s"] = 0
    cfg = config.from_dict(settings)
    world, clock, speaker = World(things, blocks), Clock(), FakeSpeaker(echo)
    assistant = Assistant(
        cfg, FakeCamera(world), FakeDetector(world), FakeLidar(world), pantilt(world),
        speaker, clock=clock, **parts)
    assistant._running = True
    return assistant, world, speaker, clock


def follow_guidance(assistant, world, clock, cycles: int = 40, rng=None, echo=False) -> int:
    """Keep doing what the assistant says until guidance ends. Returns cycles used."""
    step_length = assistant.cfg.navigation.step_length_m
    done = None
    for cycle in range(cycles):
        assistant.step()
        guide = assistant._guide
        if guide is None:
            return cycle
        world.goal = guide.target
        if guide.instruction is not None and guide.instruction is not done:
            done = guide.instruction
            if assistant.route is not None and assistant.route.legs:
                world.log("route", points=world.along(assistant.route.legs))
            world.obey(done, step_length, rng)
            if echo:
                print(f"       user: (does so: now {world.x:+.1f} m across, {world.y:.1f} m "
                      f"forward, facing {world.heading:+.0f} degrees)")
        clock.advance(20.0)
    return cycles


# Rooms for `python main.py simulate`: name -> (labelled things, anonymous blocks).
ROOMS = {
    "chair": ({"door": (0.0, 5.45, 0.45), "chair": (0.0, 2.45, 0.25)}, []),
    "opening": ({"door": (1.2, 5.0, 0.45)},
                [disc for disc in wall(-3.0, 3.0, 2.5) if not 0.5 < disc[0] < 1.9]),
}


def record(things, blocks=(), language: str = "en") -> dict:
    """Guide a simulated user to the door and return everything that happened.

    The result is plain data (it converts straight to JSON) for the demo
    page to replay. `events` is a list, in order, of:

        say    who ("user" or "assistant"), text
        look   pan: the head turns to this angle, relative to the body
        photo  the camera takes a picture where the head points
        ray    bearing (in the room), distance or None: one LiDAR reading
        route  points: the planned route, from where the user stands
        turn   heading: the user turns to face this way (0 = up the room)
        walk   to: the user walks straight to this point
    """
    assistant, world, speaker, clock = make(
        things, blocks=blocks, settings={"speech": {"language": language}})
    world.events = []
    speaker.log = world.log
    world.log("say", who="user", text="Guide me to the door")
    assistant.submit("guide me to the door")
    follow_guidance(assistant, world, clock)
    closest = world.closest_obstacle
    return {
        "events": world.events,
        "reached": assistant.profile.guided("door") > 0,
        "closest": None if math.isinf(closest) else round(closest, 2),
    }


def demonstrate(room: str = "chair", language: str = "en", seed: int = None) -> None:
    """Guide a simulated user to the door, printing what is said and a map."""
    import random

    if seed is not None:
        door, blocks = cluttered_room(random.Random(seed))
        things = {"door": door}
    else:
        things, blocks = ROOMS[room]
    assistant, world, _, clock = make(
        things, blocks=blocks, echo=True, settings={"speech": {"language": language}})
    print('       user: "guide me to the door"')
    assistant.submit("guide me to the door")
    follow_guidance(assistant, world, clock, echo=True)
    print()
    print(world.picture())
    print("S start   * path walked   E end   D door   # obstacle   (0.25 m per character)")
    print(f"Closest approach to an obstacle: {world.closest_obstacle:.2f} m")
