"""Geometry and decisions behind the spoken guidance.

Everything here is arithmetic, so it runs (and is tested) without any
hardware. Turning the results into sentences is the job of phrases.py.

Bearings are in degrees relative to where the user faces:
0 = straight ahead, positive = to the user's right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

# A spoken turn is one of three sizes. A person told to turn "slightly"
# turns about 30 degrees and one told to "turn" about 90, so each band is
# centred on those and guidance settles instead of overshooting forever.
SLIGHT_DEG = 45.0
SIDE_DEG = 135.0
TURN_DEGREES = {"slight": 30.0, "full": 90.0, "around": 180.0}
# A turn of up to this much is asked for as "slightly". The boundary sits
# midway between 30 and 90 so that either wording leaves at most 30 degrees
# to correct.
SLIGHT_TURN_DEG = 60.0
AFTER_TURN_SLACK = 1.7  # how much wider "straight ahead" is right after a turn
STEP_MARGIN = 0.85  # share of the clear distance that may be walked
MAX_STEPS = 5  # longest walk in one instruction
MAX_DRIFT_M = 0.6  # most a walk may stray sideways from the planned leg
NUDGE_DEG = 1.0  # a route leaning at least this much says which way to edge

# Typical real-world (height, width) in metres, used to judge distance from
# how big an object looks. Only objects whose size varies little are listed:
# a table or sofa can be any size, so the camera alone cannot range them.
TYPICAL_SIZE_M = {
    "door": (2.0, 0.9), "person": (1.65, 0.5), "refrigerator": (1.7, 0.7),
    "chair": (0.9, 0.5), "bottle": (0.25, 0.08), "cup": (0.1, 0.08),
}
EDGE_PX = 2  # a box this close to the frame edge is taken to be cut off
# The LiDAR and the camera must agree within this factor to be believed.
AGREEMENT = 1.5


@dataclass(frozen=True)
class Sighting:
    """An object located relative to the user."""

    label: str
    bearing: float
    distance: Optional[float] = None  # metres, None when unknown
    confidence: float = 0.0
    offset: float = 0.0  # how far off the camera axis it was seen (degrees)
    approx: bool = False  # distance is a camera estimate, not a LiDAR reading
    width: float = 0.0  # angular width in degrees
    visual: Optional[float] = None  # the camera's own distance estimate


@dataclass(frozen=True)
class Instruction:
    """One thing to tell the user to do next."""

    kind: str  # turn | move | arrived | blocked
    turn: Optional[str] = None  # slight_left, right, around, ...
    steps: int = 0  # 0 with kind "move" means "forward slowly"
    then: Optional[str] = None  # turn to make after walking


def position_of(cx: float, width: float) -> str:
    """Classify a horizontal pixel position as left / center / right."""
    third = width / 3.0
    if cx < third:
        return "left"
    if cx > 2 * third:
        return "right"
    return "center"


def pixel_bearing(cx: float, width: float, hfov_deg: float) -> float:
    """Angle of a pixel column from the camera axis, in degrees."""
    half = width / 2.0
    return math.degrees(
        math.atan((cx - half) / half * math.tan(math.radians(hfov_deg / 2.0)))
    )


def pixel_width(x1: float, x2: float, width: float, hfov_deg: float) -> float:
    """Angle an object spans between two pixel columns, in degrees."""
    return pixel_bearing(x2, width, hfov_deg) - pixel_bearing(x1, width, hfov_deg)


def normalise(bearing: float) -> float:
    """Wrap an angle into -180..180."""
    return (bearing + 180.0) % 360.0 - 180.0


def direction_of(bearing: float, deadband: float) -> str:
    """Where something is: ahead, slight_left, right, behind_left, ..."""
    angle = abs(bearing)
    if angle <= deadband:
        return "ahead"
    side = "right" if bearing > 0 else "left"
    if angle <= SLIGHT_DEG:
        return f"slight_{side}"
    if angle <= SIDE_DEG:
        return side
    return f"behind_{side}"


def turn_for(bearing: float, deadband: float) -> Optional[str]:
    """The turn that faces `bearing`, or None if already facing it."""
    angle = abs(bearing)
    if angle <= deadband:
        return None
    side = "right" if bearing > 0 else "left"
    if angle <= SLIGHT_TURN_DEG:
        return f"slight_{side}"
    if angle <= SIDE_DEG:
        return side
    return "around"


def turn_degrees(turn: Optional[str]) -> float:
    """How far a person is expected to turn for a spoken turn."""
    if turn is None:
        return 0.0
    if turn == "around":
        return TURN_DEGREES["around"]
    size = TURN_DEGREES["slight" if turn.startswith("slight") else "full"]
    return size if turn.endswith("right") else -size


def has_arrived(distance: Optional[float], nav) -> bool:
    """True when the target is close enough that another step would overshoot.

    The aim is to stop `arrive_m` from the target. Steps are whole, so the
    user is "there" once less than half a step short of that.
    """
    return distance is not None and distance <= nav.arrive_m + nav.step_length_m / 2.0


def steps_for(distance: float, step_length: float, stop_short: float = 0.0) -> int:
    """Number of steps to walk, stopping `stop_short` metres before the target."""
    return max(1, round(max(distance - stop_short, 0.0) / step_length))


def visual_distance(label: str, box, frame_width: int, frame_height: int,
                    hfov_deg: float, vfov_deg: float) -> Optional[float]:
    """Distance judged from how big an object of known size looks.

    Height is used when the whole height is in view. Up close an object is
    taller than the picture, and then its width is used instead. An object
    cut off both ways gives no estimate (it is simply very near).
    """
    size = TYPICAL_SIZE_M.get(label)
    if size is None or not frame_width or not frame_height:
        return None

    def judge(low: float, high: float, frame: int, fov_deg: float, real: float):
        if low <= EDGE_PX or high >= frame - EDGE_PX or high - low < 4:
            return None  # cut off by the frame, or too small to measure
        focal = (frame / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        return real * focal / (high - low)

    x1, y1, x2, y2 = box
    return (judge(y1, y2, frame_height, vfov_deg, size[0])
            or judge(x1, x2, frame_width, hfov_deg, size[1]))


def fuse_distance(lidar: Optional[float], visual: Optional[float]):
    """Combine the LiDAR reading and the camera estimate for one object.

    Returns (distance, approx). The LiDAR is exact but measures whatever its
    beam hits; if that disagrees with how big the object looks, the beam is
    on something else (in front of, above or below the object) and the
    camera's rougher estimate is the honest answer. Without this check, a
    chair standing in front of a door would be reported as the door.
    """
    if lidar is None:
        return visual, visual is not None
    if visual is None:
        return lidar, False
    if visual / AGREEMENT <= lidar <= visual * AGREEMENT:
        return lidar, False
    return visual, True


def best_sightings(sightings: Iterable[Sighting]) -> list:
    """Keep one sighting per label: measured distance first, then most central."""

    def rank(sighting: Sighting):
        return (sighting.distance is None, sighting.approx, sighting.offset)

    best = {}
    for sighting in sightings:
        current = best.get(sighting.label)
        if current is None or rank(sighting) < rank(current):
            best[sighting.label] = sighting
    return sorted(best.values(), key=lambda s: s.bearing)


def after_move(bearing: float, distance: Optional[float], instruction: Instruction,
               step_length: float):
    """Where a target should be once the user has done what was said.

    Returns (bearing, distance). It is only a prediction of where to look
    first; the camera then checks it.
    """
    if distance is None:
        return normalise(bearing - turn_degrees(instruction.turn)), None
    x = distance * math.sin(math.radians(bearing))
    y = distance * math.cos(math.radians(bearing))
    y -= instruction.steps * step_length
    turned = turn_degrees(instruction.turn) + turn_degrees(instruction.then)
    return (normalise(math.degrees(math.atan2(x, y)) - turned),
            max(math.hypot(x, y), 0.0))


def next_instruction(route, nav, after_turn: bool = False) -> Instruction:
    """The next spoken step along a planned route (see planner.Route).

    One action at a time: face the first leg, and only then walk it. The
    user walks the way they face, so the walk is limited to the distance
    that is clear straight ahead.

    `after_turn` says the user has just been turned. Nobody turns exactly
    30 degrees, so a second correction is asked for only if they are well
    off; otherwise a target near the edge of "straight ahead" would have
    them turned left, right, left for ever. Walking then re-planning takes
    care of the small error that is left.
    """
    if not route.legs:
        return Instruction("blocked")
    first = route.legs[0]
    deadband = nav.center_deadband_deg * (AFTER_TURN_SLACK if after_turn else 1.0)
    turn = turn_for(first.bearing, deadband)
    if turn:
        return Instruction("turn", turn=turn)

    step = nav.step_length_m
    more = len(route.legs) > 1  # another leg follows this one
    last = not more and route.reaches_goal  # this leg ends at the target
    # Steps the leg calls for, and steps there is room for. Real steps vary
    # in length, so only part of the measured room is used, and a long walk
    # is cut short and re-planned rather than trusted to the last step.
    wanted = int(first.length / step + 0.5)
    room = int(route.clear_ahead * STEP_MARGIN / step + 0.1)
    steps = min(wanted, room, MAX_STEPS)
    # Walking the way one faces instead of exactly along the leg drifts
    # sideways; stop and re-plan before the drift grows large.
    off_course = abs(math.sin(math.radians(first.bearing)))
    if off_course > 0 and steps > 1:
        steps = max(1, min(steps, int(MAX_DRIFT_M / off_course / step + 0.5)))
    if steps >= 1:
        then = None
        if steps == wanted and more:
            then = turn_for(normalise(route.legs[1].bearing - first.bearing),
                            nav.center_deadband_deg)
        return Instruction("move", steps=steps, then=then)

    if wanted < 1 and more:
        # Too short to walk: go straight to the turn that follows.
        following = turn_for(route.legs[1].bearing, nav.center_deadband_deg)
        if following:
            return Instruction("turn", turn=following)
        return Instruction("move", steps=1) if room >= 1 else Instruction("blocked")
    if last and first.length <= step:
        # Less than a step from the stopping point: close enough.
        return Instruction("arrived")
    if abs(first.bearing) >= NUDGE_DEG:
        # Facing almost the right way but brushing past something (often an
        # obstacle the user is standing beside): turn a little away from it,
        # to the side the route leans.
        side = "right" if first.bearing > 0 else "left"
        return Instruction("turn", turn=f"slight_{side}")
    return Instruction("blocked")
