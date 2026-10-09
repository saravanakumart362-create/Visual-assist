"""Shortest safe path to a target, around whatever is in the way.

A LiDAR sweep and the camera's detections become round obstacles on a
small map of the floor in front of the user. A* then finds the shortest
route that keeps a body's width away from all of them.

The map has the user at (0, 0) facing +y, with +x to the right; distances
are metres. Space nobody has measured is treated as free: the route is
re-planned from a fresh sweep after every instruction, and the user is
only ever told to walk in a direction the sweep has just covered.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable, Optional

SOFT_MARGIN_M = 0.3  # extra room the route takes when it costs little
SOFT_PENALTY = 2.0
RAY_GAP_CAP_M = 0.25
SAME_SURFACE = 2.5  # neighbouring hits this many ray-widths apart are one surface
JOIN_SPACING_M = 0.2
MAX_EXPANSIONS = 40000
MIN_PROGRESS_M = 0.3  # a partial route must get at least this much closer
TARGET_CLEARANCE_M = 0.15  # how near the target's own surface a walk may end
STRAIGHT_BONUS_M = 0.5  # extra length accepted to keep going the way one faces
MAX_STRAIGHT_STEPS = 5

_NEIGHBOURS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float
    label: str = ""  # empty for something only the LiDAR saw


@dataclass(frozen=True)
class Leg:
    bearing: float  # direction of this stretch, relative to where the user faces now
    length: float


@dataclass(frozen=True)
class Route:
    legs: tuple = ()  # empty when there is no way forward
    reaches_goal: bool = False
    clear_ahead: float = 0.0  # metres that are free straight ahead
    direct: bool = False  # the straight line to the target is free
    blocker: Optional[str] = None  # what blocks that line ("" = unidentified)


def to_xy(bearing: float, distance: float):
    angle = math.radians(bearing)
    return distance * math.sin(angle), distance * math.cos(angle)


def obstacles_from_rays(rays: Iterable, step_deg: float, max_range: float) -> list:
    """One obstacle per LiDAR hit, wide enough to join up with its neighbours.

    `rays` is a list of (bearing, distance or None). Two hits on the same
    wall are `distance * step` apart, so each is given half that as a
    radius; otherwise a wall would look like a row of posts to walk between.
    """
    gap = math.tan(math.radians(step_deg) / 2.0)
    obstacles = []
    previous = None  # the hit on the neighbouring ray, if it had one
    for bearing, distance in sorted(rays, key=lambda ray: ray[0]):
        if distance is None or distance > max_range:
            previous = None
            continue
        x, y = to_xy(bearing, distance)
        hit = Obstacle(x, y, min(distance * gap, RAY_GAP_CAP_M))
        if previous is not None:
            obstacles.extend(_join(previous, hit, step_deg))
        obstacles.append(hit)
        previous = hit
    return obstacles


def _join(a: Obstacle, b: Obstacle, step_deg: float) -> list:
    """Fill the space between two neighbouring hits on the same surface.

    A wall seen at a slant gives hits much further apart than one seen
    head-on. Hits are taken to be one surface when they are no further
    apart than a slanted wall would put them; a bigger jump is the edge of
    an object with open floor beside it, and is left open.
    """
    apart = math.hypot(b.x - a.x, b.y - a.y)
    mean_range = (math.hypot(a.x, a.y) + math.hypot(b.x, b.y)) / 2.0
    if apart > SAME_SURFACE * mean_range * math.radians(step_deg):
        return []
    pieces = int(apart / JOIN_SPACING_M)
    radius = min(a.radius, b.radius)
    return [Obstacle(a.x + (b.x - a.x) * k / (pieces + 1),
                     a.y + (b.y - a.y) * k / (pieces + 1), radius)
            for k in range(1, pieces + 1)]


def obstacle_from_sighting(sighting) -> Optional[Obstacle]:
    """A detected object as an obstacle, sized from how wide it looks."""
    if sighting.distance is None:
        return None
    half_width = sighting.distance * math.tan(math.radians(min(sighting.width, 90.0)) / 2.0)
    radius = min(max(half_width, 0.1), 0.75)
    # The distance is to the near face, so the middle is one radius further.
    x, y = to_xy(sighting.bearing, sighting.distance + radius)
    return Obstacle(x, y, radius, sighting.label)


class _Grid:
    def __init__(self, half_width: float, depth: float, back: float, res: float):
        self.res = res
        self.x0, self.y0 = -half_width, -back
        self.nx = int(round(2 * half_width / res)) + 1
        self.ny = int(round((depth + back) / res)) + 1
        self.blocked = bytearray(self.nx * self.ny)
        self.extra = [0.0] * (self.nx * self.ny)

    def copy(self) -> "_Grid":
        """A second map of the same floor that can be blocked independently."""
        twin = object.__new__(_Grid)
        twin.__dict__.update(self.__dict__)
        twin.blocked = bytearray(self.blocked)
        twin.extra = list(self.extra)
        return twin

    def index(self, x: float, y: float):
        return int(round((x - self.x0) / self.res)), int(round((y - self.y0) / self.res))

    def xy(self, i: int, j: int):
        return self.x0 + i * self.res, self.y0 + j * self.res

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny

    def is_blocked(self, i: int, j: int) -> bool:
        return not self.inside(i, j) or bool(self.blocked[j * self.nx + i])

    def _disc(self, x: float, y: float, radius: float):
        i0, j0 = self.index(x - radius, y - radius)
        i1, j1 = self.index(x + radius, y + radius)
        for j in range(max(j0, 0), min(j1, self.ny - 1) + 1):
            for i in range(max(i0, 0), min(i1, self.nx - 1) + 1):
                cx, cy = self.xy(i, j)
                yield j * self.nx + i, math.hypot(cx - x, cy - y)

    def stamp(self, obstacle: Obstacle, clearance: float) -> None:
        hard = obstacle.radius + clearance
        for cell, distance in self._disc(obstacle.x, obstacle.y, hard + SOFT_MARGIN_M):
            if distance <= hard:
                self.blocked[cell] = 1
            else:
                penalty = SOFT_PENALTY * (1.0 - (distance - hard) / SOFT_MARGIN_M)
                self.extra[cell] = max(self.extra[cell], penalty)

    def clear(self, x: float, y: float, radius: float) -> None:
        for cell, distance in self._disc(x, y, radius):
            if distance <= radius:
                self.blocked[cell] = 0

    def free_line(self, a, b) -> bool:
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        samples = max(1, int(math.ceil(length / (self.res / 2.0))))
        for k in range(samples + 1):
            t = k / samples
            if self.is_blocked(*self.index(a[0] + (b[0] - a[0]) * t,
                                           a[1] + (b[1] - a[1]) * t)):
                return False
        return True


def _search(grid: _Grid, target, arrive: float, origin=(0.0, 0.0)):
    """A* from `origin` (normally the user). Returns (cells, reached_goal)."""
    tx, ty = target

    def remaining(node) -> float:
        x, y = grid.xy(*node)
        return max(0.0, math.hypot(x - tx, y - ty) - arrive)

    start = grid.index(*origin)
    cost = {start: 0.0}
    came = {}
    start_remaining = remaining(start)
    best = (start_remaining, start)
    frontier = [(start_remaining, 0.0, start)]
    goal = None
    expansions = 0
    while frontier and expansions < MAX_EXPANSIONS:
        _, spent, node = heapq.heappop(frontier)
        if spent > cost.get(node, math.inf):
            continue
        expansions += 1
        left = remaining(node)
        if left <= 0.0:
            goal = node
            break
        if left < best[0]:
            best = (left, node)
        i, j = node
        for di, dj in _NEIGHBOURS:
            nxt = (i + di, j + dj)
            if grid.is_blocked(*nxt):
                continue
            # No squeezing diagonally between two blocked cells.
            if di and dj and (grid.is_blocked(i + di, j) or grid.is_blocked(i, j + dj)):
                continue
            step = grid.res * math.hypot(di, dj) * (1.0 + grid.extra[nxt[1] * grid.nx + nxt[0]])
            total = spent + step
            if total < cost.get(nxt, math.inf):
                cost[nxt] = total
                came[nxt] = node
                heapq.heappush(frontier, (total + remaining(nxt), total, nxt))

    if goal is None:
        # No way through: settle for the closest reachable point, if it helps.
        if start_remaining - best[0] < MIN_PROGRESS_M:
            return [], False
        goal = best[1]
    cells = [goal]
    while cells[-1] in came:
        cells.append(came[cells[-1]])
    cells.reverse()
    return cells, remaining(goal) <= 0.0


def _straighten(grid: _Grid, points: list) -> list:
    """Replace the grid's staircase with the fewest straight stretches."""
    result = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not grid.free_line(points[i], points[j]):
            j -= 1
        result.append(points[j])
        i = j
    return result


def _blocker(obstacles: list, end, clearance: float) -> str:
    """Label of the nearest obstacle on the straight line from the user to `end`."""
    length = math.hypot(*end)
    if length == 0:
        return ""
    ux, uy = end[0] / length, end[1] / length
    nearest = {}
    for obstacle in obstacles:
        along = obstacle.x * ux + obstacle.y * uy
        across = abs(obstacle.x * uy - obstacle.y * ux)
        reach = obstacle.radius + clearance
        if -reach <= along <= length + reach and across <= reach:
            named = bool(obstacle.label)
            if named not in nearest or along < nearest[named][0]:
                nearest[named] = (along, obstacle.label)
    # A name from the camera is more use to the listener than "something".
    return nearest[True][1] if True in nearest else ""


def _route_points(grid: _Grid, target, arrive: float, origin=(0.0, 0.0)):
    """Corner points of the shortest route from `origin`. Returns (points, reached)."""
    cells, reached = _search(grid, target, arrive, origin)
    points = [origin] + [grid.xy(*cell) for cell in cells[1:]]
    return _straighten(grid, points), reached


def _length(points: list) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def _straight_first(grid: _Grid, target, arrive: float, step: float,
                    clear_ahead: float, limit: float):
    """A route that starts by walking straight ahead, no longer than `limit`.

    The longest straight start that fits is chosen: fewer, longer walks are
    easier to follow than a step and a turn at a time.
    """
    for count in range(MAX_STRAIGHT_STEPS, 0, -1):
        ahead = count * step
        if ahead > clear_ahead:
            continue
        rest, reached = _route_points(grid, target, arrive, (0.0, ahead))
        points = [(0.0, 0.0)] + rest
        if reached and _length(points) <= limit:
            return points
    return None


def plan(goal_bearing: float, goal_distance: float, obstacles: Iterable,
         nav, planner, keep_straight: bool = False) -> Route:
    """Shortest route to within `nav.arrive_m` of the target.

    `nav` is a NavigationConfig and `planner` a PlannerConfig.

    `keep_straight` asks for a route that begins by walking straight ahead,
    provided one exists that is not clearly longer than the shortest. Two
    ways round an obstacle are often nearly the same length, and which is
    shorter can change with every new sweep; a user just turned to face one
    should not be turned back to the other for the sake of a few centimetres.
    """
    res, clearance, arrive = planner.resolution_m, planner.clearance_m, nav.arrive_m
    obstacles = list(obstacles)
    target = to_xy(goal_bearing, goal_distance)
    reach = max(abs(target[0]), abs(target[1]), 1.0) + 1.5
    grid = _Grid(half_width=reach, depth=reach, back=1.5, res=res)

    # The target's own surface must not wall off the spot we are heading for.
    solid, surface = [], []
    for obstacle in obstacles:
        near = math.hypot(obstacle.x - target[0], obstacle.y - target[1]) <= arrive
        (surface if near else solid).append(obstacle)
    for obstacle in solid:
        grid.stamp(obstacle, clearance)
    grid.clear(0.0, 0.0, clearance)  # the user is standing here

    # How far the user could walk straight ahead. This does count the
    # target's own surface: the route may end beside the target, but nobody
    # is to be walked into it. The target may be approached more closely
    # than an obstacle is passed, since touching distance is the point.
    everything = grid.copy()
    for obstacle in surface:
        everything.stamp(obstacle, TARGET_CLEARANCE_M)
    ahead = 0.0
    limit = min(planner.max_range_m, reach)
    while ahead < limit and (ahead <= clearance
                             or not everything.is_blocked(*everything.index(0.0, ahead))):
        ahead += res / 2.0
    clear_ahead = max(0.0, ahead - res)

    approach = to_xy(goal_bearing, max(goal_distance - arrive, 0.0))
    direct = grid.free_line((0.0, 0.0), approach)
    blocker = None if direct else _blocker(solid, approach, clearance)

    if goal_distance <= arrive:
        return Route((), True, clear_ahead, direct, blocker)  # already there
    if direct:
        # Nothing in the way: the shortest route is the straight line itself.
        points, reached = [(0.0, 0.0), approach], True
    else:
        points, reached = _route_points(grid, target, arrive)
    if len(points) < 2:
        return Route((), reached, clear_ahead, direct, blocker)  # nowhere to go
    if keep_straight and reached:
        straight = _straight_first(grid, target, arrive, nav.step_length_m, clear_ahead,
                                   _length(points) + STRAIGHT_BONUS_M)
        if straight:
            points = straight
            # If the stretch after the first is nearly in line with it, they
            # are one walk, not "one step" followed by "three more steps".
            while len(points) > 2:
                bearing = math.degrees(math.atan2(points[2][0], points[2][1]))
                if abs(bearing) > nav.center_deadband_deg or not grid.free_line(
                        points[0], points[2]):
                    break
                del points[1]

    legs = []
    for a, b in zip(points, points[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length >= res / 2.0:
            legs.append(Leg(math.degrees(math.atan2(dx, dy)), length))
    return Route(tuple(legs), reached, clear_ahead, direct, blocker)
