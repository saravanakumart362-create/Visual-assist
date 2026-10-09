"""Path planning: routes are short, safe, and spoken one step at a time."""

import math

import pytest

from visual_assist import config
from visual_assist.navigator import Instruction, next_instruction
from visual_assist.planner import (
    Leg, Obstacle, Route, obstacles_from_rays, plan, to_xy,
)

NAV = config.NavigationConfig()
PLANNER = config.PlannerConfig()


def points(route):
    """The corners of a route, starting at the user."""
    x = y = 0.0
    corners = [(x, y)]
    for leg in route.legs:
        dx, dy = to_xy(leg.bearing, leg.length)
        x, y = x + dx, y + dy
        corners.append((x, y))
    return corners


def clearance(route, obstacles):
    """Closest the route comes to the surface of any obstacle."""
    closest = math.inf
    corners = points(route)
    for (ax, ay), (bx, by) in zip(corners, corners[1:]):
        for k in range(101):
            px, py = ax + (bx - ax) * k / 100, ay + (by - ay) * k / 100
            for o in obstacles:
                closest = min(closest, math.hypot(px - o.x, py - o.y) - o.radius)
    return closest


def test_open_floor_gives_one_straight_leg():
    route = plan(20.0, 4.0, [], NAV, PLANNER)
    assert route.direct and route.reaches_goal and route.blocker is None
    assert len(route.legs) == 1
    assert route.legs[0].bearing == pytest.approx(20.0, abs=3)
    assert route.legs[0].length == pytest.approx(4.0 - NAV.arrive_m, abs=0.2)


def test_goes_around_an_obstacle_and_keeps_a_body_width_away():
    chair = Obstacle(0.0, 2.0, 0.3, "chair")
    route = plan(0.0, 4.0, [chair], NAV, PLANNER)
    assert not route.direct and route.blocker == "chair"
    assert route.reaches_goal and len(route.legs) >= 2
    assert clearance(route, [chair]) >= PLANNER.clearance_m - PLANNER.resolution_m
    # The detour should cost little more than the straight line.
    assert sum(leg.length for leg in route.legs) < (4.0 - NAV.arrive_m) + 0.6
    end = points(route)[-1]
    assert math.hypot(end[0], end[1] - 4.0) <= NAV.arrive_m + PLANNER.resolution_m


def test_finds_the_gap_in_a_wall():
    # A wall 2 m ahead with one opening, 1.2 m wide, off to the right.
    wall = [Obstacle(x / 10, 2.0, 0.1) for x in range(-30, 31, 2)
            if not 9 <= x <= 21]
    route = plan(0.0, 4.0, wall, NAV, PLANNER)
    assert route.reaches_goal and not route.direct and route.blocker == ""
    corners = points(route)
    (ax, ay), (bx, by) = next(pair for pair in zip(corners, corners[1:])
                              if pair[0][1] <= 2.0 <= pair[1][1])
    crossing = ax + (bx - ax) * (2.0 - ay) / (by - ay)
    assert 0.9 < crossing < 2.1
    assert clearance(route, wall) >= PLANNER.clearance_m - PLANNER.resolution_m


def test_no_route_when_walled_in():
    ring = [Obstacle(0.8 * math.sin(a / 10), 0.8 * math.cos(a / 10), 0.2)
            for a in range(0, 63, 2)]
    route = plan(0.0, 4.0, ring, NAV, PLANNER)
    assert route.legs == () and not route.reaches_goal
    assert next_instruction(route, NAV).kind == "blocked"


def test_heads_for_the_nearest_reachable_point_when_the_goal_is_cut_off():
    # A wide wall with no gap: the route gets as close as it can, then stops.
    wall = [Obstacle(x / 10, 3.0, 0.1) for x in range(-80, 81, 2)]
    route = plan(0.0, 5.0, wall, NAV, PLANNER)
    assert route.legs and not route.reaches_goal
    assert all(y < 3.0 - PLANNER.clearance_m for _, y in points(route))


def test_the_targets_own_surface_does_not_block_the_way_to_it():
    # LiDAR hits on the door itself and the wall it is set in.
    door_wall = [Obstacle(x / 10, 3.0, 0.1) for x in range(-20, 21, 2)]
    route = plan(0.0, 3.0, door_wall, NAV, PLANNER)
    assert route.reaches_goal
    assert len(route.legs) == 1 and route.legs[0].bearing == pytest.approx(0, abs=3)


def test_lidar_hits_on_one_wall_join_up():
    # Rays 6 degrees apart on a wall 3 m away are 0.31 m apart: too far for
    # 0.1 m posts to stop a route slipping between them, so each hit is widened.
    rays = [(b, 3.0 / math.cos(math.radians(b))) for b in range(-54, 55, 6)]
    obstacles = obstacles_from_rays(rays, 6.0, PLANNER.max_range_m)
    seen_to = 3.0 * math.tan(math.radians(54))  # the wall is known out to here
    route = plan(0.0, 5.0, obstacles, NAV, PLANNER)
    corners = points(route)
    for (ax, ay), (bx, by) in zip(corners, corners[1:]):
        if ay <= 3.0 <= by:  # where the route passes the wall line...
            crossing = ax + (bx - ax) * (3.0 - ay) / (by - ay)
            assert abs(crossing) > seen_to  # ...it is beyond the part that was seen
    assert route.clear_ahead == pytest.approx(3.0 - PLANNER.clearance_m, abs=0.3)
    assert obstacles_from_rays([(0, None), (5, 9.0)], 6.0, PLANNER.max_range_m) == []


def test_an_object_edge_is_not_joined_to_the_wall_behind_it():
    # Neighbouring rays: one on a chair 2 m away, the next on a wall 5 m away.
    obstacles = obstacles_from_rays([(0, 2.0), (6, 5.0)], 6.0, PLANNER.max_range_m)
    assert len(obstacles) == 2  # nothing filled in: the floor beside the chair is open


def test_instructions_face_the_route_before_walking_it():
    route = Route((Leg(40.0, 2.0), Leg(-10.0, 2.0)), True, clear_ahead=5.0)
    assert next_instruction(route, NAV) == Instruction("turn", turn="slight_right")


def test_instructions_walk_then_announce_the_next_turn():
    route = Route((Leg(5.0, 1.8), Leg(-70.0, 2.0)), True, clear_ahead=5.0)
    assert next_instruction(route, NAV) == Instruction("move", steps=3, then="left")


def test_walk_is_limited_to_what_is_clear_straight_ahead():
    # The route bears 10 degrees right for 3 m, but straight ahead only 1.5 m
    # is free: two steps (1.2 m), with some of the measured room kept in hand.
    route = Route((Leg(10.0, 3.0),), True, clear_ahead=1.5)
    assert next_instruction(route, NAV) == Instruction("move", steps=2)
    # With nothing free ahead, nudge towards the route instead of walking.
    route = Route((Leg(10.0, 3.0),), True, clear_ahead=0.1)
    assert next_instruction(route, NAV) == Instruction("turn", turn="slight_right")


def test_a_long_walk_is_cut_short_and_replanned():
    # Step lengths vary, so a 6 m leg is not trusted to one instruction...
    route = Route((Leg(0.0, 6.0), Leg(90.0, 2.0)), True, clear_ahead=8.0)
    assert next_instruction(route, NAV) == Instruction("move", steps=5)
    # ...and facing 14 degrees off the leg, even less: drift grows with distance.
    route = Route((Leg(14.0, 6.0),), True, clear_ahead=8.0)
    assert next_instruction(route, NAV) == Instruction("move", steps=4)


def test_no_whole_step_left_before_the_target_counts_as_arriving():
    # 0.3 m short of the stopping point, and the target's surface leaves no
    # room for a full step: that is as close as is sensible.
    route = Route((Leg(0.0, 0.3),), True, clear_ahead=0.4)
    assert next_instruction(route, NAV) == Instruction("arrived")


def test_a_route_that_stops_short_of_the_target_is_still_followed():
    # One leg that gets nearer but cannot reach (the way beyond is unknown).
    partial = Route((Leg(0.0, 1.8),), reaches_goal=False, clear_ahead=5.0)
    assert next_instruction(partial, NAV) == Instruction("move", steps=3)
    stub = Route((Leg(0.0, 0.2),), reaches_goal=False, clear_ahead=5.0)
    assert next_instruction(stub, NAV) == Instruction("blocked")


def test_after_a_turn_small_errors_are_walked_off_not_corrected():
    route = Route((Leg(20.0, 1.2),), True, clear_ahead=5.0)
    assert next_instruction(route, NAV).kind == "turn"
    assert next_instruction(route, NAV, after_turn=True) == Instruction("move", steps=2)


def test_keep_straight_starts_ahead_when_that_costs_little():
    # The target is 40 degrees to the right on open floor. The shortest route
    # turns to it at once; asked to keep straight, the route walks ahead
    # first and turns later, for under half a metre more.
    shortest = plan(40.0, 4.0, [], NAV, PLANNER)
    steady = plan(40.0, 4.0, [], NAV, PLANNER, keep_straight=True)
    assert shortest.legs[0].bearing == pytest.approx(40.0, abs=3)
    assert steady.legs[0].bearing == pytest.approx(0.0, abs=1) and len(steady.legs) == 2
    assert steady.reaches_goal
    extra = sum(l.length for l in steady.legs) - sum(l.length for l in shortest.legs)
    assert 0 < extra <= 0.5


def test_keep_straight_gives_way_when_ahead_is_blocked():
    wall = [Obstacle(x / 10, 0.8, 0.1) for x in range(-10, 11, 2)]
    steady = plan(60.0, 3.0, wall, NAV, PLANNER, keep_straight=True)
    assert steady.reaches_goal and abs(steady.legs[0].bearing) > 30
