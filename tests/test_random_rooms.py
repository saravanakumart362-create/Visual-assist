"""Guidance through many random cluttered rooms.

Two promises are checked. Safety is absolute: whatever the layout, and
however roughly the user follows instructions, they are never walked into
anything. Success is statistical: nearly every room is solved, and a room
that is not ends with the assistant saying so, not with the user stuck.
"""

import math
import random

import pytest

from visual_assist.simulation import cluttered_room, follow_guidance, make

ROOMS = 60


def walk_rooms(seed: int, imprecise: bool):
    rng = random.Random(seed)
    reached = 0
    for _ in range(ROOMS):
        door, blocks = cluttered_room(rng)
        assistant, world, speaker, clock = make({"door": door}, blocks=blocks)
        assistant.submit("guide me to the door")
        cycles = follow_guidance(assistant, world, clock, cycles=60,
                                 rng=rng if imprecise else None)

        assert world.closest_obstacle > 0.1, "walked into an obstacle"
        assert world.closest > 0.05, "walked into the door"
        assert cycles < 60, "guidance never finished"
        if speaker.said[-1].startswith("You have reached"):
            to_door = math.hypot(door[0] - world.x, door[1] - world.y) - door[2]
            assert to_door <= 1.2, "said 'reached' while still far away"
            reached += 1
        else:
            assert "Guidance stopped." in speaker.said[-1]
    return reached


@pytest.mark.parametrize("seed", [1, 2])
def test_exact_follower_reaches_the_door_safely(seed):
    assert walk_rooms(seed, imprecise=False) >= 0.95 * ROOMS


@pytest.mark.parametrize("seed", [3, 4])
def test_imprecise_follower_reaches_the_door_safely(seed):
    assert walk_rooms(seed, imprecise=True) >= 0.9 * ROOMS
