"""
Copied and adapted from https://github.com/mila-iqia/babyai.
Levels described in the Baby AI ICLR 2019 submission, with the `Put Next` instruction.
"""

from __future__ import annotations

from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import ObjDesc, PutNextInstr


class PutNextLocal(RoomGridLevel):
    """

    ## Description

    Put an object next to another object, inside a single room
    with no doors, no distractors

    ## Mission Space

    "put the {color} {type} next to the {color} {type}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Drops held item   |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent finishes the instructed task.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-PutNextLocal-v0`
    - `BabyAI-PutNextLocalS5N3-v0`
    - `BabyAI-PutNextLocalS6N4-v0``

    """

    def __init__(self, room_size=8, num_objs=8, **kwargs):
        self.num_objs = num_objs
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_objs, all_unique=True)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)

        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color), ObjDesc(o2.type, o2.color)
        )


class PutNext(RoomGridLevel):
    """

    ## Description

    Task of the form: move the A next to the B and the C next to the D.
    This task is structured to have a very large number of possible
    instructions.

    ## Mission Space

    "put the {color} {type} next to the {color} {type}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Drops held item   |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent finishes the instructed task.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-PutNextS4N1-v0`
    - `BabyAI-PutNextS5N2-v0`
    - `BabyAI-PutNextS5N1-v0`
    - `BabyAI-PutNextS6N3-v0`
    - `BabyAI-PutNextS7N4-v0`
    - `BabyAI-PutNextS5N2Carrying-v0`
    - `BabyAI-PutNextS6N3Carrying-v0`
    - `BabyAI-PutNextS7N4Carrying-v0`

    ## Additional Notes

    The BabyAI bot is unable to solve the bonus PutNextCarrying configurations.
    """

    def __init__(
        self,
        room_size,
        objs_per_room,
        start_carrying=False,
        max_steps: int | None = None,
        **kwargs,
    ):
        assert room_size >= 4
        assert objs_per_room <= 9
        self.objs_per_room = objs_per_room
        self.start_carrying = start_carrying

        if max_steps is None:
            max_steps = 8 * room_size**2

        super().__init__(
            num_rows=1, num_cols=2, room_size=room_size, max_steps=max_steps, **kwargs
        )

    def gen_mission(self):
        self.place_agent(0, 0)

        # Add objects to both the left and right rooms
        # so that we know that we have two non-adjacent set of objects
        objs_l = self.add_distractors(0, 0, self.objs_per_room)
        objs_r = self.add_distractors(1, 0, self.objs_per_room)

        # Remove the wall between the two rooms
        self.remove_wall(0, 0, 0)

        # Select objects from both subsets
        a = self._rand_elem(objs_l)
        b = self._rand_elem(objs_r)

        # Randomly flip the object to be moved
        if self._rand_bool():
            t = a
            a = b
            b = t

        self.obj_a = a

        self.instrs = PutNextInstr(ObjDesc(a.type, a.color), ObjDesc(b.type, b.color))

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        # If the agent starts off carrying the object
        if self.start_carrying:
            assert self.obj_a.init_pos is not None
            self.grid.set(*self.obj_a.init_pos, None)
            self.carrying = self.obj_a

        return obs
