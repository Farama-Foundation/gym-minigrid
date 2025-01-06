"""
Copied and adapted from https://github.com/mila-iqia/babyai.
Levels described in the Baby AI ICLR 2019 submission, with different instructions than those in other files.
"""

from __future__ import annotations

from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import (
    BeforeInstr,
    GoToInstr,
    ObjDesc,
    OpenInstr,
    PickupInstr,
    PutNextInstr,
)


class ActionObjDoor(RoomGridLevel):
    """

    ## Description

    [pick up an object] or
    [go to an object or door] or
    [open a door]
    (in the current room)

    ## Mission Space

    "pick up the {color} {type}"

    or

    "go to the {color} {type}"

    or

    "open a {color} door"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box", "door" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Open/closes door  |
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

    1. The agent finishes the instruction.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-ActionObjDoor-v0`

    """

    def __init__(self, **kwargs):
        super().__init__(room_size=7, **kwargs)

    def gen_mission(self):
        objs = self.add_distractors(1, 1, num_distractors=5)
        for _ in range(4):
            door, _ = self.add_door(1, 1, locked=False)
            objs.append(door)

        self.place_agent(1, 1)

        obj = self._rand_elem(objs)
        desc = ObjDesc(obj.type, obj.color)

        if obj.type == "door":
            if self._rand_bool():
                self.instrs = GoToInstr(desc)
            else:
                self.instrs = OpenInstr(desc)
        else:
            if self._rand_bool():
                self.instrs = GoToInstr(desc)
            else:
                self.instrs = PickupInstr(desc)


class FindObjS5(RoomGridLevel):
    """

    ## Description

    Pick up an object (in a random room)
    Rooms have a size of 5
    This level requires potentially exhaustive exploration

    ## Mission Space

    "pick up the {type}"

    {type} is the type of the object. Can be "ball", "box" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Open/closes door  |
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

    1. The agent picks up the object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-FindObjS5-v0`
    - `BabyAI-FindObjS6-v0`
    - `BabyAI-FindObjS7-v0`

    """

    def __init__(self, room_size=5, max_steps: int | None = None, **kwargs):
        if max_steps is None:
            max_steps = 20 * room_size**2

        super().__init__(room_size=room_size, max_steps=max_steps, **kwargs)

    def gen_mission(self):
        # Add a random object to a random room
        i = self._rand_int(0, self.num_rows)
        j = self._rand_int(0, self.num_cols)
        obj, _ = self.add_object(i, j)
        self.place_agent(1, 1)
        self.connect_all()

        self.instrs = PickupInstr(ObjDesc(obj.type))


class KeyCorridor(RoomGridLevel):
    """

    ## Description

    A ball is behind a locked door, the key is placed in a
    random room.

    ## Mission Space

    "pick up the ball"

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Open/closes door  |
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

    1. The agent picks up the ball.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-KeyCorridor-v0`
    - `BabyAI-KeyCorridorS3R1-v0`
    - `BabyAI-KeyCorridorS3R2-v0`
    - `BabyAI-KeyCorridorS3R3-v0`
    - `BabyAI-KeyCorridorS4R3-v0`
    - `BabyAI-KeyCorridorS5R3-v0`
    - `BabyAI-KeyCorridorS6R3-v0`

    """

    def __init__(
        self,
        num_rows=3,
        obj_type="ball",
        room_size=6,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.obj_type = obj_type

        if max_steps is None:
            max_steps = 30 * room_size**2

        super().__init__(
            room_size=room_size, num_rows=num_rows, max_steps=max_steps, **kwargs
        )

    def gen_mission(self):
        # Connect the middle column rooms into a hallway
        for j in range(1, self.num_rows):
            self.remove_wall(1, j, 3)

        # Add a locked door on the bottom right
        # Add an object behind the locked door
        room_idx = self._rand_int(0, self.num_rows)
        door, _ = self.add_door(2, room_idx, 2, locked=True)
        obj, _ = self.add_object(2, room_idx, kind=self.obj_type)

        # Add a key in a random room on the left side
        self.add_object(0, self._rand_int(0, self.num_rows), "key", door.color)

        # Place the agent in the middle
        self.place_agent(1, self.num_rows // 2)

        # Make sure all rooms are accessible
        self.connect_all()

        self.instrs = PickupInstr(ObjDesc(obj.type))


class OneRoomS8(RoomGridLevel):
    """

    ## Description

    Pick up the ball. Rooms have a size of 8.

    ## Mission Space

    "pick up the ball"

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
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

    1. The agent picks up the ball.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-OneRoomS8-v0`
    - `BabyAI-OneRoomS12-v0`
    - `BabyAI-OneRoomS16-v0`
    - `BabyAI-OneRoomS20-v0`

    """

    def __init__(self, room_size=8, **kwargs):
        super().__init__(room_size=room_size, num_rows=1, num_cols=1, **kwargs)

    def gen_mission(self):
        obj, _ = self.add_object(0, 0, kind="ball")
        self.place_agent()
        self.instrs = PickupInstr(ObjDesc(obj.type))


class MoveTwoAcross(RoomGridLevel):
    """

    ## Description

    Task of the form: move the A next to the B and the C next to the D.
    This task is structured to have a very large number of possible
    instructions.

    ## Mission Space

    "put the {color} {type} next to the {color} {type}, then put the {color} {type} next to the {color} {type}"

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
    | 4   | drop         | Unused            |
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

    1. The agent finishes the instruction.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-MoveTwoAcrossS5N2-v0`
    - `BabyAI-MoveTwoAcrossS8N9-v0`

    """

    def __init__(
        self, room_size, objs_per_room, max_steps: int | None = None, **kwargs
    ):
        assert objs_per_room <= 9
        self.objs_per_room = objs_per_room

        if max_steps is None:
            max_steps = 16 * room_size**2

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
        objs_l = self._rand_subset(objs_l, 2)
        objs_r = self._rand_subset(objs_r, 2)
        a = objs_l[0]
        b = objs_r[0]
        c = objs_r[1]
        d = objs_l[1]

        self.instrs = BeforeInstr(
            PutNextInstr(ObjDesc(a.type, a.color), ObjDesc(b.type, b.color)),
            PutNextInstr(ObjDesc(c.type, c.color), ObjDesc(d.type, d.color)),
        )
