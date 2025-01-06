"""
Copied and adapted from https://github.com/mila-iqia/babyai.
Levels described in the Baby AI ICLR 2019 submission, with the `Open` instruction.
"""

from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import (
    LOC_NAMES,
    AfterInstr,
    BeforeInstr,
    ObjDesc,
    OpenInstr,
)


class Open(RoomGridLevel):
    """

    ## Description

    Open a door, which may be in another room

    ## Mission Space

    "open a {color} door"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

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

    1. The agent opens the door.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-Open-v0`

    """

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()

        # Collect a list of all the doors in the environment
        doors = []
        for i in range(self.num_cols):
            for j in range(self.num_rows):
                room = self.get_room(i, j)
                for door in room.doors:
                    if door:
                        doors.append(door)

        door = self._rand_elem(doors)
        self.instrs = OpenInstr(ObjDesc(door.type, door.color))


class OpenRedDoor(RoomGridLevel):
    """

    ## Description

    Go to the red door
    (always unlocked, in the current room)
    Note: this level is intentionally meant for debugging and is
    intentionally kept very simple.

    ## Mission Space

    "open the red door"

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

    1. The agent opens the door.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-OpenRedDoor-v0`

    """

    def __init__(self, **kwargs):
        super().__init__(num_rows=1, num_cols=2, room_size=5, **kwargs)

    def gen_mission(self):
        obj, _ = self.add_door(0, 0, 0, "red", locked=False)
        self.place_agent(0, 0)
        self.instrs = OpenInstr(ObjDesc("door", "red"))


class OpenDoor(RoomGridLevel):
    """

    ## Description

    Go to the door
    The door to open is given by its color or by its location.
    (always unlocked, in the current room)

    ## Mission Space

    "open the {color} door"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

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

    1. The agent opens the door.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-OpenDoor-v0`
    - `BabyAI-OpenDoorDebug-v0`
    - `BabyAI-OpenDoorColor-v0`
    - `BabyAI-OpenDoorLoc-v0`

    """

    def __init__(self, debug=False, select_by=None, **kwargs):
        self.select_by = select_by
        self.debug = debug
        super().__init__(**kwargs)

    def gen_mission(self):
        door_colors = self._rand_subset(COLOR_NAMES, 4)
        objs = []

        for i, color in enumerate(door_colors):
            obj, _ = self.add_door(1, 1, door_idx=i, color=color, locked=False)
            objs.append(obj)

        select_by = self.select_by
        if select_by is None:
            select_by = self._rand_elem(["color", "loc"])
        if select_by == "color":
            object = ObjDesc(objs[0].type, color=objs[0].color)
        elif select_by == "loc":
            object = ObjDesc(objs[0].type, loc=self._rand_elem(LOC_NAMES))
        else:
            raise NotImplementedError("Not implemented.")

        self.place_agent(1, 1)
        self.instrs = OpenInstr(object, strict=self.debug)


class OpenTwoDoors(RoomGridLevel):
    """

    ## Description

    Open door X, then open door Y
    The two doors are facing opposite directions, so that the agent
    Can't see whether the door behind him is open.
    This task requires memory (recurrent policy) to be solved effectively.

    ## Mission Space

    "open the {color} door, then open the {color} door"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

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

    1. The agent opens the door.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-OpenTwoDoors-v0`
    - `BabyAI-OpenRedBlueDoors-v0`
    - `BabyAI-OpenRedBlueDoorsDebug-v0`

    """

    def __init__(
        self,
        first_color=None,
        second_color=None,
        strict=False,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.first_color = first_color
        self.second_color = second_color
        self.strict = strict

        room_size = 6
        if max_steps is None:
            max_steps = 20 * room_size**2

        super().__init__(room_size=room_size, max_steps=max_steps, **kwargs)

    def gen_mission(self):
        colors = self._rand_subset(COLOR_NAMES, 2)

        first_color = self.first_color
        if first_color is None:
            first_color = colors[0]
        second_color = self.second_color
        if second_color is None:
            second_color = colors[1]

        door1, _ = self.add_door(1, 1, 2, color=first_color, locked=False)
        door2, _ = self.add_door(1, 1, 0, color=second_color, locked=False)

        self.place_agent(1, 1)

        self.instrs = BeforeInstr(
            OpenInstr(ObjDesc(door1.type, door1.color), strict=self.strict),
            OpenInstr(ObjDesc(door2.type, door2.color)),
        )


class OpenDoorsOrder(RoomGridLevel):
    """

    ## Description

    Open one or two doors in the order specified.

    ## Mission Space

    "open the {color} door, the open the {color} door"

    or

    "open the {color} door after you open the {color} door"

    or

    "open the {color} door"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

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

    1. The agent opens the door.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-OpenDoorsOrderN2-v0`
    - `BabyAI-OpenDoorsOrderN4-v0`
    - `BabyAI-OpenDoorsOrderN2Debug-v0`
    - `BabyAI-OpenDoorsOrderN4Debug-v0`
    """

    def __init__(self, num_doors, debug=False, max_steps: int | None = None, **kwargs):
        assert num_doors >= 2
        self.num_doors = num_doors
        self.debug = debug

        room_size = 6
        if max_steps is None:
            max_steps = 20 * room_size**2

        super().__init__(room_size=room_size, max_steps=max_steps, **kwargs)

    def gen_mission(self):
        colors = self._rand_subset(COLOR_NAMES, self.num_doors)
        doors = []
        for i in range(self.num_doors):
            door, _ = self.add_door(1, 1, color=colors[i], locked=False)
            doors.append(door)
        self.place_agent(1, 1)

        door1, door2 = self._rand_subset(doors, 2)
        desc1 = ObjDesc(door1.type, door1.color)
        desc2 = ObjDesc(door2.type, door2.color)

        mode = self._rand_int(0, 3)
        if mode == 0:
            self.instrs = OpenInstr(desc1, strict=self.debug)
        elif mode == 1:
            self.instrs = BeforeInstr(
                OpenInstr(desc1, strict=self.debug), OpenInstr(desc2, strict=self.debug)
            )
        elif mode == 2:
            self.instrs = AfterInstr(
                OpenInstr(desc1, strict=self.debug), OpenInstr(desc2, strict=self.debug)
            )
        else:
            assert False
