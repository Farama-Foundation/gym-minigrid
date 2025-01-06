"""
Copied and adapted from https://github.com/mila-iqia/babyai.
Levels described in the Baby AI ICLR 2019 submission, with the `Go to` instruction.
"""

from __future__ import annotations

from minigrid.envs.babyai.core.levelgen import LevelGen
from minigrid.envs.babyai.core.roomgrid_level import RejectSampling, RoomGridLevel
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc


class GoToRedBallGrey(RoomGridLevel):
    """

    ## Description

    Go to the red ball, single room, with distractors.
    The distractors are all grey to reduce perceptual complexity.
    This level has distractors but doesn't make use of language.

    ## Mission Space

    "go to the red ball"

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

    1. The agent goes to the red ball.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoToRedBallGrey-v0`

    """

    def __init__(self, room_size=8, num_dists=7, **kwargs):
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, "ball", "red")
        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        for dist in dists:
            dist.color = "grey"

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class GoToRedBall(RoomGridLevel):
    """
    ## Description

    Go to the red ball, single room, with distractors.
    This level has distractors but doesn't make use of language.

    ## Mission Space

    "go to the red ball"

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

    1. The agent goes to the red ball.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoToRedBall-v0`

    """

    def __init__(self, room_size=8, num_dists=7, **kwargs):
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, "ball", "red")
        self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class GoToRedBallNoDists(GoToRedBall):
    """

    ## Description

    Go to the red ball. No distractors present.

    ## Mission Space

    "go to the red ball"

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

    1. The agent goes to the red ball.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoToRedBallNoDists-v0`

    """

    def __init__(self, **kwargs):
        super().__init__(room_size=8, num_dists=0, **kwargs)


class GoToObj(RoomGridLevel):
    """
    ## Description

    Go to an object, inside a single room with no doors, no distractors. The
    naming convention `GoToObjS{X}` represents a room of size `X`.

    ## Mission Space

    "go to the {color} {type}"

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

    1. The agent goes to the object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoToObj-v0`
    - `BabyAI-GoToObjS4-v0`
    - `BabyAI-GoToObjS6-v1`

    Notice: `BabyAI-GoToObjS6-v0` is no longer for use due to a bug in the registry parameters.

    """

    def __init__(self, room_size=8, **kwargs):
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=1)
        obj = objs[0]
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class GoToLocal(RoomGridLevel):
    """

    ## Description

    Go to an object, inside a single room with no doors, no distractors. The
    naming convention `GoToLocalS{X}N{Y}` represents a room of size `X` with
    distractor number `Y`.

    ## Mission Space

    "go to the {color} {type}"

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

    1. The agent goes to the object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoToLocal-v0`
    - `BabyAI-GoToLocalS5N2-v0`
    - `BabyAI-GoToLocalS6N2-v0`
    - `BabyAI-GoToLocalS6N3-v0`
    - `BabyAI-GoToLocalS6N4-v0`
    - `BabyAI-GoToLocalS7N4-v0`
    - `BabyAI-GoToLocalS7N5-v0`
    - `BabyAI-GoToLocalS8N2-v0`
    - `BabyAI-GoToLocalS8N3-v0`
    - `BabyAI-GoToLocalS8N4-v0`
    - `BabyAI-GoToLocalS8N5-v0`
    - `BabyAI-GoToLocalS8N6-v0`
    - `BabyAI-GoToLocalS8N7-v0`
    """

    def __init__(self, room_size=8, num_dists=8, **kwargs):
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class GoTo(RoomGridLevel):
    """

    ## Description

    Go to an object, the object may be in another room. Many distractors.

    ## Mission Space

    "go to a/the {color} {type}"

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

    1. The agent goes to the object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoTo-v0`
    - `BabyAI-GoToOpen-v0`
    - `BabyAI-GoToObjMaze-v0`
    - `BabyAI-GoToObjMazeOpen-v0`
    - `BabyAI-GoToObjMazeS4R2-v0`
    - `BabyAI-GoToObjMazeS4-v0`
    - `BabyAI-GoToObjMazeS5-v0`
    - `BabyAI-GoToObjMazeS6-v0`
    - `BabyAI-GoToObjMazeS7-v0`
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        doors_open=False,
        **kwargs,
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        super().__init__(
            num_rows=num_rows, num_cols=num_cols, room_size=room_size, **kwargs
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

        # If requested, open all the doors
        if self.doors_open:
            self.open_all_doors()


class GoToImpUnlock(RoomGridLevel):
    """

    ## Description

    Go to an object, which may be in a locked room.
    Competencies: Maze, GoTo, ImpUnlock
    No unblocking.

    ## Mission Space

    "go to a/the {color} {type}"

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

    1. The agent goes to the object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoToImpUnlock-v0`

    """

    def gen_mission(self):
        # Add a locked door to a random room
        id = self._rand_int(0, self.num_cols)
        jd = self._rand_int(0, self.num_rows)
        door, pos = self.add_door(id, jd, locked=True)
        locked_room = self.get_room(id, jd)

        # Add the key to a different room
        while True:
            ik = self._rand_int(0, self.num_cols)
            jk = self._rand_int(0, self.num_rows)
            if ik is id and jk is jd:
                continue
            self.add_object(ik, jk, "key", door.color)
            break

        self.connect_all()

        # Add distractors to all but the locked room.
        # We do this to speed up the reachability test,
        # which otherwise will reject all levels with
        # objects in the locked room.
        for i in range(self.num_cols):
            for j in range(self.num_rows):
                if i is not id or j is not jd:
                    self.add_distractors(i, j, num_distractors=2, all_unique=False)

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is locked_room:
                continue
            break

        self.check_objs_reachable()

        # Add a single object to the locked room
        # The instruction requires going to an object matching that description
        (obj,) = self.add_distractors(id, jd, num_distractors=1, all_unique=False)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class GoToSeq(LevelGen):
    """

    ## Description

    Sequencing of go-to-object commands.

    Competencies: Maze, GoTo, Seq
    No locked room.
    No locations.
    No unblocking.

    ## Mission Space

    "go to a/the {color} {type}" +
    "and go to a/the {color} {type}" +
    ", then go to a/the {color} {type}" +
    "and go to a/the {color} {type}"

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

    1. The agent goes to the object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoToSeq-v0`
    - `BabyAI-GoToSeqS5R2-v0`

    """

    def __init__(self, room_size=8, num_rows=3, num_cols=3, num_dists=18, **kwargs):
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            action_kinds=["goto"],
            locked_room_prob=0,
            locations=False,
            unblocking=False,
            **kwargs,
        )


class GoToRedBlueBall(RoomGridLevel):
    """

    ## Description

    Go to the red ball or to the blue ball.
    There is exactly one red or blue ball, and some distractors.
    The distractors are guaranteed not to be red or blue balls.
    Language is not required to solve this level.

    ## Mission Space

    "go to the {color} ball"

    {color} is the color of the box. Can be "red" or "blue".

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

    1. The agent goes to the ball.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoToRedBlueBall-v0`

    """

    def __init__(self, room_size=8, num_dists=7, **kwargs):
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()

        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # Ensure there is only one red or blue ball
        for dist in dists:
            if dist.type == "ball" and (dist.color == "blue" or dist.color == "red"):
                raise RejectSampling("can only have one blue or red ball")

        color = self._rand_elem(["red", "blue"])
        obj, _ = self.add_object(0, 0, "ball", color)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class GoToDoor(RoomGridLevel):
    """

    ## Description

    Go to a door
    (of a given color, in the current room)
    No distractors, no language variation

    ## Mission Space

    "go to the {color} door"

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

    1. The agent goes to the door.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoToDoor-v0`

    """

    def __init__(self, **kwargs):
        super().__init__(room_size=7, **kwargs)

    def gen_mission(self):
        objs = []
        for _ in range(4):
            door, _ = self.add_door(1, 1)
            objs.append(door)
        self.place_agent(1, 1)

        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc("door", obj.color))


class GoToObjDoor(RoomGridLevel):
    """

    ## Description

    Go to an object or door
    (of a given type and color, in the current room)

    ## Mission Space

    "go to the {color} {type}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box", "key" or "door".

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

    1. The agent goes to the object or door.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-GoToObjDoor-v0`

    """

    def __init__(self, **kwargs):
        super().__init__(room_size=8, **kwargs)

    def gen_mission(self):
        self.place_agent(1, 1)
        objs = self.add_distractors(1, 1, num_distractors=8, all_unique=False)

        for _ in range(4):
            door, _ = self.add_door(1, 1)
            objs.append(door)

        self.check_objs_reachable()

        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))
