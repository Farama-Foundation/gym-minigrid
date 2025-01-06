"""
Copied and adapted from https://github.com/mila-iqia/babyai.
Levels described in the Baby AI ICLR 2019 submission.
The instructions are a synthesis of those from `PutNext`, `Open`, `GoTo`, and `Pickup`.
"""

from __future__ import annotations

from minigrid.envs.babyai.core.levelgen import LevelGen


class Synth(LevelGen):
    """

    ## Description

    Union of all instructions from PutNext, Open, Goto and PickUp.
    The agent may need to move objects around. The agent may have
    to unlock the door, but only if it is explicitly referred by
    the instruction.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open

    ## Mission Space

    "go to the {color} {type}"

    or

    "pick up a/the {color} {type}"

    or

    "open the {color} door"

    or

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

    1. The agent achieves the task.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-Synth-v0`
    - `BabyAI-SynthS5R2-v0`

    """

    def __init__(self, room_size=8, num_rows=3, num_cols=3, num_dists=18, **kwargs):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            instr_kinds=["action"],
            locations=False,
            unblocking=True,
            implicit_unlock=False,
            **kwargs,
        )


class SynthLoc(LevelGen):
    """

    ## Description

    Like Synth, but a significant share of object descriptions involves
    location language like in PickUpLoc. No implicit unlocking.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc

    ## Mission Space

    "go to the {color} {type} {location}"

    or

    "pick up a/the {color} {type} {location}"

    or

    "open the {color} door {location}"

    or

    "put the {color} {type} {location} next to the {color} {type} {location}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    {location} can be " ", "in front of you", "behind you", "on your left"
    or "on your right"

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Drops held item   |
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

    1. The agent achieves the task.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-SynthLoc-v0`
    """

    def __init__(self, **kwargs):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            instr_kinds=["action"],
            locations=True,
            unblocking=True,
            implicit_unlock=False,
            **kwargs,
        )


class SynthSeq(LevelGen):
    """

    ## Description

    Like SynthLoc, but now with multiple commands, combined just like in GoToSeq.
    No implicit unlocking.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc, Seq

    ## Mission Space

    Action mission space:

    "go to the {color} {type} {location}"

    or

    "pick up a/the {color} {type} {location}"

    or

    "open the {color} door {location}"

    or

    "put the {color} {type} {location} next to the {color} {type} {location}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    {location} can be " ", "in front of you", "behind you", "on your left"
    or "on your right"

    And mission space:

    Two action missions concatenated with "and"

    Example:

    go to the green key
    and
    put the box next to the yellow ball

    Sequence mission space:

    Two missions, they can be action or and missions, concatenated with
    ", then" or "after you".

    Example:

    open a red door and go to the ball on your left
    after you
    put the grey ball next to a door

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Drops held item   |
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

    1. The agent achieves the task.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-SynthSeq-v0`

    """

    def __init__(self, **kwargs):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            locations=True, unblocking=True, implicit_unlock=False, **kwargs
        )


class MiniBossLevel(LevelGen):
    """

    ## Description

    Command can be any sentence drawn from the Baby Language grammar.
    Union of all competencies. This level is a superset of all other levels.
    Compared to BossLevel this has a smaller room and a lower probability of
    locked rooms.

    ## Mission Space

    Action mission space:

    "go to the {color} {type} {location}"

    or

    "pick up a/the {color} {type} {location}"

    or

    "open the {color} door {location}"

    or

    "put the {color} {type} {location} next to the {color} {type} {location}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    {location} can be " ", "in front of you", "behind you", "on your left"
    or "on your right"

    And mission space:

    Two action missions concatenated with "and"

    Example:

    go to the green key
    and
    put the box next to the yellow ball

    Sequence mission space:

    Two missions, they can be action or and missions, concatenated with
    ", then" or "after you".

    Example:

    open a red door and go to the ball on your left
    after you
    put the grey ball next to a door

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Drops held item   |
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

    1. The agent achieves the task.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-MiniBossLevel-v0`

    """

    def __init__(self, **kwargs):
        super().__init__(
            num_cols=2,
            num_rows=2,
            room_size=5,
            num_dists=7,
            locked_room_prob=0.25,
            **kwargs,
        )


class BossLevel(LevelGen):
    """

    ## Description

    Command can be any sentence drawn from the Baby Language grammar.
    Union of all competencies. This level is a superset of all other levels.

    ## Mission Space

    Action mission space:

    "go to the {color} {type} {location}"

    or

    "pick up a/the {color} {type} {location}"

    or

    "open the {color} door {location}"

    or

    "put the {color} {type} {location} next to the {color} {type} {location}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    {location} can be " ", "in front of you", "behind you", "on your left"
    or "on your right"

    And mission space:

    Two action missions concatenated with "and"

    Example:

    go to the green key
    and
    put the box next to the yellow ball

    Sequence mission space:

    Two missions, they can be action or and missions, concatenated with
    ", then" or "after you".

    Example:

    open a red door and go to the ball on your left
    after you
    put the grey ball next to a door

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Drops held item   |
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

    1. The agent achieves the task.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-BossLevel-v0`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BossLevelNoUnlock(LevelGen):
    """

    ## Description

    Command can be any sentence drawn from the Baby Language grammar.
    Union of all competencies. This level is a superset of all other levels.
    No implicit unlocking.

    ## Mission Space

    Action mission space:

    "go to the {color} {type} {location}"

    or

    "pick up a/the {color} {type} {location}"

    or

    "open the {color} door {location}"

    or

    "put the {color} {type} {location} next to the {color} {type} {location}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball", "box" or "key".

    {location} can be " ", "in front of you", "behind you", "on your left"
    or "on your right"

    And mission space:

    Two action missions concatenated with "and"

    Example:

    go to the green key
    and
    put the box next to the yellow ball

    Sequence mission space:

    Two missions, they can be action or and missions, concatenated with
    ", then" or "after you".

    Example:

    open a red door and go to the ball on your left
    after you
    put the grey ball next to a door

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Drops held item   |
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

    1. The agent achieves the task.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-BossLevelNoUnlock-v0`
    """

    def __init__(self, **kwargs):
        super().__init__(locked_room_prob=0, implicit_unlock=False, **kwargs)
