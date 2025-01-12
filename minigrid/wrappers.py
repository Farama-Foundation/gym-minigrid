from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.core import ActionWrapper, ObservationWrapper, ObsType, Wrapper

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from minigrid.core.world_object import Goal


class ReseedWrapper(Wrapper):
    """
    Wrapper to always regenerate an environment with the same set of seeds.
    This can be used to force an environment to always keep the same
    configuration when reset.

    Example:
        >>> import minigrid
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ReseedWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> _ = env.reset(seed=123)
        >>> [env.np_random.integers(10).item() for i in range(10)]
        [0, 6, 5, 0, 9, 2, 2, 1, 3, 1]
        >>> env = ReseedWrapper(env, seeds=[0, 1], seed_idx=0)
        >>> _, _ = env.reset()
        >>> [env.np_random.integers(10).item() for i in range(10)]
        [8, 6, 5, 2, 3, 0, 0, 0, 1, 8]
        >>> _, _ = env.reset()
        >>> [env.np_random.integers(10).item() for i in range(10)]
        [4, 5, 7, 9, 0, 1, 8, 9, 2, 3]
        >>> _, _ = env.reset()
        >>> [env.np_random.integers(10).item() for i in range(10)]
        [8, 6, 5, 2, 3, 0, 0, 0, 1, 8]
        >>> _, _ = env.reset()
        >>> [env.np_random.integers(10).item() for i in range(10)]
        [4, 5, 7, 9, 0, 1, 8, 9, 2, 3]
    """

    def __init__(self, env, seeds=(0,), seed_idx=0):
        """A wrapper that always regenerate an environment with the same set of seeds.

        Args:
            env: The environment to apply the wrapper
            seeds: A list of seed to be applied to the env
            seed_idx: Index of the initial seed in seeds
        """
        self.seeds = list(seeds)
        self.seed_idx = seed_idx
        super().__init__(env)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        if seed is not None:
            logger.warn(
                "A seed has been passed to `ReseedWrapper.reset` which is ignored."
            )
        seed = self.seeds[self.seed_idx]
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        return self.env.reset(seed=seed, options=options)


class ActionBonus(gym.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ActionBonus
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> _, _ = env.reset(seed=0)
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> env_bonus = ActionBonus(env)
        >>> _, _ = env_bonus.reset(seed=0)
        >>> _, reward, _, _, _ = env_bonus.step(1)
        >>> print(reward)
        1.0
        >>> _, reward, _, _, _ = env_bonus.step(1)
        >>> print(reward)
        1.0
    """

    def __init__(self, env):
        """A wrapper that adds an exploration bonus to less visited (state,action) pairs.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        """Steps through the environment with `action`."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action)

        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, terminated, truncated, info


class PositionBonus(Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.

    Note:
        This wrapper was previously called ``StateBonus``.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import PositionBonus
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> _, _ = env.reset(seed=0)
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> env_bonus = PositionBonus(env)
        >>> obs, _ = env_bonus.reset(seed=0)
        >>> obs, reward, terminated, truncated, info = env_bonus.step(1)
        >>> print(reward)
        1.0
        >>> obs, reward, terminated, truncated, info = env_bonus.step(1)
        >>> print(reward)
        0.7071067811865475
    """

    def __init__(self, env):
        """A wrapper that adds an exploration bonus to less visited positions.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        """Steps through the environment with `action`."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = tuple(env.agent_pos)

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, terminated, truncated, info


class ImgObsWrapper(ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ImgObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> obs.keys()
        dict_keys(['image', 'direction', 'mission'])
        >>> env = ImgObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (7, 7, 3)
    """

    def __init__(self, env):
        """A wrapper that makes image the only observation.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["image"]

    def observation(self, obs):
        return obs["image"]


class OneHotPartialObsWrapper(ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import OneHotPartialObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> obs["image"][0, :, :]
        array([[2, 5, 0],
               [2, 5, 0],
               [2, 5, 0],
               [2, 5, 0],
               [2, 5, 0],
               [2, 5, 0],
               [2, 5, 0]], dtype=uint8)
        >>> env = OneHotPartialObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> obs["image"][0, :, :]
        array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]],
              dtype=uint8)
    """

    def __init__(self, env, tile_size=8):
        """A wrapper that makes the image observation a one-hot encoding of a partially observable agent view.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        self.tile_size = tile_size

        obs_shape = env.observation_space["image"].shape

        # Number of bits per cell
        num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)

        new_image_space = spaces.Box(
            low=0, high=255, shape=(obs_shape[0], obs_shape[1], num_bits), dtype="uint8"
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        img = obs["image"]
        out = np.zeros(self.observation_space.spaces["image"].shape, dtype="uint8")

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                type = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]

                out[i, j, type] = 1
                out[i, j, len(OBJECT_TO_IDX) + color] = 1
                out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1

        return {**obs, "image": out}


class RGBImgObsWrapper(ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as observation,
    This can be used to have the agent to solve the gridworld in pixel space.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import RGBImgObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> plt.imshow(obs['image'])  # doctest: +SKIP
        ![NoWrapper](../figures/lavacrossing_NoWrapper.png)
        >>> env = RGBImgObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> plt.imshow(obs['image'])  # doctest: +SKIP
        ![RGBImgObsWrapper](../figures/lavacrossing_RGBImgObsWrapper.png)
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.unwrapped.height * tile_size,
                self.unwrapped.width * tile_size,
                3,
            ),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        rgb_img = self.unwrapped.get_frame(
            highlight=self.unwrapped.highlight, tile_size=self.tile_size
        )

        return {**obs, "image": rgb_img}


class RGBImgPartialObsWrapper(ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as observation.
    This can be used to have the agent to solve the gridworld in pixel space.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> plt.imshow(obs["image"])  # doctest: +SKIP
        ![NoWrapper](../figures/lavacrossing_NoWrapper.png)
        >>> env_obs = RGBImgObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> plt.imshow(obs["image"])  # doctest: +SKIP
        ![RGBImgObsWrapper](../figures/lavacrossing_RGBImgObsWrapper.png)
        >>> env_obs = RGBImgPartialObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> plt.imshow(obs["image"])  # doctest: +SKIP
        ![RGBImgPartialObsWrapper](../figures/lavacrossing_RGBImgPartialObsWrapper.png)
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        # Rendering attributes for observations
        self.tile_size = tile_size

        obs_shape = env.observation_space.spaces["image"].shape
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        rgb_img_partial = self.unwrapped.get_frame(
            tile_size=self.tile_size, agent_pov=True
        )

        return {**obs, "image": rgb_img_partial}


class FullyObsWrapper(ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding instead of the agent view.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FullyObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> obs['image'].shape
        (7, 7, 3)
        >>> env_obs = FullyObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs['image'].shape
        (11, 11, 3)
    """

    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.env.unwrapped.width,
                self.env.unwrapped.height,
                3,
            ),  # number of cells
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )

        return {**obs, "image": full_grid}


class DictObservationSpaceWrapper(ObservationWrapper):
    """
    Transforms the observation space (that has a textual component) to a fully numerical observation space,
    where the textual instructions are replaced by arrays representing the indices of each word in a fixed vocabulary.

    This wrapper is not applicable to BabyAI environments, given that these have their own language component.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import DictObservationSpaceWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> obs['mission']
        'avoid the lava and get to the green goal square'
        >>> env_obs = DictObservationSpaceWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs['mission'][:10]
        [19, 31, 17, 36, 20, 38, 31, 2, 15, 35]
    """

    def __init__(self, env, max_words_in_mission=50, word_dict=None):
        """
        max_words_in_mission is the length of the array to represent a mission, value 0 for missing words
        word_dict is a dictionary of words to use (keys=words, values=indices from 1 to < max_words_in_mission),
                  if None, use the Minigrid language
        """
        super().__init__(env)

        if word_dict is None:
            word_dict = self.get_minigrid_words()

        self.max_words_in_mission = max_words_in_mission
        self.word_dict = word_dict

        self.observation_space = spaces.Dict(
            {
                "image": env.observation_space["image"],
                "direction": spaces.Discrete(4),
                "mission": spaces.MultiDiscrete(
                    [len(self.word_dict.keys())] * max_words_in_mission
                ),
            }
        )

    @staticmethod
    def get_minigrid_words():
        colors = ["red", "green", "blue", "yellow", "purple", "grey"]
        objects = [
            "unseen",
            "empty",
            "wall",
            "floor",
            "box",
            "key",
            "ball",
            "door",
            "goal",
            "agent",
            "lava",
        ]

        verbs = [
            "pick",
            "avoid",
            "get",
            "find",
            "put",
            "use",
            "open",
            "go",
            "fetch",
            "reach",
            "unlock",
            "traverse",
        ]

        extra_words = [
            "up",
            "the",
            "a",
            "at",
            ",",
            "square",
            "and",
            "then",
            "to",
            "of",
            "rooms",
            "near",
            "opening",
            "must",
            "you",
            "matching",
            "end",
            "hallway",
            "object",
            "from",
            "room",
            "maze",
        ]

        all_words = colors + objects + verbs + extra_words
        assert len(all_words) == len(set(all_words))
        return {word: i for i, word in enumerate(all_words)}

    def string_to_indices(self, string, offset=1):
        """
        Convert a string to a list of indices.
        """
        indices = []
        # adding space before and after commas
        string = string.replace(",", " , ")
        for word in string.split():
            if word in self.word_dict.keys():
                indices.append(self.word_dict[word] + offset)
            else:
                raise ValueError(f"Unknown word: {word}")
        return indices

    def observation(self, obs):
        obs["mission"] = self.string_to_indices(obs["mission"])
        assert len(obs["mission"]) < self.max_words_in_mission
        obs["mission"] += [0] * (self.max_words_in_mission - len(obs["mission"]))

        return obs


class FlatObsWrapper(ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array.

    This wrapper is not applicable to BabyAI environments, given that these have their own language component.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FlatObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> env_obs = FlatObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs.shape
        (2835,)
    """

    def __init__(self, env, maxStrLen: int = 96):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 28

        img_size = np.prod(env.observation_space["image"].shape)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(img_size + self.numCharCodes * self.maxStrLen,),
            dtype="uint8",
        )

        self.cachedStr: str = None

    def observation(self, obs):
        image = obs["image"]
        mission = obs["mission"]

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert (
                len(mission) <= self.maxStrLen
            ), f"mission string too long ({len(mission)} chars)"
            mission = mission.lower()

            str_array = np.zeros(
                shape=(self.maxStrLen, self.numCharCodes), dtype="uint8"
            )
            # as `numCharCodes` < 255 then we can use `uint8`

            for idx, ch in enumerate(mission):
                if "a" <= ch <= "z":
                    chNo = ord(ch) - ord("a")
                elif ch == " ":
                    chNo = ord("z") - ord("a") + 1
                elif ch == ",":
                    chNo = ord("z") - ord("a") + 2
                else:
                    raise ValueError(
                        f"Character {ch} is not available in mission string."
                    )
                assert chNo < self.numCharCodes, f"{ch} : {chNo:d}"
                str_array[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = str_array

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs


class ViewSizeWrapper(ObservationWrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ViewSizeWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> obs['image'].shape
        (7, 7, 3)
        >>> env_obs = ViewSizeWrapper(env, agent_view_size=5)
        >>> obs, _ = env_obs.reset()
        >>> obs['image'].shape
        (5, 5, 3)
    """

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3

        self.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        new_image_space = gym.spaces.Box(
            low=0, high=255, shape=(agent_view_size, agent_view_size, 3), dtype="uint8"
        )

        # Override the environment's observation spaceexit
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped

        grid, vis_mask = env.gen_obs_grid(self.agent_view_size)

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        return {**obs, "image": image}


class DirectionObsWrapper(ObservationWrapper):
    """
    Provides the slope/angular direction to the goal with the observations as modeled by (y2 - y2 )/( x2 - x1)
    type = {slope , angle}

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import DirectionObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> env_obs = DirectionObsWrapper(env, type="slope")
        >>> obs, _ = env_obs.reset()
        >>> obs['goal_direction'].item()
        1.0
    """

    def __init__(self, env, type="slope"):
        super().__init__(env)
        self.goal_position: tuple = None
        self.type = type

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset()

        if not self.goal_position:
            self.goal_position = [
                x for x, y in enumerate(self.unwrapped.grid.grid) if isinstance(y, Goal)
            ]
            # in case there are multiple goals , needs to be handled for other env types
            if len(self.goal_position) >= 1:
                self.goal_position = (
                    int(self.goal_position[0] / self.unwrapped.height),
                    self.goal_position[0] % self.unwrapped.width,
                )

        return self.observation(obs), info

    def observation(self, obs):
        slope = np.divide(
            self.goal_position[1] - self.unwrapped.agent_pos[1],
            self.goal_position[0] - self.unwrapped.agent_pos[0],
        )

        if self.type == "angle":
            obs["goal_direction"] = np.arctan(slope)
        else:
            obs["goal_direction"] = slope

        return obs


class SymbolicObsWrapper(ObservationWrapper):
    """
    Fully observable grid with a symbolic state representation.
    The symbol is a triple of (X, Y, IDX), where X and Y are
    the coordinates on the grid, and IDX is the id of the object.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import SymbolicObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> obs['image'].shape
        (7, 7, 3)
        >>> env_obs = SymbolicObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs['image'].shape
        (11, 11, 3)
    """

    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=max(OBJECT_TO_IDX.values()),
            shape=(
                self.env.unwrapped.width,
                self.env.unwrapped.height,
                3,
            ),  # number of cells
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        objects = np.array(
            [
                OBJECT_TO_IDX[o.type] if o is not None else -1
                for o in self.unwrapped.grid.grid
            ]
        )
        agent_pos = self.env.unwrapped.agent_pos
        ncol, nrow = self.unwrapped.width, self.unwrapped.height
        grid = np.mgrid[:ncol, :nrow]
        _objects = np.transpose(objects.reshape(1, nrow, ncol), (0, 2, 1))

        grid = np.concatenate([grid, _objects])
        grid = np.transpose(grid, (1, 2, 0))
        grid[agent_pos[0], agent_pos[1], 2] = OBJECT_TO_IDX["agent"]
        obs["image"] = grid

        return obs


class StochasticActionWrapper(ActionWrapper):
    """
    Add stochasticity to the actions

    If a random action is provided, it is returned with probability `1 - prob`.
    Else, a random action is sampled from the action space.
    """

    def __init__(self, env=None, prob=0.9, random_action=None):
        super().__init__(env)
        self.prob = prob
        self.random_action = random_action

    def action(self, action):
        """ """
        if np.random.uniform() < self.prob:
            return action
        else:
            if self.random_action is None:
                return self.np_random.integers(0, high=6)
            else:
                return self.random_action


class NoDeath(Wrapper):
    """
    Wrapper to prevent death in specific cells (e.g., lava cells).
    Instead of dying, the agent will receive a negative reward.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import NoDeath
        >>>
        >>> env = gym.make("MiniGrid-LavaCrossingS9N1-v0")
        >>> _, _ = env.reset(seed=2)
        >>> _, _, _, _, _ = env.step(1)
        >>> _, reward, term, *_ = env.step(2)
        >>> reward, term
        (0, True)
        >>>
        >>> env = NoDeath(env, no_death_types=("lava",), death_cost=-1.0)
        >>> _, _ = env.reset(seed=2)
        >>> _, _, _, _, _ = env.step(1)
        >>> _, reward, term, *_ = env.step(2)
        >>> reward, term
        (-1.0, False)
        >>>
        >>>
        >>> env = gym.make("MiniGrid-Dynamic-Obstacles-5x5-v0")
        >>> _, _ = env.reset(seed=2)
        >>> _, reward, term, *_ = env.step(2)
        >>> reward, term
        (-1, True)
        >>>
        >>> env = NoDeath(env, no_death_types=("ball",), death_cost=-1.0)
        >>> _, _ = env.reset(seed=2)
        >>> _, reward, term, *_ = env.step(2)
        >>> reward, term
        (-2.0, False)
    """

    def __init__(self, env, no_death_types: tuple[str, ...], death_cost: float = -1.0):
        """A wrapper to prevent death in specific cells.

        Args:
            env: The environment to apply the wrapper
            no_death_types: List of strings to identify death cells
            death_cost: The negative reward received in death cells

        """
        assert "goal" not in no_death_types, "goal cannot be a death cell"

        super().__init__(env)
        self.death_cost = death_cost
        self.no_death_types = no_death_types

    def step(self, action):
        # In Dynamic-Obstacles, obstacles move after the agent moves,
        # so we need to check for collision before self.env.step()
        front_cell = self.unwrapped.grid.get(*self.unwrapped.front_pos)
        going_to_death = (
            action == self.unwrapped.actions.forward
            and front_cell is not None
            and front_cell.type in self.no_death_types
        )

        obs, reward, terminated, truncated, info = self.env.step(action)

        # We also check if the agent stays in death cells (e.g., lava)
        # without moving
        current_cell = self.unwrapped.grid.get(*self.unwrapped.agent_pos)
        in_death = current_cell is not None and current_cell.type in self.no_death_types

        if terminated and (going_to_death or in_death):
            terminated = False
            reward += self.death_cost

        return obs, reward, terminated, truncated, info


class MoveActionWrapper(Wrapper):
    """
    Wrapper to change the action space to move actions, instead of turn.

    Thus, the action space, become a Discrete(8) space, where the actions are:
    0: Move right
    1: Move down
    2: Move left
    3: Move up
    4: pickup
    5: drop
    5: toggle
    7: done

    Note:
        This wrapper is mostly useful with full observation, as the partial observation is rotated depending on the agent direction.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(8)

    def step(self, action):
        if action <= 3:
            agent_dir = self.unwrapped.agent_dir
            left_turns = (agent_dir - action) % 4
            if left_turns == 3:
                self.env.step(1)
            else:
                for _ in range(left_turns):
                    self.env.step(0)

            return self.env.step(2)
        else:
            return self.env.step(action - 1)
