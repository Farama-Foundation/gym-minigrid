from __future__ import annotations

import math
import warnings

import gymnasium as gym
import numpy as np
import pytest

from minigrid.core.actions import Actions
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.envs import EmptyEnv
from minigrid.wrappers import (
    ActionBonus,
    DictObservationSpaceWrapper,
    DirectionObsWrapper,
    FlatObsWrapper,
    FullyObsWrapper,
    ImgObsWrapper,
    MoveActionWrapper,
    NoDeath,
    OneHotPartialObsWrapper,
    PositionBonus,
    ReseedWrapper,
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper,
    StochasticActionWrapper,
    SymbolicObsWrapper,
    ViewSizeWrapper,
)
from tests.utils import all_testing_env_specs, assert_equals, minigrid_testing_env_specs

SEEDS = [100, 243, 500]
NUM_STEPS = 100


@pytest.mark.parametrize(
    "env_spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_reseed_wrapper(env_spec):
    """
    Test the ReseedWrapper with a list of SEEDS.
    """
    unwrapped_env = env_spec.make()
    env = ReseedWrapper(env_spec.make(), seeds=SEEDS)
    env.action_space.seed(0)

    for seed in SEEDS:
        env.reset()
        unwrapped_env.reset(seed=seed)
        for time_step in range(NUM_STEPS):
            action = env.action_space.sample()

            obs, rew, terminated, truncated, info = env.step(action)
            (
                unwrapped_obs,
                unwrapped_rew,
                unwrapped_terminated,
                unwrapped_truncated,
                unwrapped_info,
            ) = unwrapped_env.step(action)

            assert_equals(obs, unwrapped_obs, f"[{time_step}] ")
            assert unwrapped_env.observation_space.contains(obs)

            assert (
                rew == unwrapped_rew
            ), f"[{time_step}] reward={rew}, unwrapped reward={unwrapped_rew}"
            assert (
                terminated == unwrapped_terminated
            ), f"[{time_step}] terminated={terminated}, unwrapped terminated={unwrapped_terminated}"
            assert (
                truncated == unwrapped_truncated
            ), f"[{time_step}] truncated={truncated}, unwrapped truncated={unwrapped_truncated}"
            assert_equals(info, unwrapped_info, f"[{time_step}] ")

            # Start the next seed
            if terminated or truncated:
                break

    env.close()
    unwrapped_env.close()


@pytest.mark.parametrize("env_id", ["MiniGrid-Empty-16x16-v0"])
def test_position_bonus_wrapper(env_id):
    env = gym.make(env_id)
    wrapped_env = PositionBonus(gym.make(env_id))

    action_forward = Actions.forward
    action_left = Actions.left
    action_right = Actions.right

    for _ in range(10):
        wrapped_env.reset()
        for _ in range(5):
            wrapped_env.step(action_forward)

    # Turn lef 3 times (check that actions don't influence bonus)
    for _ in range(3):
        _, wrapped_rew, _, _, _ = wrapped_env.step(action_left)

    env.reset()
    for _ in range(5):
        env.step(action_forward)
    # Turn right 3 times
    for _ in range(3):
        _, rew, _, _, _ = env.step(action_right)

    expected_bonus_reward = rew + 1 / math.sqrt(13)

    assert expected_bonus_reward == wrapped_rew


@pytest.mark.parametrize("env_id", ["MiniGrid-Empty-16x16-v0"])
def test_action_bonus_wrapper(env_id):
    env = gym.make(env_id)
    wrapped_env = ActionBonus(gym.make(env_id))

    action = Actions.forward

    for _ in range(10):
        wrapped_env.reset()
        for _ in range(5):
            _, wrapped_rew, _, _, _ = wrapped_env.step(action)

    env.reset()
    for _ in range(5):
        _, rew, _, _, _ = env.step(action)

    expected_bonus_reward = rew + 1 / math.sqrt(10)

    assert expected_bonus_reward == wrapped_rew


@pytest.mark.parametrize(
    "env_spec",
    minigrid_testing_env_specs,
    ids=[spec.id for spec in minigrid_testing_env_specs],
)  # DictObservationSpaceWrapper is not compatible with BabyAI levels. See minigrid/wrappers.py for more details.
def test_dict_observation_space_wrapper(env_spec):
    env = env_spec.make()
    env = DictObservationSpaceWrapper(env)
    env.reset()
    mission = env.unwrapped.mission
    obs, _, _, _, _ = env.step(0)
    assert env.string_to_indices(mission) == [
        value for value in obs["mission"] if value != 0
    ]
    env.close()


@pytest.mark.parametrize(
    "wrapper",
    [
        ReseedWrapper,
        ImgObsWrapper,
        FlatObsWrapper,
        ViewSizeWrapper,
        DictObservationSpaceWrapper,
        OneHotPartialObsWrapper,
        RGBImgPartialObsWrapper,
        FullyObsWrapper,
        MoveActionWrapper,
    ],
)
@pytest.mark.parametrize(
    "env_spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_main_wrappers(wrapper, env_spec):
    if (
        wrapper in (DictObservationSpaceWrapper, FlatObsWrapper)
        and env_spec not in minigrid_testing_env_specs
    ):
        # DictObservationSpaceWrapper and FlatObsWrapper are not compatible with BabyAI levels
        # See minigrid/wrappers.py for more details
        pytest.skip()

    env = env_spec.make()
    env = wrapper(env)

    with warnings.catch_warnings():
        env.reset(seed=123)
    env.step(0)

    env.close()


@pytest.mark.parametrize(
    "wrapper",
    [
        OneHotPartialObsWrapper,
        RGBImgPartialObsWrapper,
        FullyObsWrapper,
    ],
)
@pytest.mark.parametrize(
    "env_spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_observation_space_wrappers(wrapper, env_spec):
    env = wrapper(env_spec.make(disable_env_checker=True))
    obs_space, wrapper_name = env.observation_space, wrapper.__name__
    assert isinstance(
        obs_space, gym.spaces.Dict
    ), f"Observation space for {wrapper_name} is not a Dict: {obs_space}."
    # This should not fail either
    ImgObsWrapper(env)
    env.reset()
    env.step(0)
    env.close()


class EmptyEnvWithExtraObs(EmptyEnv):
    """
    Custom environment with an extra observation
    """

    def __init__(self) -> None:
        super().__init__(size=5)
        self.observation_space["size"] = gym.spaces.Box(
            low=0, high=np.iinfo(np.uint).max, shape=(2,), dtype=np.uint
        )

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs["size"] = np.array([self.width, self.height])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs["size"] = np.array([self.width, self.height])
        return obs, reward, terminated, truncated, info


@pytest.mark.parametrize(
    "wrapper",
    [
        OneHotPartialObsWrapper,
        RGBImgObsWrapper,
        RGBImgPartialObsWrapper,
        FullyObsWrapper,
    ],
)
def test_agent_sees_method(wrapper):
    env1 = wrapper(EmptyEnvWithExtraObs())
    env2 = wrapper(gym.make("MiniGrid-Empty-5x5-v0"))

    obs1, _ = env1.reset(seed=0)
    obs2, _ = env2.reset(seed=0)
    assert "size" in obs1
    assert obs1["size"].shape == (2,)
    assert (obs1["size"] == [5, 5]).all()
    for key in obs2:
        assert np.array_equal(obs1[key], obs2[key])

    obs1, reward1, terminated1, truncated1, _ = env1.step(0)
    obs2, reward2, terminated2, truncated2, _ = env2.step(0)
    assert "size" in obs1
    assert obs1["size"].shape == (2,)
    assert (obs1["size"] == [5, 5]).all()
    for key in obs2:
        assert np.array_equal(obs1[key], obs2[key])


@pytest.mark.parametrize("view_size", [5, 7, 9])
def test_viewsize_wrapper(view_size):
    env = gym.make("MiniGrid-Empty-5x5-v0")
    env = ViewSizeWrapper(env, agent_view_size=view_size)
    env.reset()
    obs, _, _, _, _ = env.step(0)
    assert obs["image"].shape == (view_size, view_size, 3)
    env.close()


@pytest.mark.parametrize("env_id", ["MiniGrid-LavaCrossingS11N5-v0"])
@pytest.mark.parametrize("type", ["slope", "angle"])
def test_direction_obs_wrapper(env_id, type):
    env = gym.make(env_id)
    env = DirectionObsWrapper(env, type=type)
    obs, _ = env.reset()
    slope = np.divide(
        env.unwrapped.goal_position[1] - env.unwrapped.agent_pos[1],
        env.unwrapped.goal_position[0] - env.unwrapped.agent_pos[0],
    )
    if type == "slope":
        assert obs["goal_direction"] == slope
    elif type == "angle":
        assert obs["goal_direction"] == np.arctan(slope)

    obs, _, _, _, _ = env.step(0)
    slope = np.divide(
        env.unwrapped.goal_position[1] - env.unwrapped.agent_pos[1],
        env.unwrapped.goal_position[0] - env.unwrapped.agent_pos[0],
    )
    if type == "slope":
        assert obs["goal_direction"] == slope
    elif type == "angle":
        assert obs["goal_direction"] == np.arctan(slope)

    env.close()


@pytest.mark.parametrize("env_id", ["MiniGrid-DistShift1-v0"])
def test_symbolic_obs_wrapper(env_id):
    env = gym.make(env_id)

    env = SymbolicObsWrapper(env)
    obs, _ = env.reset(seed=123)
    agent_pos = env.unwrapped.agent_pos
    goal_pos = env.unwrapped.goal_pos

    assert obs["image"].shape == (env.unwrapped.width, env.unwrapped.height, 3)
    assert np.all(
        obs["image"][agent_pos[0], agent_pos[1], :]
        == np.array([agent_pos[0], agent_pos[1], OBJECT_TO_IDX["agent"]])
    )
    assert np.all(
        obs["image"][goal_pos[0], goal_pos[1], :]
        == np.array([goal_pos[0], goal_pos[1], OBJECT_TO_IDX["goal"]])
    )

    obs, _, _, _, _ = env.step(2)
    agent_pos = env.unwrapped.agent_pos
    goal_pos = env.unwrapped.goal_pos

    assert obs["image"].shape == (env.unwrapped.width, env.unwrapped.height, 3)
    assert np.all(
        obs["image"][agent_pos[0], agent_pos[1], :]
        == np.array([agent_pos[0], agent_pos[1], OBJECT_TO_IDX["agent"]])
    )
    assert np.all(
        obs["image"][goal_pos[0], goal_pos[1], :]
        == np.array([goal_pos[0], goal_pos[1], OBJECT_TO_IDX["goal"]])
    )
    env.close()


@pytest.mark.parametrize("env_id", ["MiniGrid-Empty-16x16-v0"])
def test_stochastic_action_wrapper(env_id):
    env = gym.make(env_id)
    env = StochasticActionWrapper(env, prob=0.2)
    _, _ = env.reset()
    for _ in range(20):
        _, _, _, _, _ = env.step(0)
    env.close()

    env = gym.make(env_id)
    env = StochasticActionWrapper(env, prob=0.2, random_action=1)
    _, _ = env.reset()
    for _ in range(20):
        _, _, _, _, _ = env.step(0)
    env.close()


def test_dict_observation_space_doesnt_clash_with_one_hot():
    env = gym.make("MiniGrid-Empty-5x5-v0")
    env = OneHotPartialObsWrapper(env)
    env = DictObservationSpaceWrapper(env)
    env.reset()
    obs, _, _, _, _ = env.step(0)
    assert obs["image"].shape == (7, 7, 20)
    assert env.observation_space["image"].shape == (7, 7, 20)
    env.close()


def test_no_death_wrapper():
    death_cost = -1

    env = gym.make("MiniGrid-LavaCrossingS9N1-v0")
    _, _ = env.reset(seed=2)
    _, _, _, _, _ = env.step(1)
    _, reward, term, *_ = env.step(2)

    env_wrap = NoDeath(env, ("lava",), death_cost)
    _, _ = env_wrap.reset(seed=2)
    _, _, _, _, _ = env_wrap.step(1)
    _, reward_wrap, term_wrap, *_ = env_wrap.step(2)

    assert term and not term_wrap
    assert reward_wrap == reward + death_cost
    env.close()
    env_wrap.close()

    env = gym.make("MiniGrid-Dynamic-Obstacles-5x5-v0")
    _, _ = env.reset(seed=2)
    _, reward, term, *_ = env.step(2)

    env = NoDeath(env, ("ball",), death_cost)
    _, _ = env.reset(seed=2)
    _, reward_wrap, term_wrap, *_ = env.step(2)

    assert term and not term_wrap
    assert reward_wrap == reward + death_cost
    env.close()
    env_wrap.close()


def test_non_square_RGBIMgObsWrapper():
    """
    Add test for non-square dimensions with RGBImgObsWrapper
    (https://github.com/Farama-Foundation/Minigrid/issues/444).
    """
    env = RGBImgObsWrapper(gym.make("MiniGrid-BlockedUnlockPickup-v0"))
    obs, info = env.reset()
    assert env.observation_space["image"].shape == obs["image"].shape
