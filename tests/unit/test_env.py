"""Unit tests for ballbot_gym environment."""
import pytest
import gymnasium as gym
import ballbot_gym
import numpy as np


def test_environment_registration():
    """Test that environment is registered."""
    assert "ballbot-v0.1" in gym.envs.registry.env_specs


def test_environment_creation():
    """Test that environment can be created."""
    env = gym.make("ballbot-v0.1", GUI=False, terrain_type="flat")
    assert env is not None
    env.close()


def test_reset():
    """Test environment reset."""
    env = gym.make("ballbot-v0.1", GUI=False, terrain_type="flat")
    obs, info = env.reset()
    assert obs is not None
    assert isinstance(obs, dict)
    env.close()


def test_step():
    """Test environment step."""
    env = gym.make("ballbot-v0.1", GUI=False, terrain_type="flat")
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None
    assert isinstance(reward, (int, float))
    env.close()


def test_observation_space():
    """Test observation space."""
    env = gym.make("ballbot-v0.1", GUI=False, terrain_type="flat")
    assert env.observation_space is not None
    env.close()


def test_action_space():
    """Test action space."""
    env = gym.make("ballbot-v0.1", GUI=False, terrain_type="flat")
    assert env.action_space is not None
    assert env.action_space.shape == (3,)
    env.close()


def test_environment_with_reward_config():
    """Test environment with custom reward config."""
    reward_config = {
        "type": "directional",
        "config": {"target_direction": [0.0, 1.0]}
    }
    env = gym.make(
        "ballbot-v0.1",
        GUI=False,
        terrain_type="flat",
        reward_config=reward_config
    )
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    env.close()


def test_environment_with_terrain_config():
    """Test environment with custom terrain config."""
    terrain_config = {
        "type": "flat",
        "config": {}
    }
    env = gym.make(
        "ballbot-v0.1",
        GUI=False,
        terrain_config=terrain_config
    )
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    env.close()

