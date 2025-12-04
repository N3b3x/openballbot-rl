"""Reward functions for ballbot environment."""
from ballbot_gym.core.registry import ComponentRegistry
from ballbot_gym.rewards.base import BaseReward
from ballbot_gym.rewards.directional import DirectionalReward
from ballbot_gym.rewards.distance import DistanceReward

# Auto-register components on import
ComponentRegistry.register_reward("directional", DirectionalReward)
ComponentRegistry.register_reward("distance", DistanceReward)

__all__ = ["DirectionalReward", "DistanceReward", "BaseReward"]

