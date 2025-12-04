"""Core infrastructure for component registry and factories."""
from ballbot_gym.core.registry import ComponentRegistry
from ballbot_gym.core.factories import (
    create_reward,
    create_terrain,
    create_policy,
    validate_config,
)

__all__ = [
    "ComponentRegistry",
    "create_reward",
    "create_terrain",
    "create_policy",
    "validate_config",
]

