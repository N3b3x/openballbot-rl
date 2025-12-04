"""Factory functions for creating components from configuration."""
from typing import Dict, Any, Callable, Type, Optional
import numpy as np

from ballbot_gym.core.registry import ComponentRegistry
from ballbot_gym.rewards.base import BaseReward


def create_reward(config: Dict[str, Any]) -> BaseReward:
    """
    Create a reward function from configuration dictionary.
    
    Configuration format:
        {
            "type": "directional",  # Name of registered reward
            "config": {              # Arguments for reward constructor
                "target_direction": [0.0, 1.0],
                "scale": 0.01
            }
        }
    
    Args:
        config: Configuration dictionary with "type" and optional "config" keys
        
    Returns:
        Instantiated reward function object
        
    Raises:
        ValueError: If config is invalid or reward type not found
        KeyError: If required config keys are missing
    """
    if not isinstance(config, dict):
        raise ValueError(f"Reward config must be a dictionary, got {type(config)}")
    
    reward_type = config.get("type")
    if reward_type is None:
        raise ValueError("Reward config must have 'type' key")
    
    reward_config = config.get("config", {})
    
    try:
        return ComponentRegistry.get_reward(reward_type, **reward_config)
    except ValueError as e:
        raise ValueError(f"Failed to create reward '{reward_type}': {e}")


def create_terrain(config: Dict[str, Any]) -> Callable:
    """
    Create a terrain generator function from configuration dictionary.
    
    Configuration format:
        {
            "type": "perlin",  # Name of registered terrain generator
            "config": {         # Arguments for terrain generator
                "scale": 25.0,
                "octaves": 4,
                "seed": None
            }
        }
    
    Args:
        config: Configuration dictionary with "type" and optional "config" keys
        
    Returns:
        Terrain generator function (callable that takes n: int, **kwargs) -> np.ndarray
        
    Raises:
        ValueError: If config is invalid or terrain type not found
        KeyError: If required config keys are missing
    """
    if not isinstance(config, dict):
        raise ValueError(f"Terrain config must be a dictionary, got {type(config)}")
    
    terrain_type = config.get("type")
    if terrain_type is None:
        raise ValueError("Terrain config must have 'type' key")
    
    terrain_config = config.get("config", {})
    
    try:
        terrain_fn = ComponentRegistry.get_terrain(terrain_type)
    except ValueError as e:
        raise ValueError(f"Failed to get terrain '{terrain_type}': {e}")
    
    # Return a partially configured function
    def configured_terrain(n: int, **override_kwargs) -> np.ndarray:
        """Terrain generator with pre-configured parameters."""
        # Merge config with any runtime overrides (e.g., seed)
        final_config = {**terrain_config, **override_kwargs}
        return terrain_fn(n, **final_config)
    
    return configured_terrain


def create_policy(config: Dict[str, Any]) -> Type:
    """
    Create a policy class from configuration dictionary.
    
    Configuration format:
        {
            "type": "mlp",  # Name of registered policy
            "config": {      # Policy-specific configuration
                "hidden_sizes": [128, 128, 128],
                "activation": "leaky_relu"
            }
        }
    
    Args:
        config: Configuration dictionary with "type" and optional "config" keys
        
    Returns:
        Policy class (not instantiated)
        
    Raises:
        ValueError: If config is invalid or policy type not found
        KeyError: If required config keys are missing
    """
    if not isinstance(config, dict):
        raise ValueError(f"Policy config must be a dictionary, got {type(config)}")
    
    policy_type = config.get("type")
    if policy_type is None:
        raise ValueError("Policy config must have 'type' key")
    
    try:
        return ComponentRegistry.get_policy(policy_type)
    except ValueError as e:
        raise ValueError(f"Failed to get policy '{policy_type}': {e}")


def validate_config(config: Dict[str, Any], component_type: str) -> bool:
    """
    Validate a component configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        component_type: Type of component ("reward", "terrain", "policy")
        
    Returns:
        True if config is valid
        
    Raises:
        ValueError: If config is invalid
    """
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config)}")
    
    if "type" not in config:
        raise ValueError(f"{component_type} config must have 'type' key")
    
    if component_type == "reward":
        reward_type = config["type"]
        if reward_type not in ComponentRegistry.list_rewards():
            available = ComponentRegistry.list_rewards()
            raise ValueError(
                f"Unknown reward type '{reward_type}'. Available: {available}"
            )
    
    elif component_type == "terrain":
        terrain_type = config["type"]
        if terrain_type not in ComponentRegistry.list_terrains():
            available = ComponentRegistry.list_terrains()
            raise ValueError(
                f"Unknown terrain type '{terrain_type}'. Available: {available}"
            )
    
    elif component_type == "policy":
        policy_type = config["type"]
        if policy_type not in ComponentRegistry.list_policies():
            available = ComponentRegistry.list_policies()
            raise ValueError(
                f"Unknown policy type '{policy_type}'. Available: {available}"
            )
    
    else:
        raise ValueError(
            f"Unknown component_type '{component_type}'. "
            f"Must be one of: 'reward', 'terrain', 'policy'"
        )
    
    return True

