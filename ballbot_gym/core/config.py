"""Configuration utilities and validation."""
from typing import Dict, Any, Optional
from pathlib import Path
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with config_file.open("r") as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence.
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = base.copy()
    
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def load_training_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration and merge with referenced env config.
    
    Training configs MUST have 'env_config' key pointing to an env config file.
    The env config provides terrain, reward, camera, and env settings.
    Training config provides algorithm hyperparameters and training settings.
    
    Args:
        config_path: Path to training configuration YAML file
        
    Returns:
        Merged configuration dictionary with env config merged in
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If env_config key is missing or invalid
        
    Example:
        Training config:
            env_config: "configs/env/perlin_directional.yaml"
            algo:
              ent_coef: 0.001
            total_timesteps: 10e6
        
        Env config:
            terrain:
              type: "perlin"
              config: {...}
            reward:
              type: "directional"
              config: {...}
        
        Result: Merged config with both training and env settings
    """
    config = load_config(config_path)
    
    # Require env_config key
    env_config_path = config.get("env_config")
    if not env_config_path:
        raise ValueError(
            f"Training config must specify 'env_config' key pointing to an environment config.\n"
            f"Example: env_config: 'configs/env/perlin_directional.yaml'\n"
            f"Config file: {config_path}"
        )
    
    # Resolve relative paths
    config_file = Path(config_path)
    env_config_file = Path(env_config_path)
    
    if not env_config_file.is_absolute():
        # If path starts with "configs/", resolve from repo root
        if env_config_path.startswith("configs/"):
            # Try resolving from current working directory (repo root)
            repo_root = Path.cwd()
            env_config_path = str(repo_root / env_config_path)
        else:
            # Relative to training config directory
            env_config_path = str(config_file.parent.parent / env_config_path)
    
    # Load and merge env config
    env_config = load_config(env_config_path)
    
    # Merge: training config overrides env config (allows overrides)
    merged = merge_configs(env_config, config)
    
    # Ensure problem section exists (from env config)
    if "problem" not in merged:
        merged["problem"] = {}
    
    # Move env config's terrain/reward to problem section if not already there
    if "terrain" in env_config and "terrain" not in merged["problem"]:
        merged["problem"]["terrain"] = env_config["terrain"]
    if "reward" in env_config and "reward" not in merged["problem"]:
        merged["problem"]["reward"] = env_config["reward"]
    
    # Remove env_config key from merged (it's just a reference)
    merged.pop("env_config", None)
    
    return merged


def get_component_config(
    config: Dict[str, Any],
    component_type: str,
    default_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract component configuration from full config dictionary.
    
    Args:
        config: Full configuration dictionary
        component_type: Type of component ("reward", "terrain", "policy")
        default_type: Default type to use if not specified in config
        
    Returns:
        Component configuration dictionary with "type" and "config" keys
        
    Example:
        >>> full_config = {
        ...     "problem": {
        ...         "reward": {"type": "directional", "config": {"target_direction": [0, 1]}}
        ...     }
        ... }
        >>> reward_config = get_component_config(full_config, "reward")
        >>> reward_config
        {"type": "directional", "config": {"target_direction": [0, 1]}}
    """
    problem_config = config.get("problem", {})
    component_config = problem_config.get(component_type, {})
    
    # Handle backward compatibility: if component_type is a string, treat as type
    if isinstance(component_config, str):
        return {"type": component_config, "config": {}}
    
    # If component_config is empty and default_type is provided, use it
    if not component_config and default_type:
        return {"type": default_type, "config": {}}
    
    # Ensure component_config has "type" key
    if not isinstance(component_config, dict) or "type" not in component_config:
        if default_type:
            if isinstance(component_config, dict):
                return {"type": default_type, "config": component_config}
            else:
                return {"type": default_type, "config": {}}
        else:
            raise ValueError(
                f"Component config for '{component_type}' must have 'type' key "
                f"or be a string, got: {component_config}"
            )
    
    # Ensure "config" key exists
    if "config" not in component_config:
        component_config["config"] = {}
    
    return component_config

