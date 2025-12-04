"""
Example 6: Configuration Examples

This example shows various configuration examples for different use cases.

Run this example:
    python examples/06_configuration_examples.py
"""
import yaml
from ballbot_gym.core.config import get_component_config


def print_config_example(name: str, config: dict):
    """Print a configuration example."""
    print(f"\n{name}:")
    print("-" * 60)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))


def main():
    """Show various configuration examples."""
    print("=" * 60)
    print("Example: Configuration Examples")
    print("=" * 60)
    
    # Example 1: Perlin terrain with custom parameters
    print_config_example(
        "Example 1: Perlin Terrain with Custom Parameters",
        {
            "problem": {
                "terrain": {
                    "type": "perlin",
                    "config": {
                        "scale": 30.0,  # Larger features
                        "octaves": 6,   # More detail
                        "persistence": 0.3,
                        "lacunarity": 2.0,
                        "seed": None    # Random each episode
                    }
                }
            }
        }
    )
    
    # Example 2: Stepped terrain
    print_config_example(
        "Example 2: Stepped Terrain",
        {
            "problem": {
                "terrain": {
                    "type": "stepped",
                    "config": {
                        "num_steps": 8,
                        "step_height": 0.15,
                        "seed": None
                    }
                }
            }
        }
    )
    
    # Example 3: Directional reward with custom direction
    print_config_example(
        "Example 3: Directional Reward (Custom Direction)",
        {
            "problem": {
                "reward": {
                    "type": "directional",
                    "config": {
                        "target_direction": [1.0, 0.0]  # Move in +X direction
                    }
                }
            }
        }
    )
    
    # Example 4: Distance-based reward
    print_config_example(
        "Example 4: Distance-Based Reward",
        {
            "problem": {
                "reward": {
                    "type": "distance",
                    "config": {
                        "goal_position": [5.0, 3.0],  # Target position [x, y]
                        "scale": 0.1
                    }
                }
            }
        }
    )
    
    # Example 5: Complete training config
    print_config_example(
        "Example 5: Complete Training Configuration",
        {
            "algo": {
                "name": "ppo",
                "ent_coef": 0.001,
                "clip_range": 0.015,
                "learning_rate": -1,  # Use scheduler
                "n_steps": 2048,
                "n_epochs": 5,
                "batch_sz": 256
            },
            "problem": {
                "terrain": {
                    "type": "perlin",
                    "config": {
                        "scale": 25.0,
                        "octaves": 4,
                        "seed": None
                    }
                },
                "reward": {
                    "type": "directional",
                    "config": {
                        "target_direction": [0.0, 1.0]
                    }
                },
                "policy": {
                    "type": "mlp",
                    "config": {
                        "hidden_sizes": [128, 128, 128, 128],
                        "activation": "leaky_relu"
                    }
                }
            },
            "total_timesteps": 10000000,
            "num_envs": 10,
            "seed": 42
        }
    )
    
    print("\n" + "=" * 60)
    print("Configuration examples shown!")
    print("=" * 60)
    print("\nTo use these configs:")
    print("1. Save to a YAML file (e.g., configs/train/my_config.yaml)")
    print("2. Run: ballbot-train --config configs/train/my_config.yaml")


if __name__ == "__main__":
    main()

