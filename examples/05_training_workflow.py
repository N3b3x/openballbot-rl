"""
Example 5: Complete Training Workflow

This example demonstrates a complete training workflow using the new
configuration-driven component system.

Run this example:
    python examples/05_training_workflow.py
"""
import yaml
from pathlib import Path
import gymnasium as gym
import ballbot_gym
from ballbot_gym.core.config import get_component_config


def main():
    """Demonstrate complete training workflow."""
    print("=" * 60)
    print("Example: Complete Training Workflow")
    print("=" * 60)
    
    # Step 1: Load configuration
    print("\n1. Loading configuration...")
    config_path = Path("configs/train/ppo_directional.yaml")
    if not config_path.exists():
        print(f"   ⚠ Config file not found: {config_path}")
        print("   Using default configuration...")
        config = {
            "problem": {
                "terrain": {"type": "perlin", "config": {}},
                "reward": {"type": "directional", "config": {"target_direction": [0.0, 1.0]}}
            }
        }
    else:
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        print(f"   ✓ Loaded config from {config_path}")
    
    # Step 2: Extract component configs
    print("\n2. Extracting component configurations...")
    terrain_config = get_component_config(config, "terrain", default_type="perlin")
    reward_config = get_component_config(config, "reward", default_type="directional")
    
    print(f"   Terrain config: {terrain_config}")
    print(f"   Reward config: {reward_config}")
    
    # Step 3: Create environment with configs
    print("\n3. Creating environment...")
    env = gym.make(
        "ballbot-v0.1",
        GUI=False,
        terrain_config=terrain_config,
        reward_config=reward_config,
        max_ep_steps=100
    )
    print(f"   ✓ Environment created")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space keys: {list(env.observation_space.spaces.keys())}")
    
    # Step 4: Run a short training-like loop
    print("\n4. Running training loop (simplified)...")
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    episode_length = 0
    
    for step in range(100):
        # In real training, this would come from your policy
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_length += 1
        
        if terminated or truncated:
            break
    
    print(f"   Episode length: {episode_length}")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Final position: {info['pos2d']}")
    
    env.close()
    
    # Step 5: Show how to use with training script
    print("\n5. Using with training script...")
    print("   To train with this configuration:")
    print("   ```bash")
    print("   ballbot-train --config configs/train/ppo_directional.yaml")
    print("   ```")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

