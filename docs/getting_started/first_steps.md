# First Steps Tutorial

This tutorial walks you through your first interactions with openballbot-rl.

## Step 1: Create Your First Environment

```python
import gymnasium as gym
import ballbot_gym

# Create environment
env = gym.make(
    "ballbot-v0.1",
    GUI=False,  # Set to True to visualize (requires mjpython on macOS)
    terrain_type="flat",  # Start with flat terrain
    max_ep_steps=1000
)
```

## Step 2: Understand the Environment

```python
# Check action and observation spaces
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Action space: Box(-1.0, 1.0, shape=(3,))
# - 3D continuous actions for three omniwheels
# - Values normalized to [-1, 1]

# Observation space: Dict with keys:
# - "orientation": Robot orientation (3D rotation vector)
# - "angular_vel": Angular velocity (3D)
# - "vel": Linear velocity (3D)
# - "motor_state": Wheel velocities (3D)
# - "actions": Previous action (3D)
# - "rgbd_0", "rgbd_1": RGB-D images (if cameras enabled)
```

## Step 3: Run an Episode

```python
# Reset environment
obs, info = env.reset(seed=42)
print(f"Initial position: {info['pos2d']}")

# Run episode
total_reward = 0.0
for step in range(100):
    # Sample random action (in practice, use your policy)
    action = env.action_space.sample()
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if terminated or truncated:
        print(f"Episode ended at step {step}")
        break

print(f"Total reward: {total_reward:.2f}")
print(f"Final position: {info['pos2d']}")
```

## Step 4: Use Custom Configuration

```python
# Create environment with custom reward and terrain
reward_config = {
    "type": "directional",
    "config": {"target_direction": [1.0, 0.0]}  # Move in +X direction
}

terrain_config = {
    "type": "perlin",
    "config": {"scale": 25.0, "octaves": 4, "seed": 42}
}

env = gym.make(
    "ballbot-v0.1",
    GUI=False,
    reward_config=reward_config,
    terrain_config=terrain_config
)

obs, _ = env.reset()
# ... run episode ...
env.close()
```

## Step 5: Check Available Components

```python
from ballbot_gym.core.registry import ComponentRegistry

# List available components
print("Available rewards:", ComponentRegistry.list_rewards())
print("Available terrains:", ComponentRegistry.list_terrains())
print("Available policies:", ComponentRegistry.list_policies())
```

## Common Issues

### Environment Creation Fails

- Check MuJoCo is installed correctly
- Verify `MUJOCO_PATH` is set
- Try `GUI=False` first

### Import Errors

- Ensure packages are installed: `pip install -e ballbot_gym/ ballbot_rl/`
- Check Python version: `python --version` (should be 3.9+)

## Next Steps

- [Quick Start](quick_start.md) - 5-minute guide
- [Examples](../../examples/) - More examples
- [Training Guide](../user_guides/training.md) - Train a policy

