# Quick Start Guide

Get up and running with openballbot-rl in 5 minutes!

## 1. Installation (2 minutes)

```bash
# Install MuJoCo with patch (see installation.md for details)
# Then install openballbot-rl
make install

# Verify installation
python scripts/test_pid.py
```

## 2. Basic Usage (1 minute)

```python
import gymnasium as gym
import ballbot_gym

# Create environment
env = gym.make("ballbot-v0.1", terrain_type="flat")

# Reset and step
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

## 3. Training (2 minutes)

```bash
# Train with default config
ballbot-train --config configs/train/ppo_directional.yaml

# Or use Python module
python -m ballbot_rl.training.train --config configs/train/ppo_directional.yaml
```

## 4. Evaluation

```bash
# Evaluate trained policy
ballbot-eval --algo ppo --path outputs/models/example_model.zip --n_test 5
```

## What's Next?

- **Examples**: See `examples/` directory
- **Documentation**: Read `docs/` for detailed guides
- **Custom Components**: Follow [Extension Guide](../architecture/extension_guide.md)

## Key Concepts

- **Environment**: `ballbot-v0.1` - Gymnasium environment
- **Actions**: 3D continuous [-1, 1] for three omniwheels
- **Observations**: Dict with proprioceptive and visual data
- **Rewards**: Configurable reward functions
- **Terrains**: Configurable terrain generators

## Configuration

Components are selected via YAML config:

```yaml
problem:
  terrain:
    type: "perlin"  # or "flat", "stepped"
    config: {}
  reward:
    type: "directional"
    config:
      target_direction: [0.0, 1.0]
```

## Need Help?

- [FAQ](../user_guides/faq.md)
- [Architecture](../architecture/)
- [Examples](../../examples/)

