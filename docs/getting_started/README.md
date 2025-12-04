# Getting Started with openballbot-rl

Welcome to openballbot-rl! This guide will help you get started quickly.

## Quick Links

- [Installation](installation.md) - Installation instructions
- [First Steps](first_steps.md) - Your first environment run
- [Quick Start](quick_start.md) - 5-minute quick start guide

## What is openballbot-rl?

openballbot-rl is a reinforcement learning environment for training ballbot robots to navigate uneven terrain. It features:

- **MuJoCo Physics**: Realistic robot simulation
- **RGB-D Cameras**: Visual perception for terrain navigation
- **Extensible Architecture**: Easy to add custom rewards, terrains, and policies
- **Configuration-Driven**: Switch components via YAML configs

## Installation

See [Installation Guide](installation.md) for detailed instructions.

Quick install:

```bash
make install
```

## Your First Run

See [First Steps](first_steps.md) for a step-by-step tutorial.

Quick example:

```python
import gymnasium as gym
import ballbot_gym

env = gym.make("ballbot-v0.1", terrain_type="flat")
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

## Next Steps

1. **Run Examples**: Check out `examples/` directory
2. **Read Documentation**: See `docs/` for detailed guides
3. **Understand Concepts**: 
   - [Ballbot Mechanics](../concepts/ballbot_mechanics.md) - Physics and dynamics
   - [RL Fundamentals](../concepts/rl_fundamentals.md) - MDP formulation
   - [Reward Design](../concepts/reward_design.md) - Reward engineering
   - [Observation Design](../concepts/observation_design.md) - Multi-modal fusion
4. **Choose Learning Path**: See [Learning Paths](../LEARNING_PATHS.md) for personalized guides
3. **Add Components**: Follow [Extension Guide](../architecture/extension_guide.md)
4. **Train a Policy**: See [Training Guide](../user_guides/training.md)

## Need Help?

- Check [FAQ](../user_guides/faq.md)
- See [Examples](../../examples/)
- Read [Architecture Docs](../architecture/)

