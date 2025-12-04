# Ballbot RL

Reinforcement learning training and evaluation package for ballbot navigation.

## Overview

This package provides tools for training and evaluating reinforcement learning policies for ballbot navigation in uneven terrain.

## Components

- **Training** (`ballbot_rl.training`): PPO training with configurable hyperparameters
- **Evaluation** (`ballbot_rl.evaluation`): Policy evaluation and testing
- **Data Collection** (`ballbot_rl.data`): Data collection utilities
- **Encoders** (`ballbot_rl.encoders`): Pretrained depth encoders for visual observations
- **Policies** (`ballbot_rl.policies`): Policy network architectures

## Installation

Install as an editable package:

```bash
pip install -e ballbot_rl/
```

## Usage

### Training

```bash
ballbot-train --config configs/train/ppo_directional.yaml
```

### Evaluation

```bash
ballbot-eval --algo ppo --path outputs/models/my_model.zip
```

## Dependencies

- `ballbot-gym`: Ballbot environment package
- `stable-baselines3`: RL algorithms
- `torch`: Deep learning framework
- `gymnasium`: RL environment interface

See `pyproject.toml` for complete dependency list.

