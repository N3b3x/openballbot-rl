# Configuration Files

This directory contains all configuration files for training, evaluation, and environments.

## Directory Structure

```
configs/
├── env/          # Environment configurations (terrain + reward)
├── train/        # Training configurations (algorithm + hyperparameters)
├── eval/         # Evaluation configurations
└── README.md     # This file
```

## Quick Start

### Training

1. Choose or create an environment config in `configs/env/`:
   ```yaml
   # configs/env/perlin_directional.yaml
   terrain:
     type: "perlin"
     config: {...}
   reward:
     type: "directional"
     config: {...}
   ```

2. Create or use a training config that references it:
   ```yaml
   # configs/train/ppo_directional.yaml
   env_config: "configs/env/perlin_directional.yaml"
   algo:
     name: ppo
     ent_coef: 0.001
     # ... hyperparameters
   ```

3. Train:
   ```bash
   python -m ballbot_rl.training.train --config configs/train/ppo_directional.yaml
   ```

### Evaluation

1. Use default eval config or create custom:
   ```yaml
   # configs/eval/default.yaml
   n_test_episodes: 5
   deterministic: true
   render: true
   env_config: null  # Uses model's training config
   ```

2. Evaluate:
   ```bash
   python -m ballbot_rl.evaluation.evaluate \
     --algo ppo \
     --path outputs/models/my_model.zip \
     --eval_config configs/eval/default.yaml
   ```

## Configuration System

### Environment Configs (`configs/env/`)

Define the **problem** (terrain + reward) and **environment settings** (camera, episode limits, etc.).

**Naming:** `{terrain_type}_{reward_type}.yaml`

See `configs/env/README.md` for details.

### Training Configs (`configs/train/`)

Define **algorithm hyperparameters** and reference an environment config.

**Required:** `env_config` key pointing to an env config file.

### Evaluation Configs (`configs/eval/`)

Define **evaluation settings** (number of episodes, deterministic mode, etc.).

**Optional:** `env_config` to override the training environment.

## Config Merging

When loading a training config:

1. Training config specifies `env_config: "configs/env/perlin_directional.yaml"`
2. System loads and merges the env config
3. Training config values override env config values (if conflicts)
4. Result: Complete config with both environment and training settings

## Examples

See `examples/` directory for complete usage examples.

