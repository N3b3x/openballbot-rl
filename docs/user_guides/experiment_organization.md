# Experiment Organization Guide

## Overview

All training experiments are automatically organized in dedicated folders under `outputs/experiments/runs/`. Each experiment gets its own folder with a descriptive name and contains all outputs, logs, checkpoints, and configuration files.

## Experiment Folder Structure

```
outputs/experiments/runs/
└── {timestamp}_{algo}_{env_config}_{seed}/
    ├── config.yaml              # Complete training configuration (merged)
    ├── info.txt                 # Experiment metadata (JSON)
    ├── progress.csv             # Training progress logs (from SB3)
    ├── stdout.txt               # Standard output logs
    ├── best_model/              # Best model checkpoint
    │   └── best_model.zip       # Best performing model (based on eval)
    ├── checkpoints/             # Periodic model checkpoints
    │   ├── ppo_agent_20000_steps.zip
    │   ├── ppo_agent_40000_steps.zip
    │   └── ...
    └── results/                 # Evaluation results
        └── evaluations.npz      # Evaluation metrics over time
```

## Experiment Naming Convention

Experiments are automatically named using the format:

```
{timestamp}_{algorithm}_{env_config}_{seed}
```

**Example:**
```
20241203_143022_ppo_perlin_directional_seed10
```

**Components:**
- `timestamp`: `YYYYMMDD_HHMMSS` format (e.g., `20241203_143022`)
- `algorithm`: Algorithm name from config (e.g., `ppo`, `sac`)
- `env_config`: Environment config filename without extension (e.g., `perlin_directional`, `flat_directional`)
- `seed`: Random seed used (e.g., `seed10`, `seed42`)

**Benefits:**
- ✅ Immediately identifies when experiment was run
- ✅ Shows which algorithm was used
- ✅ Shows which environment config was used
- ✅ Includes seed for reproducibility
- ✅ Unique names prevent overwrites (unless explicitly confirmed)

## What Gets Stored

### 1. Configuration Files

**`config.yaml`**
- Complete merged configuration (environment + training settings)
- Used for reproducibility and understanding experiment setup
- Can be used to resume training or recreate the experiment

**`info.txt`**
- Experiment metadata in JSON format:
  ```json
  {
    "algo": "ppo",
    "num_envs": 10,
    "out": "/path/to/experiment",
    "resume": "",
    "seed": 10
  }
  ```

### 2. Training Logs

**`progress.csv`**
- Training metrics logged by Stable-Baselines3
- Columns include:
  - `time/total_timesteps`: Total steps trained
  - `rollout/ep_len_mean`: Mean episode length
  - `rollout/ep_rew_mean`: Mean episode reward
  - `train/value_loss`: Value function loss
  - `train/policy_loss`: Policy loss
  - `train/entropy_loss`: Entropy loss
  - `train/learning_rate`: Current learning rate
  - `eval/mean_reward`: Evaluation reward
  - And more...

**`stdout.txt`**
- Standard output from training script
- Includes initialization messages, warnings, and print statements

### 3. Model Checkpoints

**`best_model/best_model.zip`**
- Best performing model based on evaluation metrics
- Automatically saved when evaluation performance improves
- Use this for final evaluation or deployment

**`checkpoints/ppo_agent_{steps}_steps.zip`**
- Periodic checkpoints saved every 20,000 steps (configurable)
- Useful for:
  - Resuming training from a specific point
  - Analyzing training progression
  - Comparing models at different training stages

### 4. Evaluation Results

**`results/evaluations.npz`**
- NumPy archive containing evaluation metrics
- Includes:
  - `timesteps`: Timesteps at which evaluations occurred
  - `results`: Evaluation rewards
  - `ep_lengths`: Episode lengths during evaluation

## Custom Output Directory

You can specify a custom output directory in the training config:

```yaml
# configs/train/ppo_directional.yaml
out: "my_experiments/my_custom_name"
```

**Behavior:**
- If `out` is a directory path: Creates `{out}/{experiment_name}/`
- If `out` is empty: Uses default `outputs/experiments/runs/{experiment_name}/`
- If `out` is a file path: Uses that exact path (not recommended)

**Example:**
```yaml
out: "outputs/experiments/ablation_study"
# Creates: outputs/experiments/ablation_study/20241203_143022_ppo_perlin_directional_seed10/
```

## Finding Experiments

### By Timestamp
```bash
ls -lt outputs/experiments/runs/ | head -10
```

### By Environment Config
```bash
ls outputs/experiments/runs/*perlin_directional*
```

### By Algorithm
```bash
ls outputs/experiments/runs/*ppo*
```

### By Seed
```bash
ls outputs/experiments/runs/*seed10*
```

## Resuming Training

To resume training from a checkpoint:

```yaml
# configs/train/ppo_directional.yaml
resume: "outputs/experiments/runs/20241203_143022_ppo_perlin_directional_seed10/checkpoints/ppo_agent_200000_steps.zip"
```

The training script will:
1. Load the model from the checkpoint
2. Continue training from that point
3. Create a new experiment folder (unless `out` is specified)

## Best Practices

1. **Use Descriptive Config Names**: Name your env configs clearly (e.g., `perlin_directional_easy.yaml`) so experiment names are informative

2. **Keep Configs**: Always keep `config.yaml` files - they're essential for reproducibility

3. **Organize by Study**: Use custom `out` paths for different studies:
   ```yaml
   out: "outputs/experiments/hyperparameter_sweep"
   out: "outputs/experiments/ablation_study"
   out: "outputs/experiments/final_models"
   ```

4. **Document Experiments**: Consider adding a `README.md` or `notes.txt` in important experiment folders

5. **Clean Up**: Periodically archive or delete old experiments to save disk space

## Disk Space Considerations

Each experiment typically uses:
- **Config files**: < 10 KB
- **Logs**: 1-10 MB (depends on training length)
- **Checkpoints**: 5-50 MB per checkpoint (depends on model size)
- **Best model**: 5-50 MB

For a full training run (10M steps):
- Total size: ~100-500 MB per experiment
- With 10 experiments: ~1-5 GB

## Troubleshooting

### Experiment Folder Already Exists

If an experiment folder already exists and is not empty, the training script will:
1. Prompt you to confirm overwrite
2. If confirmed: Delete old folder and create new one
3. If declined: Exit without training

**Solution:** Use a different `out` path or manually delete the old folder.

### Missing Checkpoints

Checkpoints are saved every 20,000 steps by default. If you need more frequent checkpoints, modify `callbacks.py`:

```python
CheckpointCallback(10000,  # Save every 10,000 steps instead
                   save_path=str(out_path / "checkpoints"),
                   name_prefix=f"{config['algo']['name']}_agent")
```

### Finding Best Model

The best model is automatically saved in `best_model/best_model.zip`. This is based on evaluation performance, not training reward.

## Related Documentation

- [Training Guide](training.md) - How to run training
- [Evaluation Guide](evaluation.md) - How to evaluate models
- [Configuration System](../../configs/README.md) - Understanding configs

