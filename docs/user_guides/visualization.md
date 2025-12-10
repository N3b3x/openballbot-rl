# 🎨 Visualization Guide

*Complete guide to visualizing environments, models, and training progress*

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Visualizing Environments](#visualizing-environments)
3. [Visualizing Trained Models](#visualizing-trained-models)
4. [Visualizing All Archived Models](#visualizing-all-archived-models)
5. [Plotting Training Progress](#plotting-training-progress)
6. [Window Titles and Identification](#window-titles-and-identification)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The Ballbot RL project provides several visualization tools for different purposes:

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `ballbot-browse-env` | Interactive environment browser | Exploring available environments, testing terrain configurations |
| `ballbot-visualize-env` | Direct environment visualization | Quick visualization of a specific config file |
| `ballbot-visualize-model` | Single model visualization | Testing a specific trained model |
| `visualize_all_archived_models.py` | Batch model visualization | Comparing multiple archived models |
| `ballbot-plot-training` | Training curve plots | Analyzing training progress |

---

## 🌍 Visualizing Environments

### Interactive Environment Browser

The **interactive browser** is the recommended way to explore environments:

```bash
# Interactive mode (recommended)
ballbot-browse-env

# Or as Python module
python -m ballbot_rl.visualization.browse_environments
```

**Features:**
- Browse predefined environment configurations
- Browse training configurations (which reference env configs)
- Select and configure terrain types interactively
- Select and configure reward types interactively
- Create fully custom environments
- Automatically detects macOS and uses `mjpython` when available

**Use Cases:**
- Exploring available terrain types
- Testing custom terrain configurations
- Understanding environment setup before training
- Debugging environment configuration issues

### Direct Environment Visualization

For quick visualization of a specific configuration:

```bash
# Visualize from environment config
ballbot-visualize-env --env_config configs/env/perlin_directional.yaml

# Visualize from training config (uses env_config from training config)
ballbot-visualize-env --train_config configs/train/ppo_directional.yaml

# Visualize with custom terrain type
ballbot-visualize-env --terrain_type flat --n_episodes 2

# Or as Python module
python -m ballbot_rl.visualization.visualize_env --env_config configs/env/perlin_directional.yaml
```

**Use Cases:**
- Quick verification of a config file
- Testing environment setup in CI/CD
- Non-interactive visualization workflows

**Key Differences:**
- `browse-env`: Interactive, exploratory, user-friendly
- `visualize-env`: Direct, scriptable, faster for known configs

---

## 🤖 Visualizing Trained Models

### Single Model Visualization

Visualize a trained model in MuJoCo:

```bash
# Basic usage
ballbot-visualize-model --model_path outputs/experiments/archived_models/2025-12-04_ppo-flat-directional-seed10/best_model.zip

# With more episodes
ballbot-visualize-model --model_path .../best_model.zip --n_episodes 5

# Keep viewer open for continuous testing
ballbot-visualize-model --model_path .../best_model.zip --keep_open

# Or as Python module
python -m ballbot_rl.visualization.visualize_model --model_path .../best_model.zip
```

**Features:**
- Automatically loads model and environment configuration
- Runs deterministic policy (no exploration)
- Shows episode rewards and step counts
- Window title shows model name and terrain type
- Supports GUI reset button for continuous testing
- Graceful Ctrl+C handling

**Window Title Format:**
```
Ballbot RL - {model_name} ({terrain_type})
```

Example: `Ballbot RL - 2025-12-04_ppo-flat-directional-seed10 (flat)`

**Keep Open Mode:**
- Use `--keep_open` to keep the viewer open after episodes complete
- Click "Reset" in MuJoCo GUI to run additional episodes
- Press Ctrl+C to exit gracefully

---

## 📊 Visualizing All Archived Models

The batch visualization tool processes all archived models:

```bash
# Show progress reports for all models
python scripts/utils/visualize_all_archived_models.py

# Show progress and plot training curves (interactive display)
python scripts/utils/visualize_all_archived_models.py --plot

# Save plots without displaying them
python scripts/utils/visualize_all_archived_models.py --save-plots

# Visualize models in MuJoCo (sequential, one at a time)
python scripts/utils/visualize_all_archived_models.py --visualize

# Visualize all models in parallel (all viewers open simultaneously)
python scripts/utils/visualize_all_archived_models.py --visualize --parallel

# Do everything: progress, plots, and visualization
python scripts/utils/visualize_all_archived_models.py --plot --visualize

# Specific model only
python scripts/utils/visualize_all_archived_models.py --model 2025-12-04_ppo-flat-directional-seed10

# Include legacy models
python scripts/utils/visualize_all_archived_models.py --include-legacy
```

**Features:**
- Lists all archived models with metadata
- Shows training progress summaries
- Generates training curve plots
- Visualizes models in MuJoCo (sequential or parallel)
- Each viewer window has a unique title for identification
- Proper process termination on Ctrl+C

**Parallel Mode:**
- Launches all viewers simultaneously
- Each window has a descriptive title
- Output is suppressed to avoid terminal clutter
- Press Ctrl+C to terminate all processes gracefully

**Sequential Mode:**
- Views one model at a time
- Automatically uses `--keep_open` for continuous testing
- Shows plots before visualization (if `--plot` not specified)
- GUI reset button works for continuous policy execution

---

## 📈 Plotting Training Progress

Generate training curve plots:

```bash
# Using CLI command (recommended)
ballbot-plot-training \
    --csv outputs/experiments/runs/.../progress.csv \
    --config outputs/experiments/runs/.../config.yaml \
    --plot_train

# Or using Python module
python -m ballbot_rl.visualization.plot_training \
    --csv outputs/experiments/runs/.../progress.csv \
    --config outputs/experiments/runs/.../config.yaml \
    --plot_train
```

**Features:**
- Plots reward over training steps
- Shows training and validation curves
- Configurable plot styling
- Can save plots to files

---

## 🏷️ Window Titles and Identification

When visualizing multiple models, each MuJoCo window has a descriptive title:

**Format:** `Ballbot RL - {model_name} ({terrain_type})`

**Examples:**
- `Ballbot RL - 2025-12-04_ppo-flat-directional-seed10 (flat)`
- `Ballbot RL - 2025-12-04_ppo-perlin-directional-seed10 (perlin)`
- `Ballbot RL - legacy (perlin)`

This makes it easy to identify which model is displayed in each window when using parallel visualization.

**Note:** Window title setting may fail silently if GLFW window access is unavailable. The viewer will still work, just without a custom title.

---

## 🔧 Troubleshooting

### MuJoCo Viewer Not Opening (macOS)

**Problem:** Viewer doesn't open or shows warning about `mjpython`.

**Solution:**
- On macOS, you must use `mjpython` instead of `python`
- The scripts automatically detect macOS and use `mjpython` when available
- If you see warnings, ensure MuJoCo is properly installed with `mjpython`

```bash
# Check if mjpython is available
which mjpython

# If not available, install MuJoCo Python bindings
# See installation guide in README.md
```

### Ctrl+C Not Working

**Problem:** Pressing Ctrl+C doesn't exit the visualization.

**Solution:**
- The scripts now have improved interrupt handling
- Press Ctrl+C once and wait a moment
- In parallel mode, all processes will be terminated
- If stuck, you can close the MuJoCo windows manually

### Multiple Windows Open

**Problem:** Too many MuJoCo windows open, hard to identify which is which.

**Solution:**
- Each window should have a unique title (see [Window Titles](#window-titles-and-identification))
- Use sequential mode (`--visualize` without `--parallel`) to view one at a time
- Close windows manually when done

### Viewer Closes Immediately

**Problem:** Viewer closes as soon as the robot fails.

**Solution:**
- Use `--keep_open` flag for `ballbot-visualize-model`
- Sequential mode in `visualize_all_archived_models.py` automatically uses `keep_open`
- Click "Reset" in MuJoCo GUI to run another episode

### No Plots Showing

**Problem:** Plots don't appear when using `--visualize`.

**Solution:**
- Use `--plot` flag explicitly: `--plot --visualize`
- Or plots will show automatically before visualization in sequential mode
- Use `--save-plots` to save plots without displaying

---

## 📝 Quick Reference

### Environment Visualization
```bash
# Interactive browser
ballbot-browse-env

# Direct visualization
ballbot-visualize-env --env_config configs/env/perlin_directional.yaml
```

### Model Visualization
```bash
# Single model
ballbot-visualize-model --model_path .../best_model.zip --keep_open

# All archived models (parallel)
python scripts/utils/visualize_all_archived_models.py --visualize --parallel

# All archived models (sequential with plots)
python scripts/utils/visualize_all_archived_models.py --plot --visualize
```

### Training Progress
```bash
ballbot-plot-training --csv .../progress.csv --config .../config.yaml --plot_train
```

---

## 🔗 Related Documentation

- [Training Guide](training.md) - How to train models
- [Experiment Organization](experiment_organization.md) - Managing training runs and archived models
- [Quick Reference](quick_reference.md) - Command cheat sheet
- [FAQ](faq.md) - Common questions

---

*Last Updated: 2025*
