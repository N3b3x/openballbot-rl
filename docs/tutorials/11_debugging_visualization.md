# 🔍 Debugging & Visualization for Reinforcement Learning

*A comprehensive guide to debugging training, analyzing policies, and visualizing RL systems*

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Understanding Training Logs](#understanding-training-logs)
3. [Visualizing Training Progress](#visualizing-training-progress)
4. [Analyzing Loss Components](#analyzing-loss-components)
5. [Debugging Policy Behavior](#debugging-policy-behavior)
6. [Reward Component Analysis](#reward-component-analysis)
7. [Common Training Issues](#common-training-issues)
8. [Performance Profiling](#performance-profiling)
9. [Modern Experiment Tracking](#modern-experiment-tracking) ⭐
10. [Advanced Visualization](#advanced-visualization) ⭐
11. [Automated Hyperparameter Tuning](#automated-hyperparameter-tuning) ⭐
12. [Best Practices](#best-practices)
13. [Summary](#summary)

---

## 🎯 Introduction

Debugging reinforcement learning systems is challenging because:
- **Delayed feedback**: Rewards come after actions
- **Non-stationary**: Policy changes during training
- **Multiple components**: Environment, policy, value function all interact
- **High-dimensional observations**: Hard to visualize

> "Debugging RL is like debugging a moving target. The system changes as you observe it."  
> — *Common wisdom in deep RL*

**Key Questions This Tutorial Answers:**
- How do I know if training is working?
- What metrics should I monitor?
- How do I debug policy failures?
- How do I visualize training progress?
- What are common failure modes and how do I fix them?

---

## 📊 Understanding Training Logs

### Log File Structure

During training, Stable-Baselines3 logs metrics to `log/progress.csv`:

```csv
# rollout/ep_len_mean,rollout/ep_rew_mean,train/learning_rate,...
1.23e+03,45.6,1e-4,...
1.25e+03,46.2,1e-4,...
...
```

### Key Metrics Explained

#### 1. Episode Metrics

**`rollout/ep_rew_mean`**: Average episode reward
- **What it means**: How well the policy performs on average
- **What to expect**: Should increase over time (if learning)
- **Typical range**: Depends on reward function (Ballbot: 0-100+)

**`rollout/ep_len_mean`**: Average episode length
- **What it means**: How long episodes last before termination
- **What to expect**: Should increase (robot stays balanced longer)
- **Typical range**: 0 to `max_ep_steps` (Ballbot: 0-4000)

**`rollout/ep_rew_std`**: Standard deviation of episode rewards
- **What it means**: Consistency of performance
- **What to expect**: Should decrease as policy stabilizes
- **Warning sign**: High std = unstable learning

#### 2. Training Losses

**`train/policy_gradient_loss`**: Policy gradient loss
- **What it means**: How much the policy is changing
- **What to expect**: Should decrease (policy improving)
- **Warning sign**: Oscillating = unstable learning

**`train/value_loss`**: Value function loss
- **What it means**: How well the critic estimates returns
- **What to expect**: Should decrease (better value estimates)
- **Warning sign**: Very high = critic not learning

**`train/entropy_loss`**: Entropy (exploration) loss
- **What it means**: How much the policy explores
- **What to expect**: Should decrease (policy becomes more deterministic)
- **Warning sign**: Too low = policy overfits, too high = not learning

#### 3. PPO-Specific Metrics

**`train/approx_kl`**: Approximate KL divergence
- **What it means**: How much policy changed in one update
- **What to expect**: Should be small (< 0.1 typically)
- **Warning sign**: Very high = policy changed too much (unstable)

**`train/clip_fraction`**: Fraction of clipped updates
- **What it means**: How often PPO clipping activates
- **What to expect**: 0.1-0.3 typically
- **Warning sign**: Very high (> 0.5) = learning rate too high

**`train/learning_rate`**: Current learning rate
- **What it means**: Step size for gradient updates
- **What to expect**: May decrease if using scheduler
- **Note**: Check if scheduler is working correctly

#### 4. Evaluation Metrics

**`eval/mean_reward`**: Average reward on evaluation episodes
- **What it means**: Performance on held-out evaluation set
- **What to expect**: Should track training reward (slightly lower)
- **Warning sign**: Much lower than training = overfitting

**`eval/mean_ep_length`**: Average episode length on evaluation
- **What it means**: How long evaluation episodes last
- **What to expect**: Should track training episode length

### Reading Logs Programmatically

```python
import csv
import numpy as np

def read_training_logs(csv_file):
    """Read and parse training logs."""
    data = {}
    
    with open(csv_file, 'r') as f:
        # Skip comment line
        first_line = f.readline()
        headers = first_line.lstrip('#').strip().split(',')
        
        reader = csv.DictReader(f, fieldnames=headers)
        
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                if value != '':
                    data[key].append(float(value))
    
    return data

# Example usage
logs = read_training_logs('log/progress.csv')
episode_rewards = logs['rollout/ep_rew_mean']
print(f"Final reward: {episode_rewards[-1]}")
print(f"Max reward: {max(episode_rewards)}")
```

---

## 📈 Visualizing Training Progress

### Visualization Module Overview

The project includes a comprehensive visualization module (`ballbot_rl/visualization/`) with the following tools:

1. **`plot_training.py`** - Plot training progress and loss curves from CSV logs
2. **`visualize_env.py`** - Visualize environment configurations before training
3. **`visualize_model.py`** - Visualize trained models and their behavior
4. **`browse_environments.py`** - Interactive browser for exploring environments

All tools are available as CLI commands (after installation) or as Python modules.

### Using Built-in Plotting Tools

The project includes visualization tools in `ballbot_rl/visualization/` for visualizing training. You can use the CLI command or Python module:

#### 1. Training vs. Evaluation Progress

**Using CLI command (recommended after installation):**
```bash
ballbot-plot-training \
    --csv outputs/experiments/runs/.../progress.csv \
    --config outputs/experiments/runs/.../config.yaml \
    --plot_train
```

**Or using Python module:**
```bash
python -m ballbot_rl.visualization.plot_training \
    --csv outputs/experiments/runs/.../progress.csv \
    --config outputs/experiments/runs/.../config.yaml \
    --plot_train
```

**What it shows:**
- **Reward plot**: Training and evaluation rewards over time
- **Episode length plot**: Training and evaluation episode lengths

**What to look for:**
- ✅ Training and eval rewards both increasing
- ✅ Eval reward tracks training (not much lower)
- ✅ Episode length increasing (robot balancing longer)
- ⚠️ Eval much lower than training = overfitting
- ⚠️ Rewards plateauing = may need more training or hyperparameter tuning

#### 2. Loss Component Analysis

**Using CLI command:**
```bash
ballbot-plot-training \
    --csv outputs/experiments/runs/.../progress.csv \
    --config outputs/experiments/runs/.../config.yaml
```

**Or using Python module:**
```bash
python -m ballbot_rl.visualization.plot_training \
    --csv outputs/experiments/runs/.../progress.csv \
    --config outputs/experiments/runs/.../config.yaml
```

**What it shows:**
- **Entropy loss**: Exploration over time
- **Policy gradient loss**: Policy improvement
- **Value loss**: Critic learning

**What to look for:**
- ✅ All losses decreasing (learning is happening)
- ✅ Losses stabilizing (converging)
- ⚠️ Losses oscillating = unstable learning
- ⚠️ Losses not decreasing = not learning

### Other Visualization Tools

#### 3. Visualize Environment Configuration

Before training, you can visualize your environment setup:

```bash
# Using CLI command
ballbot-visualize-env --env_config configs/env/perlin_directional.yaml

# Or from training config
ballbot-visualize-env --train_config configs/train/ppo_directional.yaml

# Or as Python module
python -m ballbot_rl.visualization.visualize_env --env_config configs/env/perlin_directional.yaml
```

This shows:
- Terrain generation (height map visualization)
- Robot spawn position
- Camera setup
- Environment behavior

#### 4. Visualize Trained Model

After training, visualize your trained policy:

```bash
# Using CLI command
ballbot-visualize-model --model_path outputs/experiments/runs/.../best_model/best_model.zip

# With more episodes
ballbot-visualize-model --model_path .../best_model.zip --n_episodes 5

# Or as Python module
python -m ballbot_rl.visualization.visualize_model --model_path .../best_model.zip
```

#### 5. Interactive Environment Browser

Browse and explore all available environments interactively:

```bash
# Interactive mode (recommended)
ballbot-browse-env

# Or as Python module
python -m ballbot_rl.visualization.browse_environments
```

This provides an interactive interface to:
- Browse predefined environment configurations
- Browse training configurations
- Select and configure terrain types
- Select and configure reward types
- Create custom environments

### Custom Visualization Scripts

#### Plot Reward Over Time

```python
import matplotlib.pyplot as plt
import csv

def plot_rewards(csv_file):
    """Plot episode rewards over training."""
    timesteps = []
    rewards = []
    
    with open(csv_file, 'r') as f:
        first_line = f.readline()
        headers = first_line.lstrip('#').strip().split(',')
        reader = csv.DictReader(f, fieldnames=headers)
        
        for row in reader:
            if row['rollout/ep_rew_mean'] != '':
                timesteps.append(float(row['time/total_timesteps']))
                rewards.append(float(row['rollout/ep_rew_mean']))
    
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards, linewidth=2)
    plt.xlabel('Total Timesteps')
    plt.ylabel('Mean Episode Reward')
    plt.title('Training Progress')
    plt.grid(True)
    plt.show()

plot_rewards('log/progress.csv')
```

#### Plot Multiple Metrics

```python
def plot_training_metrics(csv_file):
    """Plot multiple training metrics."""
    data = read_training_logs(csv_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode reward
    axes[0, 0].plot(data['rollout/ep_rew_mean'], linewidth=2)
    axes[0, 0].set_title('Episode Reward')
    axes[0, 0].set_xlabel('Update')
    axes[0, 0].grid(True)
    
    # Episode length
    axes[0, 1].plot(data['rollout/ep_len_mean'], linewidth=2, color='orange')
    axes[0, 1].set_title('Episode Length')
    axes[0, 1].set_xlabel('Update')
    axes[0, 1].grid(True)
    
    # Policy loss
    axes[1, 0].plot(data['train/policy_gradient_loss'], linewidth=2, color='green')
    axes[1, 0].set_title('Policy Loss')
    axes[1, 0].set_xlabel('Update')
    axes[1, 0].grid(True)
    
    # Value loss
    axes[1, 1].plot(data['train/value_loss'], linewidth=2, color='red')
    axes[1, 1].set_title('Value Loss')
    axes[1, 1].set_xlabel('Update')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
```

#### Moving Average Smoothing

```python
def smooth_curve(data, window=100):
    """Apply moving average smoothing."""
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2)
        smoothed.append(np.mean(data[start:end]))
    return smoothed

# Plot with smoothing
rewards = logs['rollout/ep_rew_mean']
smoothed_rewards = smooth_curve(rewards, window=100)

plt.plot(rewards, alpha=0.3, label='Raw')
plt.plot(smoothed_rewards, linewidth=2, label='Smoothed')
plt.legend()
plt.show()
```

---

## 📉 Analyzing Loss Components

### Understanding Loss Curves

#### Healthy Training

**Signs of good training:**
- All losses decreasing over time
- Losses stabilizing (converging)
- Episode reward increasing
- Episode length increasing

**Example healthy curves:**
```
Policy Loss:  [0.5, 0.4, 0.3, 0.25, 0.2, 0.18, ...]  ↓ decreasing
Value Loss:   [0.3, 0.25, 0.2, 0.15, 0.12, 0.1, ...]  ↓ decreasing
Entropy:      [0.1, 0.08, 0.06, 0.05, 0.04, ...]      ↓ decreasing
Reward:       [20, 30, 40, 50, 60, 70, ...]           ↑ increasing
```

#### Unstable Training

**Signs of instability:**
- Losses oscillating wildly
- Policy loss and value loss diverging
- Episode reward not improving

**Example unstable curves:**
```
Policy Loss:  [0.5, 0.3, 0.6, 0.2, 0.7, ...]  ↕ oscillating
Value Loss:   [0.3, 0.5, 0.2, 0.6, 0.1, ...]  ↕ oscillating
Reward:       [20, 25, 18, 30, 15, ...]        ↕ not improving
```

**Solutions:**
- Reduce learning rate
- Increase batch size
- Reduce clip range (PPO)
- Add gradient clipping

#### Not Learning

**Signs of not learning:**
- Losses not decreasing
- Episode reward not improving
- Episode length not increasing

**Example not learning:**
```
Policy Loss:  [0.5, 0.5, 0.5, 0.5, ...]  → constant
Value Loss:   [0.3, 0.3, 0.3, 0.3, ...]  → constant
Reward:       [20, 20, 20, 20, ...]      → constant
```

**Solutions:**
- Check if gradients are flowing (gradient clipping, vanishing gradients)
- Increase learning rate
- Check reward function (may be too sparse)
- Verify environment is working correctly

### Loss Component Analysis

#### Policy Loss Breakdown

```python
def analyze_policy_loss(csv_file):
    """Analyze policy loss components."""
    data = read_training_logs(csv_file)
    
    pg_loss = data['train/policy_gradient_loss']
    entropy = data['train/entropy_loss']
    kl = data['train/approx_kl']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Policy gradient loss
    axes[0].plot(pg_loss, linewidth=2, label='Policy Loss')
    axes[0].set_title('Policy Gradient Loss')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    axes[0].legend()
    
    # Entropy (exploration)
    axes[1].plot(entropy, linewidth=2, color='green', label='Entropy')
    axes[1].set_title('Entropy (Exploration)')
    axes[1].set_ylabel('Entropy')
    axes[1].grid(True)
    axes[1].legend()
    
    # KL divergence
    axes[2].plot(kl, linewidth=2, color='red', label='KL Divergence')
    axes[2].set_title('Approximate KL Divergence')
    axes[2].set_xlabel('Update')
    axes[2].set_ylabel('KL')
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Policy Loss - Mean: {np.mean(pg_loss):.4f}, Std: {np.std(pg_loss):.4f}")
    print(f"Entropy - Mean: {np.mean(entropy):.4f}, Std: {np.std(entropy):.4f}")
    print(f"KL Divergence - Mean: {np.mean(kl):.4f}, Max: {np.max(kl):.4f}")
```

**What to look for:**
- **Policy loss decreasing**: Policy improving
- **Entropy decreasing**: Policy becoming more deterministic (expected)
- **KL divergence small**: Policy not changing too much per update (stable)

---

## 🐛 Debugging Policy Behavior

### Visualizing Policy Actions

#### Action Distribution Analysis

```python
def analyze_policy_actions(model, env, n_episodes=10):
    """Analyze action distribution from policy."""
    actions = []
    rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_actions = []
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            episode_actions.append(action)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        
        actions.extend(episode_actions)
        rewards.append(episode_reward)
    
    actions = np.array(actions)
    
    # Plot action distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        axes[i].hist(actions[:, i], bins=50, alpha=0.7)
        axes[i].set_title(f'Action {i} Distribution')
        axes[i].set_xlabel('Action Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Action means: {np.mean(actions, axis=0)}")
    print(f"Action stds: {np.std(actions, axis=0)}")
    print(f"Action ranges: [{np.min(actions, axis=0)}, {np.max(actions, axis=0)}]")
    print(f"Mean episode reward: {np.mean(rewards):.2f}")
```

**What to look for:**
- ✅ Actions distributed across action space (exploring)
- ✅ Actions not stuck at boundaries (not saturating)
- ⚠️ Actions always zero = policy not learning
- ⚠️ Actions always at limits = policy overconfident

#### Action Trajectory Visualization

```python
def visualize_action_trajectory(model, env, n_steps=1000):
    """Visualize action trajectory over time."""
    obs, _ = env.reset()
    actions = []
    rewards = []
    
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        
        if done:
            break
    
    actions = np.array(actions)
    
    # Plot action trajectory
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for i in range(3):
        axes[i].plot(actions[:, i], linewidth=2, label=f'Action {i}')
        axes[i].set_title(f'Action {i} Over Time')
        axes[i].set_xlabel('Step')
        axes[i].set_ylabel('Action Value')
        axes[i].grid(True)
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()
```

### Debugging Episode Failures

#### Episode Replay with Annotations

```python
def replay_episode_with_info(model, env, seed=0):
    """Replay episode and collect debugging information."""
    obs, info = env.reset(seed=seed)
    done = False
    
    episode_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'info': []
    }
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        episode_data['observations'].append(obs.copy())
        episode_data['actions'].append(action.copy())
        
        obs, reward, done, truncated, info = env.step(action)
        episode_data['rewards'].append(reward)
        episode_data['info'].append(info.copy())
    
    return episode_data

# Analyze failed episode
episode = replay_episode_with_info(model, env, seed=42)

# Plot reward over episode
plt.plot(episode['rewards'])
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Reward Over Episode')
plt.grid(True)
plt.show()

# Check termination reason
last_info = episode['info'][-1]
print(f"Termination reason: {last_info}")
print(f"Episode length: {len(episode['rewards'])}")
print(f"Total reward: {sum(episode['rewards'])}")
```

#### Failure Mode Analysis

```python
def analyze_failure_modes(model, env, n_episodes=100):
    """Analyze common failure modes."""
    failure_reasons = {
        'tilt_exceeded': 0,
        'max_steps': 0,
        'other': 0
    }
    
    episode_lengths = []
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
        
        episode_lengths.append(step)
        episode_rewards.append(total_reward)
        
        # Check failure reason
        if info.get('failure', False):
            failure_reasons['tilt_exceeded'] += 1
        elif step >= env.max_ep_steps:
            failure_reasons['max_steps'] += 1
        else:
            failure_reasons['other'] += 1
    
    # Print statistics
    print("Failure Mode Analysis:")
    for reason, count in failure_reasons.items():
        print(f"  {reason}: {count} ({100*count/n_episodes:.1f}%)")
    
    print(f"\nEpisode Statistics:")
    print(f"  Mean length: {np.mean(episode_lengths):.1f}")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"  Success rate: {100*(1 - failure_reasons['tilt_exceeded']/n_episodes):.1f}%")
```

---

## 🎁 Reward Component Analysis

### Understanding Reward Components

The Ballbot environment logs reward components if `log_options["reward_terms"] = True`:

```python
# In bbot_env.py
if self.log_options["reward_terms"]:
    # Save reward components to /tmp/log_<random>/term_1.npy, term_2.npy
    # term_1: Directional reward
    # term_2: Action regularization
```

### Analyzing Reward Components

```python
import numpy as np

def analyze_reward_components(log_dir):
    """Analyze logged reward components."""
    term_1 = np.load(f"{log_dir}/term_1.npy")  # Directional reward
    term_2 = np.load(f"{log_dir}/term_2.npy")  # Action regularization
    
    # Plot reward components
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Directional reward
    axes[0].plot(term_1, linewidth=2, label='Directional Reward')
    axes[0].set_title('Directional Reward Component')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Reward')
    axes[0].grid(True)
    axes[0].legend()
    
    # Action regularization
    axes[1].plot(term_2, linewidth=2, color='red', label='Action Regularization')
    axes[1].set_title('Action Regularization Component')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Reward (negative)')
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Directional reward - Mean: {np.mean(term_1):.4f}, Std: {np.std(term_1):.4f}")
    print(f"Action regularization - Mean: {np.mean(term_2):.4f}, Std: {np.std(term_2):.4f}")
    print(f"Total reward - Mean: {np.mean(term_1 + term_2):.4f}")
```

**What to look for:**
- **Directional reward increasing**: Policy learning to move in target direction
- **Action regularization small**: Actions not too large (smooth control)
- **Balance**: Both components contributing appropriately

### Reward Shaping Debugging

```python
def debug_reward_function(env, n_steps=1000):
    """Debug reward function by computing rewards manually."""
    obs, _ = env.reset()
    rewards = []
    reward_components = []
    
    for step in range(n_steps):
        # Random action for testing
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        rewards.append(reward)
        
        # Manually compute reward components (if possible)
        # This depends on your reward function implementation
        
        if done:
            break
    
    # Plot reward distribution
    plt.hist(rewards, bins=50, alpha=0.7)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.grid(True)
    plt.show()
    
    print(f"Reward statistics:")
    print(f"  Mean: {np.mean(rewards):.4f}")
    print(f"  Std: {np.std(rewards):.4f}")
    print(f"  Min: {np.min(rewards):.4f}")
    print(f"  Max: {np.max(rewards):.4f}")
```

---

## ⚠️ Common Training Issues

### Issue 1: Training Not Improving

**Symptoms:**
- Episode reward not increasing
- Episode length not increasing
- Losses not decreasing

**Diagnosis:**
```python
# Check if environment is working
env = make_ballbot_env()()
obs, _ = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"Reward: {reward}, Done: {done}")
    if done:
        break

# Check if policy is learning
model = PPO.load("path/to/model")
print(f"Policy parameters: {sum(p.numel() for p in model.policy.parameters())}")
```

**Solutions:**
1. **Check reward function**: May be too sparse or poorly shaped
2. **Check learning rate**: May be too low (not learning) or too high (unstable)
3. **Check observation normalization**: Observations may be out of range
4. **Check action space**: Actions may be clipped incorrectly
5. **Check environment**: Environment may not be resetting correctly

### Issue 2: Unstable Training

**Symptoms:**
- Losses oscillating wildly
- Episode reward jumping around
- Policy performance degrading

**Diagnosis:**
```python
# Check learning rate
logs = read_training_logs('log/progress.csv')
plt.plot(logs['train/learning_rate'])
plt.title('Learning Rate Schedule')
plt.show()

# Check KL divergence
plt.plot(logs['train/approx_kl'])
plt.title('KL Divergence')
plt.axhline(y=0.1, color='r', linestyle='--', label='Warning threshold')
plt.legend()
plt.show()
```

**Solutions:**
1. **Reduce learning rate**: Try 10x smaller
2. **Reduce clip range**: For PPO, try smaller `clip_range`
3. **Increase batch size**: More stable gradients
4. **Add gradient clipping**: Prevent exploding gradients
5. **Reduce number of parallel environments**: May reduce variance

### Issue 3: Overfitting

**Symptoms:**
- Training reward much higher than evaluation reward
- Policy performs well in training but poorly in evaluation
- High variance in evaluation

**Diagnosis:**
```python
# Compare training vs evaluation
logs = read_training_logs('log/progress.csv')
train_rewards = logs['rollout/ep_rew_mean']
eval_rewards = logs['eval/mean_reward']

plt.plot(train_rewards, label='Training')
plt.plot(eval_rewards, label='Evaluation')
plt.legend()
plt.title('Training vs Evaluation Reward')
plt.show()

gap = np.array(train_rewards) - np.array(eval_rewards)
print(f"Mean train-eval gap: {np.mean(gap):.2f}")
```

**Solutions:**
1. **Add domain randomization**: Vary terrain, camera noise, etc.
2. **Reduce policy complexity**: Smaller network
3. **Add regularization**: Weight decay, dropout
4. **Early stopping**: Stop when eval reward plateaus
5. **Increase evaluation frequency**: Catch overfitting earlier

### Issue 4: Policy Not Exploring

**Symptoms:**
- Actions always the same
- Low entropy
- Policy stuck in local optimum

**Diagnosis:**
```python
# Check action distribution
actions = []
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=False)
    actions.append(action)
actions = np.array(actions)

print(f"Action std: {np.std(actions, axis=0)}")
print(f"Action range: [{np.min(actions)}, {np.max(actions)}]")

# Check entropy
logs = read_training_logs('log/progress.csv')
plt.plot(logs['train/entropy_loss'])
plt.title('Entropy (Exploration)')
plt.show()
```

**Solutions:**
1. **Increase entropy coefficient**: Encourage exploration
2. **Add action noise**: During training
3. **Increase learning rate**: May help escape local optima
4. **Curriculum learning**: Start with easier tasks
5. **Check reward function**: May be too sparse

---

## ⚡ Performance Profiling

### Profiling Environment Step Time

```python
import time

def profile_environment(env, n_steps=1000):
    """Profile environment step time."""
    obs, _ = env.reset()
    
    step_times = []
    render_times = []
    
    for step in range(n_steps):
        action = env.action_space.sample()
        
        # Time step
        start = time.time()
        obs, reward, done, truncated, info = env.step(action)
        step_time = time.time() - start
        step_times.append(step_time)
        
        # Time rendering (if enabled)
        if hasattr(env, 'render'):
            start = time.time()
            env.render()
            render_time = time.time() - start
            render_times.append(render_time)
        
        if done:
            obs, _ = env.reset()
    
    print(f"Step time - Mean: {np.mean(step_times)*1000:.2f}ms, Std: {np.std(step_times)*1000:.2f}ms")
    if render_times:
        print(f"Render time - Mean: {np.mean(render_times)*1000:.2f}ms")
    print(f"Steps per second: {1/np.mean(step_times):.1f}")
```

### Profiling Training Speed

```python
def profile_training_speed(model, env, n_steps=10000):
    """Profile training speed."""
    import time
    
    obs, _ = env.reset()
    
    # Profile inference
    inference_times = []
    for _ in range(1000):
        start = time.time()
        action, _ = model.predict(obs, deterministic=False)
        inference_times.append(time.time() - start)
    
    print(f"Inference time - Mean: {np.mean(inference_times)*1000:.2f}ms")
    print(f"Inference FPS: {1/np.mean(inference_times):.1f}")
    
    # Profile training step (if possible)
    # This requires access to training loop
```

---

## 🚀 Modern Experiment Tracking

### Weights & Biases (W&B) Integration ⭐

**W&B** is commonly used for experiment tracking in deep learning and RL research.

**Why Use W&B:**
- Real-time dashboards
- Automatic hyperparameter tracking
- Easy experiment comparison
- Model versioning and artifact management
- Collaborative features

**Setup:**
```bash
pip install wandb
wandb login  # One-time setup
```

**Integration with Stable-Baselines3:**
```python
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class WandbCallback(BaseCallback):
    """
    Callback for logging to Weights & Biases.
    """
    def __init__(self, project_name="ballbot-rl", config=None):
        super().__init__()
        self.project_name = project_name
        self.config = config or {}
        
        # Initialize W&B
        wandb.init(
            project=project_name,
            config=self.config,
            sync_tensorboard=True  # Also sync TensorBoard logs
        )
    
    def _on_step(self) -> bool:
        # Log metrics every N steps
        if self.n_calls % 1000 == 0:
            # Get latest metrics from logger
            if self.logger is not None:
                for key, value in self.logger.name_to_value.items():
                    wandb.log({key: value}, step=self.num_timesteps)
        return True
    
    def _on_training_end(self) -> None:
        wandb.finish()

# Usage in training
from stable_baselines3 import PPO

config = {
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 5,
    "ent_coef": 0.001,
    "vf_coef": 2.0
}

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(
    total_timesteps=10_000_000,
    callback=WandbCallback(project_name="ballbot-rl", config=config)
)
```

**Advanced W&B Features:**
```python
# Log custom metrics
wandb.log({
    "custom/reward_component_1": reward_term_1,
    "custom/reward_component_2": reward_term_2,
    "custom/tilt_angle": tilt_angle,
    "custom/episode_success": success_rate
})

# Log images (camera frames)
wandb.log({
    "outputs/visualizations/images/depth_camera_0": wandb.Image(depth_image),
    "outputs/visualizations/images/rgb_camera_1": wandb.Image(rgb_image)
})

# Log videos (episode replays)
wandb.log({
    "video/episode_replay": wandb.Video(episode_frames, fps=30)
})

# Log model checkpoints
wandb.save("models/checkpoint.zip")
```

**Hyperparameter Sweeps:**
```yaml
# sweep_config.yaml
program: train.py
method: bayes  # or grid, random
metric:
  name: eval/mean_reward
  goal: maximize
parameters:
  learning_rate:
    min: 1e-5
    max: 1e-3
    distribution: log_uniform
  n_steps:
    values: [1024, 2048, 4096]
  batch_size:
    values: [128, 256, 512]
```

```bash
# Run sweep
wandb sweep sweep_config.yaml
wandb agent <sweep_id>
```

### TensorBoard Integration

**TensorBoard** provides detailed visualization of training metrics, model graphs, and embeddings.

**Setup:**
```python
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

# Configure TensorBoard logging
log_dir = "./logs/tensorboard/"
logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
model.set_logger(logger)

# Train
model.learn(total_timesteps=10_000_000)

# View: tensorboard --logdir=./logs/tensorboard/
```

**Custom TensorBoard Logging:**
```python
from torch.utils.tensorboard import SummaryWriter

class TensorBoardCallback(BaseCallback):
    def __init__(self, log_dir="./logs/tensorboard/"):
        super().__init__()
        self.writer = SummaryWriter(log_dir)
    
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            # Log scalar metrics
            self.writer.add_scalar("Reward/Episode", episode_reward, self.num_timesteps)
            self.writer.add_scalar("Loss/Policy", policy_loss, self.num_timesteps)
            
            # Log histograms (action distributions)
            self.writer.add_histogram("Actions/Distribution", actions, self.num_timesteps)
            
            # Log images
            self.writer.add_image("Observations/Depth", depth_image, self.num_timesteps)
        
        return True
    
    def _on_training_end(self) -> None:
        self.writer.close()
```

### MLflow Integration

**MLflow** provides experiment tracking, model registry, and deployment tools.

```python
import mlflow
import mlflow.pytorch

# Start experiment
mlflow.set_experiment("ballbot-rl")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 1e-4)
    mlflow.log_param("n_steps", 2048)
    
    # Train model
    model = PPO("MultiInputPolicy", env)
    model.learn(total_timesteps=10_000_000)
    
    # Log metrics
    mlflow.log_metric("final_reward", final_reward)
    mlflow.log_metric("episode_length", episode_length)
    
    # Log model
    mlflow.pytorch.log_model(model.policy, "policy")
    
    # Log artifacts
    mlflow.log_artifact("config.yaml")
    mlflow.log_artifact("training_logs/")
```

---

## 🎨 Advanced Visualization

### Attention Visualization

**For Transformer-based models**, visualize which modalities the model attends to:

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(model, observations):
    """
    Visualize cross-modal attention weights.
    """
    # Get attention weights from model
    with torch.no_grad():
        # Forward pass with attention return
        features, attention_weights = model.forward_with_attention(observations)
    
    # attention_weights: (B, num_heads, num_modalities, num_modalities)
    # Average over heads and batch
    attn = attention_weights.mean(dim=0).mean(dim=0)  # (num_modalities, num_modalities)
    
    # Plot attention heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        attn.cpu().numpy(),
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=['Proprio', 'Vision_0', 'Vision_1'],
        yticklabels=['Proprio', 'Vision_0', 'Vision_1']
    )
    plt.title('Cross-Modal Attention Weights')
    plt.ylabel('Query Modality')
    plt.xlabel('Key Modality')
    plt.show()
```

### Feature Activation Visualization

**Visualize what features the model learns:**

```python
def visualize_feature_activations(model, observations, layer_name='fusion'):
    """
    Visualize feature activations at different layers.
    """
    # Register hooks to capture activations
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hook
    layer = dict(model.named_modules())[layer_name]
    layer.register_forward_hook(get_activation(layer_name))
    
    # Forward pass
    with torch.no_grad():
        _ = model(observations)
    
    # Visualize activations
    features = activations[layer_name]  # (B, d_model)
    
    # Plot feature distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(features.flatten().cpu().numpy(), bins=50)
    plt.title('Feature Activation Distribution')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    
    # Plot feature heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(features[:10].cpu().numpy(), cmap='viridis')
    plt.title('Feature Activations (First 10 Samples)')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Sample')
    
    plt.tight_layout()
    plt.show()
```

### Gradient Flow Analysis

**Check if gradients are flowing properly:**

```python
def visualize_gradient_flow(model):
    """
    Visualize gradient flow through the network.
    """
    gradients = {}
    
    def get_gradient(name):
        def hook(grad):
            gradients[name] = grad.detach()
        return hook
    
    # Register hooks
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(get_gradient(name))
    
    # Forward and backward pass
    loss = compute_loss(model, observations, actions)
    loss.backward()
    
    # Plot gradient norms
    grad_norms = {name: grad.norm().item() for name, grad in gradients.items()}
    
    plt.figure(figsize=(12, 6))
    names = list(grad_norms.keys())
    norms = list(grad_norms.values())
    
    plt.barh(names, norms)
    plt.xlabel('Gradient Norm')
    plt.title('Gradient Flow Analysis')
    plt.tight_layout()
    plt.show()
    
    # Check for vanishing/exploding gradients
    print(f"Min gradient norm: {min(norms):.6f}")
    print(f"Max gradient norm: {max(norms):.6f}")
    print(f"Mean gradient norm: {np.mean(norms):.6f}")
```

### Policy Uncertainty Visualization

**Visualize policy uncertainty (useful for safety-critical applications):**

```python
def visualize_policy_uncertainty(model, env, n_samples=100):
    """
    Visualize policy uncertainty by sampling multiple actions.
    """
    obs, _ = env.reset()
    
    # Sample multiple actions
    actions = []
    for _ in range(n_samples):
        action, _ = model.predict(obs, deterministic=False)
        actions.append(action)
    
    actions = np.array(actions)  # (n_samples, action_dim)
    
    # Plot action distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        axes[i].hist(actions[:, i], bins=30, alpha=0.7, edgecolor='black')
        axes[i].axvline(np.mean(actions[:, i]), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(actions[:, i]):.3f}')
        axes[i].axvline(np.mean(actions[:, i]) + np.std(actions[:, i]), 
                       color='orange', linestyle='--', label='±1 std')
        axes[i].axvline(np.mean(actions[:, i]) - np.std(actions[:, i]), 
                       color='orange', linestyle='--')
        axes[i].set_title(f'Action {i} Distribution')
        axes[i].set_xlabel('Action Value')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compute uncertainty metrics
    uncertainty = np.std(actions, axis=0)
    print(f"Action uncertainty: {uncertainty}")
    print(f"Mean uncertainty: {np.mean(uncertainty):.4f}")
```

---

## 🤖 Automated Hyperparameter Tuning

### Ray Tune Integration ⭐

**Ray Tune** provides distributed hyperparameter optimization.

**Setup:**
```bash
pip install ray[tune] optuna
```

**Example:**
```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from stable_baselines3 import PPO

def train_ppo(config):
    """Training function for Ray Tune."""
    # Create environment
    env = make_ballbot_env()()
    
    # Create model with config
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        verbose=0
    )
    
    # Train
    model.learn(total_timesteps=1_000_000)
    
    # Evaluate
    eval_reward = evaluate_policy(model, env, n_episodes=10)
    
    # Report to Tune
    tune.report(mean_reward=eval_reward)

# Define search space
config = {
    "learning_rate": tune.loguniform(1e-5, 1e-3),
    "n_steps": tune.choice([1024, 2048, 4096]),
    "batch_size": tune.choice([128, 256, 512]),
    "n_epochs": tune.choice([3, 5, 10]),
    "ent_coef": tune.loguniform(0.0001, 0.01),
    "vf_coef": tune.uniform(0.5, 5.0)
}

# Run optimization
analysis = tune.run(
    train_ppo,
    config=config,
    num_samples=50,  # Number of trials
    scheduler=ASHAScheduler(metric="mean_reward", mode="max"),
    resources_per_trial={"cpu": 4, "gpu": 0.5}
)

# Get best config
best_config = analysis.get_best_config(metric="mean_reward", mode="max")
print(f"Best config: {best_config}")
```

### Optuna Integration

**Optuna** provides efficient hyperparameter optimization with pruning.

```python
import optuna
from optuna.pruners import MedianPruner

def objective(trial):
    """Objective function for Optuna."""
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    ent_coef = trial.suggest_loguniform("ent_coef", 0.0001, 0.01)
    vf_coef = trial.suggest_uniform("vf_coef", 0.5, 5.0)
    
    # Create and train model
    env = make_ballbot_env()()
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=0
    )
    
    # Train with intermediate reporting
    for step in range(10):  # 10 checkpoints
        model.learn(total_timesteps=100_000)
        
        # Evaluate
        eval_reward = evaluate_policy(model, env, n_episodes=5)
        
        # Report intermediate value
        trial.report(eval_reward, step)
        
        # Prune if necessary
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return eval_reward

# Create study
study = optuna.create_study(
    direction="maximize",
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
)

# Optimize
study.optimize(objective, n_trials=50)

# Get best parameters
print(f"Best parameters: {study.best_params}")
print(f"Best value: {study.best_value}")

# Visualize optimization history
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
```

---

## ✅ Best Practices

### 1. Monitor Multiple Metrics

> "Don't just watch reward. Watch everything."  
> — *Common wisdom in RL*

Monitor:
- Episode reward (primary)
- Episode length
- Policy loss
- Value loss
- Entropy
- KL divergence

### 2. Use Evaluation Episodes

- Always evaluate on held-out episodes
- Compare training vs. evaluation performance
- Catch overfitting early

### 3. Log Everything

- Enable reward component logging
- Save camera frames for analysis
- Log terrain seeds for reproducibility

### 4. Visualize Regularly

- Plot training progress daily
- Analyze loss components weekly
- Debug policy behavior when performance drops

### 5. Test Policy Behavior

- Run evaluation episodes regularly
- Visualize action distributions
- Check for common failure modes

### 6. Keep Experiments Organized

- Use descriptive log directory names
- Save configuration files with logs
- Document hyperparameter changes

### 7. Use Modern Experiment Tracking ⭐

- **W&B** for real-time dashboards and collaboration
- **TensorBoard** for detailed metric visualization
- **MLflow** for model registry and deployment
- **Ray Tune/Optuna** for automated hyperparameter optimization

### 8. Visualize Model Internals

- Attention weights (for Transformer models)
- Feature activations
- Gradient flow
- Policy uncertainty

### 9. Automate Hyperparameter Tuning

- Use Ray Tune or Optuna for systematic search
- Start with broad search, then narrow down
- Use pruning to save computational resources

---

## 📝 Summary

### Key Takeaways

1. **Monitor multiple metrics**: Reward, length, losses, entropy, KL divergence
2. **Visualize training progress**: Use plotting tools to track learning
3. **Analyze loss components**: Understand what's happening during training
4. **Debug policy behavior**: Visualize actions, analyze failures
5. **Profile performance**: Identify bottlenecks
6. **Common issues**: Know how to diagnose and fix common problems

### Debugging Checklist

- [ ] Training reward increasing?
- [ ] Evaluation reward tracking training?
- [ ] Losses decreasing?
- [ ] Entropy appropriate (not too low/high)?
- [ ] KL divergence reasonable (< 0.1)?
- [ ] Actions distributed (not stuck)?
- [ ] Episode length increasing?
- [ ] No overfitting (train ≈ eval)?

### Next Steps

- Set up regular monitoring
- Create custom visualization scripts
- Debug specific issues in your training
- Profile and optimize performance

---

## 📚 Further Reading

### Papers

**Classic RL Debugging:**
- **Schulman et al. (2017)** - "Proximal Policy Optimization Algorithms"
- **Henderson et al. (2018)** - "Deep Reinforcement Learning That Matters"

**Modern Tools & Practices:**
- **Biewald (2020)** - "Experiment Tracking with Weights and Biases"
- **Liaw et al. (2018)** - "Tune: A Research Platform for Distributed Model Selection and Training"
- **Akiba et al. (2019)** - "Optuna: A Next-generation Hyperparameter Optimization Framework"

### Tutorials

- [Environment & RL Workflow](../03_environment_and_rl.md) - Training setup
- [Reward Design for Robotics](../04_reward_design_for_robotics.md) - Reward function design
- [Actor-Critic Methods](../05_actor_critic_methods.md) - Understanding losses
- [Multi-Modal Fusion](../10_multimodal_fusion.md) - Advanced fusion architectures

### Tools & Resources

- **Weights & Biases**: [wandb.ai](https://wandb.ai) - Experiment tracking
- **TensorBoard**: [tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard) - Visualization
- **Ray Tune**: [docs.ray.io/tune](https://docs.ray.io/en/latest/tune/) - Hyperparameter tuning
- **Optuna**: [optuna.org](https://optuna.org) - Hyperparameter optimization

### Code References

- `ballbot_rl/visualization/plot_training.py` - Training progress visualization
- `ballbot_rl/visualization/visualize_env.py` - Environment visualization
- `ballbot_rl/visualization/visualize_model.py` - Model visualization
- `ballbot_rl/visualization/browse_environments.py` - Interactive environment browser
- `ballbot_rl/training/train.py` - Training script with logging
- `ballbot_rl/evaluation/evaluate.py` - Policy evaluation script

---

*Last Updated: 2025*

**Happy Debugging! 🔍**

