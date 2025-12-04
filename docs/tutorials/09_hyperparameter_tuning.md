# 🎛️ Hyperparameter Tuning for PPO

*A comprehensive guide to tuning PPO hyperparameters for optimal performance*

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Understanding PPO Hyperparameters](#understanding-ppo-hyperparameters)
3. [Hyperparameter Categories](#hyperparameter-categories)
4. [Tuning Strategies](#tuning-strategies)
5. [Ballbot-Specific Considerations](#ballbot-specific-considerations)
6. [Systematic Tuning Process](#systematic-tuning-process)
7. [Common Configurations](#common-configurations)
8. [Monitoring and Evaluation](#monitoring-and-evaluation)
9. [Advanced Techniques](#advanced-techniques)
10. [Troubleshooting](#troubleshooting)
11. [Summary](#summary)

---

## 🎯 Introduction

Hyperparameter tuning is one of the most important yet challenging aspects of reinforcement learning. The right hyperparameters can make the difference between a policy that learns efficiently and one that fails to converge.

> "Hyperparameter tuning is 90% of RL engineering. The algorithm is important, but the hyperparameters determine whether it works."  
> — *Common wisdom in deep RL*

**What You'll Learn:**
- What each PPO hyperparameter does
- How hyperparameters interact with each other
- Systematic approaches to tuning
- Ballbot-specific recommendations
- How to monitor and evaluate hyperparameter choices

**Prerequisites:**
- Understanding of PPO algorithm ([Actor-Critic Methods](05_actor_critic_methods.md))
- Basic knowledge of training setup ([Complete Training Guide](13_complete_training_guide.md))
- Familiarity with configuration files

---

## 🎛️ Understanding PPO Hyperparameters

### What are Hyperparameters?

**Hyperparameters** are configuration settings that control the learning process but are not learned from data. They must be set before training begins.

**Examples:**
- Learning rate (how fast to learn)
- Batch size (how many samples per update)
- Network architecture (how many layers)

**Key Distinction:**
- **Parameters:** Learned from data (e.g., neural network weights)
- **Hyperparameters:** Set by the user (e.g., learning rate)

### Why Hyperparameters Matter

**Impact on Learning:**
- **Too aggressive:** Policy diverges, training unstable
- **Too conservative:** Slow learning, may not converge
- **Just right:** Fast, stable convergence

**Example:**
```python
# Too high learning rate
learning_rate = 1e-2  # Policy diverges immediately

# Too low learning rate
learning_rate = 1e-6  # Takes forever to learn

# Good learning rate
learning_rate = 1e-4  # Stable learning
```

---

## 📊 Hyperparameter Categories

### 1. Learning Rate Hyperparameters

#### `learning_rate` (or `lr`)

**What It Does:** Controls how much the policy changes per update.

**Typical Range:** 1e-5 to 1e-3

**Ballbot Default:** Scheduled (1e-4 → 5e-5 → 1e-5)

**Mathematical Impact:**
\[
\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t)
\]

Where α is the learning rate.

**Tuning Guidelines:**
- **Too high:** Policy oscillates, may diverge
- **Too low:** Slow learning, may get stuck
- **Scheduled:** Start high, decay over time (common in Ballbot)

**Example Configuration:**
```yaml
algo:
  learning_rate: -1  # -1 enables scheduler
  # OR
  learning_rate: 1e-4  # Fixed learning rate
```

#### Learning Rate Schedule

**Why Schedule?** Start with high learning rate (fast initial learning), then decay (fine-tuning).

**Ballbot Implementation:**
```python
# From ballbot_rl/training/train.py
def lr_schedule(progress_remaining):
    # progress_remaining: 1.0 (start) → 0.0 (end)
    if progress_remaining > 0.66:
        return 1e-4  # First third: high LR
    elif progress_remaining > 0.33:
        return 5e-5  # Middle third: medium LR
    else:
        return 1e-5  # Final third: low LR
```

**When to Use:**
- Long training runs (> 1M steps)
- Complex tasks requiring fine-tuning
- Default for Ballbot

### 2. Policy Update Hyperparameters

#### `clip_range` (PPO Clipping Parameter)

**What It Does:** Limits how much the policy can change per update.

**Typical Range:** 0.1 to 0.3 (standard PPO), 0.01 to 0.05 (conservative)

**Ballbot Default:** 0.015 (very conservative)

**Mathematical Impact:**
\[
L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
\]

Where ε is `clip_range`.

**Why Conservative in Ballbot:**
- Ballbot is unstable (falls easily)
- Small policy changes prevent catastrophic failures
- More stable but slower learning

**Tuning Guidelines:**
- **Higher (0.2-0.3):** Faster learning, less stable
- **Lower (0.01-0.05):** Slower learning, more stable
- **Ballbot:** Start with 0.015, increase if learning too slow

#### `n_steps` (Rollout Length)

**What It Does:** Number of steps collected per environment before updating.

**Typical Range:** 512 to 4096

**Ballbot Default:** 2048

**Impact:**
- **Larger:** More stable updates, slower data collection
- **Smaller:** Faster updates, less stable

**Trade-offs:**
```python
n_steps = 512   # Fast updates, less stable
n_steps = 2048  # Balanced (Ballbot default)
n_steps = 4096  # Very stable, slower
```

**Relation to Episode Length:**
- If `n_steps` > episode length, may collect multiple episodes
- Ballbot episodes: ~4000 steps (max), so 2048 is reasonable

#### `n_epochs` (Update Epochs)

**What It Does:** Number of times to update on the same batch of data.

**Typical Range:** 3 to 10

**Ballbot Default:** 5

**Impact:**
- **More epochs:** Better sample efficiency, risk of overfitting
- **Fewer epochs:** Less sample efficiency, more stable

**Mathematical Impact:**
\[
\text{Total gradient steps} = n\_epochs \times \frac{n\_steps \times n\_envs}{batch\_size}
\]

**Example:**
```python
n_steps = 2048
n_envs = 10
batch_size = 256
n_epochs = 5

# Total gradient steps per update:
gradient_steps = 5 * (2048 * 10) / 256 = 400 steps
```

#### `batch_size`

**What It Does:** Number of samples per gradient update.

**Typical Range:** 64 to 512

**Ballbot Default:** 256

**Impact:**
- **Larger:** More stable gradients, requires more memory
- **Smaller:** Less stable, faster updates

**Relation to `n_steps`:**
- `batch_size` should divide `n_steps * n_envs`
- Example: `n_steps=2048, n_envs=10` → total samples = 20,480
- `batch_size=256` → 80 batches per update

### 3. Exploration Hyperparameters

#### `ent_coef` (Entropy Coefficient)

**What It Does:** Encourages exploration by penalizing low-entropy (deterministic) policies.

**Typical Range:** 0.0001 to 0.01

**Ballbot Default:** 0.001

**Mathematical Impact:**
\[
L(\theta) = L^{CLIP}(\theta) - c_H H[\pi_\theta]
\]

Where `c_H` is `ent_coef` and `H` is entropy.

**Impact:**
- **Higher:** More exploration, less exploitation
- **Lower:** Less exploration, more exploitation

**Tuning Guidelines:**
- **Start high (0.01):** Encourage exploration early
- **Decay over time:** Reduce as policy improves
- **Ballbot:** 0.001 is moderate (balanced)

**Example:**
```python
ent_coef = 0.01   # Very exploratory
ent_coef = 0.001  # Balanced (Ballbot)
ent_coef = 0.0001 # Very exploitative
```

### 4. Value Function Hyperparameters

#### `vf_coef` (Value Function Coefficient)

**What It Does:** Weight for value function loss in total loss.

**Typical Range:** 0.5 to 2.0

**Ballbot Default:** 2.0

**Mathematical Impact:**
\[
L(\theta) = L^{CLIP}(\theta) - c_H H[\pi_\theta] + c_V L^{VF}(\theta)
\]

Where `c_V` is `vf_coef`.

**Impact:**
- **Higher:** More emphasis on value function accuracy
- **Lower:** Less emphasis on value function

**Why High in Ballbot (2.0):**
- Accurate value estimates crucial for stable learning
- Ballbot rewards are sparse (survival + directional)
- Helps with credit assignment

#### `normalize_advantage`

**What It Does:** Normalize advantages to zero mean, unit variance.

**Typical Values:** `true` or `false`

**Ballbot Default:** `false`

**Impact:**
- **True:** More stable updates, may reduce variance
- **False:** Raw advantages (Ballbot default)

**When to Use:**
- **True:** High-variance environments, unstable training
- **False:** Stable environments, want raw signal

### 5. Network Architecture Hyperparameters

#### `hidden_sz` (Hidden Layer Size)

**What It Does:** Number of neurons per hidden layer.

**Typical Range:** 64 to 512

**Ballbot Default:** 128

**Impact:**
- **Larger:** More capacity, slower training, risk of overfitting
- **Smaller:** Less capacity, faster training, may underfit

**Example Configuration:**
```yaml
hidden_sz: 64   # Small network
hidden_sz: 128  # Medium (Ballbot default)
hidden_sz: 256  # Large network
```

#### `net_arch` (Network Architecture)

**What It Does:** Defines network structure (layers, sizes).

**Ballbot Default:**
```python
net_arch = dict(
    pi=[128, 128, 128, 128],  # Policy network: 4 layers, 128 units each
    vf=[128, 128, 128, 128]   # Value network: 4 layers, 128 units each
)
```

**Tuning Guidelines:**
- **Deeper:** More capacity, slower training
- **Wider:** More capacity per layer
- **Shared:** Policy and value share lower layers (default in Ballbot)

### 6. Regularization Hyperparameters

#### `weight_decay` (L2 Regularization)

**What It Does:** Penalizes large weights to prevent overfitting.

**Typical Range:** 0.0 to 0.1

**Ballbot Default:** 0.01

**Mathematical Impact:**
\[
L(\theta) = L^{CLIP}(\theta) + \lambda \|\theta\|^2
\]

Where λ is `weight_decay`.

**Impact:**
- **Higher:** More regularization, simpler policies
- **Lower:** Less regularization, more complex policies

**When to Use:**
- **High:** Overfitting, large networks
- **Low:** Underfitting, small networks

### 7. Environment Hyperparameters

#### `num_envs` (Number of Parallel Environments)

**What It Does:** Number of environments to run in parallel.

**Typical Range:** 4 to 32

**Ballbot Default:** 10

**Impact:**
- **More:** Faster data collection, more memory
- **Fewer:** Slower data collection, less memory

**Trade-offs:**
```python
num_envs = 4   # Low memory, slower
num_envs = 10  # Balanced (Ballbot default)
num_envs = 32  # High memory, faster
```

**Relation to `n_steps`:**
- Total samples per update = `n_steps * num_envs`
- Example: `n_steps=2048, num_envs=10` → 20,480 samples

---

## 🔍 Tuning Strategies

### 1. Manual Tuning (Grid Search)

**Process:**
1. Start with default values
2. Tune one hyperparameter at a time
3. Evaluate on validation set
4. Keep best value, move to next

**Example:**
```python
# Step 1: Tune learning rate
for lr in [1e-5, 1e-4, 1e-3]:
    train_with_lr(lr)
    evaluate()
# Best: lr = 1e-4

# Step 2: Tune clip_range (with best lr)
for clip in [0.01, 0.015, 0.02]:
    train_with_clip(clip, lr=1e-4)
    evaluate()
# Best: clip = 0.015
```

**Pros:**
- Simple, interpretable
- Good for understanding impact

**Cons:**
- Time-consuming
- Doesn't capture interactions

### 2. Random Search

**Process:**
1. Sample hyperparameters randomly
2. Train multiple configurations
3. Select best performing

**Example:**
```python
import random

configs = []
for _ in range(20):
    config = {
        'lr': random.uniform(1e-5, 1e-3),
        'clip_range': random.uniform(0.01, 0.03),
        'ent_coef': random.uniform(0.0001, 0.01),
    }
    configs.append(config)

# Train all, select best
```

**Pros:**
- More efficient than grid search
- Better coverage of space

**Cons:**
- Still time-consuming
- May miss optimal regions

### 3. Bayesian Optimization

**Process:**
1. Build probabilistic model of performance
2. Suggest promising hyperparameters
3. Update model with results
4. Repeat

**Tools:**
- Optuna
- Hyperopt
- Ray Tune

**Example (Optuna):**
```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    clip_range = trial.suggest_uniform('clip_range', 0.01, 0.03)
    ent_coef = trial.suggest_loguniform('ent_coef', 0.0001, 0.01)
    
    # Train and evaluate
    return evaluate(train(lr, clip_range, ent_coef))

study = optuna.create_study()
study.optimize(objective, n_trials=50)
```

**Pros:**
- Efficient exploration
- Captures interactions

**Cons:**
- More complex setup
- Requires more infrastructure

---

## 🤖 Ballbot-Specific Considerations

### Why Conservative Hyperparameters?

**1. Instability:**
- Ballbot falls easily (tilt > 20°)
- Small policy changes can cause failures
- Conservative `clip_range` (0.015) prevents large updates

**2. Sparse Rewards:**
- Main reward: directional velocity
- Survival bonus: small (+0.02)
- Requires stable value estimates (`vf_coef=2.0`)

**3. Multi-Modal Observations:**
- Proprioceptive state + depth images
- Complex feature extraction
- Moderate network size (`hidden_sz=128`)

### Recommended Starting Point

**Conservative Configuration:**
```yaml
algo:
  learning_rate: -1  # Scheduled
  clip_range: 0.015   # Conservative
  ent_coef: 0.001     # Moderate exploration
  vf_coef: 2.0        # Emphasize value accuracy
  n_steps: 2048       # Standard
  n_epochs: 5         # Standard
  batch_sz: 256       # Standard
  weight_decay: 0.01  # Regularization

hidden_sz: 128        # Moderate network
num_envs: 10          # Balanced
```

### When to Increase Aggressiveness

**If Learning Too Slow:**
- Increase `clip_range` to 0.02-0.03
- Increase `learning_rate` to 1e-3
- Increase `ent_coef` to 0.01

**If Training Unstable:**
- Decrease `clip_range` to 0.01
- Decrease `learning_rate` to 1e-5
- Increase `vf_coef` to 3.0

---

## 🔄 Systematic Tuning Process

### Step 1: Establish Baseline

**Run with defaults:**
```bash
python ballbot_rl/training/train.py --config configs/train/ppo_directional.yaml
```

**Record metrics:**
- Final episode reward
- Training time
- Convergence speed

### Step 2: Tune Learning Rate

**Test range:** [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

**Evaluate:**
- Convergence speed
- Final performance
- Training stability

### Step 3: Tune Clip Range

**Test range:** [0.01, 0.015, 0.02, 0.03]

**Evaluate:**
- Update stability
- Learning speed
- Final performance

### Step 4: Tune Exploration

**Test range:** [0.0001, 0.001, 0.01]

**Evaluate:**
- Exploration vs. exploitation balance
- Final performance

### Step 5: Fine-Tune Architecture

**Test:**
- `hidden_sz`: [64, 128, 256]
- `net_arch`: Different depths

**Evaluate:**
- Capacity vs. speed trade-off

### Step 6: Validate on Multiple Seeds

**Run best configuration with 3-5 seeds:**
```python
for seed in [10, 42, 100, 200, 300]:
    train(config, seed=seed)
```

**Evaluate:**
- Consistency across seeds
- Robustness

---

## 📋 Common Configurations

### Configuration 1: Fast Learning (Aggressive)

**Use Case:** Quick experiments, simple tasks

```yaml
algo:
  learning_rate: 1e-3      # High
  clip_range: 0.2          # Standard PPO
  ent_coef: 0.01           # High exploration
  n_steps: 1024            # Shorter rollouts
  n_epochs: 3              # Fewer epochs

hidden_sz: 64              # Smaller network
num_envs: 16               # More parallel envs
```

**Trade-offs:**
- ✅ Fast learning
- ❌ Less stable
- ❌ May not converge

### Configuration 2: Stable Learning (Conservative)

**Use Case:** Production training, complex tasks

```yaml
algo:
  learning_rate: -1         # Scheduled
  clip_range: 0.015        # Very conservative
  ent_coef: 0.001          # Moderate exploration
  vf_coef: 2.0             # Emphasize value
  n_steps: 4096            # Longer rollouts
  n_epochs: 10              # More epochs

hidden_sz: 256             # Larger network
num_envs: 8                # Fewer parallel envs
```

**Trade-offs:**
- ✅ Very stable
- ✅ High final performance
- ❌ Slower learning

### Configuration 3: Balanced (Ballbot Default)

**Use Case:** General purpose, good starting point

```yaml
algo:
  learning_rate: -1         # Scheduled
  clip_range: 0.015        # Conservative
  ent_coef: 0.001          # Moderate
  vf_coef: 2.0             # Emphasize value
  n_steps: 2048            # Standard
  n_epochs: 5              # Standard

hidden_sz: 128             # Moderate
num_envs: 10               # Balanced
```

**Trade-offs:**
- ✅ Balanced stability/speed
- ✅ Good default
- ✅ Works for most tasks

---

## 📊 Monitoring and Evaluation

### Key Metrics to Track

**1. Episode Reward:**
- Mean reward over last 100 episodes
- Should increase over time
- Target: > 100 for Ballbot

**2. Episode Length:**
- Mean length over last 100 episodes
- Should increase (longer survival)
- Target: > 3000 steps

**3. Value Function Loss:**
- Should decrease over time
- Indicates value accuracy

**4. Policy Loss:**
- Should stabilize (not diverge)
- Indicates policy stability

**5. Entropy:**
- Should decrease over time
- Indicates policy becoming deterministic

### When to Stop Tuning

**Stop if:**
- ✅ Performance plateaus (no improvement for 1M steps)
- ✅ Consistent across multiple seeds
- ✅ Meets performance targets

**Continue if:**
- ❌ Performance still improving
- ❌ High variance across seeds
- ❌ Below performance targets

---

## 🚀 Advanced Techniques

### 1. Learning Rate Scheduling

**Why:** Start fast, finish fine-tuned

**Implementation:**
```python
def lr_schedule(progress_remaining):
    if progress_remaining > 0.66:
        return 1e-4
    elif progress_remaining > 0.33:
        return 5e-5
    else:
        return 1e-5
```

### 2. Entropy Scheduling

**Why:** Explore early, exploit later

**Implementation:**
```python
def ent_coef_schedule(progress_remaining):
    return 0.01 * progress_remaining  # Decay from 0.01 to 0
```

### 3. Adaptive Clip Range

**Why:** Larger updates early, smaller later

**Implementation:**
```python
def clip_range_schedule(progress_remaining):
    return 0.03 * progress_remaining + 0.01  # Decay from 0.03 to 0.01
```

---

## 🐛 Troubleshooting

### Problem: Policy Not Learning

**Symptoms:**
- Episode reward not increasing
- Policy loss near zero
- High entropy (random policy)

**Solutions:**
- Increase `learning_rate` (1e-4 → 1e-3)
- Increase `ent_coef` (0.001 → 0.01)
- Increase `clip_range` (0.015 → 0.02)

### Problem: Training Unstable

**Symptoms:**
- Policy loss oscillating
- Episode reward decreasing
- Value loss increasing

**Solutions:**
- Decrease `learning_rate` (1e-3 → 1e-4)
- Decrease `clip_range` (0.02 → 0.015)
- Increase `vf_coef` (1.0 → 2.0)

### Problem: Slow Learning

**Symptoms:**
- Slow reward increase
- Long training time
- Low sample efficiency

**Solutions:**
- Increase `n_steps` (2048 → 4096)
- Increase `n_epochs` (5 → 10)
- Increase `num_envs` (10 → 16)

### Problem: Overfitting

**Symptoms:**
- Training reward high, eval reward low
- Policy too deterministic
- Poor generalization

**Solutions:**
- Increase `weight_decay` (0.01 → 0.05)
- Increase `ent_coef` (0.001 → 0.01)
- Decrease network size (`hidden_sz`: 256 → 128)

---

## 📝 Summary

**Key Takeaways:**

1. **Hyperparameters** control learning but aren't learned from data
2. **Conservative values** (low `clip_range`, scheduled `lr`) work well for Ballbot
3. **Tune systematically:** one hyperparameter at a time
4. **Monitor metrics:** episode reward, value loss, entropy
5. **Validate:** test best configuration on multiple seeds

**Recommended Starting Point:**
- Use Ballbot default configuration
- Tune `learning_rate` and `clip_range` first
- Adjust based on training stability and speed

**Next Steps:**

- Review [Complete Training Guide](13_complete_training_guide.md) for end-to-end workflow
- Explore [Environment Wrappers](06_environment_wrappers.md) for environment configuration
- Try tuning hyperparameters in your own experiments

---

## 🔗 Related Resources

- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Stable-Baselines3 PPO Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Ballbot Training Configuration](../../configs/train/ppo_directional.yaml) - Default config
- [Hyperparameter Tuning Best Practices](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#id17)

