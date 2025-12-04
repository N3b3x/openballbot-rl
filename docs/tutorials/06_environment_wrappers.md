# 🔄 Environment Wrappers and Vectorization

*Understanding how to wrap, monitor, and parallelize Gymnasium environments for efficient RL training*

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [What are Environment Wrappers?](#what-are-environment-wrappers)
3. [Common Wrappers](#common-wrappers)
4. [Vectorized Environments](#vectorized-environments)
5. [Monitoring and Logging](#monitoring-and-logging)
6. [Wrapper Chaining](#wrapper-chaining)
7. [Real-World Example: Ballbot Training](#real-world-example-ballbot-training)
8. [Best Practices](#best-practices)
9. [Common Pitfalls](#common-pitfalls)
10. [Summary](#summary)

---

## 🎯 Introduction

Environment wrappers are a powerful pattern in Gymnasium that allow you to modify environment behavior without changing the base implementation. They're essential for:

- **Logging and monitoring** training progress
- **Normalizing observations** for better learning
- **Limiting episode length** to prevent infinite episodes
- **Parallelizing environments** for faster data collection
- **Recording videos** for debugging and visualization

> "Wrappers are the Swiss Army knife of RL environments. They let you compose complex behaviors from simple building blocks."  
> — *Common wisdom in RL engineering*

**What You'll Learn:**
- How wrappers work and when to use them
- Common wrapper types and their use cases
- How to vectorize environments for parallel training
- How to monitor and log training statistics
- Best practices for wrapper composition

**Prerequisites:**
- Understanding of Gymnasium environments ([Introduction to Gymnasium](01_introduction_to_gymnasium.md))
- Basic knowledge of RL training ([Actor-Critic Methods](05_actor_critic_methods.md))

---

## 🔄 What are Environment Wrappers?

### The Wrapper Pattern

A wrapper is an environment that wraps another environment, forwarding most method calls while modifying specific behaviors.

**Basic Structure:**
```python
class MyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Initialize wrapper state
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Modify observation or info
        return modified_obs, modified_info
    
    def step(self, action):
        # Modify action before passing to env
        modified_action = self._modify_action(action)
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        # Modify observation, reward, or info
        return modified_obs, modified_reward, terminated, truncated, modified_info
```

### Why Use Wrappers?

**1. Separation of Concerns:**
- Base environment handles physics/simulation
- Wrappers handle logging, normalization, etc.

**2. Composability:**
- Chain multiple wrappers for complex behaviors
- Reuse wrappers across different environments

**3. Non-Invasive:**
- Don't modify original environment code
- Easy to enable/disable features

**4. Testing:**
- Test wrappers independently
- Test base environment without wrappers

---

## 🔧 Common Wrappers

### 1. Monitor Wrapper (Stable-Baselines3)

**Purpose:** Log episode statistics (rewards, lengths) to CSV files.

**Usage:**
```python
from stable_baselines3.common.monitor import Monitor

env = gym.make("ballbot-v0.1")
env = Monitor(env, filename="./logs/monitor.csv")
```

**What It Logs:**
- Episode reward (sum of rewards)
- Episode length (number of steps)
- Episode time (wall-clock time)

**Output Format (CSV):**
```csv
r,l,t
125.3,400,2.5
98.7,350,2.1
...
```

**In Ballbot Training:**
```python
# From ballbot_rl/training/utils.py
def make_ballbot_env(...):
    def _init():
        env = gym.make("ballbot-v0.1", ...)
        return Monitor(env)  # Automatically logs to default location
    return _init
```

### 2. TimeLimit Wrapper

**Purpose:** Terminate episodes after a maximum number of steps.

**Usage:**
```python
from gymnasium.wrappers import TimeLimit

env = gym.make("ballbot-v0.1")
env = TimeLimit(env, max_episode_steps=4000)
```

**Why It's Important:**
- Prevents infinite episodes
- Ensures finite episode lengths for batching
- Matches real-world time constraints

**In Ballbot:**
- Default max episode length: 4000 steps
- Episodes terminate if robot falls (tilt > 20°) or timeout

### 3. ClipAction Wrapper

**Purpose:** Clip actions to action space bounds.

**Usage:**
```python
from gymnasium.wrappers import ClipAction

env = gym.make("ballbot-v0.1")
env = ClipAction(env)
```

**Why It's Needed:**
- Neural networks may output out-of-bounds actions
- Prevents simulation errors from invalid actions
- Common in continuous control tasks

**Example:**
```python
# Action space: [-1, 1] for each motor
action = np.array([1.5, -0.8, 0.3])  # Out of bounds
clipped_action = np.clip(action, -1, 1)  # [1.0, -0.8, 0.3]
```

### 4. NormalizeObservation Wrapper

**Purpose:** Normalize observations using running mean and standard deviation.

**Usage:**
```python
from gymnasium.wrappers import NormalizeObservation

env = gym.make("ballbot-v0.1")
env = NormalizeObservation(env)
```

**How It Works:**
- Maintains running statistics: μ (mean), σ (std)
- Normalizes: `obs_normalized = (obs - μ) / σ`
- Updates statistics during training

**Benefits:**
- Stabilizes learning (all features on similar scale)
- Reduces sensitivity to observation scale
- Common preprocessing step

**Note:** In Ballbot, we handle normalization in the feature extractor instead.

### 5. FrameStack Wrapper

**Purpose:** Stack consecutive frames for temporal information.

**Usage:**
```python
from gymnasium.wrappers import FrameStack

env = gym.make("ballbot-v0.1")
env = FrameStack(env, num_stack=4)
```

**Observation Shape:**
- Before: `(H, W, C)` for single frame
- After: `(H, W, C * num_stack)` for stacked frames

**Use Case:**
- Vision-based tasks needing temporal context
- Not used in Ballbot (we use proprioceptive state + depth)

### 6. RecordVideo Wrapper

**Purpose:** Record episodes to video files.

**Usage:**
```python
from gymnasium.wrappers import RecordVideo

env = gym.make("ballbot-v0.1")
env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: x % 100 == 0)
```

**When to Use:**
- Debugging policy behavior
- Creating demonstrations
- Visualizing training progress

---

## 🚀 Vectorized Environments

### Why Vectorize?

**Problem:** Single environment = slow data collection.

**Solution:** Run multiple environments in parallel.

**Benefits:**
- **Faster training:** Collect more data per second
- **Better sample efficiency:** More diverse experiences
- **Stability:** Parallel environments reduce variance

### Types of Vectorized Environments

#### 1. DummyVecEnv (Sequential)

**How It Works:**
- Runs environments sequentially in a single process
- Simple and compatible (works everywhere)
- Slower than true parallelization

**Usage:**
```python
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return gym.make("ballbot-v0.1")

vec_env = DummyVecEnv([make_env for _ in range(4)])
```

**When to Use:**
- macOS (multiprocessing issues with MuJoCo)
- Debugging (easier to debug single process)
- Small number of environments (< 4)

#### 2. SubprocVecEnv (Parallel)

**How It Works:**
- Runs environments in separate processes
- True parallelization (uses multiple CPU cores)
- Faster data collection

**Usage:**
```python
from stable_baselines3.common.vec_env import SubprocVecEnv

vec_env = SubprocVecEnv([make_env for _ in range(10)])
```

**When to Use:**
- Linux/Windows (multiprocessing works well)
- Large number of environments (> 4)
- Production training (maximize speed)

**In Ballbot Training:**
```python
# From ballbot_rl/training/train.py
use_subproc = config.get("use_subproc_vec_env", platform.system() != 'Darwin')
if use_subproc:
    VecEnvClass = SubprocVecEnv  # Parallel on Linux/Windows
else:
    VecEnvClass = DummyVecEnv    # Sequential on macOS
```

### Vectorized Environment API

**Key Methods:**
```python
# Reset all environments
obs = vec_env.reset()  # Shape: (n_envs, *obs_shape)

# Step all environments
obs, rewards, dones, infos = vec_env.step(actions)  # actions: (n_envs, *action_shape)

# Get number of environments
n_envs = vec_env.num_envs
```

**Important Notes:**
- All environments reset/step together
- `dones` is a boolean array (one per env)
- `infos` is a list of info dicts (one per env)

---

## 📊 Monitoring and Logging

### Episode Statistics

**Monitor Wrapper** logs:
- Episode reward (cumulative)
- Episode length (steps)
- Episode time (seconds)

**Accessing Statistics:**
```python
from stable_baselines3.common.monitor import Monitor

env = Monitor(env, filename="./logs/monitor.csv")

# After episodes, read CSV
import pandas as pd
df = pd.read_csv("./logs/monitor.csv")
mean_reward = df["r"].mean()
mean_length = df["l"].mean()
```

### TensorBoard Logging

**Stable-Baselines3** automatically logs to TensorBoard:

```python
from stable_baselines3 import PPO

model = PPO(
    "MultiInputPolicy",
    vec_env,
    tensorboard_log="./logs/tensorboard/"
)
model.learn(total_timesteps=1e6)
```

**What's Logged:**
- Episode reward (mean/std)
- Episode length
- Value function loss
- Policy loss
- Entropy
- Learning rate

**View Logs:**
```bash
tensorboard --logdir ./logs/tensorboard/
```

### Custom Callbacks

**For Advanced Logging:**
```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        # Log custom metrics
        if self.n_calls % 1000 == 0:
            print(f"Step {self.n_calls}, Reward: {self.locals['rewards']}")
        return True

model.learn(total_timesteps=1e6, callback=CustomCallback())
```

---

## 🔗 Wrapper Chaining

### How to Chain Wrappers

**Order Matters!** Apply wrappers from outermost to innermost:

```python
env = gym.make("ballbot-v0.1")

# 1. TimeLimit (outermost - terminates episodes)
env = TimeLimit(env, max_episode_steps=4000)

# 2. ClipAction (clips actions before env receives them)
env = ClipAction(env)

# 3. NormalizeObservation (normalizes observations)
env = NormalizeObservation(env)

# 4. Monitor (innermost - logs everything)
env = Monitor(env, filename="./logs/monitor.csv")
```

**Wrapper Stack:**
```
Monitor
  └─ NormalizeObservation
      └─ ClipAction
          └─ TimeLimit
              └─ BallbotEnv (base)
```

### Recommended Order

1. **TimeLimit** (outermost) - Terminates episodes
2. **ClipAction** - Ensures valid actions
3. **NormalizeObservation** - Normalizes observations
4. **FrameStack** - Stacks frames (if using)
5. **Monitor** (innermost) - Logs statistics

---

## 🎮 Real-World Example: Ballbot Training

### Complete Setup

**From `ballbot_rl/training/utils.py`:**

```python
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

def make_ballbot_env(terrain_type=None, reward_config=None, ...):
    def _init():
        env = gym.make(
            "ballbot-v0.1",
            GUI=gui,
            terrain_type=terrain_type,
            reward_config=reward_config,
            ...
        )
        return Monitor(env)  # Monitor wrapper for logging
    return _init

# Create vectorized environment
N_ENVS = 10
vec_env = DummyVecEnv([make_ballbot_env(...) for _ in range(N_ENVS)])
```

### Training Configuration

**From `ballbot_rl/training/train.py`:**

```python
# Choose vectorization strategy
use_subproc = config.get("use_subproc_vec_env", platform.system() != 'Darwin')
if use_subproc:
    VecEnvClass = SubprocVecEnv  # Parallel (Linux/Windows)
else:
    VecEnvClass = DummyVecEnv    # Sequential (macOS)

# Create training and evaluation environments
vec_env = VecEnvClass([
    make_ballbot_env(..., seed=seed) 
    for _ in range(N_ENVS)
])

eval_env = VecEnvClass([
    make_ballbot_env(..., seed=seed + N_ENVS + i, eval_env=True) 
    for i in range(N_ENVS)
])
```

### Monitoring Setup

**Automatic Logging:**
- Monitor wrapper logs to `{out_path}/logs/monitor.csv`
- TensorBoard logs to `{out_path}/logs/tensorboard/`
- EvalCallback logs evaluation metrics

---

## ✅ Best Practices

### 1. Always Use Monitor for Training

```python
env = Monitor(env, filename="./logs/monitor.csv")
```

**Why:** Essential for tracking training progress and debugging.

### 2. Use TimeLimit for Finite Episodes

```python
env = TimeLimit(env, max_episode_steps=4000)
```

**Why:** Prevents infinite episodes and ensures batching works.

### 3. Choose Vectorization Based on Platform

```python
if platform.system() == 'Darwin':
    VecEnvClass = DummyVecEnv  # macOS compatibility
else:
    VecEnvClass = SubprocVecEnv  # Linux/Windows speed
```

### 4. Apply Wrappers in Correct Order

**Order:** TimeLimit → ClipAction → NormalizeObservation → Monitor

### 5. Use Separate Environments for Evaluation

```python
eval_env = VecEnvClass([
    make_ballbot_env(..., eval_env=True) 
    for _ in range(N_ENVS)
])
```

**Why:** Evaluation environments should be deterministic and not logged.

---

## ⚠️ Common Pitfalls

### 1. Wrong Wrapper Order

**Problem:**
```python
env = Monitor(env)
env = TimeLimit(env, max_episode_steps=4000)  # Wrong order!
```

**Issue:** Monitor logs incomplete episodes if TimeLimit terminates early.

**Solution:** Apply TimeLimit before Monitor.

### 2. Forgetting to Reset Vectorized Environments

**Problem:**
```python
obs = vec_env.reset()
# ... later ...
obs = vec_env.step(actions)  # Forgot to reset after done!
```

**Issue:** Environments may be in inconsistent states.

**Solution:** Always reset after episodes terminate.

### 3. Multiprocessing Issues on macOS

**Problem:** SubprocVecEnv hangs on macOS with MuJoCo.

**Solution:** Use DummyVecEnv on macOS (automatic in Ballbot).

### 4. Not Handling `infos` Correctly

**Problem:**
```python
obs, rewards, dones, infos = vec_env.step(actions)
info = infos[0]  # Assumes single env
```

**Issue:** `infos` is a list, one dict per environment.

**Solution:** Iterate over `infos` or index correctly.

---

## 📝 Summary

**Key Takeaways:**

1. **Wrappers** modify environment behavior without changing base code
2. **Monitor** logs episode statistics for tracking progress
3. **Vectorization** speeds up training by running parallel environments
4. **Wrapper order** matters (TimeLimit → ClipAction → NormalizeObservation → Monitor)
5. **Platform matters** (DummyVecEnv on macOS, SubprocVecEnv on Linux/Windows)

**Next Steps:**

- Explore [Multi-Modal Fusion](10_multimodal_fusion.md) to understand observation processing
- Review [Complete Training Guide](13_complete_training_guide.md) for end-to-end training workflow
- Try modifying wrappers in the [Basic Usage Example](../../examples/01_basic_usage.py)

---

## 🔗 Related Resources

- [Gymnasium Wrappers Documentation](https://gymnasium.farama.org/api/wrappers/)
- [Stable-Baselines3 VecEnv Documentation](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
- [Ballbot Training Utils](../../ballbot_rl/training/utils.py) - Real implementation
- [Actor-Critic Methods Tutorial](05_actor_critic_methods.md) - How algorithms use environments

