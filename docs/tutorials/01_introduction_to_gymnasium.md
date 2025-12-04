# 🏋️ Introduction to Gymnasium

*A comprehensive guide to the Gymnasium API for reinforcement learning environments*

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [What is Gymnasium?](#what-is-gymnasium)
3. [Why Gymnasium Exists](#why-gymnasium-exists)
4. [Core API](#core-api)
5. [Real-World Example: Ballbot Environment](#real-world-example-ballbot-environment)
6. [Environment Lifecycle](#environment-lifecycle)
7. [Spaces: Action and Observation](#spaces-action-and-observation)
8. [Integration with RL Algorithms](#integration-with-rl-algorithms)
9. [Advanced Gymnasium Features](#advanced-gymnasium-features)
10. [Best Practices](#best-practices)
11. [Summary](#summary)

---

## 🎯 Introduction

Gymnasium is the **standard interface** for reinforcement learning environments. It provides a consistent API that allows RL algorithms to interact with any environment—from simple gridworlds to complex physics simulations.

> "Gymnasium provides the lingua franca of reinforcement learning. It's the interface that makes everything work together."  
> — *Greg Brockman, OpenAI (on the original Gym)*

**Key Concepts:**
- Gymnasium is an **interface**, not a simulator
- It defines a **standard contract** between agents and environments
- It enables **algorithm-environment interchangeability**

---

## 🔍 What is Gymnasium?

### The Interface Layer

Gymnasium sits between RL algorithms and simulators:

```
┌─────────────────────────────────────┐
│      RL Algorithm                   │
│  (PPO, SAC, DQN, etc.)             │
└──────────────┬──────────────────────┘
               │
               │ Uses Gymnasium API
               ▼
┌─────────────────────────────────────┐
│   Gymnasium Environment             │
│   (reset, step, spaces)             │
└──────────────┬──────────────────────┘
               │
               │ Wraps
               ▼
┌─────────────────────────────────────┐
│   Simulator / Physics Engine        │
│   (MuJoCo, Isaac, PyBullet, etc.)   │
└─────────────────────────────────────┘
```

**Critical Point:** Gymnasium is **NOT** a physics simulator. It's the **interface layer** that standardizes how algorithms interact with simulators.

### The Standard Contract

Every Gymnasium environment must provide:

1. **`reset(seed=None)`** - Initialize or reset the environment
2. **`step(action)`** - Execute one timestep
3. **`action_space`** - Definition of valid actions
4. **`observation_space`** - Definition of valid observations

That's it! This simple contract enables universal compatibility.

---

## 🎯 Why Gymnasium Exists

### 1. Interchangeability

> "The power of Gymnasium is that any algorithm can work with any environment, as long as they both speak the same language."  
> — *Common wisdom in RL community*

**Example:**
```python
# Same algorithm, different environments
env1 = gym.make("CartPole-v1")
env2 = gym.make("ballbot-v0.1")
env3 = MyCustomEnv()

# PPO works with all of them
model1 = PPO("MlpPolicy", env1)
model2 = PPO("MlpPolicy", env2)
model3 = PPO("MlpPolicy", env3)
```

### 2. Reproducibility

Standardized API ensures:
- Experiments are comparable
- Results are reproducible
- Code is shareable

### 3. Ecosystem Compatibility

Major RL libraries all use Gymnasium:
- **Stable-Baselines3** - PPO, SAC, TD3, etc.
- **CleanRL** - Clean implementations
- **RLlib** - Distributed RL
- **Acme** - Research framework

### 4. Fast Prototyping

You can create a new environment in minutes:

```python
class MyEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(-1, 1, shape=(2,))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,))
    
    def reset(self, seed=None):
        # Initialize environment
        return obs, {}
    
    def step(self, action):
        # Execute action
        return obs, reward, terminated, truncated, {}
```

---

## 📚 Core API

### 1. `reset(seed=None)`

Initializes or resets the environment to an initial state.

**Returns:**
```python
obs, info = env.reset(seed=42)
```

- **`obs`**: Initial observation
- **`info`**: Additional information (dict)

**Example:**
```python
obs, info = env.reset()
print(f"Initial observation: {obs}")
print(f"Info: {info}")
```

### 2. `step(action)`

Executes one timestep of the environment.

**Returns:**
```python
obs, reward, terminated, truncated, info = env.step(action)
```

- **`obs`**: Next observation
- **`reward`**: Scalar reward
- **`terminated`**: True if episode ended (e.g., failure, success)
- **`truncated`**: True if episode ended due to time limit
- **`info`**: Additional information

**Example:**
```python
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
```

### 3. Spaces

Define the format of actions and observations:

```python
env.action_space      # What actions are valid?
env.observation_space # What observations look like?
```

**Example:**
```python
print(env.action_space)
# Box(-1.0, 1.0, (3,), float32)

print(env.observation_space)
# Dict('orientation': Box(-3.14, 3.14, (3,), float32), ...)
```

---

## 🤖 Real-World Example: Ballbot Environment

Let's examine how the **Ballbot** environment implements the Gymnasium API:

### Environment Class Definition

```python
# From bbot_env.py
class BBotSimulation(gym.Env):
    def __init__(self, xml_path, ...):
        super().__init__()
        
        # Define spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Dict({
            "orientation": gym.spaces.Box(...),
            "angular_vel": gym.spaces.Box(...),
            # ... more components
        })
        
        # Initialize MuJoCo
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
```

### Reset Implementation

```python
def reset(self, seed=None, goal: str = "random", **kwargs):
    # Initialize random number generator
    super().reset(seed=seed)
    
    # Reset goal and reward
    self._reset_goal_and_reward_objs()
    
    # Generate new terrain
    init_height = self._reset_terrain()
    
    # Reset MuJoCo state
    mujoco.mj_resetData(self.model, self.data)
    
    # Position robot on terrain
    self.data.joint("base_free_joint").qpos[2] += init_height
    
    # Get initial observation
    obs = self._get_obs(np.zeros(3))
    info = self._get_info()
    
    return obs, info
```

### Step Implementation

```python
def step(self, omniwheel_commands):
    # Scale actions from [-1, 1] to physical range
    ctrl = omniwheel_commands * 10.0
    ctrl = np.clip(ctrl, -10, 10)
    
    # Apply to MuJoCo
    self.data.ctrl[:] = -ctrl
    mujoco.mj_step(self.model, self.data)
    
    # Get observation
    obs = self._get_obs(omniwheel_commands)
    info = self._get_info()
    
    # Compute reward
    reward = self.compute_reward(obs, omniwheel_commands)
    
    # Check termination
    terminated = self.check_termination(obs)
    truncated = False
    
    return obs, reward, terminated, truncated, info
```

### Complete Interaction Loop

```python
# Create environment
env = BBotSimulation(xml_path="bbot.xml")

# Reset
obs, info = env.reset()

# Training loop
for step in range(1000):
    # Get action from policy
    action = policy(obs)
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check if done
    if terminated or truncated:
        obs, info = env.reset()

# Cleanup
env.close()
```

---

## 🔄 Environment Lifecycle

### 1. Initialization (`__init__`)

```python
def __init__(self):
    # Define spaces
    self.action_space = ...
    self.observation_space = ...
    
    # Initialize simulator
    self.model = ...
    self.data = ...
```

### 2. Reset

```python
def reset(self, seed=None):
    # Set random seed
    super().reset(seed=seed)
    
    # Reset simulator state
    # Generate new initial conditions
    # Return initial observation
    return obs, info
```

### 3. Step Loop

```python
while not done:
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

### 4. Cleanup

```python
def close(self):
    # Close renderers
    # Free resources
    # Clean up
    pass
```

---

## 📦 Spaces: Action and Observation

### Action Space

Defines what actions the agent can take:

```python
# Continuous actions (robotics standard)
self.action_space = spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(n_actuators,),
    dtype=np.float32
)

# Discrete actions (games)
self.action_space = spaces.Discrete(n_actions)
```

### Observation Space

Defines what the agent observes:

```python
# Vector observations
self.observation_space = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(n_features,),
    dtype=np.float32
)

# Dict observations (multi-modal)
self.observation_space = spaces.Dict({
    "proprio": spaces.Box(...),
    "camera": spaces.Box(...)
})
```

---

## 🔌 Integration with RL Algorithms

### Stable-Baselines3

```python
from stable_baselines3 import PPO

env = gym.make("ballbot-v0.1")
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

### Custom Training Loop

```python
env = MyEnv()
obs, info = env.reset()

for step in range(num_steps):
    action = agent.select_action(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    agent.store_transition(obs, action, reward, next_obs, terminated)
    
    if terminated or truncated:
        obs, info = env.reset()
    else:
        obs = next_obs
```

---

## 🚀 Advanced Gymnasium Features

### Vectorized Environments (VecEnv)

**Why Vectorize?**
- **Parallel execution**: Run multiple environments simultaneously
- **Faster data collection**: Collect more samples per second
- **Better GPU utilization**: Batch observations for neural networks

**Types of VecEnv:**

**1. DummyVecEnv (Synchronous):**
```python
from stable_baselines3.common.vec_env import DummyVecEnv

# Wrap single environment
env = gym.make("ballbot-v0.1")
vec_env = DummyVecEnv([lambda: env])

# Now step returns batched observations
obs = vec_env.reset()  # Shape: (1, ...)
obs, rewards, dones, infos = vec_env.step(actions)  # actions: (1, 3)
```

**2. SubprocVecEnv (Asynchronous):**
```python
from stable_baselines3.common.vec_env import SubprocVecEnv

# Create multiple environments in separate processes
def make_env():
    return gym.make("ballbot-v0.1")

vec_env = SubprocVecEnv([make_env for _ in range(8)])  # 8 parallel envs

# Step all environments in parallel
obs = vec_env.reset()  # Shape: (8, ...)
obs, rewards, dones, infos = vec_env.step(actions)  # actions: (8, 3)
```

**Benefits of SubprocVecEnv:**
- True parallelism (uses multiple CPU cores)
- Faster than DummyVecEnv for CPU-bound environments
- Better for MuJoCo environments

**3. VecFrameStack (Frame Stacking):**
```python
from stable_baselines3.common.vec_env import VecFrameStack

# Stack frames for temporal information
vec_env = VecFrameStack(vec_env, n_stack=4)

# Observations now include last 4 frames
obs = vec_env.reset()  # Shape: (8, 4, H, W) for images
```

### Environment Wrappers

**Wrappers modify environments without changing base implementation.**

**Common Wrappers:**

**1. TimeLimit:**
```python
from gymnasium.wrappers import TimeLimit

# Limit episode length
env = TimeLimit(env, max_episode_steps=1000)
```

**2. ClipAction:**
```python
from gymnasium.wrappers import ClipAction

# Clip actions to action space bounds
env = ClipAction(env)
```

**3. NormalizeObservation:**
```python
from gymnasium.wrappers import NormalizeObservation

# Normalize observations (running mean/std)
env = NormalizeObservation(env)
```

**4. FrameStack:**
```python
from gymnasium.wrappers import FrameStack

# Stack consecutive frames
env = FrameStack(env, num_stack=4)
```

**5. RecordVideo:**
```python
from gymnasium.wrappers import RecordVideo

# Record episodes to video
env = RecordVideo(env, video_folder="./videos")
```

**6. Monitor (Stable-Baselines3):**
```python
from stable_baselines3.common.monitor import Monitor

# Log episode statistics
env = Monitor(env, filename="./logs/monitor.csv")
```

### Chaining Wrappers

**Combine Multiple Wrappers:**
```python
from gymnasium.wrappers import TimeLimit, ClipAction, NormalizeObservation
from stable_baselines3.common.monitor import Monitor

env = gym.make("ballbot-v0.1")
env = TimeLimit(env, max_episode_steps=4000)
env = ClipAction(env)
env = NormalizeObservation(env)
env = Monitor(env, filename="./logs/monitor.csv")
```

**Order Matters:**
- Apply `TimeLimit` first (outermost)
- Apply `ClipAction` before normalization
- Apply `Monitor` last (innermost)

### Environment Registration

**Register Custom Environments:**

**1. Basic Registration:**
```python
from gymnasium.envs.registration import register

register(
    id="Ballbot-v0.1",
    entry_point="ballbot_gym.ballbot_gym.bbot_env:BBotSimulation",
    max_episode_steps=4000,
    kwargs={
        "xml_path": "path/to/bbot.xml",
        "enable_cameras": True,
    }
)

# Now can use:
env = gym.make("Ballbot-v0.1")
```

**2. With Versioning:**
```python
register(
    id="Ballbot-v0",
    entry_point="ballbot_gym.ballbot_gym.bbot_env:BBotSimulation",
    max_episode_steps=4000,
)

register(
    id="Ballbot-v1",
    entry_point="ballbot_gym.ballbot_gym.bbot_env:BBotSimulation",
    max_episode_steps=5000,  # Different max steps
)
```

**3. Entry Point in setup.py:**
```python
# setup.py
setup(
    name="ballbot_gym",
    entry_points={
        "gymnasium.envs": [
            "Ballbot-v0.1=ballbot_gym.ballbot_gym.bbot_env:BBotSimulation",
        ],
    },
)
```

### Async Environments

**AsyncVecEnv (Gymnasium v0.29+):**
```python
from gymnasium.vector import AsyncVectorEnv

# Create async vectorized environment
env = AsyncVectorEnv([make_env for _ in range(8)])

# Non-blocking step (returns immediately)
obs, rewards, dones, infos = env.step_async(actions)

# Wait for results
obs, rewards, dones, infos = env.step_wait()
```

**Benefits:**
- Non-blocking operations
- Better for I/O-bound environments
- Can overlap computation and environment steps

### Modern Gymnasium API (v0.29+)

**Key Changes:**
- `done` → `terminated` and `truncated` (explicit distinction)
- Better type hints
- Improved error messages
- Vectorized environments built-in

**Migration Guide:**
```python
# Old API (Gym)
obs, reward, done, info = env.step(action)
if done:
    obs = env.reset()

# New API (Gymnasium)
obs, reward, terminated, truncated, info = env.step(action)
if terminated or truncated:
    obs, info = env.reset()
```

---

## ✅ Best Practices

### 1. Always Define Spaces

```python
# ✅ Good
self.action_space = spaces.Box(-1, 1, shape=(3,))
self.observation_space = spaces.Dict({...})

# ❌ Bad
# Missing space definitions
```

### 2. Use Proper Return Types

```python
# ✅ Good
return obs, reward, terminated, truncated, info

# ❌ Bad
return obs, reward, done  # Missing truncated, info
```

### 3. Handle Seeds Properly

```python
def reset(self, seed=None):
    super().reset(seed=seed)  # Always call super
    # ... reset logic ...
```

### 4. Document Your Environment

```python
"""
Ballbot Environment

Action Space: Box(-1.0, 1.0, shape=(3,))
- Normalized omniwheel commands

Observation Space: Dict with proprioceptive and visual data
- See observation_space for details

Reward: Directional progress + stability + efficiency
"""
```

---

## 📊 Summary

### Key Takeaways

1. **Gymnasium is an interface** - Not a simulator, but a standard API

2. **Simple contract** - `reset()`, `step()`, and spaces define everything

3. **Universal compatibility** - Any algorithm works with any environment

4. **Real-world example** - Ballbot shows complete implementation

### Gymnasium Checklist

- [ ] Implements `reset()` and `step()`
- [ ] Defines `action_space` and `observation_space`
- [ ] Returns proper tuple from `step()`
- [ ] Handles seeds in `reset()`
- [ ] Implements `close()` for cleanup
- [ ] Documented clearly

---

## 🎯 Next Steps

Now that you understand Gymnasium basics, here's what to explore next:

### Related Tutorials
- **[Action Spaces in RL](02_action_spaces_in_rl.md)** - Learn how actions are defined and structured
- **[Observation Spaces in RL](03_observation_spaces_in_rl.md)** - Understand what agents observe
- **[Reward Design](04_reward_design_for_robotics.md)** - Design effective reward functions

### Practical Examples
- **[Basic Usage Example](../../examples/01_basic_usage.py)** - Run your first environment
- **[Custom Reward Example](../../examples/02_custom_reward.py)** - Create custom rewards
- **[Training Workflow](../../examples/05_training_workflow.py)** - Complete training example

### Concepts to Explore
- **[RL Fundamentals](../concepts/rl_fundamentals.md)** - MDP formulation and RL basics
- **[Ballbot Mechanics](../concepts/ballbot_mechanics.md)** - Physics and dynamics
- **[Architecture Overview](../architecture/README.md)** - System design

### Research Papers
- **[Research Timeline](../research/timeline.md)** - Historical context
- **[Code Mapping](../research/code_mapping.md)** - Papers → code connections

**Prerequisites for Next Tutorial:**
- Understanding of Gymnasium API (this tutorial)
- Basic Python knowledge
- Familiarity with NumPy arrays

---

## 📚 Further Reading

### Documentation
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Gymnasium API Reference](https://gymnasium.farama.org/api/env/)

### Papers
- **Brockman et al. (2016)** - "OpenAI Gym" - Original Gym paper

### Code Examples
- Ballbot environment: `ballbot_gym/bbot_env.py`
- Gymnasium examples: [GitHub](https://github.com/Farama-Foundation/Gymnasium)

---

## 🎓 Exercises

1. **Create Simple Environment**: Implement a 2D point mass environment with Gymnasium API.

2. **Extend Ballbot**: Add a new observation component to the Ballbot environment.

3. **Integration Test**: Train a PPO agent on your custom environment.

---

*Next Tutorial: [Action Spaces in Reinforcement Learning](02_action_spaces_in_rl.md)*

---

**Happy Learning! 🚀**

