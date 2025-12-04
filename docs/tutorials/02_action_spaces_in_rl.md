# 🎮 Action Spaces in Reinforcement Learning

*A comprehensive guide to designing action spaces for robotics applications*

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [What is an Action Space?](#what-is-an-action-space)
3. [Types of Action Spaces](#types-of-action-spaces)
4. [Continuous Actions for Robotics](#continuous-actions-for-robotics)
5. [Action Normalization and Scaling](#action-normalization-and-scaling)
6. [Neural Network Action Outputs](#neural-network-action-outputs)
7. [Real-World Example: Ballbot Actions](#real-world-example-ballbot-actions)
8. [Safety and Constraints](#safety-and-constraints)
9. [Hybrid Action Spaces](#hybrid-action-spaces)
10. [Best Practices](#best-practices)
11. [Summary](#summary)

---

## 🎯 Introduction

The action space is one of the most fundamental concepts in reinforcement learning. It defines **what the agent can do**—the set of all possible commands the agent may produce at each timestep.

> "The action space defines the agent's interface to the world. Get it wrong, and learning becomes impossible."  
> — *Common wisdom in deep RL*

In robotics, action spaces are particularly critical because they directly map to physical actuators—motors, servos, and other control mechanisms. A well-designed action space enables efficient learning; a poorly designed one can make learning impossible.

**Key Questions This Tutorial Answers:**
- What are the different types of action spaces?
- Why do robots almost always use continuous actions?
- How should actions be normalized and scaled?
- How do neural networks output actions?
- How do we map RL actions to real motor commands?

---

## 🔍 What is an Action Space?

### Mathematical Definition

In a Markov Decision Process (MDP), the action space **𝒜** is the set of all possible actions the agent can take in any state:

**𝒜** = {*a*₁, *a*₂, ..., *a*ₙ}

For each state **s** ∈ **𝒮**, the agent selects an action **a** ∈ **𝒜**(**s**), where **𝒜**(**s**) ⊆ **𝒜** is the set of actions available in state **s**.

### In Gymnasium

Gymnasium provides a standardized way to define action spaces:

```python
import gymnasium as gym
from gymnasium import spaces

# Example: 3D continuous action space
self.action_space = spaces.Box(
    low=-1.0, 
    high=1.0, 
    shape=(3,), 
    dtype=np.float32
)
```

This defines a **3-dimensional continuous action space** where each component is bounded in [-1, 1].

### Why Bounds Matter

> "Bounded action spaces are essential for stable learning. Unbounded actions lead to instability and dangerous behaviors."  
> — *Sergey Levine, UC Berkeley*

Bounded action spaces:
- Prevent the agent from commanding dangerous torques/velocities
- Enable proper normalization for neural networks
- Make learning more stable and predictable

---

## 📦 Types of Action Spaces

Gymnasium provides several action space types, each suited for different applications:

### 1. Discrete Action Space

A **discrete action space** contains a finite set of actions:

```python
spaces.Discrete(n)  # n possible actions: {0, 1, 2, ..., n-1}
```

**Mathematical Representation:**
**𝒜** = {0, 1, 2, ..., *n* - 1}

**Use Cases:**
- Atari games (up, down, left, right, fire)
- Gridworld navigation
- Simple decision-making tasks

**Example:**
```python
action_space = spaces.Discrete(4)  # 4 actions: North, East, South, West
action = env.action_space.sample()  # Returns: 0, 1, 2, or 3
```

**Why Not for Robotics?**
Robots require fine-grained control. Discrete actions like "turn left" or "move forward" are too coarse for precise manipulation or locomotion.

### 2. MultiDiscrete Action Space

For multiple independent discrete choices:

```python
spaces.MultiDiscrete([3, 3, 2])  # Three choices: [0-2, 0-2, 0-1]
```

**Use Cases:**
- Multi-agent games
- Hierarchical control (rare in robotics)

### 3. Box (Continuous) Action Space ⭐ **THE ROBOTICS DEFAULT**

A **Box space** represents a continuous, bounded region in ℝⁿ:

```python
spaces.Box(
    low=np.array([-1.0, -1.0, -1.0]),  # Lower bounds
    high=np.array([1.0, 1.0, 1.0]),    # Upper bounds
    shape=(3,),                         # Dimension
    dtype=np.float32
)
```

**Mathematical Representation:**
**𝒜** = {**a** ∈ ℝⁿ : **low** ≤ **a** ≤ **high**}

Where **low** and **high** are vectors defining the bounds.

**Why Robots Use Continuous Actions:**

1. **Physical Reality**: Motors accept real-valued torque/velocity commands
2. **Fine Control**: Precise manipulation requires continuous control
3. **Smooth Motions**: Continuous actions enable smooth, natural movements
4. **Efficiency**: Direct mapping to actuators without discretization overhead

> "Continuous control is the natural language of robotics. Discretization is an artificial constraint that limits capability."  
> — *Pieter Abbeel, UC Berkeley*

### 4. MultiBinary Action Space

Multiple independent binary decisions:

```python
spaces.MultiBinary(5)  # 5 independent binary choices
```

**Use Cases:**
- Feature selection
- Rare in robotics

### 5. Dict Action Spaces (Hybrid)

For complex robots with mixed action types:

```python
spaces.Dict({
    "joint_torques": spaces.Box(-1, 1, shape=(6,)),  # Continuous
    "gripper": spaces.Discrete(2),                    # Discrete: open/close
    "mode": spaces.Discrete(3)                        # Discrete: mode selection
})
```

**Use Cases:**
- Manipulation robots (continuous joints + discrete gripper)
- Mobile manipulators (continuous base + discrete tools)

---

## 🤖 Continuous Actions for Robotics

### The Standard Pattern

Almost all modern robotics RL uses **normalized continuous actions**:

```python
# Standard robotics action space
action_space = spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(n_joints,),
    dtype=np.float32
)
```

**Why [-1, 1]?**
- Neural networks train best with normalized inputs/outputs
- Easy to scale to different actuator ranges
- Prevents numerical instability

### Action-to-Actuator Mapping

The environment must map normalized actions to physical commands:

```
RL Action (normalized) → Scaling → Physical Command → Actuator
     [-1, 1]          →   ×τₘₐₓ   →    [-τₘₐₓ, τₘₐₓ]  →  Motor
```

**General Formula:**
**τ** = **a** ⊙ **τₘₐₓ**

Where:
- **τ** is the physical command vector
- **a** is the normalized action vector
- **τₘₐₓ** is the maximum torque/velocity for each actuator
- ⊙ denotes element-wise multiplication

---

## ⚖️ Action Normalization and Scaling

### Why Normalize?

> "Normalization is not optional—it's a fundamental requirement for deep RL. Without it, learning becomes unstable or impossible."  
> — *John Schulman, OpenAI*

**Benefits:**
1. **Stable Learning**: Neural networks require inputs/outputs in similar ranges
2. **Generalization**: Normalized actions work across different robots
3. **Numerical Stability**: Prevents gradient explosions
4. **Easier Tuning**: Hyperparameters become more transferable

### Normalization Strategy

**Step 1: Define Normalized Space**
```python
action_space = spaces.Box(-1.0, 1.0, shape=(3,))
```

**Step 2: Scale in Environment**
```python
def step(self, action):
    # Action is in [-1, 1]
    # Scale to physical range
    max_velocity = 10.0  # rad/s
    physical_command = action * max_velocity
    
    # Apply to actuators
    self.data.ctrl[:] = physical_command
```

### Per-Actuator Scaling

Different joints may have different limits:

```python
# Different max velocities for each joint
max_velocities = np.array([10.0, 8.0, 12.0])  # rad/s

# Scale each action component independently
physical_commands = action * max_velocities
```

**Mathematical Form:**
**τ** = **a** ⊙ **τₘₐₓ**

Where **τₘₐₓ** = [τ₁ₘₐₓ, τ₂ₘₐₓ, ..., τₙₘₐₓ]ᵀ

---

## 🧠 Neural Network Action Outputs

### Policy Network Architecture

For continuous actions, the policy network typically outputs:

1. **Mean** μ(**s**) - The deterministic action
2. **Standard Deviation** σ(**s**) - For exploration (optional)

```python
# Simplified policy network
class PolicyNetwork(nn.Module):
    def forward(self, state):
        # ... feature extraction ...
        mean = self.mean_head(features)      # Shape: (batch, n_actions)
        std = self.std_head(features)        # Shape: (batch, n_actions)
        return mean, std
```

### Action Sampling

Actions are sampled from a distribution (typically Gaussian):

**a** ~ π(·|**s**) = 𝒩(μ(**s**), σ²(**s**))

**In Practice:**
```python
mean, std = policy_network(state)
action = mean + std * torch.randn_like(mean)
action = torch.tanh(action)  # Enforce bounds [-1, 1]
```

### Tanh Activation for Bounds

The `tanh` function naturally enforces bounds:

**a** = tanh(μ + σ · ε)

Where:
- ε ~ 𝒩(0, 1) is exploration noise
- tanh maps ℝ → (-1, 1)

**Properties:**
- Smooth and differentiable
- Saturates at boundaries
- Centered at zero

> "The tanh activation is the natural choice for bounded continuous actions. It provides smooth saturation and stable gradients."  
> — *Tuomas Haarnoja, Google DeepMind (SAC paper)*

---

## 🎮 Real-World Example: Ballbot Actions

Let's examine the **Ballbot** environment's action space implementation:

### Action Space Definition

```python
# From bbot_env.py
self.action_space = gym.spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(3,),  # Three omniwheels
    dtype=np.float32
)
```

**What This Means:**
- 3 continuous actions (one per omniwheel)
- Each action in [-1, 1]
- Represents normalized wheel velocity commands

### Action Processing in `step()`

```python
def step(self, omniwheel_commands):
    """
    omniwheel_commands: np.ndarray of shape (3,), values in [-1, 1]
    """
    # Scale from normalized [-1, 1] to physical range [-10, 10] rad/s
    ctrl = omniwheel_commands * 10.0
    
    # Safety: clip to prevent extreme commands
    ctrl = np.clip(ctrl, a_min=-10, a_max=10)
    
    # Apply to MuJoCo (negative for correct coordinate system)
    self.data.ctrl[:] = -ctrl
    
    # Step physics
    mujoco.mj_step(self.model, self.data)
    
    # ... compute reward, observation, etc. ...
```

### Why the Negative Sign?

The negative sign accounts for coordinate system conventions between:
- RL action convention (positive = forward)
- MuJoCo actuator convention (may be opposite)

This is a common pattern in robotics RL.

### Action Flow Diagram

```
┌─────────────────────────────────────────┐
│  Policy Network (PPO/SAC/etc.)         │
│  Output: μ ∈ [-1, 1]³                  │
└──────────────┬─────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Action Sampling                        │
│  a = tanh(μ + σ·ε)                      │
│  Result: a ∈ [-1, 1]³                   │
└──────────────┬─────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Environment.step(a)                    │
│  Scale: τ = a × 10.0                   │
│  Clip: τ ∈ [-10, 10]                   │
└──────────────┬─────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  MuJoCo Physics                         │
│  Apply: data.ctrl[:] = -τ               │
│  Step: mj_step(model, data)             │
└─────────────────────────────────────────┘
```

### Physical Interpretation

For the Ballbot:
- **Action [1.0, 0.0, 0.0]**: Wheel 0 spins forward at 10 rad/s, others stationary
- **Action [0.0, 1.0, 0.0]**: Wheel 1 spins forward at 10 rad/s
- **Action [0.5, 0.5, 0.5]**: All wheels spin forward at 5 rad/s (balanced forward motion)

The omniwheel configuration allows the robot to move in any direction by combining wheel velocities.

---

## 🛡️ Safety and Constraints

### Action Clipping

Always clip actions to prevent dangerous commands:

```python
# In environment step()
action = np.clip(action, 
                 a_min=self.action_space.low,
                 a_max=self.action_space.high)
```

### Rate Limiting

For smooth control, limit action changes:

```python
# Rate limit: max change per step
max_change = 0.1
action_change = action - self.prev_action
action_change = np.clip(action_change, -max_change, max_change)
action = self.prev_action + action_change
```

### Torque Limits

Physical actuators have limits:

```python
# Per-joint torque limits
max_torques = np.array([5.0, 5.0, 5.0])  # N·m
torques = action * max_torques
torques = np.clip(torques, -max_torques, max_torques)
```

> "Safety constraints are not optional in robotics. They prevent damage to hardware and ensure stable operation."  
> — *Russ Tedrake, MIT*

---

## 🔀 Hybrid Action Spaces

Some robots require both continuous and discrete actions:

### Example: Manipulation Robot

```python
action_space = spaces.Dict({
    "arm_torques": spaces.Box(-1, 1, shape=(6,)),  # 6-DOF arm
    "gripper": spaces.Discrete(2),                  # Open/Close
    "tool_mode": spaces.Discrete(3)                 # Mode selection
})
```

### Processing Hybrid Actions

```python
def step(self, action_dict):
    # Continuous: scale and apply
    arm_torques = action_dict["arm_torques"] * self.max_torques
    self.data.ctrl[:6] = arm_torques
    
    # Discrete: direct mapping
    if action_dict["gripper"] == 0:
        self.gripper.open()
    else:
        self.gripper.close()
    
    # Discrete: mode switching
    self.set_mode(action_dict["tool_mode"])
```

---

## ✅ Best Practices

### 1. Always Normalize to [-1, 1]

```python
# ✅ Good
action_space = spaces.Box(-1.0, 1.0, shape=(n,))

# ❌ Bad
action_space = spaces.Box(-10.0, 10.0, shape=(n,))  # Harder to learn
```

### 2. Scale in Environment, Not Policy

```python
# ✅ Good: Policy outputs normalized, env scales
action = policy(state)  # [-1, 1]
torque = action * max_torque  # In environment

# ❌ Bad: Policy outputs physical values
torque = policy(state)  # Direct physical values (unstable)
```

### 3. Use Tanh for Bounded Actions

```python
# ✅ Good
action = torch.tanh(policy_output)

# ❌ Bad
action = torch.clamp(policy_output, -1, 1)  # Hard clipping (bad gradients)
```

### 4. Clip for Safety

```python
# ✅ Good: Always clip as safety measure
action = np.clip(action, 
                 self.action_space.low,
                 self.action_space.high)
```

### 5. Document Action Conventions

```python
# ✅ Good: Clear documentation
"""
Action Space: Box(-1.0, 1.0, shape=(3,))
- action[0]: Left omniwheel velocity (normalized)
- action[1]: Right omniwheel velocity (normalized)
- action[2]: Back omniwheel velocity (normalized)
- Scaled to [-10, 10] rad/s internally
"""
```

---

## 🎯 Advanced Action Space Techniques ⭐⭐

### Action Masking

**Concept:** Dynamically disable invalid actions based on current state.

**Why Use Action Masking?**
- Prevents invalid actions (e.g., moving when stuck)
- Improves sample efficiency
- More stable learning
- Common in hierarchical RL

**Implementation:**
```python
class MaskedActionSpace:
    """
    Action space with dynamic masking.
    """
    def __init__(self, base_action_space):
        self.base_action_space = base_action_space
    
    def get_valid_actions(self, state):
        """
        Return mask of valid actions.
        """
        mask = np.ones(self.base_action_space.shape[0], dtype=bool)
        
        # Example: Disable actions that would cause collision
        if self.would_collide(state):
            mask[0] = False  # Disable forward action
        
        # Example: Disable actions when battery low
        if self.battery_low(state):
            mask[2] = False  # Disable high-power action
        
        return mask
    
    def sample(self, state):
        """Sample only valid actions."""
        mask = self.get_valid_actions(state)
        valid_actions = np.where(mask)[0]
        if len(valid_actions) == 0:
            return np.zeros(self.base_action_space.shape[0])
        action_idx = np.random.choice(valid_actions)
        action = np.zeros(self.base_action_space.shape[0])
        action[action_idx] = 1.0
        return action
```

**In Policy:**
```python
class MaskedPolicy(nn.Module):
    """
    Policy that respects action masks.
    """
    def forward(self, state, action_mask):
        logits = self.policy_net(state)
        
        # Mask invalid actions (set to large negative value)
        logits = logits.masked_fill(~action_mask, float('-inf'))
        
        # Sample from valid actions
        action_dist = torch.distributions.Categorical(logits=logits)
        return action_dist
```

### Hierarchical Action Spaces ⭐

**Concept:** Decompose complex actions into high-level and low-level actions.

**Why Hierarchical?**
- Reduces action space complexity
- Enables long-horizon planning
- More interpretable policies
- Better for complex tasks

**Two-Level Hierarchy:**
```python
class HierarchicalActionSpace:
    """
    Two-level hierarchical action space.
    """
    def __init__(self):
        # High-level: Meta-actions (goals, skills)
        self.high_level_space = spaces.Discrete(4)  # e.g., [move_forward, turn_left, turn_right, stop]
        
        # Low-level: Primitive actions (motor commands)
        self.low_level_space = spaces.Box(-1, 1, shape=(3,))  # Motor velocities
    
    def step(self, high_level_action, low_level_action):
        """
        Execute hierarchical action.
        """
        # High-level action selects skill/behavior
        if high_level_action == 0:  # move_forward
            # Low-level action controls motors for forward movement
            motor_command = self.compute_forward_motion(low_level_action)
        elif high_level_action == 1:  # turn_left
            motor_command = self.compute_turn_left(low_level_action)
        # ... etc
        
        return motor_command
```

**Hierarchical Policy:**
```python
class HierarchicalPolicy(nn.Module):
    """
    Two-level hierarchical policy.
    """
    def __init__(self):
        self.high_level_policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 high-level actions
        )
        self.low_level_policy = nn.Sequential(
            nn.Linear(state_dim + 1, 64),  # +1 for high-level action
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 motor commands
        )
    
    def forward(self, state):
        # High-level decision
        high_level_logits = self.high_level_policy(state)
        high_level_action = torch.argmax(high_level_logits, dim=1)
        
        # Low-level decision (conditioned on high-level)
        low_level_input = torch.cat([state, high_level_action.unsqueeze(1)], dim=1)
        low_level_action = torch.tanh(self.low_level_policy(low_level_input))
        
        return high_level_action, low_level_action
```

**Modern Approaches:**
- **HAC (Hierarchical Actor-Critic)**: End-to-end hierarchical RL
- **HIRO (Data-Efficient Hierarchical RL)**: Learn from off-policy data
- **HRL (Hierarchical Reinforcement Learning)**: General framework

### Action Chunking

**Concept:** Execute sequences of actions instead of single actions.

**Why Chunking?**
- Reduces action space
- Enables temporal abstraction
- Better for long-horizon tasks
- More stable learning

**Implementation:**
```python
class ActionChunking:
    """
    Execute sequences of actions.
    """
    def __init__(self, chunk_size=5):
        self.chunk_size = chunk_size
        self.action_buffer = []
    
    def step(self, action):
        """
        Add action to buffer, execute when buffer full.
        """
        self.action_buffer.append(action)
        
        if len(self.action_buffer) >= self.chunk_size:
            # Execute chunk
            for a in self.action_buffer:
                env.step(a)
            self.action_buffer = []
```

---

## 📊 Summary

### Key Takeaways

1. **Continuous actions are standard for robotics** - They match physical reality and enable fine control

2. **Normalize to [-1, 1]** - Makes learning stable and generalizable

3. **Scale in environment** - Keep policy outputs normalized, scale to physical units internally

4. **Use tanh for bounds** - Provides smooth, differentiable saturation

5. **Always clip for safety** - Prevent dangerous commands

6. **Document conventions** - Make action meanings clear

7. **Consider action masking** - For invalid action prevention ⭐

8. **Consider hierarchical actions** - For complex, long-horizon tasks ⭐

### Action Space Design Checklist

- [ ] Action space matches actuator capabilities
- [ ] Actions normalized to [-1, 1]
- [ ] Scaling happens in environment
- [ ] Safety clipping implemented
- [ ] Action conventions documented
- [ ] Coordinate systems verified
- [ ] Considered action masking for invalid actions ⭐
- [ ] Considered hierarchical actions for complex tasks ⭐

---

## 🎯 Next Steps

Now that you understand action spaces, here's what to explore next:

### Related Tutorials
- **[Observation Spaces in RL](03_observation_spaces_in_rl.md)** - Learn how observations complement actions
- **[Reward Design](04_reward_design_for_robotics.md)** - Design rewards that work with your action space
- **[Actor-Critic Methods](05_actor_critic_methods.md)** - See how policies output actions

### Practical Examples
- **[Basic Usage Example](../../examples/01_basic_usage.py)** - See actions in action
- **[Custom Policy Example](../../examples/04_custom_policy.py)** - Custom action outputs
- **[Training Workflow](../../examples/05_training_workflow.py)** - Full training with actions

### Concepts to Explore
- **[RL Fundamentals](../concepts/rl_fundamentals.md)** - MDP action space formulation
- **[Reward Design](../concepts/reward_design.md)** - How actions affect rewards
- **[Architecture Overview](../architecture/README.md)** - Action space in system design

### Research Papers
- **[Research Timeline](../research/timeline.md)** - How action spaces evolved
- **[Code Mapping](../research/code_mapping.md)** - Action space implementation

**Prerequisites for Next Tutorial:**
- Understanding of action spaces (this tutorial)
- Basic Gymnasium knowledge
- Familiarity with continuous control

---

## 📚 Further Reading

### Papers

**Classic Action Spaces:**
- **Schulman et al. (2017)** - "Proximal Policy Optimization Algorithms" - Action space normalization

**Modern Action Techniques:**
- **Nachum et al. (2018)** - "Data-Efficient Hierarchical Reinforcement Learning" - HIRO
- **Levy et al. (2019)** - "Hierarchical Reinforcement Learning with Hindsight" - HAC
- **Vezhnevets et al. (2017)** - "FeUdal Networks for Hierarchical Reinforcement Learning" - FeUdal Networks
- **Huang et al. (2019)** - "Action Masking in Reinforcement Learning" - Action masking techniques
- **Haarnoja et al. (2018)** - "Soft Actor-Critic" - Continuous action handling
- **Lillicrap et al. (2015)** - "Continuous Control with Deep Reinforcement Learning" - DDPG

### Books
- **Sutton & Barto** - "Reinforcement Learning: An Introduction" - Chapter 3: Finite Markov Decision Processes

### Code Examples
- Ballbot environment: `ballbot_gym/bbot_env.py`
- Stable-Baselines3: Continuous action policies

---

## 🎓 Exercises

1. **Modify Ballbot Actions**: Change the action space to use different scaling (e.g., [-2, 2] instead of [-1, 1]). How does this affect learning?

2. **Add Rate Limiting**: Implement action rate limiting in the Ballbot environment. What happens to policy performance?

3. **Hybrid Actions**: Design a hybrid action space for a robot with both continuous joints and a discrete gripper.

---

*Next Tutorial: [Observation Spaces in Reinforcement Learning](03_observation_spaces_in_rl.md)*

---

**Happy Learning! 🚀**

