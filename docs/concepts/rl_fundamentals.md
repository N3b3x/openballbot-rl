# RL Fundamentals: MDP Formulation for Ballbot

*How classical ballbot mechanics translate into a reinforcement learning problem*

---

## Overview

This document explains how the ballbot control problem is formulated as a Markov Decision Process (MDP) for reinforcement learning. It bridges the gap between classical mechanics and RL by showing how physical states, actions, and objectives map to RL components.

---

## MDP Formulation

### What is an MDP?

A **Markov Decision Process** (MDP) is a mathematical framework for modeling decision-making in situations where:
- Outcomes are partially random
- Decisions affect future states
- Agent wants to maximize cumulative reward

**MDP Components:**
- **State space** (S): All possible states the agent can be in
- **Action space** (A): All possible actions the agent can take
- **Transition function** (P): Probability of transitioning from state s to s' given action a
- **Reward function** (R): Reward for taking action a in state s
- **Discount factor** (γ): How much to value future rewards vs. immediate rewards

---

## Ballbot MDP Formulation

### State Space (S)

**What the agent observes:**

The state space combines **proprioceptive** (internal) and **exteroceptive** (external) information:

**Proprioceptive Observations:**
- **Orientation**: Roll (φ), pitch (θ), yaw (ψ) angles
- **Angular velocities**: (φ̇, θ̇, ψ̇)
- **Linear velocities**: (ẋ, ẏ, ż)
- **Position**: (x, y, z) - though z is usually fixed

**Exteroceptive Observations:**
- **Depth images**: From RGB-D cameras
- **Encoded depth features**: Processed by CNN encoder

**Why This Design?**
- Proprioception: Needed for balance control (know tilt angles)
- Vision: Needed for terrain navigation (see slopes ahead)
- Both: Required for robust navigation on uneven terrain

**Connection to Mechanics:**
- Orientation and velocities come directly from Lagrangian dynamics (Nagarajan 2014)
- These are the state variables (q, q̇) from the equations of motion

**Code Location:**
- `ballbot_gym/envs/observation_spaces.py` - Defines observation space
- `ballbot_gym/envs/ballbot_env.py` - Extracts observations from MuJoCo state

---

### Action Space (A)

**What the agent controls:**

**Actions:**
- Three wheel torques: [τ₀, τ₁, τ₂]
- Normalized to [-1, 1] range
- Scaled internally to [-10, 10] rad/s

**Why Three Actions?**
- Matches physical design: three omniwheels (Lauwers 2006)
- Underactuated: 6 DOF but only 3 inputs
- Agent must learn to exploit dynamics

**Action Constraints:**
- Torque limits (saturation)
- Smoothness (penalized in reward)
- Safety (large actions → falls)

**Connection to Mechanics:**
- Actions are the control inputs (τ) from Lagrangian dynamics
- Agent learns to generate appropriate torques to achieve desired motion

**Code Location:**
- `ballbot_gym/envs/ballbot_env.py` - Action space definition
- `ballbot_gym/envs/ballbot_env.py` - Action application to motors

---

### Transition Function (P)

**How states evolve:**

**Physics Simulation:**
- MuJoCo simulates physics based on Lagrangian dynamics
- Given current state and action, computes next state
- Includes:
  - Ball-ground contact dynamics
  - Omniwheel-ball friction (anisotropic)
  - Robot body dynamics
  - Terrain interaction

**Stochasticity:**
- Terrain variation (Perlin noise)
- Initial conditions (random spawn)
- Sensor noise (if modeled)

**Why MuJoCo?**
- Accurate physics simulation
- Handles complex contact dynamics
- Fast enough for RL training
- Supports anisotropic friction (via patch)

**Connection to Mechanics:**
- Transition function is the Lagrangian equations of motion
- MuJoCo solves these equations numerically
- Agent doesn't need explicit equations—learns from experience

**Code Location:**
- `ballbot_gym/models/ballbot.xml` - Physics model
- MuJoCo engine - Solves dynamics

---

### Reward Function (R)

**What the agent optimizes:**

**Reward Components:**

1. **Directional Reward** (α₁):
   ```
   r_direction = (v_xy · g_target) / 100
   ```
   - Rewards velocity toward target direction
   - Encourages goal-directed navigation

2. **Action Penalty** (α₂):
   ```
   r_action = -0.0001 ||a||²
   ```
   - Penalizes large actions
   - Encourages efficient control
   - Prevents excessive energy use

3. **Survival Bonus** (α₃):
   ```
   r_survival = 0.02 · 𝟙[upright]
   ```
   - Rewards staying upright
   - Encourages balance maintenance
   - Prevents falls

**Total Reward:**
```
r = r_direction + r_action + r_survival
```

**Why This Design?**
- **Directional**: Enables goal-directed navigation
- **Action penalty**: Prevents reward hacking (e.g., spinning in place)
- **Survival**: Ensures agent learns to balance

**Connection to Mechanics:**
- Survival bonus encodes balance constraint (Carius 2022)
- Action penalty prevents excessive control effort
- Directional reward enables navigation

**Code Location:**
- `ballbot_gym/rewards/directional.py` - Reward implementation
- `ballbot_gym/envs/ballbot_env.py` - Reward computation

---

### Discount Factor (γ)

**How much to value future rewards:**

**Typical Value:** γ = 0.99 or 0.999

**Why Discount?**
- Prevents infinite reward accumulation
- Encourages faster task completion
- Mathematically necessary for convergence

**For Ballbot:**
- Balance is immediate (survival bonus)
- Navigation is long-term (directional reward)
- Discount balances immediate vs. long-term goals

---

## Policy Architecture

### What is a Policy?

A **policy** π(a|s) is a function that maps states to actions (or action distributions).

**For Ballbot:**
- Input: Multi-modal observations (proprioception + vision)
- Output: Three wheel torques
- Architecture: Neural network (MLP with feature extractors)

### Policy Components

**1. Feature Extractors:**
- **Proprioceptive**: Flatten and concatenate
- **Visual**: CNN encoder (pretrained or trainable)
- **Fusion**: Concatenate features

**2. Policy Network:**
- Multi-layer perceptron (MLP)
- Outputs action distribution (mean + std for continuous actions)

**Why This Architecture?**
- Handles multi-modal inputs
- Learns to fuse proprioception and vision
- Generalizes across terrains

**Code Location:**
- `ballbot_rl/policies/mlp_policy.py` - Policy architecture
- Uses Stable-Baselines3's `BaseFeaturesExtractor`

---

## Training Process

### Algorithm: PPO (Proximal Policy Optimization)

**Why PPO?**
- Stable training
- Good sample efficiency
- Works well with continuous control
- Less sensitive to hyperparameters

**Training Loop:**
1. Collect rollouts (state-action-reward sequences)
2. Compute advantages using GAE
3. Update policy to maximize advantage
4. Update value function to predict returns
5. Repeat

**Key Hyperparameters:**
- Learning rate
- Batch size
- Number of environments (parallel)
- Entropy coefficient (exploration)

**Code Location:**
- `ballbot_rl/training/train.py` - Training script
- Uses Stable-Baselines3 PPO implementation

---

## Key Insights

### 1. State Space Design

**Principle:** Include all information needed for control

**For Ballbot:**
- Proprioception: Balance control
- Vision: Terrain navigation
- Both: Robust performance

**Connection to Mechanics:**
- State variables from Lagrangian dynamics (q, q̇)
- Plus visual information for terrain awareness

### 2. Action Space Design

**Principle:** Match physical actuators

**For Ballbot:**
- Three wheel torques
- Continuous actions
- Normalized for neural network

**Connection to Mechanics:**
- Control inputs from dynamics equations
- Agent learns to generate appropriate torques

### 3. Reward Design

**Principle:** Encode objectives and constraints

**For Ballbot:**
- Objectives: Navigate toward goal, maintain balance
- Constraints: Stay upright, don't use excessive control

**Connection to Mechanics:**
- Constraints become penalties (Carius 2022)
- Objectives become positive rewards
- Balance trade-offs via weights

### 4. Policy Architecture

**Principle:** Handle multi-modal inputs

**For Ballbot:**
- Separate extractors for each modality
- Fusion layer combines features
- Policy network outputs actions

**Connection to Mechanics:**
- Proprioception extracts state variables
- Vision extracts terrain features
- Policy learns to combine them

---

## Summary

**MDP Formulation:**
- **State**: Proprioception + vision
- **Action**: Three wheel torques
- **Reward**: Directional + action penalty + survival
- **Policy**: Neural network with multi-modal fusion

**Connection to Mechanics:**
- States come from Lagrangian dynamics
- Actions are control inputs
- Rewards encode constraints and objectives
- Policy learns to exploit dynamics

**Why RL Works:**
- Learns from experience (no explicit equations needed)
- Handles nonlinearity and uncertainty
- Generalizes to new scenarios
- Optimizes multiple objectives simultaneously

---

## Next Steps

- Read [Reward Design](reward_design.md) for detailed reward engineering
- Read [Observation Design](observation_design.md) for multi-modal fusion
- Read [Mechanics to RL Guide](../research/mechanics_to_rl.md) for mathematical details
- Explore [Code Walkthrough](../api/code_walkthrough.md) to see MDP in code

---

*Last Updated: 2025*

