# Research Papers → Code Implementation Mapping

*How each research paper's contributions map to specific code files and design decisions*

---

## Overview

This document maps each foundational research paper to its implementation in the openballbot-rl codebase. Understanding these mappings helps you see how theoretical insights translate into practical code.

---

## Mapping Structure

For each paper, we provide:
1. **Key Contributions** - What the paper introduced
2. **Code Files** - Where it's implemented
3. **Design Decisions** - How it influenced the architecture
4. **Mathematical Connections** - Links to formulas and equations

---

## Lauwers 2006 → Physical Design

### Paper: "A dynamically stable single-wheeled mobile robot with inverse mouse-ball drive"

**Key Contributions:**
- Three-omniwheel configuration at 120° angles
- Inverse mouse-ball drive mechanism
- Dynamic stability through active control
- Underactuated system concept

### Code Implementation

**Physical Model:**
- **File**: `ballbot_gym/models/ballbot.xml`
- **What**: MuJoCo XML model defining robot structure
- **Details**:
  - Three omniwheels at 120° spacing
  - Ball as single contact point
  - Robot body on top of ball
  - Matches Lauwers' prototype design exactly

**Action Space:**
- **File**: `ballbot_gym/envs/ballbot_env.py`
- **What**: Three continuous actions (wheel torques)
- **Details**:
  - Action space: `Box(-1, 1, shape=(3,))`
  - Maps to three wheel motors
  - Normalized to [-1, 1], scaled to [-10, 10] rad/s internally

**Anisotropic Friction:**
- **File**: `tools/mujoco_fix.patch`
- **What**: MuJoCo patch for omniwheel simulation
- **Details**:
  - Enables directional friction needed for omniwheels
  - Without patch, omniwheels don't work correctly
  - Matches physical behavior described in paper

### Design Decisions

**Why Three Omniwheels?**
- Lauwers showed this is the minimum for omnidirectional control
- 120° spacing provides optimal force distribution
- Enables control in any horizontal direction

**Why Underactuation?**
- 6 DOF but only 3 inputs
- Creates the challenge that RL solves
- Forces agent to exploit dynamics

**Code References:**
- `ballbot_gym/models/ballbot.xml` - Lines defining omniwheel geometry
- `ballbot_gym/envs/ballbot_env.py` - Action space and motor control

---

## Nagarajan 2014 → Dynamics & Observation Space

### Paper: "The ballbot: An omnidirectional balancing mobile robot"

**Key Contributions:**
- Complete Lagrangian dynamics derivation
- Control architecture (balance + trajectory + yaw)
- Yaw decoupling from balance
- State space formulation

### Code Implementation

**Observation Space:**
- **File**: `ballbot_gym/envs/observation_spaces.py`
- **File**: `ballbot_gym/envs/ballbot_env.py` (observation extraction)
- **What**: Proprioceptive observations match Nagarajan's state variables
- **Details**:
  - Orientation: (φ, θ, ψ) - Roll, pitch, yaw angles
  - Angular velocities: (φ̇, θ̇, ψ̇)
  - Linear velocities: (ẋ, ẏ, ż)
  - These are exactly the state variables (q, q̇) from Lagrangian dynamics

**Dynamics:**
- **File**: `ballbot_gym/models/ballbot.xml`
- **What**: MuJoCo internally solves Lagrangian equations
- **Details**:
  - MuJoCo computes inertia matrices M(q)
  - Solves Euler-Lagrange equations
  - Handles Coriolis forces C(q, q̇)
  - Computes gravitational forces G(q)

**Yaw Decoupling:**
- **File**: `ballbot_gym/envs/ballbot_env.py`
- **What**: Yaw control independent of balance
- **Details**:
  - Yaw angle (ψ) doesn't affect balance
  - Can rotate while maintaining balance
  - Simplifies control problem

### Mathematical Connections

**Lagrangian Dynamics:**
```
L = T - V = ½q̇ᵀM(q)q̇ - mgh(q)
```

**Euler-Lagrange:**
```
M(q)q̈ + C(q, q̇)q̇ + G(q) = τ
```

**Code Connection:**
- MuJoCo solves these equations numerically
- Observation space extracts q and q̇
- Action space provides τ

**Code References:**
- `ballbot_gym/envs/ballbot_env.py` - `_get_obs()` extracts state variables
- `ballbot_gym/models/ballbot.xml` - Defines dynamics parameters

---

## Carius 2022 → Reward Design

### Paper: "Constrained path integral control for motion planning"

**Key Contributions:**
- Constraint-aware control theory
- Encoding constraints in control objective
- Soft vs. hard constraints
- Path integral formulation

### Code Implementation

**Reward Function:**
- **File**: `ballbot_gym/rewards/directional.py`
- **What**: Reward function encodes constraints as penalties
- **Details**:
  - Survival bonus: Encourages staying upright (constraint satisfaction)
  - Action penalty: Prevents excessive control effort
  - Directional reward: Encourages goal-directed motion

**Constraint Handling:**
- **File**: `ballbot_gym/envs/ballbot_env.py`
- **What**: Termination conditions enforce hard constraints
- **Details**:
  - Large tilt → termination (hard constraint)
  - Reward penalties for approaching limits (soft constraint)
  - Matches Carius' constraint-aware approach

**Reward Components:**
```python
r = α₁(v_xy · g_target) - α₂||a||² + α₃·𝟙[upright]
```

**Connection to Carius:**
- Hard constraints → termination conditions
- Soft constraints → reward penalties
- Control objective → reward function

### Design Decisions

**Why Soft Constraints?**
- Allows exploration
- Agent can learn from mistakes
- More robust than hard constraints alone

**Why Reward Penalties?**
- Encodes constraints without preventing learning
- Balances multiple objectives
- Enables multi-objective optimization

**Code References:**
- `ballbot_gym/rewards/directional.py` - Reward implementation
- `ballbot_gym/envs/ballbot_env.py` - Termination conditions

---

## Salehi 2025 → Complete RL System

### Paper: "Reinforcement Learning for Ballbot Navigation in Uneven Terrain"

**Key Contributions:**
- RL formulation for ballbot navigation
- Multi-modal observations (proprioception + vision)
- Depth encoder pretraining
- PPO training pipeline
- Generalization to uneven terrain

### Code Implementation

**Training Pipeline:**
- **File**: `ballbot_rl/training/train.py`
- **What**: Complete PPO training workflow
- **Details**:
  - Environment creation
  - Policy initialization
  - Training loop
  - Evaluation
  - Checkpointing

**Multi-Modal Policy:**
- **File**: `ballbot_rl/policies/mlp_policy.py`
- **What**: Feature extractor combining proprioception and vision
- **Details**:
  - Proprioceptive extractor (flatten)
  - Visual extractor (CNN encoder)
  - Fusion (concatenation + MLP)

**Depth Encoder:**
- **File**: `ballbot_rl/encoders/models.py`
- **File**: `ballbot_rl/encoders/pretrain.py`
- **What**: CNN encoder for depth images
- **Details**:
  - Pretrained on diverse terrain data
  - Reduces 128×128 depth → 20 dimensions
  - Frozen during RL training

**Reward Function:**
- **File**: `ballbot_gym/rewards/directional.py`
- **What**: Directional reward + action penalty + survival bonus
- **Details**:
  - Matches paper's reward formulation
  - Enables goal-directed navigation
  - Prevents reward hacking

**Terrain Generation:**
- **File**: `ballbot_gym/terrain/perlin.py`
- **What**: Perlin noise terrain generation
- **Details**:
  - Procedural generation
  - Infinite terrain variety
  - Enables generalization

### Design Decisions

**Why Multi-Modal?**
- Proprioception: Balance control
- Vision: Terrain navigation
- Both: Robust performance

**Why Depth Encoder?**
- Reduces dimensionality
- Extracts relevant features
- Enables efficient learning

**Why PPO?**
- Stable training
- Good sample efficiency
- Works well with continuous control

**Code References:**
- `ballbot_rl/training/train.py` - Training pipeline
- `ballbot_rl/policies/mlp_policy.py` - Policy architecture
- `ballbot_rl/encoders/models.py` - Depth encoder
- `ballbot_gym/rewards/directional.py` - Reward function

---

## Complete Mapping Table

| Paper | Contribution | Code File | Key Implementation |
|-------|-------------|-----------|-------------------|
| **Lauwers 2006** | Physical design | `ballbot_gym/models/ballbot.xml` | Three omniwheels, ball contact |
| **Lauwers 2006** | Action space | `ballbot_gym/envs/ballbot_env.py` | Three wheel torques |
| **Lauwers 2006** | Anisotropic friction | `tools/mujoco_fix.patch` | Omniwheel simulation |
| **Nagarajan 2014** | Observation space | `ballbot_gym/envs/observation_spaces.py` | State variables (q, q̇) |
| **Nagarajan 2014** | Dynamics | `ballbot_gym/models/ballbot.xml` | MuJoCo solves Lagrangian eqs |
| **Nagarajan 2014** | Yaw decoupling | `ballbot_gym/envs/ballbot_env.py` | Independent yaw control |
| **Carius 2022** | Reward design | `ballbot_gym/rewards/directional.py` | Constraints → penalties |
| **Carius 2022** | Constraint handling | `ballbot_gym/envs/ballbot_env.py` | Termination conditions |
| **Salehi 2025** | Training pipeline | `ballbot_rl/training/train.py` | PPO training |
| **Salehi 2025** | Multi-modal policy | `ballbot_rl/policies/mlp_policy.py` | Feature fusion |
| **Salehi 2025** | Depth encoder | `ballbot_rl/encoders/models.py` | CNN encoder |
| **Salehi 2025** | Terrain generation | `ballbot_gym/terrain/perlin.py` | Perlin noise |

---

## Mathematical Formulations

### Lauwers 2006: Inverse Mouse-Ball Drive

**Physical Principle:**
- Three omniwheels contact ball at 120° angles
- Each wheel can apply torque
- Combined torques create omnidirectional motion

**Code:**
- `ballbot_gym/models/ballbot.xml` - Omniwheel geometry
- `ballbot_gym/envs/ballbot_env.py` - Torque application

---

### Nagarajan 2014: Lagrangian Dynamics

**Equations:**
```
L = T - V = ½q̇ᵀM(q)q̇ - mgh(q)

M(q)q̈ + C(q, q̇)q̇ + G(q) = τ
```

**Code:**
- MuJoCo solves these equations
- `ballbot_gym/envs/ballbot_env.py` extracts q and q̇
- `ballbot_gym/envs/ballbot_env.py` applies τ

---

### Carius 2022: Constraint-Aware Control

**Control Objective:**
```
J = ∫[L(x, u) + λ·c(x, u)] dt
```

**RL Translation:**
```
r = objective_reward - penalty·constraint_violation
```

**Code:**
- `ballbot_gym/rewards/directional.py` - Reward function
- `ballbot_gym/envs/ballbot_env.py` - Constraint checking

---

### Salehi 2025: RL Formulation

**MDP:**
- State: Proprioception + vision
- Action: Three wheel torques
- Reward: Directional + penalty + survival

**Code:**
- `ballbot_gym/envs/observation_spaces.py` - State space
- `ballbot_gym/envs/ballbot_env.py` - Action space
- `ballbot_gym/rewards/directional.py` - Reward function

---

## Design Pattern Connections

### 1. Underactuation → RL Challenge

**Paper Insight:** 6 DOF, 3 inputs → cannot directly control all states

**Code Impact:**
- Agent must learn to exploit dynamics
- Cannot use simple position control
- RL learns complex control strategies

**Files:**
- `ballbot_gym/envs/ballbot_env.py` - Underactuated action space
- `ballbot_rl/policies/mlp_policy.py` - Learns complex policies

---

### 2. Dynamic Stability → Reward Design

**Paper Insight:** Must actively maintain balance

**Code Impact:**
- Survival bonus encourages balance
- Termination on large tilts
- Reward shaped for stability

**Files:**
- `ballbot_gym/rewards/directional.py` - Survival bonus
- `ballbot_gym/envs/ballbot_env.py` - Termination conditions

---

### 3. Coupled Dynamics → Multi-Modal Observations

**Paper Insight:** Balance and motion are coupled

**Code Impact:**
- Need both proprioception (balance) and vision (motion)
- Multi-modal fusion required
- Separate extractors for each modality

**Files:**
- `ballbot_gym/envs/observation_spaces.py` - Multi-modal space
- `ballbot_rl/policies/mlp_policy.py` - Fusion architecture

---

## Key Insights

### 1. Papers Provide Foundation

**Lauwers:** Physical design → code structure
**Nagarajan:** Dynamics → observation space
**Carius:** Constraints → reward design
**Salehi:** RL → complete system

### 2. Theory Informs Implementation

**Understanding papers helps:**
- Design observation spaces
- Design reward functions
- Understand why choices were made
- Debug issues

### 3. Code Implements Theory

**Each paper's insights appear in code:**
- Physical design → XML model
- Dynamics → observation extraction
- Constraints → reward penalties
- RL → training pipeline

---

## Using This Mapping

### For Understanding

1. **Read paper** → Find contribution
2. **Check mapping** → Find code file
3. **Read code** → See implementation
4. **Understand connection** → See how theory → practice

### For Modification

1. **Want to change X?** → Check which paper introduced it
2. **Understand theory** → Read paper section
3. **Modify code** → Update implementation
4. **Validate** → Ensure theory still holds

### For Debugging

1. **Issue in component X?** → Check paper mapping
2. **Understand design intent** → Read paper
3. **Debug accordingly** → Fix implementation
4. **Verify** → Ensure matches theory

---

## Summary

**Mapping Summary:**

- **Lauwers 2006** → Physical model, action space, friction
- **Nagarajan 2014** → Observation space, dynamics, yaw decoupling
- **Carius 2022** → Reward design, constraint handling
- **Salehi 2025** → Training pipeline, multi-modal policy, encoder

**Key Takeaway:**
Understanding these mappings helps you see how theoretical insights translate into practical code, making it easier to understand, modify, and extend the system.

---

## Next Steps

- Read [Research Timeline](timeline.md) for detailed paper summaries
- Read [Mechanics to RL Guide](mechanics_to_rl.md) for mathematical details
- Explore [Code Walkthrough](../api/code_walkthrough.md) to see implementations
- Read [Architecture Overview](../architecture/README.md) for system design

---

*Last Updated: 2025*

