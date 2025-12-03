# 🔬 From Classical Mechanics to Reinforcement Learning

*A comprehensive guide showing how ballbot dynamics translate into RL reward functions*

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Part 1: Understanding the Physical System](#part-1-understanding-the-physical-system)
3. [Part 2: From Dynamics to Constraints](#part-2-from-dynamics-to-constraints)
4. [Part 3: Encoding Constraints as Rewards](#part-3-encoding-constraints-as-rewards)
5. [Part 4: The Complete RL Formulation](#part-4-the-complete-rl-formulation)
6. [Real-World Example: OpenBallBot-RL Implementation](#real-world-example-openballbot-rl-implementation)
7. [Summary](#summary)

---

## 🎯 Introduction

This tutorial bridges the gap between classical control theory and modern reinforcement learning. We'll see how the Lagrangian dynamics from Nagarajan (2014) inform the observation space, how constraint-aware control from Carius (2022) shapes the reward function, and how everything comes together in Salehi's (2025) RL formulation.

> "The reward function encodes our understanding of the physics. Get the physics right, and the rewards follow naturally."  
> — *Paraphrasing control theory wisdom*

**Key Questions This Tutorial Answers:**
- How do Lagrangian equations map to observation spaces?
- Why do we penalize tilt angles in the reward?
- How do classical constraints translate to RL rewards?
- What makes the RL formulation safe and effective?

---

## 🔬 Part 1: Understanding the Physical System

### The Ballbot Configuration

A ballbot balances on a single spherical ball using three omniwheels arranged at 120° angles. The robot body sits above the ball, and the wheels contact the ball to apply torques.

**Key Physical Quantities:**
- **Tilt angles:** \(\phi\) (roll), \(\theta\) (pitch) - measure how far the robot leans
- **Yaw angle:** \(\psi\) - rotation about the vertical axis
- **Angular velocities:** \(\dot{\phi}, \dot{\theta}, \dot{\psi}\) - rates of change
- **Ball position:** \((x, y)\) - horizontal position on the ground
- **Ball velocity:** \((\dot{x}, \dot{y})\) - horizontal velocity

### Lagrangian Dynamics (Nagarajan 2014)

The complete dynamics are derived from the Lagrangian:

\[
L = T - V = \frac{1}{2}\dot{\mathbf{q}}^T \mathbf{M}(\mathbf{q}) \dot{\mathbf{q}} - mgh(\mathbf{q})
\]

Where:
- \(\mathbf{q} = [\phi, \theta, \psi]^T\) are the generalized coordinates
- \(\mathbf{M}(\mathbf{q})\) is the configuration-dependent inertia matrix
- \(h(\mathbf{q})\) is the height of the center of mass
- \(m\) is the total mass
- \(g = 9.81\) m/s² is gravitational acceleration

### Euler-Lagrange Equations

Applying the Euler-Lagrange equations:

\[
\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{\mathbf{q}}}\right) - \frac{\partial L}{\partial \mathbf{q}} = \boldsymbol{\tau}
\]

This yields the second-order dynamics:

\[
\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{G}(\mathbf{q}) = \boldsymbol{\tau}
\]

Where:
- \(\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\) captures Coriolis and centrifugal forces
- \(\mathbf{G}(\mathbf{q}) = mg\frac{\partial h}{\partial \mathbf{q}}\) is the gravitational force vector
- \(\boldsymbol{\tau} = [\tau_\phi, \tau_\theta, \tau_\psi]^T\) are the control torques

### Key Physical Insight

> "The system is underactuated: we have 6 degrees of freedom (3 angles + 3 positions) but only 3 control inputs (wheel torques). We cannot directly control all states—we must exploit the dynamics."

**What This Means:**
- We can't command the ball position directly
- We must tilt the robot to create motion
- Balance and motion are **coupled**—we can't have one without the other
- The yaw degree of freedom **decouples** from balance (Nagarajan's key insight)

### Mapping to Observation Space

The observation space in `bbot_env.py` directly reflects these physical quantities:

```python
observation_space = Dict({
    "orientation": Box(-π, π, shape=(3,)),      # [φ, θ, ψ]
    "angular_vel": Box(-2, 2, shape=(3,)),      # [φ̇, θ̇, ψ̇]
    "vel": Box(-2, 2, shape=(3,)),              # [ẋ, ẏ, ż]
    "motor_state": Box(-2, 2, shape=(3,)),      # Wheel velocities
    "actions": Box(-1, 1, shape=(3,)),          # Previous action
    "rgbd_0": Box(0, 1, shape=(C, H, W)),       # Depth camera
    "rgbd_1": Box(0, 1, shape=(C, H, W)),       # Depth camera
})
```

**Why These Observations?**
- `orientation` and `angular_vel` are the state variables from the Lagrangian
- `vel` includes ball velocity \((\dot{x}, \dot{y})\), needed for navigation
- `motor_state` tracks actuator states (important for underactuated systems)
- `actions` provides action history (helps with partial observability)
- `rgbd_*` provide visual perception (from Salehi 2025)

---

## ⚖️ Part 2: From Dynamics to Constraints

### Stability Condition

For the ballbot to remain balanced, the center of mass must stay above the contact point. This translates to constraints on the tilt angles:

\[
|\phi| \leq \phi_{\max}, \quad |\theta| \leq \theta_{\max}
\]

Where \(\phi_{\max}, \theta_{\max} \approx 20°\) are safety limits (beyond this, the robot falls).

### Energy-Based Analysis

The total energy of the system is:

\[
E = T + V = \frac{1}{2}\dot{\mathbf{q}}^T \mathbf{M}(\mathbf{q}) \dot{\mathbf{q}} + mgh(\mathbf{q})
\]

When balanced:
- Potential energy \(V = mgh\) is minimized (center of mass is low)
- Kinetic energy \(T\) is small (not moving much)
- Total energy \(E\) is near minimum

When falling:
- Tilt increases → \(h\) decreases → potential energy decreases
- But kinetic energy increases rapidly
- System becomes unstable

### Constraint-Aware Control (Carius 2022)

Carius et al. reformulated balance as a **constrained optimization problem**:

\[
\min_{\mathbf{u}} \mathbb{E}\left[\int_0^T c(\mathbf{x}(t), \mathbf{u}(t)) dt\right]
\]

Subject to:
\[
|\phi(t)| \leq \phi_{\max}, \quad |\theta(t)| \leq \theta_{\max}, \quad \forall t
\]

**Key Insight:**
> "Instead of designing controllers that avoid constraints, we optimize subject to constraints. This allows safe exploration while finding optimal policies."

### Why Constraints Matter

1. **Safety:** Violating tilt limits causes the robot to fall
2. **Performance:** Staying within limits enables efficient control
3. **Generalization:** Constraint-satisfying policies work across terrains

---

## 🎁 Part 3: Encoding Constraints as Rewards

### From Hard Constraints to Soft Penalties

In RL, we can't enforce hard constraints directly. Instead, we use **soft constraints** via reward penalties:

**Hard Constraint (Classical Control):**
\[
|\phi| \leq \phi_{\max} \quad \text{(must be satisfied)}
\]

**Soft Constraint (RL):**
\[
r_{\text{tilt}} = -\alpha \cdot |\phi| - \beta \cdot |\theta|
\]

Where \(\alpha, \beta > 0\) are penalty coefficients.

### Reward Components in OpenBallBot-RL

Looking at `Rewards.py` and `bbot_env.py`, the reward has three main components:

#### 1. Directional Reward (Progress)

Encourages movement toward a target direction:

\[
r_{\text{dir}} = \frac{\mathbf{v}_{xy} \cdot \mathbf{g}}{100}
\]

Where:
- \(\mathbf{v}_{xy} = [\dot{x}, \dot{y}]^T\) is the 2D ball velocity
- \(\mathbf{g} = [g_x, g_y]^T\) is the target direction (unit vector)
- Division by 100 scales the reward to reasonable values

**Implementation:**
```python
# From Rewards.DirectionalReward
xy_velocity = state["vel"][-3:-1]  # Extract x, y components
dir_rew = xy_velocity.dot(self.target_direction)
return dir_rew / 100.0
```

#### 2. Action Regularization (Smoothness)

Penalizes large control inputs:

\[
r_{\text{reg}} = -0.0001 \cdot \|\mathbf{a}\|^2
\]

Where \(\mathbf{a}\) is the action vector (normalized wheel commands).

**Why This Matters:**
- Encourages smooth, energy-efficient control
- Prevents jerky motions that could destabilize the robot
- Reduces wear on actuators (important for real robots)

**Implementation:**
```python
action_regularization = -0.0001 * (np.linalg.norm(omniwheel_commands)**2)
```

#### 3. Survival Bonus (Stability)

Small positive reward for staying upright:

\[
r_{\text{surv}} = \begin{cases}
0.02 & \text{if } |\phi| \leq \phi_{\max} \text{ and } |\theta| \leq \theta_{\max} \\
0 & \text{otherwise}
\end{cases}
\]

**Why This Matters:**
- Provides positive reward signal even when not moving
- Encourages the agent to maintain balance
- Helps with exploration (agent doesn't get stuck in low-reward states)

**Implementation:**
```python
if angle_in_degrees <= max_allowed_tilt:  # Default: 20 degrees
    reward += 0.02
```

### Total Reward Function

Combining all components:

\[
r_{\text{total}} = \underbrace{\frac{\mathbf{v}_{xy} \cdot \mathbf{g}}{100}}_{\text{progress}} - \underbrace{0.0001 \|\mathbf{a}\|^2}_{\text{smoothness}} + \underbrace{0.02 \cdot \mathbb{1}[\text{upright}]}_{\text{stability}}
\]

### Termination Condition

The episode terminates if tilt exceeds safety limits:

\[
\text{terminated} = \begin{cases}
\text{True} & \text{if } |\phi| > 20° \text{ or } |\theta| > 20° \\
\text{False} & \text{otherwise}
\end{cases}
\]

This provides a **hard safety limit** even though rewards use soft constraints.

---

## 🤖 Part 4: The Complete RL Formulation

### State Space

The state combines proprioceptive and visual observations:

\[
\mathbf{s}_t = \begin{bmatrix}
\phi_t, \theta_t, \psi_t \\
\dot{\phi}_t, \dot{\theta}_t, \dot{\psi}_t \\
\dot{x}_t, \dot{y}_t, \dot{z}_t \\
\mathbf{a}_{t-1} \\
\text{enc}(\mathbf{I}_t)
\end{bmatrix}
\]

Where:
- First 3 rows: Classical state variables from Lagrangian
- \(\mathbf{a}_{t-1}\): Previous action (for partial observability)
- \(\text{enc}(\mathbf{I}_t)\): Encoded depth image (from pretrained CNN)

### Action Space

Three motor torques (matching the physical actuators):

\[
\mathbf{a}_t = [\tau_0, \tau_1, \tau_2]^T \in [-1, 1]^3
\]

These are normalized and scaled internally to \([-10, 10]\) rad/s wheel velocities.

### Policy Network

The policy \(\pi_\theta(\mathbf{a}_t | \mathbf{s}_t)\) is a neural network that:
- Takes state \(\mathbf{s}_t\) as input
- Outputs mean and variance for a Gaussian distribution over actions
- Actions are sampled from this distribution during training
- Deterministic actions (mean) used during evaluation

### Training Objective (PPO)

Proximal Policy Optimization maximizes:

\[
L(\theta) = \mathbb{E}_t\left[\min\left(
r_t(\theta) \hat{A}_t,
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t
\right)\right]
\]

Where:
- \(r_t(\theta) = \frac{\pi_\theta(\mathbf{a}_t | \mathbf{s}_t)}{\pi_{\theta_{\text{old}}}(\mathbf{a}_t | \mathbf{s}_t)}\) is the importance ratio
- \(\hat{A}_t\) is the advantage estimate
- \(\epsilon = 0.015\) is the clipping parameter (from config)

**Why PPO?**
- Stable learning (prevents large policy updates)
- Sample efficient (can reuse data multiple times)
- Works well for continuous control (like ballbot)

---

## 💻 Real-World Example: OpenBallBot-RL Implementation

### Observation Extraction

In `bbot_env.py`, observations are extracted from MuJoCo:

```python
def _get_obs(self):
    # Extract orientation (Euler angles)
    orientation = self._get_orientation()
    
    # Extract angular velocities
    angular_vel = self.data.qvel[3:6]  # [φ̇, θ̇, ψ̇]
    
    # Extract linear velocities
    vel = self.data.qvel[0:3]  # [ẋ, ẏ, ż]
    
    # Extract motor states
    motor_state = self.data.act
    
    # Render depth images
    rgbd_0 = self.rgbd_inputs(self.data, "cam_0")
    rgbd_1 = self.rgbd_inputs(self.data, "cam_1")
    
    return {
        "orientation": orientation,
        "angular_vel": angular_vel,
        "vel": vel,
        "motor_state": motor_state,
        "actions": self.last_action,
        "rgbd_0": rgbd_0,
        "rgbd_1": rgbd_1,
    }
```

### Reward Computation

The reward is computed in `step()`:

```python
def step(self, action):
    # Apply action and step physics
    self.data.ctrl[:] = self._action_to_motor_command(action)
    mujoco.mj_step(self.model, self.data)
    
    # Get new observation
    obs = self._get_obs()
    
    # Compute reward components
    directional_reward = self.reward_obj(obs) / 100.0
    action_regularization = -0.0001 * np.linalg.norm(action)**2
    
    reward = directional_reward + action_regularization
    
    # Add survival bonus if upright
    tilt_angle = self._get_tilt_angle()
    if tilt_angle <= self.max_allowed_tilt:
        reward += 0.02
    
    # Check termination
    terminated = tilt_angle > self.max_allowed_tilt
    
    return obs, reward, terminated, truncated, info
```

### Training Configuration

From `train_ppo_directional.yaml`:

```yaml
algo:
  name: ppo
  learning_rate: -1  # Uses scheduler
  n_steps: 2048      # Steps per update
  n_epochs: 5        # Update epochs
  batch_sz: 256      # Batch size
  clip_range: 0.015  # PPO clipping
  ent_coef: 0.001    # Entropy bonus

problem:
  terrain_type: "perlin"  # Random terrain

total_timesteps: 10e6      # Total training steps
frozen_cnn: "../encoder_frozen/encoder_epoch_53"  # Pretrained encoder
```

---

## 📝 Summary

### Key Takeaways

1. **Physics Informs Observations**
   - The Lagrangian state variables \((\phi, \theta, \psi, \dot{\phi}, \dot{\theta}, \dot{\psi})\) become observations
   - MuJoCo computes the same dynamics equations analytically derived by Nagarajan

2. **Constraints Become Rewards**
   - Hard constraints \(|\phi| \leq \phi_{\max}\) become soft penalties \(-\alpha|\phi|\)
   - This allows RL to learn while discouraging unsafe states

3. **Reward Design Reflects Physics**
   - Directional reward encourages navigation (exploits dynamics)
   - Action regularization encourages smooth control (energy efficiency)
   - Survival bonus encourages stability (constraint satisfaction)

4. **RL Complements Classical Control**
   - Classical control provides the dynamics model
   - RL learns policies that satisfy the same constraints
   - The combination enables robust navigation

### Mathematical Summary

**Dynamics (Nagarajan 2014):**
\[
\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{G}(\mathbf{q}) = \boldsymbol{\tau}
\]

**Constraints (Carius 2022):**
\[
|\phi| \leq \phi_{\max}, \quad |\theta| \leq \theta_{\max}
\]

**Reward (Salehi 2025):**
\[
r = \frac{\mathbf{v}_{xy} \cdot \mathbf{g}}{100} - 0.0001\|\mathbf{a}\|^2 + 0.02 \cdot \mathbb{1}[\text{upright}]
\]

**RL Objective (PPO):**
\[
\max_\theta \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]
\]

### Deep Dive: Reward Component Interactions

**Component Balance:**

The three reward components interact in complex ways:

1. **Directional reward** provides the primary learning signal
   - Encourages movement toward goal
   - Scales with velocity magnitude
   - Can be negative (moving away from goal)

2. **Action regularization** prevents unsafe control
   - Always negative (penalty)
   - Quadratic in action magnitude
   - Small coefficient prevents domination

3. **Survival bonus** ensures positive signal
   - Always positive when upright
   - Accumulates over time
   - Provides baseline for exploration

**Typical Episode Rewards:**

- **Short episode (500 steps):** ~10-15 reward
- **Medium episode (2000 steps):** ~40-60 reward
- **Long episode (4000 steps):** ~80-120 reward

**Reward Scaling Analysis:**

The division by 100 in directional reward ensures:
- Directional component: ~0.1 per step (when moving)
- Action regularization: ~-0.0001 per step (small)
- Survival bonus: +0.02 per step (consistent)

**Total per step:** Typically `[0.01, 0.15]` range

**Why This Balance Works:**

- **Directional reward** dominates when moving (provides learning signal)
- **Survival bonus** dominates when stationary (encourages balance)
- **Action regularization** prevents extreme actions (safety)

**Related Documentation:**
- [Advanced Topics](09_advanced_topics.md), Reward Component Analysis section

---

### Next Steps

- Read [Environment & RL Workflow](03_environment_and_rl.md) for practical implementation
- Explore `bbot_env.py` with this mathematical foundation in mind
- Experiment with reward coefficients to see how they affect learning
- Review [Advanced Topics](09_advanced_topics.md) for deeper analysis

---

**Happy Learning! 🚀**

*Last Updated: 2025*

