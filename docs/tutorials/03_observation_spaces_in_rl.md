# 👁️ Observation Spaces in Reinforcement Learning

*A comprehensive guide to designing observation spaces for robotics applications*

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [What is an Observation Space?](#what-is-an-observation-space)
3. [Types of Observations](#types-of-observations)
4. [Observation Space Types in Gymnasium](#observation-space-types-in-gymnasium)
5. [Normalization Strategies](#normalization-strategies)
6. [Real-World Example: Ballbot Observations](#real-world-example-ballbot-observations)
7. [Partial Observability](#partial-observability)
8. [Multi-Modal Observations](#multi-modal-observations)
9. [Design Principles](#design-principles)
10. [Common Pitfalls](#common-pitfalls)
11. [Summary](#summary)

---

## 🎯 Introduction

The observation space defines **what the agent sees**—the format and content of information available to the agent at each timestep. Unlike the true state of the world (which is often unknown), observations are the **limited measurements** the agent can actually perceive.

> "The observation space is the agent's window into the world. Design it carefully, for it shapes everything the agent can learn."  
> — *Sergey Levine, UC Berkeley*

In robotics, observation design is particularly critical because:
- Real robots have **limited sensors** (unlike simulators that can access ground truth)
- Observations must be **realistic** (only include what real sensors provide)
- **Normalization** is essential for neural network training
- **Multi-modal** observations (proprioception + vision) are common

**Key Questions This Tutorial Answers:**
- What should go into an observation?
- How do we normalize different sensor types?
- What's the difference between state and observation?
- How do we handle partial observability?
- How do we combine multiple sensor modalities?

---

## 🔍 What is an Observation Space?

### Mathematical Definition

In a Partially Observable Markov Decision Process (POMDP), the observation space **Ω** defines the set of all possible observations:

**Ω** = {**o**₁, **o**₂, ..., **o**ₘ}

At each timestep *t*, the agent receives an observation **o**ₜ ∈ **Ω** that is a (possibly noisy) function of the true state:

**o**ₜ = *O*(**s**ₜ, **a**ₜ₋₁)

Where:
- **s**ₜ is the true (hidden) state
- **a**ₜ₋₁ is the previous action
- *O* is the observation function

### State vs. Observation

**Critical Distinction:**

| **State** | **Observation** |
|-----------|-----------------|
| Full underlying truth | Limited measurement |
| Usually **not available** | What agent **actually sees** |
| Complete world description | Sensor readings only |
| Used in MDPs | Used in POMDPs |

> "In real-world robotics, we almost never have access to the true state. We only have observations—noisy, partial, and delayed."  
> — *Chelsea Finn, Stanford*

### In Gymnasium

```python
import gymnasium as gym
from gymnasium import spaces

# Example: Dict observation space (common in robotics)
self.observation_space = spaces.Dict({
    "proprio": spaces.Box(-np.inf, np.inf, shape=(12,)),
    "camera": spaces.Box(0, 1, shape=(3, 64, 64))
})
```

---

## 📦 Types of Observations

### 1. Proprioceptive Observations (Internal Sensing)

**Proprioception** refers to sensing the robot's own state:

#### Joint Positions
```python
q = self.data.qpos[joint_ids]  # Joint angles (radians)
```

#### Joint Velocities
```python
qdot = self.data.qvel[joint_ids]  # Joint velocities (rad/s)
```

#### IMU Data
```python
# Orientation (quaternion or rotation vector)
orientation = quaternion.as_rotation_vector(self.data.xquat[body_id])

# Angular velocity (gyroscope)
angular_vel = self.data.qvel[gyro_joint_id]

# Linear acceleration (accelerometer)
accel = self.data.qacc[body_id]
```

#### Motor States
```python
motor_velocities = self.data.qvel[motor_joint_ids]
motor_torques = self.data.actuator_force
```

**Why Proprioception Matters:**
- Always available (no external dependencies)
- High frequency (can update every physics step)
- Essential for balance and control

### 2. Exteroceptive Observations (External Sensing)

**Exteroception** refers to sensing the environment:

#### RGB Images
```python
rgb = renderer.render()  # Shape: (H, W, 3), values [0, 255]
rgb_normalized = rgb / 255.0  # Normalize to [0, 1]
```

#### Depth Maps
```python
depth = depth_renderer.render()  # Shape: (H, W), values [0, 1]
depth = depth[..., None]  # Add channel: (H, W, 1)
```

#### LiDAR
```python
lidar_ranges = self.get_lidar_scan()  # Array of distances
```

#### Segmentation Masks
```python
segmentation = get_segmentation_image(model, data)  # Integer IDs
```

**Why Exteroception Matters:**
- Enables navigation and manipulation
- Provides spatial awareness
- Essential for tasks requiring environment understanding

### 3. Task-Related Observations

Additional information specific to the task:

#### Goal Information
```python
goal_position = target_pos - robot_pos  # Relative goal
goal_direction = goal_position / np.linalg.norm(goal_position)
```

#### Previous Actions
```python
previous_action = self.last_action  # Often included for temporal context
```

#### Time/Phase Information
```python
phase = np.sin(2 * np.pi * self.time / period)  # For periodic tasks
```

---

## 🏗️ Observation Space Types in Gymnasium

### 1. Box Space (Vector Observations)

For continuous, vector-valued observations:

```python
spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(n,),
    dtype=np.float32
)
```

**Use Cases:**
- Proprioceptive data (joints, IMU)
- Flattened feature vectors
- Low-dimensional state representations

**Example:**
```python
# Joint positions and velocities
proprio_obs = spaces.Box(
    low=-np.pi,      # Joint angle limits
    high=np.pi,
    shape=(12,),     # 6 joints × 2 (pos + vel)
    dtype=np.float32
)
```

### 2. Dict Space (Multi-Modal Observations) ⭐ **ROBOTICS STANDARD**

For combining different observation types:

```python
spaces.Dict({
    "proprio": spaces.Box(-np.inf, np.inf, shape=(12,)),
    "camera": spaces.Box(0, 1, shape=(3, 64, 64)),
    "goal": spaces.Box(-10, 10, shape=(3,))
})
```

**Why Dict Spaces?**
- Natural mapping to multi-encoder networks
- Clear separation of modalities
- Easy to add/remove sensors
- Matches real robot sensor architecture

**Network Architecture:**
```
┌─────────────┐
│ Proprio     │ → MLP Encoder → ┐
└─────────────┘                 │
┌─────────────┐                 ├→ Concatenate → Policy Head
│ Camera      │ → CNN Encoder → ┘
└─────────────┘
┌─────────────┐
│ Goal        │ → MLP Encoder → ┘
└─────────────┘
```

### 3. Image Space (Vision Observations)

Specialized Box space for images:

```python
spaces.Box(
    low=0.0,
    high=1.0,
    shape=(C, H, W),  # Channels-first format
    dtype=np.float32
)
```

**Format Convention:**
- **Channels-first**: (C, H, W) - PyTorch standard
- **Channels-last**: (H, W, C) - TensorFlow standard

**Gymnasium/Stable-Baselines3**: Prefer channels-first for consistency.

---

## ⚖️ Normalization Strategies

### Why Normalize?

> "Normalization is not optional—it's a fundamental requirement. Without it, neural networks struggle to learn effectively."  
> — *Ilya Sutskever, OpenAI*

**Benefits:**
1. **Stable Gradients**: Prevents gradient explosions/vanishing
2. **Faster Convergence**: Optimizers work better with normalized inputs
3. **Generalization**: Works across different robots/environments
4. **Numerical Stability**: Prevents overflow/underflow

### Normalization by Sensor Type

#### Joint Angles
```python
# Option 1: Scale to [-1, 1] based on limits
q_normalized = (q - q_center) / q_range

# Option 2: Use tanh for periodic angles
q_normalized = np.tanh(q)
```

#### Velocities
```python
# Divide by maximum expected velocity
v_normalized = v / v_max

# Clip to reasonable range
v_normalized = np.clip(v_normalized, -2.0, 2.0)
```

#### Images
```python
# RGB: Divide by 255
rgb_normalized = rgb.astype(np.float32) / 255.0

# Depth: Already in [0, 1] from renderer
depth_normalized = depth  # Usually already normalized
```

#### Quaternions
```python
# Quaternions are already normalized (unit quaternions)
# Just convert to rotation vector if needed
rot_vec = quaternion.as_rotation_vector(q)
```

#### Positions
```python
# Option 1: Relative to origin
pos_normalized = pos / max_distance

# Option 2: Relative to goal
relative_pos = (goal_pos - robot_pos) / max_distance
```

### Standard Normalization Formula

For a sensor reading *x* with range [*x*ₘᵢₙ, *x*ₘₐₓ]:

**x**ₙₒᵣₘ = 2 · (**x** - **x**ₘᵢₙ) / (**x**ₘₐₓ - **x**ₘᵢₙ) - 1

This maps to [-1, 1].

---

## 🤖 Real-World Example: Ballbot Observations

Let's examine the **Ballbot** environment's observation space:

### Observation Space Definition

```python
# From bbot_env.py
self.observation_space = gym.spaces.Dict({
    "orientation": gym.spaces.Box(
        low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32
    ),
    "angular_vel": gym.spaces.Box(
        low=-2, high=2, shape=(3,), dtype=np.float32
    ),
    "vel": gym.spaces.Box(
        low=-2, high=2, shape=(3,), dtype=np.float32
    ),
    "motor_state": gym.spaces.Box(
        -2.0, 2.0, shape=(3,), dtype=np.float32
    ),
    "actions": gym.spaces.Box(
        -1.0, 1.0, shape=(3,), dtype=np.float32
    ),
    "rgbd_0": gym.spaces.Box(
        low=0.0, high=1.0,
        shape=(num_channels, 64, 64),
        dtype=np.float32
    ),
    "rgbd_1": gym.spaces.Box(
        low=0.0, high=1.0,
        shape=(num_channels, 64, 64),
        dtype=np.float32
    ),
    "relative_image_timestamp": gym.spaces.Box(
        low=0.0, high=0.1, shape=(1,), dtype=np.float32
    ),
})
```

### Observation Components Explained

#### 1. Orientation (Rotation Vector)
```python
# Convert quaternion to rotation vector
orientation_quat = quaternion.quaternion(*self.data.xquat[body_id])
rot_vec = quaternion.as_rotation_vector(orientation_quat)
```

**Why Rotation Vector?**
- More stable than Euler angles (no gimbal lock)
- Easier for neural networks than quaternions
- Compact representation (3D vs 4D for quaternions)

#### 2. Angular Velocity
```python
# Compute using matrix logarithm (more accurate than Euler derivatives)
R_1 = quaternion.as_rotation_matrix(self.prev_orientation)
R_2 = quaternion.as_rotation_matrix(orientation_quat)
W = logm(R_1.T @ R_2).real  # Relative rotation matrix
angular_vel = vee(W) / self.opt_timestep  # Extract angular velocity
```

**Mathematical Derivation:**
The angular velocity **ω** is related to the rotation matrix **R** by:

**ω** = (1/Δt) · vee(log(**R**₁ᵀ **R**₂))

Where vee maps skew-symmetric matrices to vectors.

#### 3. Linear Velocity
```python
# Finite difference
vel = (position - self.prev_pos) / self.opt_timestep
vel = np.clip(vel, a_min=-2.0, a_max=2.0)
```

**Why Finite Difference?**
- Direct from position measurements
- No need for velocity sensors
- Works in simulation and can approximate real IMU integration

#### 4. Motor State
```python
motor_state = np.array([
    self.data.qvel[self.model.joint(f"wheel_joint_{i}").id]
    for i in range(3)
])
motor_state /= 10  # Normalize by max velocity (10 rad/s)
motor_state = np.clip(motor_state, -2.0, 2.0)
```

**Why Include Motor State?**
- Provides feedback about actuator response
- Helps learn dynamics
- Enables impedance control

#### 5. Previous Actions
```python
"actions": last_ctrl  # Previous action for temporal context
```

**Why Include Previous Action?**
- Provides temporal context
- Helps with action smoothing
- Common in PPO and other algorithms

#### 6. RGB-D Images
```python
# Render from cameras
rgbd_0 = self.rgbd_inputs(self.data, "cam_0")
rgbd_1 = self.rgbd_inputs(self.data, "cam_1")

# Convert to channels-first
rgbd_0 = rgbd_0.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
```

**Why Two Cameras?**
- Stereo vision for depth estimation
- Wider field of view
- Redundancy

#### 7. Relative Image Timestamp
```python
"relative_image_timestamp": np.array([
    self.data.time - self.prev_im_pair.ts
])
```

**Why Include Timestamp?**
- Cameras update at lower frequency than proprioception
- Agent needs to know how "stale" visual data is
- Important for temporal fusion

### Observation Construction Flow

```
┌─────────────────────────────────────┐
│  MuJoCo Physics State               │
│  - Positions, orientations          │
│  - Velocities, accelerations        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Extract Proprioceptive Data       │
│  - Orientation (quaternion → rot_vec)│
│  - Velocities (finite diff / logm)  │
│  - Motor states                     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Render Visual Observations         │
│  (if camera update needed)         │
│  - RGB-D from cam_0                │
│  - RGB-D from cam_1                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Normalize All Components           │
│  - Clip to bounds                   │
│  - Scale appropriately               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Construct Dict Observation         │
│  {proprio, visual, temporal}        │
└─────────────────────────────────────┘
```

---

## 🔒 Partial Observability

### What is Partial Observability?

In a **Partially Observable MDP (POMDP)**, the agent cannot directly observe the true state. Instead, it receives observations that are a (possibly noisy) function of the state.

**Mathematical Definition:**
**o**ₜ = *O*(**s**ₜ, **a**ₜ₋₁, **n**ₜ)

Where **n**ₜ is observation noise.

### Why Robotics is Partially Observable

1. **Limited Sensors**: Real robots can't see everything
2. **Sensor Noise**: Measurements are imperfect
3. **Occlusion**: Objects block vision
4. **Delayed Information**: Sensors update at different rates
5. **Hidden State**: Internal dynamics not directly measurable

> "Partial observability is not a bug—it's a feature of the real world. Our algorithms must handle it."  
> — *David Silver, DeepMind*

### Handling Partial Observability

#### 1. History Buffers
```python
# Store last N observations
self.obs_history = deque(maxlen=10)
self.obs_history.append(obs)
obs_with_history = np.concatenate(list(self.obs_history))
```

#### 2. Recurrent Networks
```python
# LSTM/GRU to maintain hidden state
class RecurrentPolicy(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_dim)
```

#### 3. Frame Stacking
```python
# Stack multiple frames
obs_stacked = np.stack([obs_t, obs_t-1, obs_t-2], axis=0)
```

#### 4. State Estimation
```python
# Kalman filter or similar
estimated_state = kalman_filter.update(observation)
```

---

## 🎨 Multi-Modal Observations

### Combining Modalities

Multi-modal observations combine different sensor types:

```python
obs = {
    "proprio": proprioceptive_data,  # Vector
    "camera": camera_image,          # Image
    "audio": audio_features,          # Vector (if applicable)
    "goal": goal_info                # Vector
}
```

### Network Architecture

```python
class MultiModalPolicy(nn.Module):
    def __init__(self):
        # Encoders for each modality
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.camera_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Fusion layer
        self.fusion = nn.Linear(64 + 64, 256)
        self.policy_head = nn.Linear(256, action_dim)
    
    def forward(self, obs):
        proprio_features = self.proprio_encoder(obs["proprio"])
        camera_features = self.camera_encoder(obs["camera"])
        
        # Concatenate
        combined = torch.cat([proprio_features, camera_features], dim=1)
        
        # Policy
        action = self.policy_head(self.fusion(combined))
        return action
```

---

## ✅ Design Principles

### 1. Only Include Realistic Sensors

> "If you wouldn't have it on a real robot, don't put it in simulation."  
> — *Sergey Levine, UC Berkeley*

**❌ Bad:**
```python
# Including ground truth that real robot can't measure
obs = {
    "true_position": self.data.qpos,  # Real robot doesn't know this!
    "contact_forces": self.data.cfrc_ext  # No force sensors
}
```

**✅ Good:**
```python
# Only realistic sensors
obs = {
    "joint_angles": self.data.qpos,  # Encoders provide this
    "imu": self.get_imu_reading(),     # IMU provides this
    "camera": self.render_camera()    # Camera provides this
}
```

### 2. Normalize Consistently

```python
# ✅ Good: All in similar ranges
obs = {
    "joints": joints / np.pi,        # [-1, 1]
    "velocities": velocities / 10.0,  # [-1, 1]
    "images": images / 255.0          # [0, 1]
}
```

### 3. Use Dict Spaces for Multi-Modal

```python
# ✅ Good: Clear separation
obs_space = spaces.Dict({
    "proprio": spaces.Box(...),
    "camera": spaces.Box(...)
})
```

### 4. Document Observation Format

```python
"""
Observation Space:
- "orientation": Rotation vector (3,), [-π, π]
- "angular_vel": Angular velocity (3,), [-2, 2] rad/s
- "vel": Linear velocity (3,), [-2, 2] m/s
- "motor_state": Wheel velocities (3,), [-2, 2] normalized
- "actions": Previous action (3,), [-1, 1]
- "rgbd_0": RGB-D image (C, 64, 64), [0, 1]
- "rgbd_1": RGB-D image (C, 64, 64), [0, 1]
- "relative_image_timestamp": Camera lag (1,), [0, 0.1] seconds
"""
```

---

## ⚠️ Common Pitfalls

### 1. Information Leakage

**❌ Including Future Information:**
```python
obs = {
    "next_state": self.get_next_state(),  # Agent shouldn't know future!
}
```

### 2. Inconsistent Normalization

**❌ Mixing Normalization Schemes:**
```python
obs = {
    "joints": joints,              # Raw values (could be large)
    "velocities": velocities / 10  # Normalized (small)
}
```

### 3. Missing Temporal Context

**❌ No History:**
```python
# Agent can't tell if it's moving or stationary
obs = {"position": pos}  # Missing velocity!
```

### 4. Overly Complex Observations

**❌ Too Much Information:**
```python
# Agent gets overwhelmed
obs = {
    "everything": np.concatenate([pos, vel, accel, joints, ...])  # Too much!
}
```

---

## 🧠 Advanced Observation Techniques ⭐⭐

### Learned Representations

**Concept:** Instead of hand-designed features, learn representations from raw observations.

**Why Use Learned Representations?**
- Automatically discovers useful features
- Better generalization
- Reduces manual engineering
- Can learn hierarchical features

**Self-Supervised Learning:**
```python
class ContrastiveObservationEncoder(nn.Module):
    """
    Learn observation representations via contrastive learning.
    """
    def __init__(self, obs_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, obs):
        return F.normalize(self.encoder(obs), dim=1)
    
    def contrastive_loss(self, obs_anchor, obs_positive, obs_negative):
        """
        Learn representations where similar observations are close,
        different observations are far.
        """
        z_anchor = self.forward(obs_anchor)
        z_positive = self.forward(obs_positive)
        z_negative = self.forward(obs_negative)
        
        # Positive pair should be similar
        pos_sim = F.cosine_similarity(z_anchor, z_positive)
        
        # Negative pair should be different
        neg_sim = F.cosine_similarity(z_anchor, z_negative)
        
        # Contrastive loss
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
        return loss
```

**Benefits:**
- Better feature extraction
- More robust to observation noise
- Can pretrain on unlabeled data

### Attention-Based Observations ⭐

**Concept:** Use attention to focus on important parts of observations.

**Why Attention?**
- Handles variable-length observations
- Focuses on relevant information
- Interpretable (can visualize attention)
- State-of-the-art performance

**Implementation:**
```python
class AttentionObservationEncoder(nn.Module):
    """
    Attention-based observation encoder.
    """
    def __init__(self, obs_dim, d_model=128, n_heads=8):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(self, observations):
        """
        observations: (B, T, obs_dim) - sequence of observations
        """
        # Project to model dimension
        x = self.input_proj(observations)  # (B, T, d_model)
        
        # Self-attention
        attended, attention_weights = self.attention(x, x, x)
        
        # Pool (mean or use last timestep)
        pooled = attended.mean(dim=1)  # (B, d_model)
        
        # Output projection
        output = self.output_proj(pooled)
        return output, attention_weights
```

**Visualizing Attention:**
```python
def visualize_attention(attention_weights, observations):
    """
    Visualize which observations the model attends to.
    """
    # attention_weights: (B, n_heads, T, T)
    # Average over heads and batch
    attn = attention_weights.mean(dim=0).mean(dim=0)  # (T, T)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn.cpu().numpy(), annot=True, fmt='.2f', cmap='Blues')
    plt.title('Attention Weights')
    plt.xlabel('Key Observation')
    plt.ylabel('Query Observation')
    plt.show()
```

### Contrastive Learning for Observations

**Concept:** Learn representations by contrasting similar vs. different observations.

**SimCLR-style Contrastive Learning:**
```python
class SimCLRObservationEncoder(nn.Module):
    """
    SimCLR-style contrastive learning for observations.
    """
    def __init__(self, obs_dim, projection_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.projection = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
    
    def forward(self, obs):
        h = self.encoder(obs)
        z = F.normalize(self.projection(h), dim=1)
        return z
    
    def contrastive_loss(self, obs_1, obs_2, temperature=0.07):
        """
        Contrastive loss: augmented versions should be similar.
        """
        z_1 = self.forward(obs_1)
        z_2 = self.forward(obs_2)
        
        # Compute similarity
        sim = F.cosine_similarity(z_1, z_2.unsqueeze(0), dim=1) / temperature
        
        # Contrastive loss
        labels = torch.arange(len(obs_1)).to(obs_1.device)
        loss = F.cross_entropy(sim, labels)
        return loss
```

### Observation Augmentation

**Concept:** Augment observations during training for robustness.

**Common Augmentations:**
```python
class ObservationAugmentation:
    """
    Augment observations for robustness.
    """
    def __init__(self):
        self.noise_std = 0.1
        self.dropout_prob = 0.1
    
    def augment(self, obs):
        # Add noise
        obs = obs + torch.randn_like(obs) * self.noise_std
        
        # Random dropout (mask some features)
        mask = torch.rand_like(obs) > self.dropout_prob
        obs = obs * mask
        
        # Normalize
        obs = F.normalize(obs, dim=-1)
        return obs
```

**Benefits:**
- More robust to sensor noise
- Better generalization
- Sim-to-real transfer

---

## 📊 Summary

### Key Takeaways

1. **Observations ≠ State**: Observations are limited measurements, not ground truth

2. **Normalize Everything**: Consistent normalization is critical for learning

3. **Use Dict Spaces**: Natural for multi-modal observations

4. **Only Realistic Sensors**: Don't include information a real robot can't measure

5. **Handle Partial Observability**: Use history, RNNs, or state estimation

6. **Document Clearly**: Make observation format explicit

7. **Consider Learned Representations** - Can improve performance and reduce engineering ⭐

8. **Use Attention for Complex Observations** - Focus on important information ⭐

### Observation Design Checklist

- [ ] Only includes realistic sensors
- [ ] All components normalized consistently
- [ ] Dict space for multi-modal observations
- [ ] Temporal context included (velocity, history)
- [ ] No information leakage (no future info)
- [ ] Format clearly documented
- [ ] Bounds are reasonable and enforced
- [ ] Considered learned representations for complex observations ⭐
- [ ] Considered attention mechanisms for variable-length observations ⭐

---

## 🎯 Next Steps

Now that you understand observation spaces, here's what to explore next:

### Related Tutorials
- **[Reward Design](04_reward_design_for_robotics.md)** - Design rewards that work with observations
- **[Multi-Modal Fusion](10_multimodal_fusion.md)** - Combine multiple observation modalities
- **[Actor-Critic Methods](05_actor_critic_methods.md)** - See how observations feed into policies

### Practical Examples
- **[Basic Usage Example](../../examples/01_basic_usage.py)** - See observations in action
- **[Custom Policy Example](../../examples/04_custom_policy.py)** - Custom observation processing
- **[Training Workflow](../../examples/05_training_workflow.py)** - Full training with observations

### Concepts to Explore
- **[Observation Design](../concepts/observation_design.md)** - Deep dive into observation design principles
- **[RL Fundamentals](../concepts/rl_fundamentals.md)** - MDP observation space formulation
- **[Multi-Modal Fusion](../tutorials/10_multimodal_fusion.md)** - Combining proprioception and vision

### Research Papers
- **[Research Timeline](../research/timeline.md)** - How observation spaces evolved
- **[Code Mapping](../research/code_mapping.md)** - Observation space implementation

**Prerequisites for Next Tutorial:**
- Understanding of observation spaces (this tutorial)
- Basic Gymnasium knowledge
- Familiarity with multi-modal inputs

---

## 📚 Further Reading

### Papers

**Classic Observation Design:**
- **Mnih et al. (2015)** - "Human-level control through deep reinforcement learning" - Atari observations
- **Lillicrap et al. (2015)** - "Continuous control with deep RL" - Proprioceptive observations
- **Levine et al. (2016)** - "End-to-end training of deep visuomotor policies" - Visual observations

**Modern Observation Learning:**
- **Chen et al. (2020)** - "A Simple Framework for Contrastive Learning of Visual Representations" - SimCLR
- **Vaswani et al. (2017)** - "Attention Is All You Need" - Transformer attention
- **Schwarzer et al. (2021)** - "Data-Efficient Reinforcement Learning with Self-Predictive Representations" - SPR
- **Laskin et al. (2020)** - "CURL: Contrastive Unsupervised Representations for Reinforcement Learning"
- **Yarats et al. (2021)** - "Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels"

### Books
- **Sutton & Barto** - "Reinforcement Learning: An Introduction" - Chapter 17: Planning and Learning with Tabular Methods

### Code Examples
- Ballbot environment: `ballbot_gym/bbot_env.py` - `_get_obs()` method

---

## 🎓 Exercises

1. **Modify Ballbot Observations**: Remove one observation component (e.g., motor_state). How does this affect learning?

2. **Add New Sensor**: Add a "goal_distance" observation to the Ballbot. How do you normalize it?

3. **Temporal Context**: Implement a history buffer for Ballbot observations. Use the last 3 observations.

---

*Next Tutorial: [Reward Design for Robotics](04_reward_design_for_robotics.md)*

---

**Happy Learning! 🚀**

