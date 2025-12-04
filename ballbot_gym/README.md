# 🤖 BallbotGym: Deep Dive Documentation

*A comprehensive guide to the Ballbot reinforcement learning environment*

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [High-Level Overview](#high-level-overview)
3. [Installation & Quick Start](#installation--quick-start)
4. [Architecture & Design](#architecture--design)
5. [Terrain & World Setup](#terrain--world-setup)
6. [Cameras & RGB-D Rendering](#cameras--rgb-d-rendering)
7. [Observation Space](#observation-space)
8. [Action Space](#action-space)
9. [Rewards & Goals](#rewards--goals)
10. [Step Logic & Dynamics](#step-logic--dynamics)
11. [Termination & Safety](#termination--safety)
12. [Logging & Diagnostics](#logging--diagnostics)
13. [Integration with RL Algorithms](#integration-with-rl-algorithms)
14. [Advanced Topics](#advanced-topics)
15. [API Reference](#api-reference)
16. [Troubleshooting](#troubleshooting)

---

## 🎯 Introduction

**BallbotGym** is a Gymnasium-compatible reinforcement learning environment for training a dynamically balanced ballbot robot. The ballbot is a 3-wheeled omni-drive robot that moves on a ball, requiring careful control to maintain balance while navigating terrain.

> "The ballbot environment demonstrates how to build a research-grade RL environment that bridges simulation and real robotics."  
> — *Designed for practical robotics applications with realistic physics and sensor simulation*

**Key Features:**
- ✅ **MuJoCo Physics Simulation** - Realistic robot dynamics
- ✅ **RGB-D Camera Support** - Visual observations from multiple cameras
- ✅ **Procedural Terrain Generation** - Perlin noise or flat terrain
- ✅ **Directional Reward System** - Goal-directed navigation
- ✅ **Comprehensive Logging** - Camera frames, rewards, terrain seeds
- ✅ **Gymnasium API** - Compatible with PPO, SAC, TD3, and other RL algorithms

**Related Tutorials:**
- [Introduction to Gymnasium](../Tutorials/01_introduction_to_gymnasium.md) - Learn the Gymnasium API
- [Action Spaces in RL](../Tutorials/02_action_spaces_in_rl.md) - Understanding continuous actions
- [Observation Spaces in RL](../Tutorials/03_observation_spaces_in_rl.md) - Multi-modal observations
- [Reward Design for Robotics](../Tutorials/04_reward_design_for_robotics.md) - Reward shaping principles
- [Camera Rendering in MuJoCo](../Tutorials/07_camera_rendering_in_mujoco.md) - RGB-D rendering details
- [Working with MuJoCo Simulation State](../Tutorials/08_working_with_mujoco_simulation_state.md) - State access, estimation, and sensor timing

---

## 🏗️ High-Level Overview

### What is BallbotGym?

Conceptually, this environment is:

> A **3-wheeled omni-drive robot** on a **random terrain**, with **RGB-D cameras**, implemented as a **Gymnasium-compatible environment** on top of **MuJoCo**.

**Key Ideas:**
- Follows the **Gymnasium Env API** → works with PPO/SAC/etc.
- Uses **MuJoCo** for physics simulation
- Observations = combination of:
  - Proprioception (pose, velocities, motor state)
  - Action history
  - Visual input (RGB-D or depth-only from 2 cameras)
- Terrain is randomized (Perlin) to encourage **robust policies**
- Custom reward and termination logic

### System Architecture

```
┌─────────────────────────────────────┐
│      RL Algorithm                   │
│  (PPO, SAC, TD3, etc.)              │
└──────────────┬──────────────────────┘
               │
               │ Gymnasium API
               ▼
┌─────────────────────────────────────┐
│   BBotSimulation (Gymnasium Env)    │
│   - reset(), step(), spaces         │
└──────────────┬──────────────────────┘
               │
               ├──► MuJoCo Physics
               ├──► RGBDInputs (Cameras)
               ├──► Terrain Generation
               └──► Reward Computation
```

---

## 🚀 Installation & Quick Start

### Prerequisites

- **Python 3.8+**
- **MuJoCo** (patched version required - see project README)
- **NumPy, Gymnasium, quaternion, scipy, opencv-python**

### Installation

```bash
# Install the package in development mode
cd ballbotgym
pip install -e .

# Or install dependencies manually
pip install gymnasium numpy-quaternion matplotlib scipy opencv-python termcolor
```

### Quick Start Example

```python
import gymnasium as gym
import ballbotgym  # Registers the environment

# Create environment
env = gym.make("ballbot-v0.1")

# Reset to initial state
obs, info = env.reset(seed=42)

# Run a simple episode
for step in range(1000):
    # Sample random action (in practice, use your policy)
    action = env.action_space.sample()
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check if episode ended
    if terminated or truncated:
        obs, info = env.reset()

# Cleanup
env.close()
```

### Environment Configuration

```python
# Custom configuration
env = gym.make(
    "ballbot-v0.1",
    disable_cameras=False,      # Enable RGB-D cameras
    depth_only=True,            # Depth-only (faster) or RGB-D
    im_shape={"h": 64, "w": 64}, # Camera resolution
    terrain_type="perlin",      # "perlin" or "flat"
    max_ep_steps=4000,          # Maximum episode length
    GUI=False                   # Enable MuJoCo viewer
)
```

---

## 🏛️ Architecture & Design

### Class Structure

The environment is implemented in `BBotSimulation`, which inherits from `gym.Env`:

```python
class BBotSimulation(gym.Env):
    """
    Main environment class implementing Gymnasium API
    """
    def __init__(self, xml_path, ...):
        # Initialize spaces, MuJoCo, cameras, terrain
        
    def reset(self, seed=None, ...):
        # Reset to initial state
        
    def step(self, omniwheel_commands):
        # Execute one timestep
        
    def close(self):
        # Cleanup resources
```

### Key Components

1. **MuJoCo Model & Data**
   - `self.model`: Physics model (masses, inertias, geometries)
   - `self.data`: Simulation state (positions, velocities)

2. **RGBDInputs**
   - Manages RGB-D camera rendering
   - Handles multiple cameras with configurable frame rates

3. **Terrain Generator**
   - Perlin noise or flat terrain
   - Dynamic heightfield updates

4. **Reward System**
   - `Rewards.DirectionalReward`: Encourages movement in target direction

5. **State Tracking**
   - Previous positions, orientations for velocity computation
   - Camera frame history for temporal consistency

---

## 🌍 Terrain & World Setup

### Terrain Generation: `_reset_terrain`

This method determines **what the robot stands on** at the start of each episode.

#### What It Does

1. **Reads MuJoCo Heightfield Parameters:**
   ```python
   nrows = self.model.hfield_nrow.item()  # Resolution
   ncols = self.model.hfield_ncol.item()
   sz = self.model.hfield_size[0, 0]        # Physical size
   hfield_height_coef = self.model.hfield_size[0, 2]  # Height scaling
   ```

2. **Generates Terrain Based on Type:**
   - `"perlin"`: Procedural Perlin noise terrain (randomized each episode)
   - `"flat"`: Zero heightfield (flat ground)

3. **Updates MuJoCo Heightfield:**
   ```python
   if self.terrain_type == "perlin":
       r_seed = self._np_random.integers(0, 10000)
       self.model.hfield_data = terrain.generate_perlin_terrain(
           nrows, seed=r_seed
       )
   elif self.terrain_type == "flat":
       self.model.hfield_data = np.zeros(nrows**2)
   ```

4. **Computes Initial Robot Height:**
   - Locates the ball geometry (`"the_ball"`) which marks spawn region
   - Computes AABB (axis-aligned bounding box) of the ball
   - Extracts terrain subregion under the robot
   - Finds maximum height and adds epsilon for safety
   - Returns `init_robot_height_offset` to position robot on terrain

#### Why This Matters

- Ensures robot is **spawned above terrain**, not intersecting it
- Supports arbitrary terrain shapes via dynamic height sampling
- Critical for physics stability and simulation realism

#### Connection to RL

- **Terrain randomization = domain randomization** → more robust policies
- Each episode gets different terrain if `"perlin"` + random seed
- Encourages policies that generalize to unseen terrain

**Related Tutorial:** See [Terrain Generation](../Tutorials/README.md) (if available) for more on procedural generation.

---

## 📷 Cameras & RGB-D Rendering

### RGBDInputs Class

The `RGBDInputs` class manages RGB-D camera rendering for MuJoCo environments.

#### Purpose

- Owns one or two **MuJoCo renderers**:
  - `_renderer_rgb`: For color images (optional)
  - `_renderer_d`: For depth images (always enabled)
- Supports **multiple named cameras**: `cams=["cam_0", "cam_1"]`

#### Usage

```python
# Initialize
rgbd_inputs = RGBDInputs(
    mjc_model,
    height=64,
    width=64,
    cams=["cam_0", "cam_1"],
    disable_rgb=True  # Depth-only mode
)

# Render images
rgbd_0 = rgbd_inputs(data, "cam_0")  # Returns (H, W, C) array
```

#### Rendering Process

1. **Update Scene:**
   ```python
   self._renderer_d.update_scene(data, camera=cam_name)
   ```

2. **Render Depth:**
   ```python
   depth = np.expand_dims(self._renderer_d.render(), axis=-1)
   depth[depth >= 1.0] = 1.0  # Clip extreme values (sky, background)
   ```

3. **Render RGB (if enabled):**
   ```python
   if self._renderer_rgb is not None:
       self._renderer_rgb.update_scene(data, camera=cam_name)
       rgb = self._renderer_rgb.render().astype(np.float32) / 255
   ```

4. **Combine Channels:**
   ```python
   if rgb_enabled:
       arr = np.concatenate([rgb, depth], -1)  # (H, W, 4)
   else:
       arr = depth  # (H, W, 1)
   ```

#### Why Separate Class?

- Encapsulates messy renderer setup/reset/close
- Makes camera handling modular and reusable
- Allows environment to treat cameras as black box

**Related Tutorial:** [Camera Rendering in MuJoCo](../Tutorials/07_camera_rendering_in_mujoco.md) - Deep dive into RGB-D rendering.

---

### StampedImPair: Camera Timing & History

```python
@dataclass
class StampedImPair:
    im_0: np.ndarray  # Last frame from cam_0
    im_1: np.ndarray  # Last frame from cam_1
    ts: float         # Timestamp when frames were taken
```

#### Purpose

The environment runs at MuJoCo timestep (e.g., 500 Hz), but cameras update at lower frequency:

```python
self.camera_frame_rate = 90  # Hz (slower than physics)
```

#### Update Logic

In `_get_obs`, cameras only re-render if enough time has passed:

```python
delta_time = self.data.time - self.prev_im_pair.ts

if self.prev_im_pair.im_0 is None or delta_time >= 1.0 / self.camera_frame_rate:
    # Render new images
    rgbd_0 = self.rgbd_inputs(self.data, "cam_0")
    rgbd_1 = self.rgbd_inputs(self.data, "cam_1")
    self.prev_im_pair = StampedImPair(im_0=rgbd_0, im_1=rgbd_1, ts=self.data.time)
else:
    # Reuse previous images
    rgbd_0 = self.prev_im_pair.im_0.copy()
    rgbd_1 = self.prev_im_pair.im_1.copy()
```

Also stores `relative_image_timestamp` = `self.data.time - self.prev_im_pair.ts`.

#### Why This Matters

- **Realistic sensor timing**: Real cameras don't run at physics update rates
- **Sensor lag**: Policy sees both images and how stale they are
- **Computational efficiency**: Reduces expensive rendering operations
- **Temporal context**: Policy learns to handle asynchronous sensor updates

This is a nice touch for realistic **sensor timing effects** in robotics.

**Related Tutorial:** [Working with MuJoCo Simulation State](../Tutorials/08_working_with_mujoco_simulation_state.md) - Deep dive into sensor timing and asynchronous updates.

---

## 👁️ Observation Space

### The Observation Builder: `_get_obs`

This is the **core of the state representation** that the RL algorithm sees. It fuses:

1. **Vision** (RGB-D images)
2. **Orientation & angular velocity**
3. **Linear velocity**
4. **Motor state** (wheel speeds)
5. **Action history**
6. **Camera timestamp lag**

### Observation Components

#### 1. Camera Observations (if enabled)

```python
# Update cameras at lower frequency
if delta_time >= 1.0 / self.camera_frame_rate:
    rgbd_0 = self.rgbd_inputs(self.data, "cam_0")
    rgbd_1 = self.rgbd_inputs(self.data, "cam_1")
    
# Convert to channels-first format for PyTorch CNNs
obs["rgbd_0"] = rgbd_0.transpose(2, 0, 1)  # (C, H, W)
obs["rgbd_1"] = rgbd_1.transpose(2, 0, 1)

# Store camera lag
obs["relative_image_timestamp"] = np.array([self.data.time - self.prev_im_pair.ts])
```

**Shape:** `(C, H, W)` where C=1 (depth-only) or 4 (RGB-D)`

#### 2. Body State (Orientation)

```python
# Get base body position and orientation
body_id = self.model.body("base").id
position = self.data.xpos[body_id]
orientation_quat = quaternion.quaternion(*self.data.xquat[body_id])

# Convert quaternion to rotation vector (3D representation)
rot_vec = quaternion.as_rotation_vector(orientation_quat)
obs["orientation"] = rot_vec  # Shape: (3,)
```

**Why rotation vector?** More stable than Euler angles, easier for neural networks than quaternions.

#### 3. Motor State

```python
# Read wheel velocities
motor_state = np.array([
    self.data.qvel[self.model.joint(f"wheel_joint_{i}").id]
    for i in range(3)
])

# Normalize and clip
motor_state /= 10  # Normalize by typical max (10 rad/s)
motor_state = np.clip(motor_state, -2.0, 2.0)
obs["motor_state"] = motor_state  # Shape: (3,)
```

#### 4. Linear Velocity

The environment uses MuJoCo's automatically computed body velocities from `data.cvel`:

```python
# Get body velocities from MuJoCo (computed automatically)
# data.cvel provides 6D velocity: [linear_vel (3D), angular_vel (3D)]
cvel = self.data.cvel[body_id]
linear_vel = cvel[:3].copy()  # Linear velocity in world frame [m/s]

# Clip to reasonable ranges for observations
vel = np.clip(linear_vel, -2.0, 2.0)
obs["vel"] = vel  # Shape: (3,)
```

**Advantages:**
- More accurate (computed from kinematic structure, not numerical differentiation)
- Simpler code (no need to store previous state)
- Available immediately after `mj_step()`

**Related Tutorial:** [Working with MuJoCo Simulation State](../Tutorials/08_working_with_mujoco_simulation_state.md) - Learn about `data.cvel` and MuJoCo's velocity computation.

#### 5. Angular Velocity

The environment uses MuJoCo's automatically computed angular velocity from `data.cvel`:

```python
# Get angular velocity from MuJoCo (computed automatically)
cvel = self.data.cvel[body_id]
angular_vel = cvel[3:].copy()  # Angular velocity in world frame [rad/s]

# Clip to reasonable ranges for observations
angular_vel = np.clip(angular_vel, -2.0, 2.0)
obs["angular_vel"] = angular_vel  # Shape: (3,)
```

**Advantages:**
- More accurate (computed from kinematic structure)
- Simpler code (no need for matrix logarithm or previous state)
- Avoids Euler angle singularities automatically
- Available immediately after `mj_step()`

**Note:** MuJoCo computes angular velocity in the world frame at the body's center of mass, which is appropriate for most RL applications.

#### 6. Previous Action

```python
obs["actions"] = last_ctrl  # Shape: (3,)
```

Provides temporal context for the policy.

### Complete Observation Dictionary

**With cameras enabled:**
```python
{
    "orientation": np.array([...]),           # (3,)
    "angular_vel": np.array([...]),          # (3,)
    "vel": np.array([...]),                  # (3,)
    "motor_state": np.array([...]),          # (3,)
    "actions": np.array([...]),              # (3,)
    "rgbd_0": np.array([...]),               # (C, H, W)
    "rgbd_1": np.array([...]),               # (C, H, W)
    "relative_image_timestamp": np.array([...])  # (1,)
}
```

**Without cameras:**
```python
{
    "orientation": np.array([...]),
    "angular_vel": np.array([...]),
    "vel": np.array([...]),
    "motor_state": np.array([...]),
    "actions": np.array([...]),
    "relative_image_timestamp": np.array([...])
}
```

**Related Tutorial:** [Observation Spaces in RL](../Tutorials/03_observation_spaces_in_rl.md) - Understanding multi-modal observations.

---

## 🎮 Action Space

### Action Definition

```python
self.action_space = gym.spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(3,),
    dtype=np.float32
)
```

- **3D continuous actions** for three omniwheels
- **Normalized to [-1, 1]**
- **Scaled internally** to [-10, 10] rad/s

### Action Processing in `step()`

```python
def step(self, omniwheel_commands):
    # Scale from [-1, 1] to [-10, 10] rad/s
    ctrl = omniwheel_commands * 10
    ctrl = np.clip(ctrl, -10, 10)  # Safety clipping
    
    # Apply to MuJoCo (negative for coordinate system)
    self.data.ctrl[:] = -ctrl
    
    # Step physics
    mujoco.mj_step(self.model, self.data)
```

### Action Interpretation

- Each action component controls one omniwheel
- Positive values rotate wheel in one direction, negative in opposite
- The negative sign in `self.data.ctrl[:] = -ctrl` accounts for coordinate system conventions

**Related Tutorial:** [Action Spaces in RL](../Tutorials/02_action_spaces_in_rl.md) - Understanding continuous action spaces.

---

## 🎯 Rewards & Goals

### Goal System

Currently, the goal is fixed to `[0.0, 1.0]` (positive Y direction):

```python
def _reset_goal_and_reward_objs(self):
    self.goal_2d = [0.0, 1.0]  # Target direction
    self.reward_obj = Rewards.DirectionalReward(
        target_direction=self.goal_2d
    )
```

**Future Extension:** Could support random or specified goal directions for more diverse training.

### Reward Structure

The reward consists of three components:

#### 1. Directional Reward

```python
# Encourages velocity in target direction
reward = self.reward_obj(obs) / 100
```

**Implementation:**
```python
# From Rewards.DirectionalReward
dir_rew = state["vel"][-3:-1].dot(self.target_direction)
```

This is the **dot product** of 2D velocity with target direction, scaled by 1/100.

**Mathematical Form:**
\[
r_{\text{dir}} = \frac{\mathbf{v}_{xy} \cdot \mathbf{g}}{100}
\]
where \(\mathbf{v}_{xy}\) is 2D velocity and \(\mathbf{g}\) is goal direction.

#### 2. Action Regularization

```python
# Penalizes large actions (energy efficiency)
action_regularization = -0.0001 * (np.linalg.norm(omniwheel_commands)**2)
reward += action_regularization
```

**Mathematical Form:**
\[
r_{\text{reg}} = -0.0001 \|\mathbf{a}\|^2
\]

This encourages **smooth, energy-efficient** control.

#### 3. Survival Bonus

```python
# Small positive reward for staying upright
if angle_in_degrees <= max_allowed_tilt:
    reward += 0.02
```

**Purpose:** Provides positive reward signal even when not moving, encouraging stability.

### Total Reward

\[
r_{\text{total}} = \frac{\mathbf{v}_{xy} \cdot \mathbf{g}}{100} - 0.0001 \|\mathbf{a}\|^2 + 0.02 \cdot \mathbb{1}[\text{upright}]
\]

### Reward Shaping Philosophy

This follows classic RL reward shaping:
- **Progress term**: Directional velocity
- **Smooth actions**: L2 penalty
- **Alive bonus**: Constant when upright
- **Failure penalty**: Via termination (no reward when fallen)

**Related Tutorial:** [Reward Design for Robotics](../Tutorials/04_reward_design_for_robotics.md) - Reward shaping principles.

---

## ⚙️ Step Logic & Dynamics

### The `step()` Method: Complete Flow

The `step()` method is the core of the environment. Here's the complete flow:

#### 1. Handle GUI Reset

```python
# Detect manual reset from viewer
if self.data.time == 0.0 and self.step_counter > 0.0:
    print("RESET DUE TO RESET FROM GUI")
    self.reset()
```

#### 2. Map Actions to Controls

```python
# Scale actions from [-1, 1] to [-10, 10] rad/s
ctrl = omniwheel_commands * 10
ctrl = np.clip(ctrl, -10, 10)

# Apply to MuJoCo (negative for coordinate system)
self.data.ctrl[:] = -ctrl
```

#### 3. Advance Simulation

```python
# Step physics forward
with warnings_stdout_off():  # Suppress benign MuJoCo warnings
    mujoco.mj_step(self.model, self.data)
```

#### 4. Reset Applied Forces

```python
# Clear external forces (prevent accumulation)
self.data.xfrc_applied[self.model.body("base").id, :3] = 0
```

#### 5. Compute Observation & Info

```python
obs = self._get_obs(omniwheel_commands.astype(_default_dtype))
info = self._get_info()
```

#### 6. Compute Reward

```python
# Directional reward
reward = self.reward_obj(obs) / 100

# Action regularization
action_regularization = -0.0001 * (np.linalg.norm(omniwheel_commands)**2)
reward += action_regularization

# Survival bonus (if upright)
if angle_in_degrees <= max_allowed_tilt:
    reward += 0.02
```

#### 7. Update Episodic Return

```python
self.G_tau += reward  # Accumulate for logging
```

#### 8. Check Termination

```python
# Max steps
if self.step_counter >= self.max_ep_steps:
    terminated = True

# Tilt check (see Termination section)
if angle_in_degrees > max_allowed_tilt:
    terminated = True
    info["failure"] = True
```

#### 9. Return Gymnasium Tuple

```python
return obs, reward, terminated, truncated, info
```

### Physics Timestep

The environment uses MuJoCo's default timestep (typically 0.002s = 500 Hz):

```python
@property
def opt_timestep(self):
    return self.model.opt.timestep  # Usually 0.002s
```

This is much faster than camera updates (90 Hz), ensuring smooth physics simulation.

---

## 🛑 Termination & Safety

### Termination Conditions

The episode can end in two ways:

#### 1. Maximum Episode Length

```python
if self.step_counter >= self.max_ep_steps:
    terminated = True
```

**Default:** `max_ep_steps = 4000` (8 seconds at 500 Hz)

#### 2. Tilt-Based Failure

The robot is considered "fallen" if it tilts too much:

```python
# Compute tilt angle
gravity = np.array([0, 0, -1.0])  # Global gravity

# Transform to robot's local frame
R_local = quaternion.as_rotation_matrix(
    quaternion.from_rotation_vector(obs["orientation"][-3:])
)
gravity_local = (R_local.T @ gravity).reshape(3)

# Robot's local up-axis
up_axis_local = np.array([0, 0, 1])

# Compute angle between up-axis and negative gravity
angle_in_degrees = np.arccos(
    up_axis_local.dot(-gravity_local)
) * 180 / np.pi

# Check threshold
max_allowed_tilt = 20  # degrees
if angle_in_degrees > max_allowed_tilt:
    terminated = True
    info["failure"] = True
```

**Interpretation:**
- When upright: `up_axis_local ≈ -gravity_local`, angle ≈ 0°
- When fallen: `up_axis_local ⊥ -gravity_local`, angle ≈ 90°
- Threshold: 20° (configurable)

### Safety Considerations

1. **Action Clipping:** Actions are clipped to [-10, 10] rad/s to prevent extreme commands
2. **Force Reset:** External forces are cleared each step to prevent accumulation
3. **Tilt Monitoring:** Continuous monitoring prevents dangerous orientations
4. **Physics Stability:** Terrain generation ensures robot spawns on surface, not embedded

### Termination vs. Truncation

- **`terminated=True`**: Episode ended due to failure or max steps (used for value function bootstrapping)
- **`truncated=True`**: Reserved for time limits (currently always False)

**Gymnasium Convention:** Use `terminated` for natural episode endings, `truncated` for artificial time limits.

---

## 📊 Logging & Diagnostics

### Logging System

The environment supports comprehensive logging for analysis and debugging.

### Logged Data

#### 1. Camera Frames

If `log_options["cams"] = True`:

```python
# Saved to: /tmp/log_<random>/rgbd_log_episode_X/
# - rgb/rbgd_a_*.png, rgb/rbgd_b_*.png (RGB images)
# - depth/depth_a_*.png, depth/depth_b_*.png (depth images)
```

**Format:**
- RGB: BGR → RGB conversion for OpenCV compatibility
- Depth: Normalized to [0, 255] for visualization

#### 2. Reward Components

If `log_options["reward_terms"] = True`:

```python
# Saved to: /tmp/log_<random>/term_1.npy, term_2.npy
# - term_1: Directional reward history
# - term_2: Action regularization history
```

#### 3. Terrain Seeds

```python
# Saved to: /tmp/log_<random>/terrain_seed_history
# One seed per line (for reproducibility)
```

### Logging Directory

Created automatically in `reset()`:

```python
# Random directory name
rand_str = ''.join(
    self._np_random.permutation(
        list(string.ascii_letters + string.digits)
    )[:12]
)
self.log_dir = "/tmp/log_" + rand_str
```

**Note:** Logs are only saved during training (not evaluation mode).

### Diagnostic Information

The `info` dictionary provides:

```python
info = {
    "success": False,        # Currently unused
    "failure": False,        # True if robot fell
    "step_counter": int,     # Current step in episode
    "pos2d": np.array([x, y]) # 2D position
}
```

### Verbose Mode

Enable verbose output for debugging:

```python
env.verbose = True
```

This prints:
- Effective camera frame rate
- Termination reasons
- Episode returns
- Failure details

---

## 🤖 Integration with RL Algorithms

### How This Fits Actor-Critic RL

From the perspective of PPO/SAC/etc:

#### Observation Space → NN Input

The observation dictionary defines what the neural network sees:

```python
# For vision-based policies
obs = {
    "rgbd_0": (C, H, W),      # CNN input
    "rgbd_1": (C, H, W),      # CNN input
    "orientation": (3,),      # MLP input
    "angular_vel": (3,),      # MLP input
    ...
}
```

**Architecture:**
- **CNN** processes RGB-D images
- **MLP** processes proprioceptive data
- **Fusion** layer combines features

#### Action Space → NN Output

```python
# Policy network outputs
action = policy_network(obs)  # Shape: (3,), range: [-1, 1]
```

#### Environment Interaction

```python
# Standard RL loop
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

### Algorithm Compatibility

#### PPO (Proximal Policy Optimization)

```python
from stable_baselines3 import PPO

env = gym.make("ballbot-v0.1")
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

**Why it works:**
- On-policy algorithm
- Uses GAE (Generalized Advantage Estimation)
- Handles Dict observation spaces

#### SAC (Soft Actor-Critic)

```python
from stable_baselines3 import SAC

env = gym.make("ballbot-v0.1")
model = SAC("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

**Why it works:**
- Off-policy algorithm
- Uses replay buffer
- Handles continuous actions

#### Custom Training Loop

```python
# Example: Simple policy gradient
env = gym.make("ballbot-v0.1")
obs, info = env.reset()

for episode in range(num_episodes):
    episode_rewards = []
    episode_obs = []
    episode_actions = []
    
    while True:
        action = policy(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        episode_rewards.append(reward)
        episode_obs.append(obs)
        episode_actions.append(action)
        
        if terminated or truncated:
            # Update policy using episode data
            update_policy(episode_obs, episode_actions, episode_rewards)
            obs, info = env.reset()
            break
        else:
            obs = next_obs
```

**Related Tutorial:** [Actor-Critic Methods in RL](../Tutorials/05_actor_critic_methods.md) - PPO, SAC, and value estimation.

---

## 🔬 Advanced Topics

### Effective Camera Frame Rate

Cameras update at lower frequency than physics. The effective rate accounts for discrete timesteps:

```python
def effective_camera_frame_rate(self):
    dt_mj = self.opt_timestep  # Physics timestep (e.g., 0.002s)
    desired_cam_dt = 1 / self.camera_frame_rate  # Desired interval (e.g., 1/90s)
    
    # Find number of physics steps needed
    N = np.ceil(desired_cam_dt / dt_mj)
    
    # Effective rate is inverse of actual interval
    effective_framre_rate = 1.0 / (N * dt_mj)
    return effective_framre_rate
```

**Example:**
- Physics: 500 Hz (0.002s timestep)
- Desired camera: 90 Hz (0.011s interval)
- Actual: ~83.3 Hz (12 physics steps per camera update)

### Velocity Computation

The environment uses MuJoCo's automatically computed body velocities:

```python
# Get 6D body velocity from MuJoCo (computed automatically)
body_id = self.model.body("base").id
cvel = self.data.cvel[body_id]  # Shape: (6,)

# Extract linear and angular components
linear_vel = cvel[:3]      # Linear velocity [m/s] in world frame
angular_vel = cvel[3:]      # Angular velocity [rad/s] in world frame
```

**Why use `data.cvel`?**
- More accurate: Computed from kinematic structure, not numerical differentiation
- Simpler: No need to store previous state or compute matrix logarithms
- Efficient: Available immediately after `mj_step()`
- Reliable: Avoids numerical issues with finite difference or Euler angles

**Related Tutorial:** [Working with MuJoCo Simulation State](../Tutorials/08_working_with_mujoco_simulation_state.md) - Learn about `data.cvel` and MuJoCo's velocity computation.

### Terrain Generation Details

Perlin noise terrain uses:

```python
terrain = generate_perlin_terrain(
    n=nrows,
    scale=25.0,        # Spatial frequency
    octaves=4,         # Detail levels
    persistence=0.2,   # Amplitude decay
    lacunarity=2,      # Frequency increase
    seed=r_seed
)
```

**Parameters:**
- `scale`: Larger = smoother terrain
- `octaves`: More = more detail
- `persistence`: Lower = smoother
- `lacunarity`: Higher = more variation

### Evaluation Mode

For reproducible evaluation:

```python
env = gym.make(
    "ballbot-v0.1",
    eval_env=[True, seed=42]  # Fixed seed for reproducibility
)
```

This uses a fixed random seed for terrain generation.

---

## 📖 API Reference

### BBotSimulation

#### `__init__(xml_path, ...)`

**Parameters:**
- `xml_path` (str): Path to MuJoCo XML model file
- `GUI` (bool): Enable MuJoCo viewer (default: False)
- `im_shape` (dict): Camera dimensions `{"h": 64, "w": 64}`
- `disable_cameras` (bool): Disable camera rendering (default: False)
- `depth_only` (bool): Depth-only mode (default: True)
- `log_options` (dict): `{"cams": False, "reward_terms": False}`
- `max_ep_steps` (int): Maximum episode length (default: 4000)
- `terrain_type` (str): `"perlin"` or `"flat"` (default: "perlin")
- `eval_env` (list): `[is_eval, seed]` for evaluation mode

#### `reset(seed=None, goal="random", **kwargs)`

**Returns:** `(observation, info)`

**Parameters:**
- `seed` (int): Random seed for reproducibility
- `goal` (str): Goal specification (currently unused)

#### `step(omniwheel_commands)`

**Parameters:**
- `omniwheel_commands` (np.ndarray): Shape (3,), range [-1, 1]

**Returns:** `(observation, reward, terminated, truncated, info)`

#### `close()`

Cleanup resources (cameras, viewer, MuJoCo structures).

### RGBDInputs

#### `__init__(mjc_model, height, width, cams, disable_rgb)`

Initialize RGB-D renderer system.

#### `__call__(data, cam_name)`

Render RGB-D or depth-only image from specified camera.

**Returns:** `np.ndarray` shape `(H, W, C)` where C=1 (depth) or 4 (RGB-D)

### StampedImPair

Dataclass storing image pair with timestamp:
- `im_0`: First image
- `im_1`: Second image
- `ts`: Timestamp

---

## 🔧 Troubleshooting

### Common Issues

#### 1. GUI Not Working on macOS

**Problem:** `RuntimeError: mjpython required`

**Solution:**
```python
# Use mjpython instead of python
mjpython your_script.py

# Or disable GUI
env = gym.make("ballbot-v0.1", GUI=False)
```

#### 2. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'ballbotgym'`

**Solution:**
```bash
# Install in development mode
cd ballbotgym
pip install -e .
```

#### 3. Camera Rendering Slow

**Problem:** Environment runs slowly with cameras enabled

**Solution:**
```python
# Use depth-only mode
env = gym.make("ballbot-v0.1", depth_only=True)

# Or reduce resolution
env = gym.make("ballbot-v0.1", im_shape={"h": 32, "w": 32})

# Or disable cameras entirely
env = gym.make("ballbot-v0.1", disable_cameras=True)
```

#### 4. Robot Spawning Inside Terrain

**Problem:** Robot intersects terrain at start

**Solution:**
- Check that `"the_ball"` geometry exists in XML
- Verify ball is centered at origin (0, 0)
- Ensure terrain generation returns valid height offset

#### 5. MuJoCo Warnings

**Problem:** Console cluttered with MuJoCo warnings

**Solution:**
- Warnings are suppressed automatically in `step()`
- If persistent, check XML file for configuration issues

### Performance Tips

1. **Disable cameras** if not needed: `disable_cameras=True`
2. **Use depth-only** mode: `depth_only=True`
3. **Reduce camera resolution**: `im_shape={"h": 32, "w": 32}`
4. **Use flat terrain** for faster training: `terrain_type="flat"`
5. **Disable logging** during training: `log_options={"cams": False, "reward_terms": False}`

### Debugging

Enable verbose mode:

```python
env = gym.make("ballbot-v0.1")
env.verbose = True
```

This prints:
- Effective camera frame rate
- Termination reasons
- Episode returns
- Failure details

---

## 📚 Further Reading

### Documentation
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

### Related Tutorials
- [Introduction to Gymnasium](../Tutorials/01_introduction_to_gymnasium.md)
- [Action Spaces in RL](../Tutorials/02_action_spaces_in_rl.md)
- [Observation Spaces in RL](../Tutorials/03_observation_spaces_in_rl.md)
- [Reward Design for Robotics](../Tutorials/04_reward_design_for_robotics.md)
- [Actor-Critic Methods](../Tutorials/05_actor_critic_methods.md)
- [Camera Rendering in MuJoCo](../Tutorials/07_camera_rendering_in_mujoco.md)
- [Working with MuJoCo Simulation State](../Tutorials/08_working_with_mujoco_simulation_state.md)

### Papers
- **Todorov et al. (2012)** - "MuJoCo: A physics engine for model-based control"
- **Schulman et al. (2017)** - "Proximal Policy Optimization"
- **Haarnoja et al. (2018)** - "Soft Actor-Critic"

---

## 🎓 Summary

### Key Takeaways

1. **BallbotGym is a research-grade RL environment** that demonstrates best practices for robotics simulation

2. **Multi-modal observations** combine proprioception and vision for rich state representation

3. **Uses MuJoCo's built-in velocities** (`data.cvel`) for accurate, efficient velocity computation

4. **Realistic sensor timing** with asynchronous camera updates matches real-world robotics

5. **Domain randomization** via terrain generation encourages robust policies

6. **Comprehensive logging** enables detailed analysis and debugging

7. **Gymnasium compatibility** ensures seamless integration with modern RL algorithms

### Environment Checklist

- ✅ Implements Gymnasium API (`reset()`, `step()`, `spaces`)
- ✅ Multi-modal observations (proprioception + vision)
- ✅ Uses MuJoCo's built-in velocities (`data.cvel`) for accuracy
- ✅ Realistic sensor timing (camera frame rates)
- ✅ Domain randomization (terrain generation)
- ✅ Comprehensive reward shaping
- ✅ Safety mechanisms (tilt monitoring, action clipping)
- ✅ Extensive logging support
- ✅ Well-documented and maintainable

---

**Happy Learning! 🚀**

*For questions or contributions, see the main project README.*

