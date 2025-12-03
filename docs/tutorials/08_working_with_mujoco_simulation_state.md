# ⚙️ Working with MuJoCo Simulation State

*A comprehensive guide to accessing, processing, and managing simulation state in MuJoCo for robotics RL*

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [MuJoCo Model & Data Architecture](#mujoco-model--data-architecture)
3. [Accessing Simulation State](#accessing-simulation-state)
4. [State Estimation Techniques](#state-estimation-techniques)
5. [Sensor Timing & Asynchronous Updates](#sensor-timing--asynchronous-updates)
6. [Real-World Example: Ballbot Environment](#real-world-example-ballbot-environment)
7. [Best Practices](#best-practices)
8. [Common Pitfalls](#common-pitfalls)
9. [Advanced Topics](#advanced-topics)
10. [Summary](#summary)

---

## 🎯 Introduction

Understanding how to access and process simulation state is fundamental to building effective RL environments. MuJoCo provides a rich API for extracting positions, velocities, orientations, and other state information—but using it effectively requires understanding the architecture and best practices.

> "The state representation is the foundation of any RL system. How you extract and process simulation data determines what your agent can learn."  
> — *Common wisdom in robotics RL*

**Key Concepts:**
- MuJoCo separates **model** (static) from **data** (dynamic)
- State estimation requires careful numerical methods
- Sensor timing must match realistic hardware constraints
- Proper state processing enables sim-to-real transfer

**What You'll Learn:**
- How MuJoCo's model/data architecture works
- Accessing positions, velocities, and orientations
- Computing derived quantities (velocities from positions)
- Handling asynchronous sensor updates
- Best practices for state extraction

---

## 🏗️ MuJoCo Model & Data Architecture

### The Fundamental Distinction

MuJoCo separates simulation into two key components:

```
┌─────────────────────────────────────┐
│      MuJoCo Model (MjModel)         │
│  - Static properties                │
│  - Loaded from XML                  │
│  - Never changes during simulation  │
└──────────────┬──────────────────────┘
               │
               │ Defines structure
               ▼
┌─────────────────────────────────────┐
│      MuJoCo Data (MjData)            │
│  - Dynamic state                     │
│  - Created from model                │
│  - Updated every timestep            │
└─────────────────────────────────────┘
```

### Model: Static Properties

The **model** contains everything that doesn't change during simulation:

```python
model = mujoco.MjModel.from_xml_path("robot.xml")

# Model properties (static)
model.nbody      # Number of bodies
model.ngeom      # Number of geometries
model.njoint     # Number of joints
model.nq         # Number of position coordinates
model.nv         # Number of velocity coordinates

# Physical properties
model.body_mass      # Mass of each body
model.geom_size      # Size of each geometry
model.joint_range    # Joint limits
model.opt.timestep   # Physics timestep
```

**Key Point:** The model is loaded once and never modified during simulation.

### Data: Dynamic State

The **data** structure holds the current simulation state:

```python
data = mujoco.MjData(model)

# State arrays (updated every timestep)
data.qpos      # Position coordinates (joint angles, positions)
data.qvel      # Velocity coordinates (joint velocities)
data.qacc      # Acceleration coordinates
data.ctrl      # Control inputs (actuator commands)

# Body state
data.xpos      # Body positions [x, y, z] in world frame
data.xquat     # Body orientations (quaternions)
data.xmat      # Body rotation matrices
data.cvel      # Body velocities (6D: linear + angular)

# Contact and forces
data.contact   # Contact information
data.qfrc_applied  # Applied forces
```

**Key Point:** The data structure is updated every physics step via `mj_step()`.

### Forward Kinematics

After modifying positions, you must call forward kinematics to update derived quantities:

```python
# Set joint positions
data.qpos[joint_id] = angle

# Update forward kinematics
mujoco.mj_forward(model, data)

# Now these are valid:
position = data.xpos[body_id]      # Body position
orientation = data.xquat[body_id]  # Body orientation
```

**Critical:** Always call `mj_forward()` after modifying `qpos` or `qvel` before accessing `xpos`, `xquat`, etc.

---

## 📊 Accessing Simulation State

### Accessing Bodies

Bodies represent rigid objects in the simulation:

```python
# Get body ID by name
body_id = model.body("base").id

# Access body position (world frame)
position = data.xpos[body_id]  # Shape: (3,) [x, y, z]

# Access body orientation (quaternion)
orientation = data.xquat[body_id]  # Shape: (4,) [w, x, y, z]

# Access body rotation matrix
rotation_matrix = data.xmat[body_id].reshape(3, 3)  # Shape: (3, 3)

# Access body velocity (6D: 3D linear + 3D angular)
velocity = data.cvel[body_id]  # Shape: (6,)
linear_vel = velocity[:3]     # Linear velocity
angular_vel = velocity[3:]     # Angular velocity
```

### Accessing Joints

Joints connect bodies and define degrees of freedom:

```python
# Get joint ID by name
joint_id = model.joint("shoulder_joint").id

# Access joint position (angle for revolute, position for prismatic)
joint_angle = data.qpos[joint_id]

# Access joint velocity
joint_velocity = data.qvel[joint_id]

# Get joint limits
joint_min = model.joint_range[joint_id, 0]
joint_max = model.joint_range[joint_id, 1]
```

### Accessing Actuators

Actuators apply forces/torques:

```python
# Get actuator ID by name
actuator_id = model.actuator("motor_0").id

# Set control input
data.ctrl[actuator_id] = torque_value

# Get actuator force (after mj_forward)
actuator_force = data.actuator_force[actuator_id]
```

### Accessing Geoms

Geometries define collision shapes:

```python
# Get geom ID by name
geom_id = model.geom("base_link").id

# Access geom position
geom_pos = data.geom_xpos[geom_id]  # Shape: (3,)

# Access geom orientation
geom_quat = data.geom_xquat[geom_id]  # Shape: (4,)

# Access geom size
geom_size = model.geom_size[geom_id]  # Shape: (3,) for box, (1,) for sphere
```

### Complete Example: Extracting Robot State

```python
import mujoco
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

# Step simulation
mujoco.mj_step(model, data)

# Extract base body state
base_id = model.body("base").id
base_pos = data.xpos[base_id].copy()        # Position
base_quat = data.xquat[base_id].copy()      # Orientation (quaternion)
base_vel = data.cvel[base_id].copy()        # 6D velocity

# Extract joint states
joint_states = []
for i in range(model.njoint):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    joint_angle = data.qpos[model.joint(joint_name).id]
    joint_vel = data.qvel[model.joint(joint_name).id]
    joint_states.append({
        "name": joint_name,
        "angle": joint_angle,
        "velocity": joint_vel
    })
```

---

## 🔬 Accessing Velocities and Accelerations

### MuJoCo Computes Velocities Automatically

**Important:** MuJoCo automatically computes velocities during simulation. You should **use MuJoCo's built-in velocities** (`data.cvel` and `data.qvel`) as your primary method. Alternative methods (like finite difference) should only be used for specific cases.

### Body Velocities: `data.cvel`

MuJoCo computes 6D body velocities automatically in `data.cvel`:

```python
# Get body velocity (computed automatically by MuJoCo)
body_id = model.body("base").id
cvel = data.cvel[body_id]  # Shape: (6,)

# Extract linear and angular components
linear_vel = cvel[:3]      # Linear velocity in world frame [m/s]
angular_vel = cvel[3:]     # Angular velocity in world frame [rad/s]
```

**Key Properties:**
- **Computed automatically** after `mj_step()` or `mj_forward()`
- **World frame**: Velocities are in the global coordinate frame
- **Center of Mass (COM)**: Velocities are computed at the body's center of mass
- **Accurate**: Computed from the kinematic structure, not numerical differentiation

**When Available:**
- After calling `mj_step(model, data)` (simulation step)
- After calling `mj_forward(model, data)` (forward kinematics)

### Joint Velocities: `data.qvel`

Joint velocities are also computed automatically:

```python
# Get joint velocity
joint_id = model.joint("shoulder_joint").id
joint_velocity = data.qvel[joint_id]  # Joint velocity [rad/s or m/s]
```

### Complete Velocity Example

```python
import mujoco
import numpy as np

# Load and step simulation
model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

# Extract velocities (MuJoCo computes these automatically!)
body_id = model.body("base").id

# Body velocities (6D: linear + angular)
cvel = data.cvel[body_id]
linear_velocity = cvel[:3].copy()    # [m/s] in world frame
angular_velocity = cvel[3:].copy()   # [rad/s] in world frame

# Joint velocities
joint_id = model.joint("shoulder_joint").id
joint_velocity = data.qvel[joint_id]  # [rad/s]

print(f"Linear velocity: {linear_velocity}")
print(f"Angular velocity: {angular_velocity}")
print(f"Joint velocity: {joint_velocity}")
```

### Accelerations: `data.qacc`

MuJoCo also computes joint accelerations:

```python
# Get joint acceleration (computed automatically)
joint_id = model.joint("shoulder_joint").id
joint_acceleration = data.qacc[joint_id]  # Joint acceleration [rad/s² or m/s²]
```

**Note:** Body accelerations are not directly available in `data`, but can be computed from joint accelerations using Jacobians (see Advanced Topics section).

### When to Use Alternative Methods

You might use finite difference or other estimation methods when:

1. **Validation**: Comparing with MuJoCo's computed velocities to verify correctness
2. **Different Coordinate Frame**: Need velocities at a specific point (not COM) or in a different frame
3. **Custom Processing**: Need velocities with specific filtering or processing
4. **Educational Purposes**: Understanding how velocities relate to positions

**However, for most applications, use MuJoCo's built-in velocities directly.**

---

## 🔬 Alternative State Estimation Techniques

### Why Use Alternative Methods?

While MuJoCo computes velocities automatically, sometimes you need alternative methods for:
- **Validation**: Verifying MuJoCo's computed velocities
- **Different Frames**: Computing velocities at specific points or in different coordinate frames
- **Custom Processing**: Applying filters or transformations
- **Educational Understanding**: Learning how velocities relate to positions

### Linear Velocity: Finite Difference Method

Finite difference estimates velocity by differentiating position over time:

```python
# Store previous position
prev_position = None
dt = model.opt.timestep  # Physics timestep

def estimate_linear_velocity(current_pos, prev_pos, dt):
    """Estimate linear velocity using finite difference."""
    if prev_pos is None:
        return np.zeros_like(current_pos)
    
    velocity = (current_pos - prev_pos) / dt
    return velocity

# Usage in simulation loop
for step in range(num_steps):
    mujoco.mj_step(model, data)
    
    # Get current position
    body_id = model.body("base").id
    current_pos = data.xpos[body_id].copy()
    
    # Estimate velocity
    if prev_position is not None:
        estimated_vel = estimate_linear_velocity(
            current_pos, prev_position, dt
        )
    else:
        estimated_vel = np.zeros(3)
    
    # Store for next iteration
    prev_position = current_pos.copy()
```

**Mathematical Form:**
\[
\mathbf{v}(t) \approx \frac{\mathbf{x}(t) - \mathbf{x}(t-\Delta t)}{\Delta t}
\]

**Advantages:**
- Simple and intuitive
- Works for any position data
- Can compute velocity at any point (not just COM)

**Disadvantages:**
- Sensitive to numerical noise
- Requires storing previous state
- First step has no velocity
- Less accurate than MuJoCo's kinematic computation

**Recommendation:** Use `data.cvel[:3]` instead unless you need velocity at a specific point.

### Angular Velocity: Alternative Methods

MuJoCo computes angular velocity automatically in `data.cvel[body_id][3:]`. However, for educational purposes or specific applications, here are alternative methods:

#### Method 1: Using MuJoCo's Built-in (Recommended)

```python
# Use MuJoCo's computed angular velocity (RECOMMENDED)
body_id = model.body("base").id
angular_vel = data.cvel[body_id][3:]  # Already computed, accurate
```

#### Method 2: Matrix Logarithm (Advanced, for Custom Applications)

For cases where you need to compute angular velocity from orientation history, use the matrix logarithm:

```python
import numpy as np
from scipy.linalg import logm
import quaternion

def estimate_angular_velocity_using_logm(
    current_quat, prev_quat, dt
):
    """
    Estimate angular velocity using matrix logarithm.
    
    This method is more accurate than differentiating Euler angles,
    which can have singularities and numerical issues.
    """
    if prev_quat is None:
        return np.zeros(3)
    
    # Convert quaternions to rotation matrices
    R_prev = quaternion.as_rotation_matrix(
        quaternion.quaternion(*prev_quat)
    )
    R_curr = quaternion.as_rotation_matrix(
        quaternion.quaternion(*current_quat)
    )
    
    # Compute relative rotation: R_rel = R_prev^T * R_curr
    # This gives the rotation that occurred between timesteps
    R_rel = R_prev.T @ R_curr
    
    # Matrix logarithm gives skew-symmetric matrix (Lie algebra so(3))
    W = logm(R_rel).real
    
    # Extract angular velocity vector using "vee" map
    # The vee map converts skew-symmetric matrix to 3D vector
    def vee(S):
        """Vee map: converts skew-symmetric matrix to vector."""
        return np.array([S[2, 1], S[0, 2], S[1, 0]])
    
    angular_vel_vec = vee(W)
    
    # Divide by timestep to get angular velocity (rad/s)
    angular_velocity = angular_vel_vec / dt
    
    return angular_velocity

# Usage
prev_orientation = None
dt = model.opt.timestep

for step in range(num_steps):
    mujoco.mj_step(model, data)
    
    body_id = model.body("base").id
    current_quat = data.xquat[body_id].copy()
    
    if prev_orientation is not None:
        angular_vel = estimate_angular_velocity_using_logm(
            current_quat, prev_orientation, dt
        )
    else:
        angular_vel = np.zeros(3)
    
    prev_orientation = current_quat.copy()
```

**Mathematical Form:**
\[
\boldsymbol{\omega} = \frac{1}{\Delta t} \text{vee}(\log(\mathbf{R}_1^T \mathbf{R}_2))
\]

where:
- \(\mathbf{R}_1, \mathbf{R}_2\) are rotation matrices
- \(\log\) is the matrix logarithm (Lie algebra)
- \(\text{vee}\) extracts the angular velocity vector

**Why Matrix Logarithm?**
- Avoids Euler angle singularities (gimbal lock)
- More numerically stable than Euler angle differentiation
- Gives true angular velocity in body frame
- Mathematically rigorous (Lie group theory)

**Note:** This is mainly useful for educational purposes or when you need angular velocity in a specific frame. For most applications, use `data.cvel[body_id][3:]` directly.

#### Method 3: Quaternion Differentiation (Alternative)

For completeness, here's quaternion-based differentiation:

```python
def estimate_angular_velocity_quaternion(
    current_quat, prev_quat, dt
):
    """Estimate angular velocity using quaternion differentiation."""
    if prev_quat is None:
        return np.zeros(3)
    
    # Convert to quaternion objects
    q_prev = quaternion.quaternion(*prev_quat)
    q_curr = quaternion.quaternion(*current_quat)
    
    # Relative quaternion: q_rel = q_prev^-1 * q_curr
    q_rel = q_prev.inverse() * q_curr
    
    # For small rotations, angular velocity ≈ 2 * q_rel.vec / dt
    # where q_rel.vec is the vector part of the quaternion
    angular_vel = 2.0 * q_rel.vec / dt
    
    return angular_vel
```

### Rotation Representations

Different representations have different properties:

#### Quaternions
```python
# MuJoCo stores orientations as quaternions [w, x, y, z]
quat = data.xquat[body_id]  # Shape: (4,)

# Advantages:
# - No singularities (gimbal lock)
# - Compact (4 numbers)
# - Smooth interpolation
# - Efficient composition

# Disadvantages:
# - Not intuitive
# - Double cover (q and -q represent same rotation)
```

#### Rotation Vectors
```python
import quaternion

# Convert quaternion to rotation vector (axis-angle)
quat_obj = quaternion.quaternion(*data.xquat[body_id])
rot_vec = quaternion.as_rotation_vector(quat_obj)  # Shape: (3,)

# Advantages:
# - Compact (3 numbers)
# - Intuitive (axis * angle)
# - Good for neural networks

# Disadvantages:
# - Singularity at 180° rotation
# - Less stable for large rotations
```

#### Euler Angles
```python
import quaternion

# Convert quaternion to Euler angles
quat_obj = quaternion.quaternion(*data.xquat[body_id])
euler = quaternion.as_euler_angles(quat_obj)  # Shape: (3,)

# Advantages:
# - Intuitive (roll, pitch, yaw)
# - Compact (3 numbers)

# Disadvantages:
# - Gimbal lock singularities
# - Order-dependent
# - Not suitable for optimization
```

**Recommendation:** Use quaternions internally, convert to rotation vectors for neural network inputs.

---

## ⏱️ Sensor Timing & Asynchronous Updates

### The Problem

Real robots have sensors that update at different rates:
- **Physics simulation**: 500-1000 Hz (very fast)
- **IMU**: 100-200 Hz
- **Cameras**: 30-90 Hz (much slower)
- **LIDAR**: 10-20 Hz

If you render cameras at physics rate, you're:
1. Wasting computation
2. Creating unrealistic observations
3. Not matching real hardware

### Solution: Asynchronous Sensor Updates

Update sensors only when enough time has passed:

```python
class SensorManager:
    """Manages sensors with different update rates."""
    
    def __init__(self, model, physics_timestep):
        self.model = model
        self.physics_timestep = physics_timestep
        self.sensor_timestamps = {}
        self.sensor_data = {}
        
    def register_sensor(self, name, update_rate_hz):
        """Register a sensor with its update rate."""
        self.sensor_timestamps[name] = 0.0
        self.sensor_data[name] = None
        self.sensor_update_interval = 1.0 / update_rate_hz
        
    def should_update(self, name, current_time):
        """Check if sensor should update."""
        time_since_update = current_time - self.sensor_timestamps[name]
        return time_since_update >= self.sensor_update_interval
    
    def update_sensor(self, name, current_time, new_data):
        """Update sensor data and timestamp."""
        self.sensor_data[name] = new_data
        self.sensor_timestamps[name] = current_time
    
    def get_sensor_data(self, name, current_time):
        """Get sensor data (may be stale)."""
        return self.sensor_data[name], current_time - self.sensor_timestamps[name]
```

### Camera Timing Example

```python
class CameraSystem:
    """Manages camera updates at realistic frame rates."""
    
    def __init__(self, model, camera_name, frame_rate_hz=30):
        self.model = model
        self.camera_name = camera_name
        self.frame_rate = frame_rate_hz
        self.update_interval = 1.0 / frame_rate_hz
        
        # Initialize renderer
        self.renderer = mujoco.Renderer(model, width=640, height=480)
        
        # Store last frame and timestamp
        self.last_frame = None
        self.last_timestamp = 0.0
    
    def get_frame(self, data, current_time):
        """
        Get camera frame, updating only if enough time has passed.
        
        Returns:
            frame: Camera image (may be stale)
            lag: Time since last update (for observation)
        """
        time_since_update = current_time - self.last_timestamp
        
        if self.last_frame is None or time_since_update >= self.update_interval:
            # Update frame
            self.renderer.update_scene(data, camera=self.camera_name)
            self.last_frame = self.renderer.render().copy()
            self.last_timestamp = current_time
            lag = 0.0
        else:
            # Reuse previous frame
            lag = time_since_update
        
        return self.last_frame.copy(), lag
```

### Effective Frame Rate Calculation

Due to discrete physics timesteps, the actual frame rate may differ from desired:

```python
def calculate_effective_frame_rate(desired_rate_hz, physics_timestep):
    """
    Calculate effective frame rate given physics timestep constraints.
    
    Args:
        desired_rate_hz: Desired sensor update rate (Hz)
        physics_timestep: Physics simulation timestep (seconds)
    
    Returns:
        effective_rate: Actual achievable frame rate (Hz)
        steps_per_update: Number of physics steps per sensor update
    """
    desired_interval = 1.0 / desired_rate_hz
    
    # Find number of physics steps needed to exceed desired interval
    steps_per_update = np.ceil(desired_interval / physics_timestep)
    
    # Effective interval is steps_per_update * physics_timestep
    effective_interval = steps_per_update * physics_timestep
    effective_rate = 1.0 / effective_interval
    
    return effective_rate, int(steps_per_update)

# Example
desired_rate = 90  # Hz
physics_dt = 0.002  # 500 Hz physics

effective_rate, steps = calculate_effective_frame_rate(desired_rate, physics_dt)
print(f"Desired: {desired_rate} Hz")
print(f"Effective: {effective_rate:.2f} Hz")
print(f"Steps per update: {steps}")
# Output:
# Desired: 90 Hz
# Effective: 83.33 Hz
# Steps per update: 12
```

### Including Sensor Lag in Observations

It's often useful to include sensor lag in observations so the policy knows how stale the data is:

```python
def get_observation_with_timing(self, data):
    """Get observation including sensor timing information."""
    current_time = data.time
    
    # Get camera frame (may be stale)
    camera_frame, camera_lag = self.camera_system.get_frame(data, current_time)
    
    # Get IMU data (updated every step, so lag ≈ 0)
    imu_data = self.get_imu_data(data)
    imu_lag = 0.0
    
    # Construct observation
    obs = {
        "camera": camera_frame,
        "camera_lag": np.array([camera_lag]),  # Include lag!
        "imu": imu_data,
        "imu_lag": np.array([imu_lag]),
        # ... other sensors
    }
    
    return obs
```

**Why Include Lag?**
- Policy can learn to handle stale data
- More realistic (real sensors have lag)
- Enables better sim-to-real transfer

---

## 🤖 Real-World Example: Ballbot Environment

Let's see how the Ballbot environment combines all these concepts:

### Model & Data Setup

```python
class BBotSimulation(gym.Env):
    def __init__(self, xml_path, ...):
        # Load model (static properties)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        
        # Create data (dynamic state)
        self.data = mujoco.MjData(self.model)
        
        # Store physics timestep
        self.opt_timestep = self.model.opt.timestep
```

### State Access

```python
def _get_obs(self, last_ctrl):
    """Extract observation from simulation state."""
    
    # Access body state
    body_id = self.model.body("base").id
    position = self.data.xpos[body_id].copy()
    orientation_quat = quaternion.quaternion(*self.data.xquat[body_id])
    
    # Access joint velocities
    motor_state = np.array([
        self.data.qvel[self.model.joint(f"wheel_joint_{i}").id]
        for i in range(3)
    ])
```

### Velocity Extraction

**Note:** The Ballbot environment currently uses finite difference for linear velocity and matrix logarithm for angular velocity. However, MuJoCo provides these automatically. Here are both approaches:

#### Recommended: Using MuJoCo's Built-in Velocities

```python
    # Use MuJoCo's automatically computed velocities (RECOMMENDED)
    cvel = self.data.cvel[body_id]
    linear_vel = cvel[:3].copy()      # Linear velocity [m/s] in world frame
    angular_vel = cvel[3:].copy()    # Angular velocity [rad/s] in world frame
    
    # Clip to reasonable ranges for observations
    linear_vel = np.clip(linear_vel, a_min=-2.0, a_max=2.0)
    angular_vel = np.clip(angular_vel, a_min=-2.0, a_max=2.0)
```

**Advantages:**
- More accurate (computed from kinematic structure)
- No need to store previous state
- Available immediately after `mj_step()`
- Simpler code

#### Alternative: Finite Difference (Current Ballbot Implementation)

The current Ballbot code uses finite difference, which works but is less accurate:

```python
    # Compute linear velocity using finite difference
    # NOTE: data.cvel[:3] would be simpler and more accurate
    if self.prev_pos is not None:
        vel = (position - self.prev_pos) / self.opt_timestep
    else:
        vel = np.zeros_like(position)
    vel = np.clip(vel, a_min=-2.0, a_max=2.0)
    self.prev_pos = position.copy()
    
    # Compute angular velocity using matrix logarithm
    # NOTE: data.cvel[3:] would be simpler, but this shows the method
    if self.prev_orientation is not None:
        R_1 = quaternion.as_rotation_matrix(self.prev_orientation)
        R_2 = quaternion.as_rotation_matrix(orientation_quat)
        R_rel = R_1.T @ R_2
        W = logm(R_rel).real
        vee = lambda S: np.array([S[2, 1], S[0, 2], S[1, 0]])
        angular_vel_vec = vee(W)
        angular_vel = angular_vel_vec / self.opt_timestep
        angular_vel = np.clip(angular_vel, a_min=-2.0, a_max=2.0)
    else:
        angular_vel = np.zeros(3)
    self.prev_orientation = orientation_quat.copy()
```

**When to Use Finite Difference:**
- Need velocity at a specific point (not COM)
- Need velocity in a different coordinate frame
- Validating MuJoCo's computed velocities

### Sensor Timing

```python
    # Camera updates at lower frequency
    self.camera_frame_rate = 90  # Hz
    
    # Check if cameras should update
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
    
    # Include sensor lag in observation
    obs["relative_image_timestamp"] = np.array([
        self.data.time - self.prev_im_pair.ts
    ])
```

### Complete Observation Construction

```python
    # Construct final observation
    obs = {
        "orientation": rot_vec,              # From state access
        "angular_vel": angular_vel,          # From state estimation
        "vel": vel,                          # From state estimation
        "motor_state": motor_state,           # From state access
        "actions": last_ctrl,
        "rgbd_0": rgbd_0.transpose(2, 0, 1), # From sensor timing
        "rgbd_1": rgbd_1.transpose(2, 0, 1),
        "relative_image_timestamp": ...,     # Sensor lag
    }
    
    return obs
```

This example shows how all three concepts work together:
1. **State Access**: Getting positions, orientations from `data`
2. **Velocity Extraction**: Using `data.cvel` (recommended) or finite difference (alternative)
3. **Sensor Timing**: Managing camera updates at realistic rates

**Key Takeaway:** Use `data.cvel` for velocities whenever possible. It's more accurate and simpler than finite difference.

---

## ✅ Best Practices

### 1. Always Call `mj_forward()` After Modifying State

```python
# ❌ Bad: Accessing xpos before forward kinematics
data.qpos[joint_id] = angle
position = data.xpos[body_id]  # WRONG! xpos not updated yet

# ✅ Good: Call forward kinematics first
data.qpos[joint_id] = angle
mujoco.mj_forward(model, data)
position = data.xpos[body_id]  # Correct!
```

### 2. Copy Arrays When Storing State

```python
# ❌ Bad: Storing reference (will change!)
self.prev_pos = data.xpos[body_id]

# ✅ Good: Copy the array
self.prev_pos = data.xpos[body_id].copy()
```

### 3. Use Appropriate Rotation Representations

```python
# For neural network inputs: rotation vectors
rot_vec = quaternion.as_rotation_vector(quat)

# For internal computation: quaternions
quat = quaternion.quaternion(*data.xquat[body_id])

# For visualization: Euler angles (if needed)
euler = quaternion.as_euler_angles(quat)
```

### 4. Handle First Step Gracefully

```python
# Always check if previous state exists
if self.prev_pos is not None:
    velocity = (current_pos - self.prev_pos) / dt
else:
    velocity = np.zeros_like(current_pos)  # First step: no velocity
```

### 5. Match Sensor Rates to Real Hardware

```python
# Realistic rates
camera_rate = 30  # Hz (typical camera)
imu_rate = 100   # Hz (typical IMU)
lidar_rate = 10  # Hz (typical LIDAR)

# Don't update sensors at physics rate (unrealistic!)
```

### 6. Include Sensor Lag in Observations

```python
# Policy should know how stale data is
obs["camera_lag"] = np.array([time_since_update])
```

### 7. Use MuJoCo's Built-in Velocities

```python
# ✅ Best: Use MuJoCo's computed velocities (most accurate)
body_id = model.body("base").id
cvel = data.cvel[body_id]
linear_vel = cvel[:3]      # Linear velocity
angular_vel = cvel[3:]     # Angular velocity

# ✅ Good: Matrix logarithm (if you need custom processing)
angular_vel = compute_angular_vel_logm(R_prev, R_curr, dt)

# ❌ Bad: Differentiating Euler angles (singularities!)
euler_prev = quaternion.as_euler_angles(q_prev)
euler_curr = quaternion.as_euler_angles(q_curr)
angular_vel = (euler_curr - euler_prev) / dt  # WRONG!
```

### 8. Access Accelerations When Needed

```python
# Joint accelerations are available in data.qacc
joint_id = model.joint("shoulder_joint").id
joint_acceleration = data.qacc[joint_id]  # [rad/s² or m/s²]

# Body accelerations require Jacobian computation (see Advanced Topics)
```

---

## ⚠️ Common Pitfalls

### Pitfall 1: Forgetting Forward Kinematics

```python
# This is a common mistake
data.qpos[joint_id] = new_angle
position = data.xpos[body_id]  # OLD position! Not updated!
```

**Solution:** Always call `mj_forward()` after modifying `qpos` or `qvel`.

### Pitfall 2: Storing References Instead of Copies

```python
# This stores a reference - data.xpos will change!
self.prev_pos = data.xpos[body_id]

# Later...
mujoco.mj_step(model, data)  # data.xpos changes
# Now self.prev_pos also changed! (same array)
```

**Solution:** Always use `.copy()` when storing state.

### Pitfall 3: Not Using MuJoCo's Built-in Velocities

```python
# This is unnecessarily complex!
prev_pos = data.xpos[body_id]
# ... store, wait for next step ...
vel = (current_pos - prev_pos) / dt  # Finite difference

# When you could just use:
vel = data.cvel[body_id][:3]  # Already computed!
```

**Solution:** Use `data.cvel` directly. It's more accurate and simpler.

### Pitfall 4: Using Euler Angles for Angular Velocity

```python
# This has singularities!
euler_prev = quaternion.as_euler_angles(q_prev)
euler_curr = quaternion.as_euler_angles(q_curr)
angular_vel = (euler_curr - euler_prev) / dt  # WRONG at singularities!
```

**Solution:** Use `data.cvel[body_id][3:]` (recommended) or matrix logarithm.

### Pitfall 4: Updating Sensors at Physics Rate

```python
# This is unrealistic and wasteful
for step in range(num_steps):
    mujoco.mj_step(model, data)
    camera_frame = renderer.render()  # Rendering every step!
```

**Solution:** Update sensors only when enough time has passed.

### Pitfall 5: Not Handling First Step

```python
# This crashes on first step!
velocity = (current_pos - self.prev_pos) / dt  # prev_pos is None!
```

**Solution:** Always check if previous state exists.

---

## 🔬 Advanced Topics

### Computing Body Accelerations

MuJoCo provides joint accelerations in `data.qacc`, but body accelerations require Jacobian computation:

```python
import mujoco
import numpy as np

def compute_body_acceleration(model, data, body_id):
    """
    Compute body linear acceleration from joint accelerations.
    
    Uses the Jacobian to transform joint accelerations to body frame.
    """
    # Allocate Jacobian matrices
    jacp = np.zeros((3, model.nv))  # Position Jacobian
    jacr = np.zeros((3, model.nv))  # Rotation Jacobian
    
    # Compute Jacobian at body's center of mass
    mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
    
    # Body linear acceleration = J_p @ qacc
    linear_accel = jacp @ data.qacc
    
    # Body angular acceleration = J_r @ qacc
    angular_accel = jacr @ data.qacc
    
    return linear_accel, angular_accel

# Usage
body_id = model.body("base").id
linear_accel, angular_accel = compute_body_acceleration(model, data, body_id)
```

**Note:** This is advanced and rarely needed. Most applications can use joint accelerations (`data.qacc`) directly.

### Custom State Extractors

Create reusable state extraction functions:

```python
class StateExtractor:
    """Extracts and processes robot state."""
    
    def __init__(self, model):
        self.model = model
        self.prev_states = {}
    
    def extract_body_state(self, data, body_name):
        """Extract complete body state."""
        body_id = self.model.body(body_name).id
        
        state = {
            "position": data.xpos[body_id].copy(),
            "orientation": data.xquat[body_id].copy(),
            "linear_vel": data.cvel[body_id][:3].copy(),
            "angular_vel": data.cvel[body_id][3:].copy(),
        }
        
        return state
    
    def extract_joint_state(self, data, joint_name):
        """Extract joint state."""
        joint_id = self.model.joint(joint_name).id
        
        state = {
            "position": data.qpos[joint_id],
            "velocity": data.qvel[joint_id],
            "limits": self.model.joint_range[joint_id].copy(),
        }
        
        return state
```

### State Normalization

Normalize state for neural network inputs:

```python
def normalize_state(state, stats):
    """Normalize state using statistics."""
    normalized = {}
    
    for key, value in state.items():
        if key in stats:
            mean = stats[key]["mean"]
            std = stats[key]["std"]
            normalized[key] = (value - mean) / (std + 1e-8)
        else:
            normalized[key] = value
    
    return normalized
```

### State History Buffers

Maintain history for temporal features:

```python
class StateHistory:
    """Maintains state history buffer."""
    
    def __init__(self, max_length=10):
        self.max_length = max_length
        self.history = []
    
    def add(self, state):
        """Add state to history."""
        self.history.append(state)
        if len(self.history) > self.max_length:
            self.history.pop(0)
    
    def get_history(self):
        """Get full history."""
        return self.history.copy()
    
    def get_recent(self, n):
        """Get n most recent states."""
        return self.history[-n:].copy()
```

---

## 📊 Summary

### Key Takeaways

1. **Model vs Data**: Model is static, Data is dynamic. Understand the distinction.

2. **Forward Kinematics**: Always call `mj_forward()` after modifying positions before accessing derived quantities.

3. **Use MuJoCo's Built-in Velocities**: `data.cvel` provides accurate 6D body velocities (linear + angular) automatically. Use this as your primary method.

4. **Joint Velocities**: `data.qvel` provides joint velocities automatically.

5. **Accelerations**: `data.qacc` provides joint accelerations. Body accelerations require Jacobian computation.

6. **Alternative Methods**: Finite difference and matrix logarithm are useful for validation or specific applications, but MuJoCo's built-in velocities are preferred.

7. **Sensor Timing**: Match sensor update rates to real hardware. Update asynchronously.

8. **Copy Arrays**: Always use `.copy()` when storing state to avoid reference issues.

9. **Rotation Representations**: Use quaternions internally, rotation vectors for neural networks.

10. **Include Sensor Lag**: Policy should know how stale sensor data is.

### The Complete Pipeline

```
1. Access State (Model & Data)
   ↓
2. Process State (Estimation)
   ↓
3. Manage Timing (Sensors)
   ↓
4. Construct Observation
```

### Checklist

- [ ] Understand model vs data distinction
- [ ] Always call `mj_forward()` after state modifications
- [ ] Use `data.cvel` for body velocities (primary method)
- [ ] Use `data.qvel` for joint velocities
- [ ] Use `data.qacc` for joint accelerations when needed
- [ ] Copy arrays when storing state
- [ ] Use appropriate rotation representations
- [ ] Match sensor rates to real hardware
- [ ] Include sensor lag in observations
- [ ] Prefer MuJoCo's built-in velocities over finite difference

---

## 📚 Further Reading

### Documentation
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MuJoCo API Reference](https://mujoco.readthedocs.io/en/latest/APIreference/index.html)
- [Quaternion Library](https://github.com/moble/quaternion)

### Papers
- **Todorov et al. (2012)** - "MuJoCo: A physics engine for model-based control"
- **Murray et al. (1994)** - "A Mathematical Introduction to Robotic Manipulation" (Lie groups)

### Related Tutorials
- [Introduction to Gymnasium](01_introduction_to_gymnasium.md)
- [Observation Spaces in RL](03_observation_spaces_in_rl.md)
- [Camera Rendering in MuJoCo](07_camera_rendering_in_mujoco.md)

---

## 🎓 Exercises

1. **State Access**: Write a function that extracts all joint positions and velocities from a MuJoCo model.

2. **Velocity Estimation**: Implement finite difference velocity estimation and compare with MuJoCo's built-in velocities.

3. **Sensor Timing**: Create a sensor manager that handles multiple sensors with different update rates.

4. **Angular Velocity**: Implement matrix logarithm angular velocity estimation and test on various rotations.

5. **Complete System**: Build a simple observation extractor that combines state access, estimation, and sensor timing.

---

*Next Tutorial: [Terrain Generation & Heightfields in MuJoCo](09_terrain_generation_in_mujoco.md)*

---

**Happy Learning! 🚀**

