# 💻 Code Walkthrough: Understanding the Implementation

*A detailed guide to the key code files and their connections in openballbot-rl*

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Entry Points](#entry-points)
3. [Core Environment: `bbot_env.py`](#core-environment-bbot_envpy)
4. [Reward System: `Rewards.py`](#reward-system-rewardspy)
5. [Terrain Generation: `terrain.py`](#terrain-generation-terrainpy)
6. [Policy Network: `policies.py`](#policy-network-policiespy)
7. [Training Script: `train.py`](#training-script-trainpy)
8. [Evaluation Script: `test.py`](#evaluation-script-testpy)
9. [Utilities: `utils.py`](#utilities-utilspy)
10. [Configuration: `train_ppo_directional.yaml`](#configuration-train_ppo_directionalyaml)
11. [MuJoCo Model: `bbot.xml`](#mujoco-model-bbotxml)
12. [Data Collection: `gather_data.py`](#data-collection-gather_datapy)
13. [Encoder Pretraining: `pretrain_cnn.py`](#encoder-pretraining-pretrain_cnnpy)
14. [Code Flow Examples](#code-flow-examples)

---

## 🎯 Overview

This walkthrough explains how the codebase is organized and how different components interact. We'll trace through the code from training initialization to policy execution, showing how research concepts translate into implementation.

**Key Files:**
- `ballbot_gym/bbot_env.py` - Main environment (1200+ lines)
- `ballbot_gym/Rewards.py` - Reward functions
- `ballbot_gym/terrain.py` - Terrain generation
- `ballbot_rl/policies/policies.py` - Neural network architectures
- `ballbot_rl/training/train.py` - Training pipeline
- `ballbot_rl/training/callbacks.py` - Training callbacks (video recording, checkpoints)
- `ballbot_rl/evaluation/evaluate.py` - Evaluation script
- `ballbot_rl/training/utils.py` - Environment factory
- `ballbot_rl/visualization/` - Visualization tools (plot_training, visualize_env, visualize_model, browse_environments)
- `configs/train_ppo_directional.yaml` - Hyperparameters

---

## 🚪 Entry Points

### Training Entry Point: `ballbot_rl/training/train.py`

**Purpose:** Main script for training RL policies using PPO.

**Key Functions:**

```python
def main(config, seed):
    # 1. Create environment factory
    vec_env = VecEnvClass([
        make_ballbot_env(...) for _ in range(N_ENVS)
    ])
    
    # 2. Initialize PPO agent
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        ...
    )
    
    # 3. Train
    model.learn(total_timesteps=config["total_timesteps"])
```

**Flow:**
1. Loads YAML configuration
2. Creates parallel environments via `make_ballbot_env()`
3. Initializes PPO with custom feature extractor
4. Trains policy with callbacks (checkpointing, evaluation)
5. Saves final model and logs

**Key Parameters:**
- `num_envs`: Number of parallel environments (default: 10)
- `total_timesteps`: Total training steps (default: 10M)
- `frozen_cnn`: Path to pretrained depth encoder

### Evaluation Entry Point: `ballbot_rl/evaluation/evaluate.py`

**Purpose:** Load trained policy and evaluate performance.

**Key Functions:**

```python
def test_policy(model_path, n_test=10):
    # 1. Load trained model
    model = PPO.load(model_path)
    
    # 2. Create evaluation environment
    env = make_ballbot_env(eval_env=True)
    
    # 3. Run episodes
    for episode in range(n_test):
        obs, info = env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
```

**Flow:**
1. Loads trained policy from checkpoint
2. Creates evaluation environment (deterministic)
3. Runs episodes with deterministic actions
4. Logs metrics (episode length, reward, success rate)

---

## 🏗️ Core Environment: `bbot_env.py`

**File:** `ballbot_gym/bbot_env.py` (1215 lines)

**Purpose:** Implements the Gymnasium environment interface for the ballbot.

### Class Structure

```python
class BBotSimulation(gym.Env):
    def __init__(self, xml_path, GUI=False, ...):
        # Initialize MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Define spaces
        self.action_space = Box(-1.0, 1.0, shape=(3,))
        self.observation_space = Dict({...})
        
        # Initialize components
        self.reward_obj = Rewards.DirectionalReward(...)
        self.terrain_gen = terrain.TerrainGenerator(...)
        self.rgbd_inputs = RGBDInputs(...)
```

### Key Methods

#### `__init__()` - Initialization

**What it does:**
1. Loads MuJoCo XML model (`bbot.xml`)
2. Defines action and observation spaces
3. Initializes reward function (`DirectionalReward`)
4. Sets up terrain generator
5. Configures RGB-D cameras (if enabled)
6. Sets up logging

**Key Code Snippets:**

```python
# Load MuJoCo model
self.model = mujoco.MjModel.from_xml_path(xml_path)
self.data = mujoco.MjData(self.model)

# Define action space: 3D continuous for three wheels
self.action_space = spaces.Box(
    low=-1.0, high=1.0, shape=(3,), dtype=np.float32
)

# Initialize reward function
target_direction = np.array([1.0, 0.0])  # Default: move in +x
self.reward_obj = Rewards.DirectionalReward(target_direction)

# Initialize terrain generator
self.terrain_gen = terrain.TerrainGenerator(
    self.model, terrain_type=terrain_type
)
```

#### `reset()` - Episode Initialization

**What it does:**
1. Generates new terrain (if `terrain_type="perlin"`)
2. Resets MuJoCo state to initial configuration
3. Sets random initial conditions (orientation, velocities)
4. Renders initial observations
5. Returns observation dictionary

**Key Code Flow:**

```python
def reset(self, seed=None):
    # Generate terrain
    if self.terrain_type == "perlin":
        hfield_data = self.terrain_gen.generate()
        self.model.geom("terrain").hfield_nrow = ...
        self.model.geom("terrain").hfield_data[:] = hfield_data
    
    # Reset MuJoCo state
    mujoco.mj_resetData(self.model, self.data)
    
    # Set random initial orientation
    quat = self._sample_initial_orientation()
    self.data.qpos[3:7] = quat
    
    # Get initial observation
    obs = self._get_obs()
    
    return obs, {}
```

**Important Details:**
- Terrain is regenerated each episode (for diversity)
- Initial orientation is sampled from a small range (near upright)
- Camera renderers are reset if model changed

#### `step()` - Single Simulation Step

**What it does:**
1. Converts normalized action to motor commands
2. Applies control inputs to MuJoCo
3. Steps physics simulation
4. Extracts new observations
5. Computes reward
6. Checks termination conditions
7. Returns (obs, reward, terminated, truncated, info)

**Key Code Flow:**

```python
def step(self, action):
    # Convert action to motor commands
    motor_commands = self._action_to_motor_command(action)
    self.data.ctrl[:] = motor_commands
    
    # Step physics
    mujoco.mj_step(self.model, self.data)
    
    # Get observations
    obs = self._get_obs()
    
    # Compute reward
    reward = self._compute_reward(obs, action)
    
    # Check termination
    terminated = self._check_termination()
    truncated = (self.step_count >= self.max_ep_steps)
    
    return obs, reward, terminated, truncated, {}
```

**Action Conversion:**

```python
def _action_to_motor_command(self, action):
    # Scale from [-1, 1] to [-10, 10] rad/s
    return action * 10.0
```

#### `_get_obs()` - Observation Extraction

**What it does:**
1. Extracts orientation (Euler angles from quaternion)
2. Extracts angular velocities
3. Extracts linear velocities
4. Gets motor states
5. Renders RGB-D images (if cameras enabled)
6. Combines into observation dictionary

**Key Code:**

```python
def _get_obs(self):
    # Extract orientation (convert quaternion to Euler)
    orientation = self._get_orientation()  # [φ, θ, ψ]
    
    # Extract velocities
    angular_vel = self.data.qvel[3:6]  # [φ̇, θ̇, ψ̇]
    vel = self.data.qvel[0:3]          # [ẋ, ẏ, ż]
    
    # Extract motor states
    motor_state = self.data.act
    
    # Render cameras (if enabled)
    if not self.disable_cameras:
        rgbd_0 = self.rgbd_inputs(self.data, "cam_0")
        rgbd_1 = self.rgbd_inputs(self.data, "cam_1")
    
    return {
        "orientation": orientation,
        "angular_vel": angular_vel,
        "vel": vel,
        "motor_state": motor_state,
        "actions": self.last_action,
        "rgbd_0": rgbd_0,  # if cameras enabled
        "rgbd_1": rgbd_1,  # if cameras enabled
    }
```

**Orientation Extraction:**

```python
def _get_orientation(self):
    # Convert quaternion to rotation matrix
    quat = self.data.qpos[3:7]  # [w, x, y, z]
    R = quaternion.as_rotation_matrix(quat)
    
    # Extract Euler angles
    phi = np.arctan2(R[2, 1], R[2, 2])      # Roll
    theta = np.arctan2(-R[2, 0], 
                      np.sqrt(R[2,1]**2 + R[2,2]**2))  # Pitch
    psi = np.arctan2(R[1, 0], R[0, 0])      # Yaw
    
    return np.array([phi, theta, psi])
```

#### `_compute_reward()` - Reward Calculation

**What it does:**
1. Computes directional reward (velocity in target direction)
2. Adds action regularization penalty
3. Adds survival bonus (if upright)
4. Returns total reward

**Key Code:**

```python
def _compute_reward(self, obs, action):
    # Directional reward
    directional_reward = self.reward_obj(obs) / 100.0
    
    # Action regularization
    action_regularization = -0.0001 * np.linalg.norm(action)**2
    
    # Total reward
    reward = directional_reward + action_regularization
    
    # Survival bonus
    tilt_angle = self._get_tilt_angle()
    if tilt_angle <= self.max_allowed_tilt:  # Default: 20 degrees
        reward += 0.02
    
    return reward
```

**Tilt Angle Calculation:**

```python
def _get_tilt_angle(self):
    # Get gravity vector in local frame
    quat = self.data.qpos[3:7]
    R = quaternion.as_rotation_matrix(quat)
    gravity_local = R.T @ np.array([0, 0, -1])
    
    # Compute angle from vertical
    up_axis = np.array([0, 0, 1])
    tilt_angle = np.arccos(np.clip(
        up_axis.dot(-gravity_local), -1, 1
    )) * 180 / np.pi  # Convert to degrees
    
    return tilt_angle
```

#### `_check_termination()` - Termination Logic

**What it does:**
1. Computes current tilt angle
2. Checks if tilt exceeds safety limit (20°)
3. Returns True if robot has fallen

**Key Code:**

```python
def _check_termination(self):
    tilt_angle = self._get_tilt_angle()
    return tilt_angle > self.max_allowed_tilt
```

### RGBDInputs Class

**Purpose:** Manages RGB-D camera rendering.

**Key Methods:**

```python
class RGBDInputs:
    def __init__(self, mjc_model, height, width, cams, disable_rgb):
        # Initialize renderers
        self._renderer_rgb = mujoco.Renderer(...) if not disable_rgb else None
        self._renderer_d = mujoco.Renderer(...)
        self._renderer_d.enable_depth_rendering()
    
    def __call__(self, data, cam_name):
        # Render RGB (if enabled)
        if self._renderer_rgb is not None:
            self._renderer_rgb.update_scene(data, camera=cam_name)
            rgb = self._renderer_rgb.render() / 255.0
        
        # Render depth
        self._renderer_d.update_scene(data, camera=cam_name)
        depth = self._renderer_d.render()
        depth = np.clip(depth, 0, 1.0)  # Clip extreme values
        
        # Combine
        if self._renderer_rgb is not None:
            return np.concatenate([rgb, depth], axis=-1)  # (H, W, 4)
        else:
            return depth  # (H, W, 1)
```

---

## 🎁 Reward System: `Rewards.py`

**File:** `ballbot_gym/Rewards.py` (54 lines)

**Purpose:** Implements reward functions for the ballbot environment.

### DirectionalReward Class

**Purpose:** Rewards movement in a target direction.

**Implementation:**

```python
class DirectionalReward:
    def __init__(self, target_direction):
        # target_direction: np.ndarray of shape (2,), unit vector
        self.target_direction = target_direction
    
    def __call__(self, state):
        # Extract x-y velocity
        xy_velocity = state["vel"][-3:-1]  # Last 2 elements
        
        # Dot product with target direction
        dir_rew = xy_velocity.dot(self.target_direction)
        
        return dir_rew
```

**Mathematical Form:**
\[
r_{\text{dir}} = \mathbf{v}_{xy} \cdot \mathbf{g}
\]

Where:
- \(\mathbf{v}_{xy} = [\dot{x}, \dot{y}]^T\) is the 2D ball velocity
- \(\mathbf{g} = [g_x, g_y]^T\) is the target direction (unit vector)

**Usage in Environment:**

```python
# In bbot_env.py
target_direction = np.array([1.0, 0.0])  # Move in +x direction
self.reward_obj = Rewards.DirectionalReward(target_direction)

# In step()
directional_reward = self.reward_obj(obs) / 100.0  # Scale by 100
```

**Why Divide by 100:**
- Keeps reward magnitudes reasonable (typically \([-0.1, 0.1]\))
- Prevents reward scaling issues in PPO
- Makes hyperparameter tuning easier

---

## 🌍 Terrain Generation: `terrain.py`

**File:** `ballbot_gym/terrain.py` (92 lines)

**Purpose:** Generates procedural terrain using Perlin noise.

### Key Function: `generate_perlin_terrain()`

**Purpose:** Creates a heightfield using Perlin noise.

**Implementation:**

```python
def generate_perlin_terrain(
    n: int,              # Grid size (should be odd)
    scale: float = 25.0,  # Feature size (higher = larger features)
    octaves: int = 4,     # Noise complexity
    persistence: float = 0.2,  # Amplitude scaling
    lacunarity: float = 2.0,   # Frequency scaling
    seed: int = 0         # Random seed
) -> np.ndarray:
    # Initialize terrain
    terrain = np.zeros((n, n))
    
    # Fill with Perlin noise
    for i in range(n):
        for j in range(n):
            x = i / scale
            y = j / scale
            terrain[i][j] = snoise2(
                x, y,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                base=seed
            )
    
    # Normalize to [0, 1]
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    
    # Return as flat array (row-major) for MuJoCo
    return terrain.flatten()
```

**Usage:**

```python
# In bbot_env.py
from . import terrain

class TerrainGenerator:
    def generate(self):
        return terrain.generate_perlin_terrain(
            n=129,           # 129x129 grid
            scale=25.0,      # Medium-sized features
            seed=self.seed  # Random seed
        )
```

**MuJoCo Integration:**

```python
# In reset()
hfield_data = self.terrain_gen.generate()
self.model.geom("terrain").hfield_data[:] = hfield_data
```

**Key Parameters:**
- `n`: Grid size (must be odd, typically 129)
- `scale`: Controls feature size (higher = smoother terrain)
- `octaves`: Number of noise layers (more = more detail)
- `seed`: Random seed (different seed = different terrain)

---

## 🧠 Policy Network: `policies.py`

**File:** `ballbot_rl/policies/policies.py` (198 lines)

**Purpose:** Defines neural network architectures for the RL policy.

### Extractor Class

**Purpose:** Custom feature extractor for Stable-Baselines3 that handles multi-modal observations.

**Key Components:**

```python
class Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, frozen_encoder_path=""):
        extractors = {}
        total_concat_size = 0
        
        # Process each observation component
        for key, subspace in observation_space.spaces.items():
            if "rgbd_" in key:
                # RGB-D images: use CNN encoder
                if frozen_encoder_path:
                    # Load pretrained frozen encoder
                    extractors[key] = torch.load(frozen_encoder_path)
                    # Freeze parameters
                    for param in extractors[key].parameters():
                        param.requires_grad = False
                else:
                    # Train encoder from scratch
                    extractors[key] = torch.nn.Sequential(
                        torch.nn.Conv2d(1, 32, kernel_size=3, stride=2),
                        torch.nn.BatchNorm2d(32),
                        torch.nn.LeakyReLU(),
                        torch.nn.Conv2d(32, 32, kernel_size=3, stride=2),
                        torch.nn.BatchNorm2d(32),
                        torch.nn.LeakyReLU(),
                        torch.nn.Flatten(),
                        torch.nn.Linear(..., 20),
                        torch.nn.Tanh()
                    )
                total_concat_size += 20
            else:
                # Proprioceptive: just flatten
                extractors[key] = torch.nn.Flatten()
                total_concat_size += subspace.shape[0]
        
        self.extractors = torch.nn.ModuleDict(extractors)
        self._features_dim = total_concat_size
    
    def forward(self, observations):
        encoded_list = []
        for key, extractor in self.extractors.items():
            encoded_list.append(extractor(observations[key]))
        return torch.cat(encoded_list, dim=1)
```

**Architecture:**
- **RGB-D images:** CNN encoder (pretrained or trainable) → 20-dim feature vector
- **Proprioceptive:** Flatten → concatenate
- **Output:** Concatenated feature vector → policy/value networks

**Usage in Training:**

```python
# In train.py
policy_kwargs = dict(
    features_extractor_class=policies.Extractor,
    features_extractor_kwargs={
        "frozen_encoder_path": config["frozen_cnn"]
    },
    ...
)

model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
```

### PID Class (Baseline)

**Purpose:** Classical PID controller for comparison.

**Location:** `ballbot_gym/controllers/pid.py`

**Implementation:**

```python
from ballbot_gym.controllers import PID

class PID:
    """Classical PID controller (not RL-specific)."""
    def __init__(self, dt, k_p, k_i, k_d):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.integral = torch.zeros(2)
        self.prev_err = torch.zeros(2)
    
    def act(self, R_mat, setpoint_r=0, setpoint_p=0):
        # Compute error (pitch, roll)
        error = [setpoint_p - pitch, setpoint_r - roll]
        
        # PID control
        self.integral += error * self.dt
        derivative = (error - self.prev_err) / self.dt
        u = self.k_p * error + self.k_i * self.integral + self.k_d * derivative
        
        # Convert to motor commands
        ctrl = self._pitch_roll_to_motors(u)
        return ctrl
```

**Note:** PID is a classical controller, not part of the RL system. It's located in `ballbot_gym/controllers/` to keep it separate from RL components.

---

## 🚀 Training Script: `train.py`

**File:** `ballbot_rl/training/train.py` (277 lines)

**Purpose:** Main training script using Stable-Baselines3 PPO.

### Key Functions

#### `main(config, seed)` - Main Training Loop

**Flow:**

```python
def main(config, seed):
    # 1. Create environments
    # Extract terrain config using safe access pattern
    terrain_config = config.get("problem", {}).get("terrain", {})
    vec_env = VecEnvClass([
        make_ballbot_env(
            terrain_config=terrain_config,
            seed=seed + i
        ) for i in range(N_ENVS)
    ])
    
    # 2. Define policy architecture
    policy_kwargs = dict(
        features_extractor_class=policies.Extractor,
        features_extractor_kwargs={
            "frozen_encoder_path": config["frozen_cnn"]
        },
        net_arch=dict(pi=[128, 128, 128, 128],
                     vf=[128, 128, 128, 128]),
        ...
    )
    
    # 3. Create PPO agent
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,
        n_steps=config["algo"]["n_steps"],
        batch_size=config["algo"]["batch_sz"],
        n_epochs=config["algo"]["n_epochs"],
        ...
    )
    
    # 4. Setup callbacks
    callbacks = [
        CheckpointCallback(...),
        EvalCallback(...)
    ]
    
    # 5. Train
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks
    )
```

**Key Configuration:**

```python
# Learning rate schedule
def lr_schedule(progress_remaining):
    if progress_remaining > 0.7:
        return 1e-4
    elif progress_remaining > 0.5:
        return 5e-5
    else:
        return 1e-5
```

**Callbacks:**
- **CheckpointCallback:** Saves model periodically
- **EvalCallback:** Evaluates policy during training
- **VideoRecorderCallback:** Records videos of best models (async, non-blocking)
- **RenderModeWrapper:** Ensures render_mode compatibility for video recording

---

## 📊 Evaluation Script: `test.py`

**File:** `ballbot_rl/evaluation/evaluate.py`

**Purpose:** Load trained policy and evaluate performance.

**Key Flow:**

```python
def test_policy(model_path, n_test=10):
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = make_ballbot_env(eval_env=True)
    
    # Run episodes
    for episode in range(n_test):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode}: reward={episode_reward}, length={episode_length}")
```

---

## 🛠️ Utilities: `utils.py`

**File:** `ballbot_rl/training/utils.py` (52 lines)

**Purpose:** Utility functions and environment factory.

### `make_ballbot_env()` - Environment Factory

**Purpose:** Creates a Gymnasium environment factory function.

**Implementation:**

```python
def make_ballbot_env(
    terrain_type,
    gui=False,
    disable_cams=False,
    seed=0,
    log_options={"cams": False, "reward_terms": False},
    eval_env=False
):
    def _init():
        env = gym.make(
            "ballbot-v0.1",
            GUI=gui,
            log_options=log_options,
            terrain_type=terrain_type,
            eval_env=[eval_env, seed]
        )
        return Monitor(env)  # Wrapper for logging
    
    return _init
```

**Usage:**

```python
# In train.py
vec_env = SubprocVecEnv([
    make_ballbot_env(terrain_type="perlin", seed=seed + i)
    for i in range(N_ENVS)
])
```

**Why Factory Function:**
- Stable-Baselines3's `VecEnv` requires factory functions (not instances)
- Allows each environment to have different seeds
- Enables proper multiprocessing

---

## ⚙️ Configuration: `train_ppo_directional.yaml`

**File:** `configs/train_ppo_directional.yaml`

**Purpose:** Defines hyperparameters and training configuration.

**Structure:**

```yaml
algo:
  name: ppo
  ent_coef: 0.001          # Entropy coefficient (exploration)
  clip_range: 0.015        # PPO clipping parameter
  target_kl: 0.3           # Target KL divergence
  vf_coef: 2.0             # Value function coefficient
  learning_rate: -1         # Uses scheduler
  n_steps: 2048            # Steps per update
  weight_decay: 0.01       # L2 regularization
  n_epochs: 5              # Update epochs
  batch_sz: 256            # Batch size

problem:
  terrain_type: "perlin"   # Terrain generation type

total_timesteps: 10e6      # Total training steps
frozen_cnn: "../outputs/encoders/encoder_epoch_53"  # Pretrained encoder
hidden_sz: 128             # Hidden layer size
num_envs: 10               # Parallel environments
seed: 10                   # Random seed
```

**Key Parameters:**
- `n_steps`: Number of steps collected before policy update
- `n_epochs`: Number of update epochs per batch
- `batch_sz`: Mini-batch size for updates
- `clip_range`: PPO clipping parameter (prevents large updates)
- `ent_coef`: Entropy bonus (encourages exploration)

---

## 🤖 MuJoCo Model: `bbot.xml`

**File:** `ballbot_gym/assets/bbot.xml`

**Purpose:** Defines the physical robot model in MuJoCo XML format.

**Key Components:**

```xml
<!-- Robot body -->
<body name="robot_body">
    <geom name="body" type="box" size="0.1 0.1 0.2"/>
    <sensor name="imu_acc" type="accelerometer"/>
    <sensor name="imu_gyro" type="gyroscope"/>
</body>

<!-- Three omniwheels -->
<body name="wheel_0">
    <geom name="wheel_mesh_0" type="capsule" size="0.025 0.02"/>
    <joint name="wheel_joint_0" type="hinge" axis="..."/>
</body>
<!-- Similar for wheel_1 and wheel_2 -->

<!-- Ball -->
<body name="ball">
    <geom name="the_ball" type="sphere" size="0.1"/>
</body>

<!-- Contact pairs with anisotropic friction -->
<contact>
    <pair name="non-isotropic0" 
          geom1="the_ball" 
          geom2="wheel_mesh_0" 
          condim="3" 
          friction="0.001 1.0"/>
    <!-- Similar for wheel_1 and wheel_2 -->
</contact>
```

**Key Features:**
- **Anisotropic friction:** `condim="3"` and `friction="0.001 1.0"`
- **Three wheels:** Arranged at 120° angles
- **Sensors:** IMU (accelerometer + gyroscope)
- **Cameras:** Two RGB-D cameras (defined in XML)

---

## 📸 Data Collection: `gather_data.py`

**File:** `ballbot_rl/data/collect.py` (66 lines)

**Purpose:** Collects depth images from the environment for encoder pretraining.

### Key Functions

**Main Function:**

```python
def main(args):
    # Create parallel environments
    vec_env = SubprocVecEnv([
        make_ballbot_env(
            goal_type="fixed_dir",
            seed=args.seed,
            log_options={"cams": True, "reward_terms": False}
        ) for _ in range(num_envs)
    ])
    
    # Load policy (for realistic data collection)
    model = PPO.load(args.policy)
    
    # Collect data
    for step in range(args.n_steps):
        action = model.predict(obs, deterministic=True)
        obs, _, _, _ = vec_env.step(action[0])
```

**What It Does:**

1. **Creates environments:** Multiple parallel environments for efficient data collection
2. **Loads policy:** Uses trained policy to generate realistic trajectories
3. **Collects images:** Saves depth images with logging enabled
4. **Organizes data:** Images saved in structured directories

**Data Structure:**

```
log_0/
  rgbd_log_episode_0/
    depth/
      frame_0000.png
      frame_0001.png
      ...
  rgbd_log_episode_1/
    ...
log_1/
  ...
```

**Usage:**

```bash
python ballbot_rl/data/collect.py \
    --n_steps 10000 \
    --n_envs 10 \
    --policy path/to/trained_policy.zip \
    --seed 42
```

**Why Use Trained Policy?**

- **Realistic data:** Policy generates realistic robot trajectories
- **Diverse:** Different episodes provide varied terrain views
- **Efficient:** Faster than random actions

---

## 🎨 Encoder Pretraining: `pretrain_cnn.py`

**File:** `ballbot_rl/encoders/pretrain.py` (280 lines)

**Purpose:** Trains an autoencoder to compress depth images into feature vectors.

### Key Components

**DepthImageDataset Class:**

```python
class DepthImageDataset(Dataset):
    def __init__(self, image_data_dict):
        self.samples = []
        for log_id, episodes in image_data_dict.items():
            for ep_id, images in episodes.items():
                for img in images:
                    self.samples.append((log_id, ep_id, img))
    
    def __getitem__(self, idx):
        _, _, img = self.samples[idx]
        img = torch.from_numpy(img).float().reshape(1, H, W)
        return img / 255.0  # Normalize to [0, 1]
```

**TinyAutoencoder Class:**

```python
class TinyAutoencoder(nn.Module):
    def __init__(self, H, W, in_c=1, out_sz=20):
        # Encoder: 64×64 → 20-dim
        self.encoder = nn.Sequential(
            Conv2d(1→32, stride=2),  # 32×32
            BatchNorm + LeakyReLU,
            Conv2d(32→32, stride=2),  # 16×16
            BatchNorm + LeakyReLU,
            Flatten,
            Linear(8192 → 20),
            BatchNorm + Tanh
        )
        
        # Decoder: 20-dim → 64×64
        self.decoder = nn.Sequential(
            Linear(20 → 8192),
            Unflatten,
            ConvTranspose2d(32→32, stride=2),
            ConvTranspose2d(32→1, stride=2),
            Sigmoid
        )
```

**Training Function:**

```python
def train_autoencoder(model, train_loader, val_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Training
        for x in train_loader:
            out = model(x)
            loss = F.mse_loss(out, x)  # Reconstruction loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        val_loss = evaluate(model, val_loader)
        
        # Save best encoder
        if val_loss < best_val:
            torch.save(model.encoder, f"encoder_epoch_{epoch}")
```

**Usage:**

```bash
python ballbot_rl/encoders/pretrain.py \
    --data_path data/depth_images \
    --save_encoder_to encoder_frozen \
    --save_dataset_as_pickle data/dataset.pkl
```

**Key Details:**

- **Loss:** Mean Squared Error (MSE) between input and reconstructed image
- **Optimizer:** Adam with learning rate 1e-3
- **Validation:** 20% of data held out
- **Checkpointing:** Saves encoder when validation loss improves
- **Verification:** Saves `p_sum` attribute for integrity checking

**Why Autoencoder?**

- **Compression:** Forces encoder to learn essential features
- **Unsupervised:** Doesn't require labeled data
- **Reconstruction:** Ensures encoder captures important information

**Related Documentation:**
- [Advanced Topics](09_advanced_topics.md), Depth Encoder Pretraining section

---

## 🔄 Code Flow Examples

### Example 1: Single Step Execution

```
1. Agent calls env.step(action)
   ↓
2. bbot_env.py: step()
   - Converts action to motor commands
   - Applies to MuJoCo: data.ctrl[:] = motor_commands
   ↓
3. MuJoCo: mj_step()
   - Computes dynamics
   - Updates state (qpos, qvel)
   ↓
4. bbot_env.py: _get_obs()
   - Extracts orientation, velocities
   - Renders cameras (if enabled)
   - Returns observation dict
   ↓
5. bbot_env.py: _compute_reward()
   - Computes directional reward
   - Adds action regularization
   - Adds survival bonus
   ↓
6. Returns (obs, reward, terminated, truncated, info)
```

### Example 2: Training Initialization

```
1. train.py: main()
   - Loads YAML config
   ↓
2. train.py: Creates environments
   - Calls make_ballbot_env() N times
   - Wraps in VecEnv
   ↓
3. train.py: Creates PPO agent
   - Defines policy_kwargs with Extractor
   - Initializes PPO with config
   ↓
4. train.py: model.learn()
   - Collects rollouts
   - Updates policy
   - Saves checkpoints
```

### Example 3: Observation Processing

```
1. MuJoCo state (data.qpos, data.qvel)
   ↓
2. bbot_env.py: _get_obs()
   - Extracts proprioceptive data
   - Calls rgbd_inputs() for cameras
   ↓
3. RGBDInputs: __call__()
   - Renders RGB images
   - Renders depth images
   - Combines into RGB-D tensor
   ↓
4. policies.py: Extractor.forward()
   - Processes RGB-D with CNN
   - Flattens proprioceptive
   - Concatenates features
   ↓
5. Policy network input
```

---

## 📝 Summary

### Key Takeaways

1. **Environment (`bbot_env.py`):** Core Gymnasium implementation
   - Handles MuJoCo physics
   - Manages observations and rewards
   - Implements episode lifecycle

2. **Rewards (`Rewards.py`):** Simple directional reward
   - Encourages movement in target direction
   - Scaled appropriately for RL

3. **Terrain (`terrain.py`):** Procedural generation
   - Perlin noise for diversity
   - Regenerated each episode

4. **Policy (`policies.py`):** Neural network architecture
   - Multi-modal feature extraction
   - Supports frozen pretrained encoders

5. **Training (`train.py`):** PPO pipeline
   - Parallel environments
   - Checkpointing and evaluation
   - Configurable hyperparameters

### File Dependencies

```
train.py
  ├── utils.py (make_ballbot_env)
  │     └── bbot_env.py
  │           ├── Rewards.py
  │           ├── terrain.py
  │           └── assets/bbot.xml
  └── policies.py
        └── outputs/encoders/encoder_epoch_53
```

---

**Happy Coding! 💻**

*Last Updated: 2025*

