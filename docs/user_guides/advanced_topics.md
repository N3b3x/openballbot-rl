# 🔬 Advanced Topics: Deep Dive into Key Components

*In-depth analysis of advanced concepts and implementation details*

---

## 📋 Table of Contents

1. [Depth Encoder Pretraining](#depth-encoder-pretraining)
2. [MuJoCo Anisotropic Friction Patch: Technical Details](#mujoco-anisotropic-friction-patch-technical-details)
3. [Terrain Generation and Robot Placement](#terrain-generation-and-robot-placement)
4. [Camera Rendering and RGB-D Processing](#camera-rendering-and-rgb-d-processing)
5. [Observation Normalization and Preprocessing](#observation-normalization-and-preprocessing)
6. [Reward Component Analysis](#reward-component-analysis)
7. [Training Callbacks and Evaluation](#training-callbacks-and-evaluation)
8. [Neural Network Architecture Details](#neural-network-architecture-details)

---

## 🎨 Depth Encoder Pretraining

### Why Pretrain the Depth Encoder?

**Problem:** Training a CNN encoder from scratch during RL is computationally expensive and unstable.

**Solution:** Pretrain the encoder using an **autoencoder** approach on depth images collected from the environment, then freeze it during RL training.

**Benefits:**
- **Faster RL training:** Encoder weights don't change, reducing computation
- **Stable learning:** Encoder learns useful features before RL starts
- **Better features:** Autoencoder learns to compress depth information efficiently

### Autoencoder Architecture

The pretraining uses a **TinyAutoencoder** with encoder-decoder structure:

**Encoder (Compression):**
```
Input: Depth image (64×64×1)
  ↓
Conv2d(1→32, kernel=3, stride=2) + BatchNorm + LeakyReLU
  ↓ (32×32×32)
Conv2d(32→32, kernel=3, stride=2) + BatchNorm + LeakyReLU
  ↓ (16×16×32)
Flatten → Linear(8192 → 20) + BatchNorm + Tanh
  ↓
Output: Feature vector (20-dim)
```

**Decoder (Reconstruction):**
```
Input: Feature vector (20-dim)
  ↓
Linear(20 → 8192) + BatchNorm + LeakyReLU
  ↓
Unflatten → (32×16×16)
ConvTranspose2d(32→32, kernel=3, stride=2) + BatchNorm + LeakyReLU
  ↓ (32×32×32)
ConvTranspose2d(32→1, kernel=3, stride=2) + Sigmoid
  ↓
Output: Reconstructed depth image (64×64×1)
```

**Key Design Choices:**
- **Small bottleneck (20-dim):** Forces compression, learning essential features
- **Tanh activation:** Keeps features in [-1, 1] range
- **BatchNorm:** Stabilizes training
- **LeakyReLU:** Prevents dead neurons

### Pretraining Process

**Step 1: Data Collection**

```python
# Run gather_data.py to collect depth images
python ballbot_rl/data/collect.py \
    --num_episodes 100 \
    --log_dir data/depth_images
```

This script:
- Runs the ballbot environment with random actions
- Saves depth images from cameras
- Organizes images by episode and log directory

**Step 2: Dataset Preparation**

```python
# In pretrain_cnn.py
image_paths = collect_depth_image_paths(root_directory)
image_data = load_depth_images(image_paths)
dataset = DepthImageDataset(image_data)
```

The dataset structure:
```
{
    'log_0': {
        0: [depth_image_0, depth_image_1, ...],
        1: [depth_image_0, depth_image_1, ...],
        ...
    },
    'log_1': {...},
    ...
}
```

**Step 3: Training**

```python
# Train autoencoder
train_autoencoder(
    model=TinyAutoencoder(H=64, W=64),
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    lr=1e-3
)
```

**Training Details:**
- **Loss:** Mean Squared Error (MSE) between input and reconstructed image
- **Optimizer:** Adam with learning rate 1e-3
- **Validation:** 20% of data held out
- **Checkpointing:** Saves encoder when validation loss improves

**Mathematical Formulation:**

The autoencoder learns to minimize:
\[
\mathcal{L} = \frac{1}{N}\sum_{i=1}^N \|\mathbf{x}_i - \text{decoder}(\text{encoder}(\mathbf{x}_i))\|^2
\]

Where:
- \(\mathbf{x}_i\) is a depth image
- \(\text{encoder}(\cdot)\) compresses to 20-dim feature vector
- \(\text{decoder}(\cdot)\) reconstructs the image

**Why This Works:**

1. **Compression forces learning:** The 20-dim bottleneck forces the encoder to learn essential features
2. **Reconstruction ensures quality:** If the decoder can reconstruct, the encoder captured important information
3. **Depth-specific:** Training on depth images ensures features are relevant for terrain perception

### Using Pretrained Encoder in RL

**Loading the Encoder:**

```python
# In policies.py Extractor class
if frozen_encoder_path:
    extractors[key] = torch.load(frozen_encoder_path)
    # Freeze parameters
    for param in extractors[key].parameters():
        param.requires_grad = False
```

**Why Freeze?**

- **Stability:** Prevents encoder from forgetting useful features
- **Efficiency:** Fewer parameters to update during RL
- **Focus:** RL can focus on learning policy, not perception

**Verification:**

The encoder saves a `p_sum` attribute (sum of parameter absolute values) for verification:

```python
p_sum = sum([param.abs().sum().item() 
             for param in encoder.parameters()])
assert abs(p_sum - encoder.p_sum) < tolerance
```

This ensures the encoder wasn't corrupted during loading.

---

## 🔧 MuJoCo Anisotropic Friction Patch: Technical Details

### The Problem

**Standard MuJoCo Contact Model:**

MuJoCo's contact model assumes **isotropic friction** (same in all directions). For sphere-capsule contacts (ball-wheel), it computes:

1. Contact normal (perpendicular to surfaces)
2. Contact frame (tangent plane)
3. Friction forces (same magnitude in all tangent directions)

**Why This Breaks Omniwheels:**

Omniwheels have **directional friction:**
- **Low friction** along rolling direction (rollers allow smooth motion)
- **High friction** perpendicular to rolling direction (prevents slipping)

Standard MuJoCo applies the same friction in all tangent directions, breaking the physics.

### The Solution: Patch Details

**File:** `mujoco_fix.patch`

**Location:** `src/engine/engine_collision_primitive.c`

**Function:** `mjraw_SphereCapsule()` - Handles sphere-capsule collision detection

**Original Code:**
```c
int mjraw_SphereCapsule(mjContact* con, mjtNum margin, ...) {
    // ... find nearest point on segment ...
    mju_scl3(vec, axis, x);
    mju_addTo3(vec, pos2);
    return mjraw_SphereSphere(con, margin, pos1, mat1, size1, vec, mat2, size2);
}
```

**Patched Code:**
```c
int mjraw_SphereCapsule(mjContact* con, mjtNum margin, ...) {
    // ... find nearest point on segment ...
    mju_scl3(vec, axis, x);
    mju_addTo3(vec, pos2);
    
    int ncon = mjraw_SphereSphere(con, margin, pos1, mat1, size1, vec, mat2, size2);
    
    // align contact frame second axis with capsule
    if (ncon) {
        mju_copy3(con->frame+3, axis);
    }
    
    return ncon;
}
```

**What the Patch Does:**

1. **Computes contact normally:** Calls `mjraw_SphereSphere()` to find contact point
2. **Aligns contact frame:** Sets the second axis of the contact frame (`con->frame+3`) to the capsule axis
3. **Enables anisotropic friction:** With aligned frame, MuJoCo can apply different friction coefficients along different axes

**Contact Frame Structure:**

The contact frame `con->frame` is a 9-element array:
- `frame[0:3]`: First tangent axis (default: arbitrary)
- `frame[3:6]`: Second tangent axis (now aligned with capsule axis)
- `frame[6:9]`: Normal axis (perpendicular to contact)

**Why This Works:**

When `condim="3"` is specified in XML:
- MuJoCo uses a 3D contact model
- Friction coefficients `friction="0.001 1.0"` are interpreted as:
  - First value (0.001): Friction along first tangent axis
  - Second value (1.0): Friction along second tangent axis (aligned with capsule)
- The patch ensures the second axis aligns with the wheel's rolling direction

**XML Configuration:**

```xml
<contact>
    <pair name="non-isotropic0" 
          geom1="the_ball" 
          geom2="wheel_mesh_0" 
          condim="3" 
          friction="0.001 1.0"/>
</contact>
```

- `condim="3"`: Enables 3D contact with anisotropic friction
- `friction="0.001 1.0"`: Low tangential (0.001), high normal (1.0)

### Physical Interpretation

**What IS Modeled:**
- Wheel geometry (capsule shape)
- Wheel mass and inertia
- Hinge joints (wheel rotation)
- Contact forces (normal and friction)

**What's NOT Modeled:**
- Individual rollers inside omniwheel
- Roller rotation dynamics
- Multi-body contact between rollers and ball

**How Anisotropic Friction Captures Omniwheel Behavior:**

The net effect of omniwheel rollers is:
- **Low resistance** to motion along the wheel's axis (rollers roll)
- **High resistance** to motion perpendicular to the axis (rollers don't roll)

Anisotropic friction with `friction="0.001 1.0"` captures this **net effect** without modeling individual rollers.

**Computational Benefits:**

- **Without patch:** Would need to model hundreds of rollers (computationally expensive)
- **With patch:** Single contact pair with directional friction (efficient)

---

## 🌍 Terrain Generation and Robot Placement

### Perlin Noise Terrain Generation

**Algorithm:** Uses Simplex noise (via `snoise2` from `noise` library)

**Parameters:**
- `n`: Grid size (must be odd, typically 129)
- `scale`: Feature size (higher = smoother terrain, default: 25.0)
- `octaves`: Number of noise layers (more = more detail, default: 4)
- `persistence`: Amplitude scaling between octaves (default: 0.2)
- `lacunarity`: Frequency scaling between octaves (default: 2.0)
- `seed`: Random seed for reproducibility

**Generation Process:**

```python
def generate_perlin_terrain(n, scale=25.0, octaves=4, seed=0):
    terrain = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            x = i / scale  # Normalize coordinates
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
    
    return terrain.flatten()  # Row-major for MuJoCo
```

**Why Perlin Noise?**

- **Smooth:** No sharp discontinuities (realistic terrain)
- **Multiscale:** Multiple octaves create natural variation
- **Controllable:** Parameters allow tuning terrain difficulty
- **Reproducible:** Fixed seed gives same terrain

**Terrain Difficulty:**

- **Low `scale` (e.g., 10):** High-frequency variation (rough terrain)
- **High `scale` (e.g., 50):** Low-frequency variation (smooth terrain)
- **More `octaves`:** More detail at different scales

### Robot Placement on Terrain

**Challenge:** Place robot on terrain surface without embedding it.

**Process:**

1. **Generate terrain heightfield**
2. **Compute ball's axis-aligned bounding box (AABB)**
3. **Sample terrain heights under ball**
4. **Set robot height to max terrain height + epsilon**

**Code Flow:**

```python
def _reset_terrain(self):
    # 1. Generate terrain
    hfield_data = terrain.generate_perlin_terrain(nrows, seed=r_seed)
    self.model.hfield_data = hfield_data
    
    # 2. Get ball position and size
    ball_pos = self.data.geom_xpos[ball_id]  # [x, y, z]
    ball_size = self.model.geom_size[ball_id]  # radius
    
    # 3. Compute AABB
    aabb_min = ball_pos - ball_size[0]
    aabb_max = ball_pos + ball_size[0]
    
    # 4. Convert to terrain cell indices
    cell_size = sz / nrows
    center_idx = nrows // 2
    
    # 5. Extract terrain region under ball
    terrain_mat = hfield_data.reshape(nrows, ncols)
    sub_terr = terrain_mat[x_min_idx:x_max_idx, y_min_idx:y_max_idx]
    
    # 6. Compute initial height
    eps = 0.01  # 1cm safety margin
    init_height = sub_terr.max() * hfield_height_coef + eps
    
    return init_height
```

**Key Assumptions:**

- Ball is centered at origin `(0, 0)` in XY plane
- Terrain is square (`nrows == ncols`)
- Terrain size matches heightfield dimensions

**Why This Matters:**

- **Realistic initialization:** Robot starts on terrain, not floating or embedded
- **Diverse training:** Different terrain each episode forces generalization
- **Safety:** Epsilon prevents collision detection issues

---

## 📷 Camera Rendering and RGB-D Processing

### Camera Configuration

**Two RGB-D Cameras:**
- `cam_0`: Front-facing camera
- `cam_1`: Side-facing camera (or different angle)

**Resolution:** 64×64 pixels (configurable via `im_shape`)

**Channels:**
- **RGB mode:** 4 channels (R, G, B, Depth)
- **Depth-only mode:** 1 channel (Depth only)

### Rendering Process

**RGBDInputs Class:**

```python
class RGBDInputs:
    def __init__(self, mjc_model, height, width, cams, disable_rgb):
        # Initialize RGB renderer (if enabled)
        self._renderer_rgb = mujoco.Renderer(...) if not disable_rgb else None
        
        # Always initialize depth renderer
        self._renderer_d = mujoco.Renderer(...)
        self._renderer_d.enable_depth_rendering()
    
    def __call__(self, data, cam_name):
        # Render RGB (if enabled)
        if self._renderer_rgb is not None:
            self._renderer_rgb.update_scene(data, camera=cam_name)
            rgb = self._renderer_rgb.render() / 255.0  # Normalize to [0, 1]
        
        # Render depth
        self._renderer_d.update_scene(data, camera=cam_name)
        depth = self._renderer_d.render()  # Raw depth values
        
        # Clip extreme values (sky, background)
        depth[depth >= 1.0] = 1.0
        
        # Combine channels
        if self._renderer_rgb is not None:
            return np.concatenate([rgb, depth], axis=-1)  # (H, W, 4)
        else:
            return depth  # (H, W, 1)
```

**Key Details:**

1. **Scene Update:** `update_scene()` updates renderer with current simulation state
2. **RGB Normalization:** Divides by 255 to get [0, 1] range
3. **Depth Clipping:** Clips values ≥ 1.0 (sky, background) to prevent outliers
4. **Channel Concatenation:** RGB and depth combined into single tensor

### Camera Frame Rate

**Physics Timestep:** ~0.002s (500 Hz)

**Camera Frame Rate:** 90 Hz (configurable via `camera_frame_rate`)

**Effective Rate:** Computed to align with physics steps:

```python
def effective_camera_frame_rate(self):
    dt_mj = self.opt_timestep  # 0.002s
    desired_cam_dt = 1 / self.camera_frame_rate  # 1/90s
    
    # Find number of physics steps needed
    N = np.ceil(desired_cam_dt / dt_mj)
    
    # Effective rate
    return 1.0 / (N * dt_mj)
```

**Why Lower Frame Rate?**

- **Computational efficiency:** Rendering is expensive
- **Sufficient for navigation:** 90 Hz captures terrain changes adequately
- **Reduces memory:** Fewer images to store and process

### Depth Image Processing

**Raw Depth Values:**
- Distance from camera to surface (in meters)
- Larger values = farther objects
- Clipped at 1.0 (background/sky)

**Normalization:**
- Depth images normalized to [0, 1] range
- Encoder expects normalized inputs

**Why Depth-Only Mode?**

- **Memory efficient:** 1 channel vs 4 channels
- **Sufficient for terrain:** Depth captures terrain shape
- **Faster training:** Less data to process

---

## 📊 Observation Normalization and Preprocessing

### Observation Components

**Proprioceptive Observations:**
- `orientation`: Euler angles `[φ, θ, ψ]` ∈ `[-π, π]`
- `angular_vel`: Angular velocities `[φ̇, θ̇, ψ̇]` ∈ `[-2, 2]` rad/s
- `vel`: Linear velocities `[ẋ, ẏ, ż]` ∈ `[-2, 2]` m/s
- `motor_state`: Wheel velocities ∈ `[-2, 2]` rad/s
- `actions`: Previous action ∈ `[-1, 1]`

**Visual Observations:**
- `rgbd_0`: RGB-D image from camera 0 ∈ `[0, 1]`
- `rgbd_1`: RGB-D image from camera 1 ∈ `[0, 1]`

### Normalization Strategy

**Why Normalize?**

- **Neural network stability:** Networks train better with normalized inputs
- **Gradient flow:** Prevents vanishing/exploding gradients
- **Feature scaling:** Ensures all features contribute equally

**Normalization Methods:**

1. **Hard bounds:** Observations clipped to specified ranges
2. **Feature scaling:** Some observations scaled by constants
3. **Batch normalization:** Applied in neural network layers

**Example: Orientation Extraction**

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

**Why Euler Angles?**

- **Interpretable:** Direct physical meaning (roll, pitch, yaw)
- **Bounded:** Natural bounds `[-π, π]`
- **Compatible:** Works well with neural networks

**Velocity Normalization:**

Velocities are clipped to `[-2, 2]` range:
- Prevents extreme values from dominating
- Assumes reasonable operating speeds
- Can be adjusted based on robot capabilities

---

## 🎁 Reward Component Analysis

### Component Breakdown

**Total Reward:**
\[
r = r_{\text{dir}} + r_{\text{reg}} + r_{\text{surv}}
\]

### 1. Directional Reward

**Formula:**
\[
r_{\text{dir}} = \frac{\mathbf{v}_{xy} \cdot \mathbf{g}}{100}
\]

**Analysis:**

- **Purpose:** Encourages movement toward target direction
- **Scaling:** Division by 100 keeps rewards in reasonable range
- **Range:** Typically `[-0.1, 0.1]` for normal velocities
- **Gradient:** Smooth, continuous function

**Why Dot Product?**

- **Alignment:** Maximum when velocity aligns with target
- **Magnitude:** Larger velocities give larger rewards
- **Direction:** Negative reward for moving away

**Scaling Factor (100):**

- **Empirical choice:** Balances with other reward components
- **Too large:** Dominates other terms, unstable learning
- **Too small:** Insufficient signal, slow learning

### 2. Action Regularization

**Formula:**
\[
r_{\text{reg}} = -0.0001 \|\mathbf{a}\|^2
\]

**Analysis:**

- **Purpose:** Encourages smooth, energy-efficient control
- **Magnitude:** Small coefficient (0.0001) prevents domination
- **Effect:** Quadratic penalty on action magnitude

**Why L2 Norm?**

- **Smooth:** Differentiable everywhere
- **Penalizes large actions:** Quadratic grows faster than linear
- **Energy interpretation:** Related to control effort

**Coefficient Choice (0.0001):**

- **Small enough:** Doesn't prevent necessary large actions
- **Large enough:** Discourages unnecessary large actions
- **Balanced:** Works well with other components

### 3. Survival Bonus

**Formula:**
\[
r_{\text{surv}} = \begin{cases}
0.02 & \text{if } |\phi| \leq 20° \text{ and } |\theta| \leq 20° \\
0 & \text{otherwise}
\end{cases}
\]

**Analysis:**

- **Purpose:** Provides positive reward signal for staying upright
- **Magnitude:** Small (0.02) but consistent
- **Effect:** Accumulates over time (2000 steps = 40 reward)

**Why This Matters:**

- **Exploration:** Prevents agent from getting stuck in low-reward states
- **Stability:** Encourages balance maintenance
- **Baseline:** Provides positive signal even when not moving

**Tilt Threshold (20°):**

- **Safety limit:** Beyond this, recovery is very difficult
- **Matches termination:** Same threshold as episode termination
- **Realistic:** Based on physical constraints

### Reward Interaction

**Component Weights:**

- **Directional:** ~0.1 per step (when moving)
- **Regularization:** ~-0.0001 per step (small)
- **Survival:** +0.02 per step (consistent)

**Total per Step:** Typically `[0.01, 0.15]` range

**Episode Reward:** ~50-100 for successful episodes (2000-4000 steps)

**Why This Works:**

1. **Directional reward** provides learning signal for navigation
2. **Action regularization** prevents unsafe control
3. **Survival bonus** ensures positive signal for balance

**Potential Issues:**

- **Reward hacking:** Agent might exploit reward structure
- **Scaling:** Components must be balanced
- **Sparse rewards:** If survival bonus dominates, navigation might be ignored

---

## 📈 Training Callbacks and Evaluation

### Callback System

**Stable-Baselines3 Callbacks:**

1. **CheckpointCallback:** Saves model periodically
2. **EvalCallback:** Evaluates policy during training

**CheckpointCallback:**

```python
CheckpointCallback(
    save_freq=10000,  # Save every 10k steps
    save_path=f'{config["out"]}/checkpoints',
    name_prefix='ppo_agent'
)
```

**Purpose:**
- **Resume training:** Can restart from checkpoints
- **Model selection:** Choose best checkpoint
- **Backup:** Prevents loss of progress

**EvalCallback:**

```python
EvalCallback(
    eval_env=eval_env,
    best_model_save_path=f'{config["out"]}/best_model',
    log_path=f"{config['out']}/results/",
    eval_freq=5000,  # Evaluate every 5k steps
    n_eval_episodes=8,
    deterministic=True
)
```

**Purpose:**
- **Monitor progress:** Track policy performance
- **Early stopping:** Can stop if performance plateaus
- **Best model:** Saves best performing policy

**VideoRecorderCallback:**

The project includes a custom `VideoRecorderCallback` that automatically records videos when new best models are found:

```python
from ballbot_rl.training.callbacks import VideoRecorderCallback

VideoRecorderCallback(
    eval_env=eval_env,
    video_folder=f'{config["out"]}/videos',
    video_length=4000,  # Max steps per video
    name_prefix="best_model",
    async_recording=True  # Non-blocking
)
```

**Functionality:**
- **Automatic video recording:** Records videos when new best models are found
- **Asynchronous recording:** Videos recorded in background thread (doesn't slow training)
- **RenderModeWrapper:** Handles render_mode compatibility for VecVideoRecorder

**Configuration:**
Video recording is configured in `configs/train/ppo_directional.yaml`:
```yaml
visualization:
  record_videos: true
  video_freq: "on_new_best"  # or "every_eval" or integer N
  video_episodes: 1
```

### Evaluation Metrics

**Logged Metrics:**

- **Episode reward:** Total reward per episode
- **Episode length:** Steps per episode
- **Success rate:** Fraction of episodes that don't fail
- **Average velocity:** Mean speed in target direction

**Evaluation Process:**

1. **Deterministic actions:** No exploration during evaluation
2. **Fixed seeds:** Reproducible evaluation
3. **Multiple episodes:** Average over multiple runs

**Why Deterministic?**

- **True performance:** Shows what policy actually does
- **Reproducible:** Same actions given same observations
- **Fair comparison:** Consistent evaluation across runs

---

## 🧠 Neural Network Architecture Details

### Policy Network

**Architecture:**

```
Input: Multi-modal observations
  ↓
Feature Extractor (Extractor class)
  ├─ Proprioceptive: Flatten → Concatenate
  └─ RGB-D: Frozen CNN Encoder → 20-dim
  ↓
Concatenated Features (N-dim)
  ↓
Policy Head (MLP)
  ├─ Hidden: [128, 128, 128, 128]
  ├─ Activation: LeakyReLU
  └─ Output: Gaussian distribution (mean, std)
```

**Value Network:**

```
Input: Same multi-modal observations
  ↓
Feature Extractor (shared with policy)
  ↓
Concatenated Features (N-dim)
  ↓
Value Head (MLP)
  ├─ Hidden: [128, 128, 128, 128]
  ├─ Activation: LeakyReLU
  └─ Output: Scalar value estimate
```

**Key Design Choices:**

1. **Shared feature extractor:** Policy and value share encoder (efficiency)
2. **Separate heads:** Policy and value have separate MLPs (specialization)
3. **LeakyReLU:** Prevents dead neurons
4. **BatchNorm:** Stabilizes training (in encoder)

**Why This Architecture?**

- **Multi-modal:** Handles both proprioceptive and visual inputs
- **Efficient:** Shared encoder reduces computation
- **Flexible:** Separate heads allow different representations

---

**Happy Exploring! 🔬**

*Last Updated: 2025*

