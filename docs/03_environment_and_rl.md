# 🏗️ Environment & RL Workflow: Complete Implementation Guide

*A step-by-step tutorial for setting up, training, and evaluating the OpenBallBot-RL system*

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Step 1: MuJoCo Setup with Anisotropic Friction](#step-1-mujoco-setup-with-anisotropic-friction)
3. [Step 2: Understanding the Environment Architecture](#step-2-understanding-the-environment-architecture)
4. [Step 3: Reward System Deep Dive](#step-3-reward-system-deep-dive)
5. [Step 4: Training PPO Policies](#step-4-training-ppo-policies)
6. [Step 5: Evaluation and Analysis](#step-5-evaluation-and-analysis)
7. [Step 6: Troubleshooting Common Issues](#step-6-troubleshooting-common-issues)
8. [Real-World Example: Complete Training Run](#real-world-example-complete-training-run)
9. [Summary](#summary)

---

## 🎯 Introduction

This tutorial provides a complete walkthrough of the OpenBallBot-RL implementation, from building MuJoCo with the required patches to training and evaluating RL policies. Each step includes explanations of why things are done this way, referencing the research papers that informed the design.

> "Understanding the implementation details helps you appreciate the engineering that makes RL work in practice."  
> — *Common wisdom in RL engineering*

**What You'll Learn:**
- How to build MuJoCo with anisotropic friction support
- The architecture of the Gymnasium environment
- How rewards are computed and logged
- How to train PPO policies effectively
- How to evaluate and visualize results

---

## 🔧 Step 1: MuJoCo Setup with Anisotropic Friction

### Why Anisotropic Friction Matters

The ballbot uses **omniwheels** that have different friction coefficients in different directions:
- **Tangential friction:** Low (allows the ball to roll)
- **Normal friction:** High (prevents slipping)

Standard MuJoCo only supports **isotropic friction** (same in all directions), which breaks the physics. The fix from Zakka et al. (2025) enables anisotropic friction.

**Important Clarification:**
- **The wheels ARE physically modeled** as capsule geometries with hinge joints in `bbot.xml`
- Each wheel has mass, inertia, and can be actuated via motors
- **What's NOT modeled:** The complex internal roller mechanism of real omniwheels (hundreds of individual rollers)
- **Anisotropic friction captures the net effect:** Low tangential friction mimics rollers allowing smooth rolling, while high normal friction prevents slipping—all without modeling every roller
- This is a **physics simplification** that makes simulation tractable while preserving realistic behavior

### Building MuJoCo from Source

#### 1.1 Clone and Checkout

```bash
# Clone MuJoCo repository
git clone https://github.com/deepmind/mujoco.git
cd mujoco

# Checkout recommended commit (optional but recommended)
git checkout 99490163df46f65a0cabcf8efef61b3164faa620
```

#### 1.2 Apply the Anisotropic Friction Patch

```bash
# Copy patch from OpenBallBot-RL repository
cp /path/to/OpenBallBot-RL/mujoco_fix.patch .

# Apply the patch
patch -p1 < mujoco_fix.patch
```

**What the Patch Does:**
- Enables `condim="3"` contact pairs (3D contact with anisotropic friction)
- Allows `friction="0.001 1.0"` syntax (tangential, normal coefficients)
- Fixes contact force computation for omniwheel-ball interaction
- Aligns the contact frame with the capsule axis so friction is applied correctly relative to wheel orientation
- **Key benefit:** Captures omniwheel directional friction behavior without modeling the complex internal roller mechanism

#### 1.3 Build MuJoCo

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/mujoco_install

# Build
cmake --build .

# Install
cmake --install .
```

#### 1.4 Build Python Bindings

```bash
# Navigate to Python bindings directory
cd ../mujoco/python

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Generate source distribution
bash make_sdist.sh

# Install with MuJoCo path
cd dist
export MUJOCO_PATH=/path/to/mujoco_install
export MUJOCO_PLUGIN_PATH=/path/to/mujoco_install/plugin
pip install mujoco-x.y.z.tar.gz  # Replace x.y.z with version
```

**Note for Conda Users:**
```bash
conda install -c conda-forge libstdcxx  # Avoids gxx issues
```

### Verifying the Installation

```python
import mujoco
import mujoco.viewer

# Test anisotropic friction
model = mujoco.MjModel.from_xml_path("ballbot.xml")
data = mujoco.MjData(model)

# Check contact pairs
for i in range(model.npair):
    pair = model.pair(i)
    if pair.condim == 3:
        print(f"Pair {i}: Anisotropic friction enabled")
        print(f"  Friction: {pair.friction}")
```

---

## 🏛️ Step 2: Understanding the Environment Architecture

### Environment Class Structure

The `BBotSimulation` class in `bbot_env.py` implements the Gymnasium interface:

```python
class BBotSimulation(gym.Env):
    def __init__(self, xml_path, ...):
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Define spaces
        self.action_space = Box(-1.0, 1.0, shape=(3,))
        self.observation_space = Dict({...})
        
        # Initialize reward function
        self.reward_obj = Rewards.DirectionalReward(...)
        
        # Terrain generation
        self.terrain_gen = terrain.TerrainGenerator(...)
```

### Key Components

#### 2.1 MuJoCo Model Loading

The XML file (`assets/bbot.xml`) defines:
- **Robot body:** Platform with IMU sensors
- **Three omniwheels:** Arranged at 120° angles
- **Ball:** Free-moving sphere
- **Actuators:** Three motors controlling wheel rotation
- **Contact pairs:** Anisotropic friction between wheels and ball
- **Sensors:** IMU (accelerometer + gyroscope)

#### 2.2 Observation Space Design

```python
observation_space = Dict({
    "orientation": Box(-π, π, shape=(3,)),      # [φ, θ, ψ]
    "angular_vel": Box(-2, 2, shape=(3,)),       # [φ̇, θ̇, ψ̇]
    "vel": Box(-2, 2, shape=(3,)),               # [ẋ, ẏ, ż]
    "motor_state": Box(-2, 2, shape=(3,)),       # Wheel velocities
    "actions": Box(-1, 1, shape=(3,)),           # Previous action
    "rgbd_0": Box(0, 1, shape=(C, H, W)),       # Depth camera 0
    "rgbd_1": Box(0, 1, shape=(C, H, W)),       # Depth camera 1
    "relative_image_timestamp": Box(0, 0.1, shape=(1,))
})
```

**Why Each Component:**
- `orientation` + `angular_vel`: State variables from Lagrangian dynamics
- `vel`: Ball velocity for navigation
- `motor_state`: Actuator feedback (important for underactuated systems)
- `actions`: Action history (helps with partial observability)
- `rgbd_*`: Visual perception (from Salehi 2025)

#### 2.3 Action Space

```python
action_space = Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
```

Actions are normalized to \([-1, 1]\) and scaled internally:

```python
def _action_to_motor_command(self, action):
    # Scale from [-1, 1] to [-10, 10] rad/s
    return action * 10.0
```

**Why Normalization:**
- Neural networks train better with normalized inputs/outputs
- Prevents one motor from dominating
- Makes hyperparameter tuning easier

### Environment Lifecycle

```python
# 1. Reset
obs, info = env.reset(seed=42)

# 2. Step loop
for step in range(max_steps):
    action = agent.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

# 3. Close
env.close()
```

---

## 🎁 Step 3: Reward System Deep Dive

### Reward Components

The reward function combines three terms (see `Rewards.py` and `bbot_env.py`):

#### 3.1 Directional Reward

Encourages movement toward target direction:

```python
class DirectionalReward:
    def __call__(self, state):
        xy_velocity = state["vel"][-3:-1]  # Extract [ẋ, ẏ]
        dir_rew = xy_velocity.dot(self.target_direction)
        return dir_rew / 100.0  # Scale to reasonable values
```

**Mathematical Form:**
\[
r_{\text{dir}} = \frac{\mathbf{v}_{xy} \cdot \mathbf{g}}{100}
\]

Where:
- \(\mathbf{v}_{xy} = [\dot{x}, \dot{y}]^T\) is 2D ball velocity
- \(\mathbf{g} = [g_x, g_y]^T\) is target direction (unit vector)

**Why Divide by 100:**
- Keeps reward magnitudes reasonable (typically \([-0.1, 0.1]\))
- Prevents reward scaling issues in PPO
- Makes hyperparameter tuning easier

#### 3.2 Action Regularization

Penalizes large control inputs:

```python
action_regularization = -0.0001 * np.linalg.norm(action)**2
```

**Mathematical Form:**
\[
r_{\text{reg}} = -0.0001 \|\mathbf{a}\|^2
\]

**Why This Matters:**
- Encourages smooth, energy-efficient control
- Prevents jerky motions that destabilize the robot
- Reduces actuator wear (important for real robots)

#### 3.3 Survival Bonus

Small positive reward for staying upright:

```python
tilt_angle = self._get_tilt_angle()  # Compute from orientation
if tilt_angle <= self.max_allowed_tilt:  # Default: 20 degrees
    reward += 0.02
```

**Mathematical Form:**
\[
r_{\text{surv}} = \begin{cases}
0.02 & \text{if } |\phi| \leq 20° \text{ and } |\theta| \leq 20° \\
0 & \text{otherwise}
\end{cases}
\]

**Why This Matters:**
- Provides positive reward signal even when not moving
- Encourages balance maintenance
- Helps with exploration (doesn't get stuck in low-reward states)

### Total Reward Computation

```python
def compute_reward(self, obs, action):
    # Directional reward
    directional_reward = self.reward_obj(obs) / 100.0
    
    # Action regularization
    action_regularization = -0.0001 * np.linalg.norm(action)**2
    
    # Total reward
    reward = directional_reward + action_regularization
    
    # Add survival bonus if upright
    tilt_angle = self._get_tilt_angle()
    if tilt_angle <= self.max_allowed_tilt:
        reward += 0.02
    
    return reward
```

**Mathematical Summary:**
\[
r_{\text{total}} = \underbrace{\frac{\mathbf{v}_{xy} \cdot \mathbf{g}}{100}}_{\text{progress}} - \underbrace{0.0001\|\mathbf{a}\|^2}_{\text{smoothness}} + \underbrace{0.02 \cdot \mathbb{1}[\text{upright}]}_{\text{stability}}
\]

### Termination Condition

```python
def _check_termination(self):
    tilt_angle = self._get_tilt_angle()
    return tilt_angle > self.max_allowed_tilt  # Default: 20 degrees
```

**Why 20 Degrees:**
- Beyond this angle, recovery becomes very difficult
- Matches safety limits from classical control literature
- Provides hard safety limit (complements soft reward penalties)

---

## 🚀 Step 4: Training PPO Policies

### Configuration File

The training configuration is in `config/train_ppo_directional.yaml`:

```yaml
algo:
  name: ppo
  ent_coef: 0.001          # Entropy coefficient (exploration)
  clip_range: 0.015        # PPO clipping parameter
  target_kl: 0.3           # Target KL divergence
  vf_coef: 2.0             # Value function coefficient
  learning_rate: -1        # Uses learning rate scheduler
  n_steps: 2048            # Steps per environment before update
  weight_decay: 0.01       # L2 regularization
  n_epochs: 5              # Update epochs per batch
  batch_sz: 256            # Batch size
  normalize_advantage: false

problem:
  terrain_type: "perlin"   # Random terrain generation

total_timesteps: 10e6      # Total training steps
frozen_cnn: "../encoder_frozen/encoder_epoch_53"  # Pretrained encoder
hidden_sz: 128             # Hidden layer size
num_envs: 10               # Parallel environments
resume: ""                  # Resume from checkpoint
out: ./log                  # Output directory
seed: 10                    # Random seed
```

### Training Script Overview

The `scripts/train.py` script:

1. **Loads configuration** from YAML file
2. **Creates environment** with specified parameters
3. **Loads pretrained encoder** (if specified)
4. **Creates PPO agent** with Stable-Baselines3
5. **Trains policy** for specified timesteps
6. **Saves checkpoints** periodically
7. **Logs metrics** to CSV files

### Running Training

```bash
cd scripts
python3 train.py --config ../config/train_ppo_directional.yaml
```

**What Happens:**
- Environment instances are created (default: 10 parallel)
- PPO agent collects rollouts from all environments
- After `n_steps=2048` steps, policy is updated for `n_epochs=5` epochs
- Process repeats until `total_timesteps=10e6` reached
- Checkpoints saved to `scripts/log/checkpoints/`
- Progress logged to `scripts/log/progress.csv`

### Monitoring Training

#### Real-Time Progress

The training script prints metrics:
```
| rollout/           |          |
|    ep_len_mean     | 1.23e+03 |
|    ep_rew_mean     | 45.6     |
| time/              |          |
|    fps             | 1234     |
|    iterations      | 100      |
|    time_elapsed    | 1234     |
|    total_timesteps | 204800   |
```

#### Logged Metrics

The `progress.csv` file contains:
- `rollout/ep_len_mean`: Average episode length
- `rollout/ep_rew_mean`: Average episode reward
- `train/learning_rate`: Current learning rate
- `train/policy_loss`: Policy loss
- `train/value_loss`: Value function loss
- `train/entropy_loss`: Entropy (exploration) loss
- `train/approx_kl`: Approximate KL divergence
- `train/clip_fraction`: Fraction of clipped updates

### Visualizing Training Progress

Use the plotting tools:

```bash
python3 ../utils/plotting_tools.py \
    --csv log/progress.csv \
    --config log/config.yaml \
    --plot_train
```

This generates plots showing:
- Episode reward over time
- Episode length over time
- Policy/value losses
- Learning rate schedule

---

## 📊 Step 5: Evaluation and Analysis

### Loading a Trained Policy

```bash
python3 test.py \
    --algo ppo \
    --n_test 10 \
    --path log/checkpoints/ppo_agent_200000_steps.zip
```

**What This Does:**
- Loads the trained policy from checkpoint
- Runs `n_test=10` evaluation episodes
- Uses deterministic actions (no exploration)
- Logs metrics (episode length, reward, etc.)

### Evaluation Metrics

The evaluation script reports:
- **Episode length:** How long the robot stayed balanced
- **Episode reward:** Total reward accumulated
- **Success rate:** Fraction of episodes that didn't fail
- **Average velocity:** Mean speed in target direction

### Comparing to Baseline

The repository includes a PID controller baseline:

```bash
python3 test_pid.py
```

This runs a classical PID controller on flat terrain, providing a comparison point for RL performance.

### Visualizing Results

#### GIF Generation

The training script can generate GIFs of policy rollouts:

```python
# In test.py or custom script
env = BBotSimulation(..., log_options={"cams": True})
# ... run episodes ...
# GIFs saved to images/ directory
```

#### Plotting Tools

```bash
python3 ../utils/plotting_tools.py \
    --csv log/progress.csv \
    --config log/config.yaml \
    --plot_train \
    --plot_eval
```

---

## 🔍 Step 6: Troubleshooting Common Issues

### Issue 1: MuJoCo Build Fails

**Symptoms:** `ImportError: cannot import name 'MjModel'`

**Solutions:**
- Verify `MUJOCO_PATH` is set correctly
- Rebuild Python bindings: `pip uninstall mujoco && pip install mujoco-x.y.z.tar.gz`
- Check that patch was applied: `grep -r "condim" mujoco/src/engine/`

### Issue 2: Anisotropic Friction Not Working

**Symptoms:** Ballbot behaves unrealistically, wheels slip

**Solutions:**
- Verify `mujoco_fix.patch` was applied
- Check XML file has `condim="3"` and `friction="0.001 1.0"`
- Test with simple script to verify friction behavior

### Issue 3: Training Doesn't Converge

**Symptoms:** Reward doesn't increase, policy doesn't improve

**Solutions:**
- Check reward scaling (should be in \([-1, 1]\) range typically)
- Verify observation normalization
- Try different learning rates
- Check that terrain generation is working (not always flat)
- Ensure survival bonus is being applied

### Issue 4: Out of Memory

**Symptoms:** `CUDA out of memory` or system runs out of RAM

**Solutions:**
- Reduce `num_envs` (fewer parallel environments)
- Reduce `n_steps` (smaller rollout batches)
- Disable cameras: `disable_cameras=True`
- Use depth-only: `depth_only=True`

### Issue 5: Policy Doesn't Generalize

**Symptoms:** Works in training but fails in evaluation

**Solutions:**
- Train longer (`total_timesteps` too low)
- Use more diverse terrain (check `terrain_type="perlin"`)
- Reduce overfitting (increase `ent_coef` for more exploration)
- Check that evaluation uses same observation normalization

### Issue 6: Depth Encoder Not Loading

**Symptoms:** `Warning: Parameter sum mismatch` or encoder fails to load

**Solutions:**
- Check encoder file path is correct
- Verify encoder was saved with `p_sum` attribute
- Check device compatibility (CUDA vs CPU)
- Try loading encoder separately to verify integrity

**Related Documentation:**
- [Advanced Topics](09_advanced_topics.md), Depth Encoder Pretraining section

---

## 💻 Real-World Example: Complete Training Run

### Setup Phase

```bash
# 1. Build MuJoCo
cd mujoco
git checkout 99490163df46f65a0cabcf8efef61b3164faa620
patch -p1 < ../OpenBallBot-RL/mujoco_fix.patch
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/mujoco_install
cmake --build . && cmake --install .

# 2. Install Python bindings
cd ../python
python -m venv venv && source venv/bin/activate
bash make_sdist.sh
cd dist
export MUJOCO_PATH=$HOME/mujoco_install
pip install mujoco-*.tar.gz

# 3. Install OpenBallBot-RL
cd ../../OpenBallBot-RL
pip install -r requirements.txt
pip install -e ballbotgym/
```

### Training Phase

```bash
# Start training
cd scripts
python3 train.py --config ../config/train_ppo_directional.yaml

# Monitor progress (in another terminal)
tail -f log/progress.csv

# Generate plots periodically
python3 ../utils/plotting_tools.py \
    --csv log/progress.csv \
    --config log/config.yaml \
    --plot_train
```

### Evaluation Phase

```bash
# Evaluate trained policy
python3 test.py \
    --algo ppo \
    --n_test 20 \
    --path log/checkpoints/ppo_agent_200000_steps.zip

# Compare to PID baseline
python3 test_pid.py
```

### Expected Results

After training for 10M steps:
- **Episode length:** ~2000-4000 steps (depending on terrain)
- **Episode reward:** ~50-100 (depends on movement)
- **Success rate:** >80% (most episodes complete without falling)
- **Average velocity:** ~0.1-0.3 m/s in target direction

---

## 📝 Summary

### Key Takeaways

1. **MuJoCo Setup is Critical**
   - Anisotropic friction patch is essential for realistic physics
   - Follow build instructions carefully
   - Verify installation with test script

2. **Environment Design Reflects Research**
   - Observation space matches Lagrangian state variables
   - Action space matches physical actuators
   - Reward function encodes constraint-aware control philosophy

3. **Training Requires Careful Configuration**
   - PPO hyperparameters matter (clip_range, learning_rate, etc.)
   - Parallel environments speed up training
   - Checkpointing enables resuming long experiments

4. **Evaluation Reveals Policy Quality**
   - Deterministic evaluation shows true performance
   - Compare to baseline (PID controller)
   - Visualize with plots and GIFs

### Complete Workflow Checklist

- [ ] Build MuJoCo with anisotropic friction patch
- [ ] Install Python bindings with correct paths
- [ ] Install OpenBallBot-RL dependencies
- [ ] Verify environment creation works
- [ ] Check reward computation (run test episode)
- [ ] Configure training parameters
- [ ] Start training and monitor progress
- [ ] Evaluate trained policy
- [ ] Visualize results and compare to baseline

### Next Steps

- Experiment with reward coefficients
- Try different terrain types
- Modify observation space (add/remove sensors)
- Implement custom reward functions
- Transfer to real robot (if available)

---

**Happy Training! 🚀**

*Last Updated: 2025*

