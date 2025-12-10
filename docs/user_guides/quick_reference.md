# ⚡ Quick Reference Card

*Essential formulas, commands, and information at a glance*

---

## 🔢 Key Formulas

### Lagrangian Dynamics
\[
L = T - V = \frac{1}{2}\dot{\mathbf{q}}^T \mathbf{M}(\mathbf{q}) \dot{\mathbf{q}} - mgh(\mathbf{q})
\]

\[
\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{G}(\mathbf{q}) = \boldsymbol{\tau}
\]

### Reward Function
\[
r = \frac{\mathbf{v}_{xy} \cdot \mathbf{g}}{100} - 0.0001\|\mathbf{a}\|^2 + 0.02 \cdot \mathbb{1}[\text{upright}]
\]

### PPO Objective
\[
L(\theta) = \mathbb{E}_t\left[\min\left(
r_t(\theta) \hat{A}_t,
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t
\right)\right]
\]

---

## 📊 Observation Space

```python
Dict({
    "orientation": Box(-π, π, shape=(3,)),      # [φ, θ, ψ]
    "angular_vel": Box(-2, 2, shape=(3,)),      # [φ̇, θ̇, ψ̇]
    "vel": Box(-2, 2, shape=(3,)),              # [ẋ, ẏ, ż]
    "motor_state": Box(-2, 2, shape=(3,)),      # Wheel velocities
    "actions": Box(-1, 1, shape=(3,)),           # Previous action
    "rgbd_0": Box(0, 1, shape=(C, H, W)),       # Depth camera 0
    "rgbd_1": Box(0, 1, shape=(C, H, W)),       # Depth camera 1
})
```

---

## 🎮 Action Space

```python
Box(-1.0, 1.0, shape=(3,), dtype=float32)
# Scaled internally to [-10, 10] rad/s
```

---

## 🚀 Common Commands

### Build MuJoCo
```bash
cd mujoco
git checkout 99490163df46f65a0cabcf8efef61b3164faa620
patch -p1 < ../openballbot-rl/tools/mujoco_fix.patch
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/mujoco_install
cmake --build . && cmake --install .
```

### Install Python Bindings
```bash
cd mujoco/python
python -m venv venv && source venv/bin/activate
bash make_sdist.sh
cd dist
export MUJOCO_PATH=/path/to/mujoco_install
pip install mujoco-*.tar.gz
```

### Train Policy
```bash
cd scripts
python train.py --config ../configs/train_ppo_directional.yaml
```

### Evaluate Policy
```bash
cd scripts
python test.py \
    --algo ppo \
    --n_test 10 \
    --path log/checkpoints/ppo_agent_200000_steps.zip
```

### Plot Training Progress
```bash
# Using CLI command (recommended)
ballbot-plot-training \
    --csv outputs/experiments/runs/.../progress.csv \
    --config outputs/experiments/runs/.../config.yaml \
    --plot_train

# Or using Python module
python -m ballbot_rl.visualization.plot_training \
    --csv outputs/experiments/runs/.../progress.csv \
    --config outputs/experiments/runs/.../config.yaml \
    --plot_train
```

---

## ⚙️ Key Hyperparameters

### PPO Defaults
- `n_steps`: 2048 (steps per update)
- `n_epochs`: 5 (update epochs)
- `batch_sz`: 256 (batch size)
- `clip_range`: 0.015 (PPO clipping)
- `ent_coef`: 0.001 (entropy bonus)
- `learning_rate`: Scheduled (1e-4 → 5e-5 → 1e-5)

### Environment Defaults
- `num_envs`: 10 (parallel environments)
- `max_ep_steps`: 4000 (max episode length)
- `terrain_type`: "perlin" (terrain generation)
- `im_shape`: {"h": 64, "w": 64} (camera resolution)

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `bbot_env.py` | Main Gymnasium environment |
| `Rewards.py` | Reward function definitions |
| `terrain.py` | Terrain generation |
| `policies.py` | Neural network architectures |
| `train.py` | Training script |
| `test.py` | Evaluation script |
| `utils.py` | Environment factory |
| `bbot.xml` | MuJoCo robot model |
| `mujoco_fix.patch` | Anisotropic friction patch |

---

## 🔑 Key Concepts

### Anisotropic Friction
- **Low tangential:** 0.001 (allows rolling)
- **High normal:** 1.0 (prevents slipping)
- **Enables:** Realistic omniwheel simulation

### Reward Components
1. **Directional:** Velocity in target direction / 100
2. **Action regularization:** -0.0001 × ||action||²
3. **Survival bonus:** +0.02 if upright

### Termination
- **Tilt angle:** > 20° (robot falls)
- **Max steps:** 4000 (episode timeout)

---

## 📚 Research Papers

| Paper | Key Contribution |
|-------|------------------|
| Lauwers 2006 | Inverse mouse-ball drive mechanism |
| Nagarajan 2014 | Lagrangian dynamics derivation |
| Carius 2022 | Constraint-aware control theory |
| Salehi 2025 | Complete RL navigation system |
| Zakka 2025 | MuJoCo anisotropic friction fix |

---

## 🐍 Python Snippets

### Create Environment
```python
from utils import make_ballbot_env
import gymnasium as gym

env = gym.make(
    "ballbot-v0.1",
    terrain_type="perlin",
    disable_cameras=False
)
obs, info = env.reset()
```

### Run Episode
```python
for step in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### Load Trained Policy
```python
from stable_baselines3 import PPO

model = PPO.load("path/to/model.zip")
action, _ = model.predict(obs, deterministic=True)
```

---

## 🔍 Debugging Tips

### Check MuJoCo Installation
```python
import mujoco
model = mujoco.MjModel.from_xml_path("bbot.xml")
print("MuJoCo loaded successfully!")
```

### Verify Anisotropic Friction
```python
# Check contact pairs
for i in range(model.npair):
    pair = model.pair(i)
    if pair.condim == 3:
        print(f"Pair {i}: Anisotropic friction enabled")
```

### Monitor Reward Components
```python
env.log_options = {"reward_terms": True}
# Run episode and check logged components
```

---

## 📏 Physical Constants

- **Gravity:** \(g = 9.81\) m/s²
- **Max tilt:** 20° (safety limit)
- **Ball radius:** 0.1 m (from `bbot.xml`)
- **Wheel radius:** 0.025 m (from `bbot.xml`)
- **Control frequency:** ~1000 Hz (MuJoCo timestep)

---

## 🎯 Training Checklist

- [ ] MuJoCo built with anisotropic friction patch
- [ ] Python bindings installed correctly
- [ ] Environment creates successfully
- [ ] Reward computation works
- [ ] Configuration file set up
- [ ] Training script runs
- [ ] Checkpoints saving
- [ ] Logs being written
- [ ] Evaluation script works

---

## 🔗 Quick Links

- [Research Timeline](01_research_timeline.md)
- [Mechanics to RL Guide](02_mechanics_to_rl.md)
- [Environment & RL Workflow](03_environment_and_rl.md)
- [Visual Diagrams](04_visual_diagrams.md)
- [Code Walkthrough](05_code_walkthrough.md)
- [Glossary](06_glossary.md)
- [FAQ](07_faq.md)
- [Advanced Topics](09_advanced_topics.md)

---

**Print this page for quick reference! 📄**

*Last Updated: 2025*

