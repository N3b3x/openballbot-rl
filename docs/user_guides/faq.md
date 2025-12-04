# ❓ Frequently Asked Questions (FAQ)

*Common questions and detailed answers about openballbot-rl*

---

## 📋 Table of Contents

1. [General Questions](#general-questions)
2. [Physics & Simulation](#physics--simulation)
3. [Reinforcement Learning](#reinforcement-learning)
4. [Implementation](#implementation)
5. [Training & Evaluation](#training--evaluation)
6. [Troubleshooting](#troubleshooting)

---

## 🌐 General Questions

### Q1: What is a ballbot?

**A:** A ballbot is a dynamically balanced mobile robot that balances on a single spherical ball. It uses three omniwheels arranged at 120° angles to control the ball's motion. Unlike traditional robots with static bases, the ballbot must actively maintain balance through control, making it an underactuated system.

**Key Characteristics:**
- Balances on a single point (the ball)
- Uses inverse mouse-ball drive mechanism
- Can move omnidirectionally
- Requires active control to maintain balance

**Related Documentation:**
- [Research Timeline](01_research_timeline.md), Section 2006
- [Mechanics to RL Guide](02_mechanics_to_rl.md), Part 1

---

### Q2: Why use reinforcement learning instead of classical control?

**A:** Classical control (e.g., PID controllers) works well for flat terrain but struggles with:
- **Uneven terrain:** Hand-tuning controllers for every terrain variation is impractical
- **Generalization:** Classical controllers don't adapt to new environments
- **Complex dynamics:** The underactuated nature makes analytical control design difficult

RL learns policies that:
- **Generalize** to unseen terrain (trained on random Perlin noise)
- **Adapt** to different conditions automatically
- **Handle complexity** through learning rather than hand-tuning

**Trade-offs:**
- RL requires more training time
- Classical control is more interpretable
- RL can discover novel behaviors

**Related Documentation:**
- [Research Timeline](01_research_timeline.md), Section 2025
- [Environment & RL Workflow](03_environment_and_rl.md), Step 4

---

### Q3: What makes this project different from other ballbot implementations?

**A:** This project combines several key innovations:

1. **RL-based navigation:** Uses PPO to learn navigation policies (not just stabilization)
2. **Uneven terrain:** Trains on procedurally generated Perlin noise terrain
3. **Visual perception:** Uses RGB-D cameras for terrain perception
4. **Anisotropic friction:** Properly models omniwheel physics via MuJoCo patch
5. **Constraint-aware rewards:** Encodes safety constraints in the reward function

**Previous work focused on:**
- Classical control for flat terrain
- Hand-tuned controllers
- Proprioceptive-only sensing

**Related Documentation:**
- [Research Timeline](01_research_timeline.md), Section 2025
- [Mechanics to RL Guide](02_mechanics_to_rl.md), Part 3

---

## ⚙️ Physics & Simulation

### Q4: Why is anisotropic friction necessary?

**A:** Omniwheels have different friction coefficients in different directions:
- **Low tangential friction:** Allows the ball to roll smoothly along the wheel's axis (like the rollers on an omniwheel)
- **High normal friction:** Prevents slipping perpendicular to the rolling direction

Standard MuJoCo only supports **isotropic friction** (same in all directions), which breaks the physics. Without anisotropic friction:
- The ballbot simulation doesn't match real-world behavior
- The RL agent learns unrealistic control strategies
- Sim-to-real transfer fails

**The patch enables:**
- `condim="3"` contact pairs (3D contact)
- `friction="0.001 1.0"` syntax (tangential, normal)
- Proper omniwheel behavior without modeling individual rollers

**Related Documentation:**
- [Research Timeline](01_research_timeline.md), Section 2025 (MuJoCo Physics Fixes)
- [Environment & RL Workflow](03_environment_and_rl.md), Step 1

---

### Q5: Are the wheels physically modeled or just abstract friction points?

**A:** The wheels **are physically modeled** as capsule geometries with:
- Mass and inertia
- Hinge joints
- Physical contact with the ball

**What's NOT modeled:**
- Individual rollers inside the omniwheel
- Complex multi-body dynamics of the roller assembly

**Anisotropic friction captures the net effect:**
- Low tangential friction mimics rollers allowing smooth rolling
- High normal friction prevents slipping
- This is a **physics simplification** that makes simulation tractable while preserving realistic behavior

**Why this matters:**
- Modeling every roller would be computationally expensive
- Anisotropic friction provides realistic behavior with much less computation
- The wheels exist as physical bodies, but directional friction makes them behave like omniwheels

**Related Documentation:**
- [Research Timeline](01_research_timeline.md), Section 2025 (Key Insight: Physics Simplification)
- [Environment & RL Workflow](03_environment_and_rl.md), Step 1

---

### Q6: How does the Lagrangian dynamics relate to the MuJoCo simulation?

**A:** MuJoCo internally computes the same dynamics equations that Nagarajan (2014) derived analytically:

**Analytical (Nagarajan 2014):**
\[
\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{G}(\mathbf{q}) = \boldsymbol{\tau}
\]

**MuJoCo (Numerical):**
- Computes inertia matrix \(\mathbf{M}(\mathbf{q})\) from robot geometry
- Computes Coriolis/centrifugal forces \(\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\)
- Computes gravitational forces \(\mathbf{G}(\mathbf{q})\)
- Solves the dynamics equation numerically

**Key Difference:**
- **Analytical:** Explicit formulas (useful for control design)
- **Numerical:** Computed automatically (useful for simulation)

**Why this matters:**
- The observation space uses the same state variables \((\phi, \theta, \psi, \dot{\phi}, \dot{\theta}, \dot{\psi})\) from the Lagrangian
- The action space maps directly to the control inputs \(\boldsymbol{\tau}\)
- Understanding the Lagrangian helps design good rewards

**Related Documentation:**
- [Mechanics to RL Guide](02_mechanics_to_rl.md), Part 1
- [Research Timeline](01_research_timeline.md), Section 2014

---

## 🤖 Reinforcement Learning

### Q7: Why use PPO instead of other RL algorithms?

**A:** PPO is well-suited for continuous control tasks like ballbot navigation:

**Advantages:**
- **Stable learning:** Clipping prevents large policy updates
- **Sample efficient:** Can reuse data multiple times (n_epochs)
- **Works well for continuous actions:** Handles the 3D action space naturally
- **Mature implementation:** Stable-Baselines3 provides reliable code

**Alternatives considered:**
- **SAC (Soft Actor-Critic):** More sample efficient but more complex
- **TD3:** Good for continuous control but requires more tuning
- **DDPG:** Less stable, prone to overestimation

**Why PPO fits this task:**
- The reward function is well-shaped (doesn't need off-policy learning)
- Continuous action space (3D motor commands)
- Need for stable, reproducible training

**Related Documentation:**
- [Environment & RL Workflow](03_environment_and_rl.md), Step 4
- [Code Walkthrough](05_code_walkthrough.md), Training Script

---

### Q8: How does the reward function encourage safe behavior?

**A:** The reward function uses **soft constraints** via penalties:

**Components:**
1. **Directional reward:** Encourages movement toward target
2. **Action regularization:** Penalizes large actions (smooth control)
3. **Survival bonus:** Rewards staying upright (+0.02 per step)

**Safety mechanisms:**
- **Soft constraint:** Large tilt angles reduce reward (but don't terminate immediately)
- **Hard constraint:** Episode terminates if tilt > 20° (safety limit)
- **Action penalty:** Prevents jerky, destabilizing motions

**Why soft constraints:**
- Allows RL to explore recovery behaviors
- Discourages dangerous states without hard blocking
- Enables learning from near-failure experiences

**Related Documentation:**
- [Mechanics to RL Guide](02_mechanics_to_rl.md), Part 3
- [Research Timeline](01_research_timeline.md), Section 2022

---

### Q9: Why include RGB-D cameras if proprioceptive sensors are sufficient?

**A:** RGB-D cameras enable **terrain perception**:

**Proprioceptive sensors (IMU, encoders):**
- Measure robot's own state (orientation, velocity)
- Don't provide information about terrain ahead
- Sufficient for balance but not for navigation

**RGB-D cameras:**
- Provide visual information about terrain
- Enable obstacle avoidance
- Help with path planning on uneven terrain

**Why both:**
- **Proprioceptive:** Essential for balance (internal state)
- **Visual:** Essential for navigation (external environment)
- **Combined:** Enables robust navigation on uneven terrain

**Note:** The depth encoder is pretrained separately, then frozen during RL training.

**Related Documentation:**
- [Code Walkthrough](05_code_walkthrough.md), Policy Network
- [Environment & RL Workflow](03_environment_and_rl.md), Step 2

---

## 💻 Implementation

### Q10: How do I modify the reward function?

**A:** The reward function is defined in `Rewards.py` and used in `bbot_env.py`:

**Step 1: Modify `Rewards.py`**
```python
class DirectionalReward:
    def __call__(self, state):
        xy_velocity = state["vel"][-3:-1]
        dir_rew = xy_velocity.dot(self.target_direction)
        return dir_rew  # Modify scaling or add terms
```

**Step 2: Modify `bbot_env.py`**
```python
def _compute_reward(self, obs, action):
    directional_reward = self.reward_obj(obs) / 100.0
    action_regularization = -0.0001 * np.linalg.norm(action)**2
    reward = directional_reward + action_regularization
    
    # Add your custom reward terms here
    # reward += custom_term
    
    if tilt_angle <= self.max_allowed_tilt:
        reward += 0.02
    
    return reward
```

**Tips:**
- Keep reward magnitudes reasonable (typically \([-1, 1]\))
- Test reward scaling (divide by constants if needed)
- Monitor reward components during training

**Related Documentation:**
- [Code Walkthrough](05_code_walkthrough.md), Reward System
- [Mechanics to RL Guide](02_mechanics_to_rl.md), Part 3

---

### Q11: How do I add new observations?

**A:** Add observations in `bbot_env.py`:

**Step 1: Extract new observation in `_get_obs()`**
```python
def _get_obs(self):
    # ... existing observations ...
    
    # Add new observation
    new_obs = self._extract_new_observation()
    
    return {
        # ... existing observations ...
        "new_obs": new_obs,
    }
```

**Step 2: Update observation space in `__init__()`**
```python
self.observation_space = Dict({
    # ... existing spaces ...
    "new_obs": Box(low=-1, high=1, shape=(N,)),
})
```

**Step 3: Update feature extractor in `policies.py`**
```python
class Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, ...):
        # ... existing extractors ...
        
        if "new_obs" in observation_space.spaces:
            subspace = observation_space.spaces["new_obs"]
            extractors["new_obs"] = torch.nn.Flatten()
            total_concat_size += subspace.shape[0]
```

**Related Documentation:**
- [Code Walkthrough](05_code_walkthrough.md), Core Environment
- [Environment & RL Workflow](03_environment_and_rl.md), Step 2

---

### Q12: How do I change the terrain generation?

**A:** Modify `terrain.py` or change terrain type:

**Option 1: Change terrain type**
```python
# In train.py or test.py
env = make_ballbot_env(terrain_type="flat")  # or "perlin"
```

**Option 2: Modify Perlin noise parameters**
```python
# In terrain.py
def generate_perlin_terrain(
    n=129,
    scale=25.0,      # Increase for smoother terrain
    octaves=4,       # Increase for more detail
    persistence=0.2, # Increase for more variation
    ...
):
```

**Option 3: Add new terrain type**
```python
# In terrain.py
def generate_custom_terrain(...):
    # Your custom generation code
    return terrain_array

# In bbot_env.py
if self.terrain_type == "custom":
    hfield_data = terrain.generate_custom_terrain(...)
```

**Related Documentation:**
- [Code Walkthrough](05_code_walkthrough.md), Terrain Generation
- [Environment & RL Workflow](03_environment_and_rl.md), Step 2

---

## 🚀 Training & Evaluation

### Q13: How long does training take?

**A:** Training time depends on several factors:

**Typical training:**
- **Total timesteps:** 10M (default)
- **Parallel environments:** 10 (default)
- **Time per timestep:** ~0.001-0.01 seconds (depends on hardware)
- **Total time:** ~3-10 hours (on GPU)

**Factors affecting time:**
- **Hardware:** GPU vs CPU, number of cores
- **Parallel environments:** More envs = faster (but more memory)
- **Camera rendering:** Disabling cameras speeds up significantly
- **Terrain generation:** Perlin noise is fast, custom terrain may be slower

**Optimization tips:**
- Use GPU for neural network training
- Increase `num_envs` (if memory allows)
- Disable cameras if not needed (`disable_cameras=True`)
- Use depth-only instead of RGB-D (`depth_only=True`)

**Related Documentation:**
- [Environment & RL Workflow](03_environment_and_rl.md), Step 4
- [Code Walkthrough](05_code_walkthrough.md), Training Script

---

### Q14: How do I know if training is working?

**A:** Monitor these metrics during training:

**Key metrics:**
- **`rollout/ep_rew_mean`:** Average episode reward (should increase)
- **`rollout/ep_len_mean`:** Average episode length (should increase)
- **`train/policy_loss`:** Policy loss (should decrease)
- **`train/value_loss`:** Value function loss (should decrease)

**Signs of good training:**
- Episode reward steadily increases
- Episode length increases (robot stays balanced longer)
- Policy loss decreases (policy improving)
- Value loss decreases (better value estimates)

**Warning signs:**
- Reward plateaus early (may need reward tuning)
- Episode length stays low (robot keeps falling)
- Losses increase (learning rate too high or reward scaling issue)

**Visualization:**
```bash
python ballbot_rl/training/plotting_tools.py \
    --csv scripts/log/progress.csv \
    --plot_train
```

**Related Documentation:**
- [Environment & RL Workflow](03_environment_and_rl.md), Step 4
- [Code Walkthrough](05_code_walkthrough.md), Training Script

---

### Q15: How do I evaluate a trained policy?

**A:** Use the `test.py` script:

**Basic evaluation:**
```bash
python ballbot_rl/evaluation/evaluate.py \
    --algo ppo \
    --n_test 10 \
    --path scripts/log/checkpoints/ppo_agent_200000_steps.zip
```

**What it does:**
- Loads trained policy from checkpoint
- Runs `n_test` episodes with deterministic actions
- Logs metrics (episode length, reward, success rate)

**Custom evaluation:**
```python
from stable_baselines3 import PPO
from utils import make_ballbot_env

# Load model
model = PPO.load("path/to/model.zip")

# Create environment
env = make_ballbot_env(eval_env=True)

# Run episodes
for episode in range(10):
    obs, info = env.reset()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
```

**Related Documentation:**
- [Environment & RL Workflow](03_environment_and_rl.md), Step 5
- [Code Walkthrough](05_code_walkthrough.md), Evaluation Script

---

## 🔧 Troubleshooting

### Q16: MuJoCo build fails with "cannot import name 'MjModel'"

**A:** This usually means MuJoCo Python bindings aren't installed correctly:

**Solution 1: Verify MuJoCo path**
```bash
export MUJOCO_PATH=/path/to/mujoco_install
export MUJOCO_PLUGIN_PATH=/path/to/mujoco_install/plugin
```

**Solution 2: Rebuild Python bindings**
```bash
cd mujoco/python
bash make_sdist.sh
cd dist
pip install mujoco-*.tar.gz
```

**Solution 3: Check patch was applied**
```bash
cd mujoco
grep -r "condim" src/engine/  # Should find anisotropic friction code
```

**Related Documentation:**
- [Environment & RL Workflow](03_environment_and_rl.md), Step 1
- [Environment & RL Workflow](03_environment_and_rl.md), Step 6

---

### Q17: Training doesn't converge (reward doesn't increase)

**A:** Check these common issues:

**1. Reward scaling:**
- Rewards should be in reasonable range (typically \([-1, 1]\))
- Check if directional reward is too large (divide by larger constant)

**2. Learning rate:**
- Try different learning rates (default uses scheduler)
- Check `train/learning_rate` in logs

**3. Observation normalization:**
- Ensure observations are normalized (check observation ranges)

**4. Terrain generation:**
- Verify terrain is actually changing (not always flat)
- Check `terrain_type="perlin"` in config

**5. Survival bonus:**
- Ensure survival bonus is being applied (check reward components)

**Debugging:**
```python
# Check reward components
env.log_options = {"reward_terms": True}
# Run episode and check logged reward components
```

**Related Documentation:**
- [Environment & RL Workflow](03_environment_and_rl.md), Step 6
- [Code Walkthrough](05_code_walkthrough.md), Core Environment

---

### Q18: Out of memory errors during training

**A:** Reduce memory usage:

**1. Reduce parallel environments:**
```yaml
# In config file
num_envs: 5  # Instead of 10
```

**2. Disable cameras:**
```python
env = make_ballbot_env(disable_cams=True)
```

**3. Use depth-only:**
```python
env = make_ballbot_env(depth_only=True)
```

**4. Reduce rollout length:**
```yaml
# In config file
algo:
  n_steps: 1024  # Instead of 2048
```

**5. Reduce batch size:**
```yaml
# In config file
algo:
  batch_sz: 128  # Instead of 256
```

**Related Documentation:**
- [Environment & RL Workflow](03_environment_and_rl.md), Step 6
- [Code Walkthrough](05_code_walkthrough.md), Training Script

---

### Q19: Policy works in training but fails in evaluation

**A:** This suggests overfitting or evaluation mismatch:

**1. Train longer:**
- Increase `total_timesteps` (may need more than 10M)

**2. Check evaluation setup:**
- Ensure evaluation uses same observation normalization
- Use deterministic actions: `model.predict(obs, deterministic=True)`

**3. Increase terrain diversity:**
- Use more diverse terrain during training
- Check `terrain_type="perlin"` with different seeds

**4. Check for distribution shift:**
- Training and evaluation should use similar conditions
- Verify terrain generation is working in both

**5. Monitor evaluation metrics:**
- Check if evaluation episodes are actually different
- Compare training vs evaluation episode lengths

**Related Documentation:**
- [Environment & RL Workflow](03_environment_and_rl.md), Step 5
- [Code Walkthrough](05_code_walkthrough.md), Evaluation Script

---

### Q20: How do I pretrain my own depth encoder?

**A:** Follow these steps:

**Step 1: Collect Depth Images**
```bash
python ballbot_rl/data/collect.py \
    --n_steps 10000 \
    --n_envs 10 \
    --policy path/to/trained_policy.zip \
    --seed 42
```

This saves depth images to `log_*/rgbd_log_episode_*/depth/` directories.

**Step 2: Train Autoencoder**
```bash
python ballbot_rl/encoders/pretrain.py \
    --data_path path/to/log_directories \
    --save_encoder_to encoder_frozen \
    --save_dataset_as_pickle dataset.pkl
```

**Step 3: Use in Training**
Update config file:
```yaml
frozen_cnn: "outputs/encoders/encoder_epoch_53"
```

**Key Parameters:**
- `epochs`: Number of training epochs (default: 100)
- `lr`: Learning rate (default: 1e-3)
- `batch_size`: Batch size (default: 64)

**Related Documentation:**
- [Advanced Topics](09_advanced_topics.md), Depth Encoder Pretraining section
- [Code Walkthrough](05_code_walkthrough.md), Encoder Pretraining section

---

### Q21: Why is the depth encoder frozen during RL training?

**A:** Freezing the encoder provides several benefits:

**1. Stability:**
- Prevents encoder from forgetting useful features
- RL updates don't corrupt pretrained features

**2. Efficiency:**
- Fewer parameters to update (only policy/value networks)
- Faster training iterations

**3. Focus:**
- RL can focus on learning control policy
- Perception is already handled by pretrained encoder

**4. Proven Approach:**
- Common in vision-based RL (e.g., Atari, robotics)
- Separates perception learning from control learning

**When to Unfreeze:**
- If encoder features are insufficient for task
- If you want end-to-end learning (slower but potentially better)
- If you have domain-specific requirements

**Related Documentation:**
- [Advanced Topics](09_advanced_topics.md), Depth Encoder Pretraining section

---

### Q22: How does the MuJoCo patch actually work?

**A:** The patch modifies the contact frame alignment:

**Original Behavior:**
- Contact frame axes are arbitrary
- Friction applied equally in all tangent directions

**Patched Behavior:**
- Second contact frame axis aligned with capsule (wheel) axis
- Friction coefficients interpreted relative to this axis:
  - `friction="0.001 1.0"` means:
    - 0.001 friction along wheel's rolling direction
    - 1.0 friction perpendicular to rolling direction

**Code Change:**
```c
// After computing contact
if (ncon) {
    mju_copy3(con->frame+3, axis);  // Align second axis with capsule
}
```

**Why This Works:**
- Omniwheels have directional friction (low along axis, high perpendicular)
- Aligning contact frame with wheel axis enables proper friction modeling
- No need to model individual rollers

**Related Documentation:**
- [Advanced Topics](09_advanced_topics.md), MuJoCo Anisotropic Friction Patch section
- [Research Timeline](01_research_timeline.md), Section 2025 (MuJoCo Physics Fixes)

---

### Q23: Where can I get more help?

**A:** Resources for additional support:

**1. Documentation:**
- Read through all docs in `docs/` folder
- Check code comments in source files
- Review [Advanced Topics](09_advanced_topics.md) for deep dives

**2. Research Papers:**
- Read the papers in `research_papers/` folder
- Understand the theoretical foundations

**3. Code Exploration:**
- Read through `bbot_env.py` with documentation open
- Trace through training script step-by-step
- Use [Code Walkthrough](05_code_walkthrough.md) as guide

**4. Experimentation:**
- Try modifying reward functions
- Experiment with different hyperparameters
- Visualize training progress

**5. Community:**
- Check GitHub issues (if repository is public)
- Ask questions in RL/robotics forums

**Related Documentation:**
- [Documentation README](README.md)
- [Code Walkthrough](05_code_walkthrough.md)
- [FAQ](07_faq.md) (this document)

---

**Still have questions?** Check the other documentation files or explore the codebase!

**Happy Learning! 🚀**

*Last Updated: 2025*

