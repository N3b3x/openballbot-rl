# 🚀 Complete Training Pipeline: From Setup to Trained Policy

*A step-by-step guide to training RL policies from start to finish*

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites & Setup](#prerequisites--setup)
3. [Step 1: Environment Setup](#step-1-environment-setup)
4. [Step 2: Configuration](#step-2-configuration)
5. [Step 3: Pretraining (Optional)](#step-3-pretraining-optional)
6. [Step 4: Training](#step-4-training)
7. [Step 5: Monitoring](#step-5-monitoring)
8. [Step 6: Evaluation](#step-6-evaluation)
9. [Step 7: Troubleshooting](#step-7-troubleshooting)
10. [Step 8: Deployment](#step-8-deployment)
11. [Complete Workflow Example](#complete-workflow-example)
12. [Summary](#summary)

---

## 🎯 Introduction

This tutorial provides a **complete, end-to-end guide** for training RL policies on the Ballbot environment. From initial setup to deploying a trained policy, we'll walk through every step with practical examples.

> "Training RL policies is part art, part science. This guide provides the science; experience provides the art."  
> — *Common wisdom in deep RL*

**What You'll Learn:**
- How to set up the training environment
- How to configure hyperparameters
- How to monitor training progress
- How to evaluate trained policies
- How to troubleshoot common issues
- How to deploy policies

**Prerequisites:**
- Basic understanding of RL (PPO, Gymnasium)
- Python 3.8+ installed
- MuJoCo built and installed
- Familiarity with command line

---

## 🔧 Prerequisites & Setup

### System Requirements

**Hardware:**
- CPU: Multi-core recommended (4+ cores)
- RAM: 8GB+ recommended
- GPU: Optional but recommended for faster training
- Storage: 10GB+ for logs and checkpoints

**Software:**
- Python 3.8+
- MuJoCo (patched version)
- CUDA (if using GPU)

### Installation Steps

**1. Clone Repository:**
```bash
git clone <repository_url>
cd openballbot-rl
```

**2. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**3. Install Ballbot Environment:**
```bash
cd ballbot_gym
pip install -e .
```

**4. Verify Installation:**
```bash
cd ../scripts
python test_pid.py  # Should run without errors
```

---

## 📝 Step 1: Environment Setup

### Build MuJoCo with Anisotropic Friction

**Why:** Ballbot requires anisotropic friction for omniwheels.

**Steps:**
```bash
# 1. Clone MuJoCo
git clone https://github.com/deepmind/mujoco.git
cd mujoco

# 2. Checkout specific commit (recommended)
git checkout 99490163df46f65a0cabcf8efef61b3164faa620

# 3. Apply patch
cp <path_to>/mujoco_fix.patch .
patch -p1 < mujoco_fix.patch

# 4. Build MuJoCo
mkdir build && cd build
cmake ..
cmake --build .
cmake --install . --prefix <install_dir>

# 5. Build Python bindings
cd ../python
bash make_sdist.sh
cd dist
export MUJOCO_PATH=<install_dir>
pip install mujoco-*.tar.gz
```

**Verify:**
```python
import mujoco
# Should import without errors
```

### Environment Variables

**Set Required Paths:**
```bash
export MUJOCO_PATH=/path/to/mujoco/install
export MUJOCO_PLUGIN_PATH=/path/to/mujoco/plugin
```

---

## ⚙️ Step 2: Configuration

### Understanding the Config File

**Location:** `configs/train_ppo_directional.yaml`

**Structure:**
```yaml
algo:
  name: ppo
  ent_coef: 0.001          # Entropy coefficient (exploration)
  clip_range: 0.015        # PPO clipping range
  target_kl: 0.3           # Target KL divergence
  vf_coef: 2.0             # Value function coefficient
  learning_rate: -1        # -1 = use scheduler
  n_steps: 2048            # Steps per environment before update
  n_epochs: 5              # Optimization epochs per update
  batch_sz: 256            # Batch size
  weight_decay: 0.01       # L2 regularization
  normalize_advantage: false

problem:
  terrain_type: "perlin"   # "perlin" or "flat"

total_timesteps: 10e6      # Total training steps
frozen_cnn: "../outputs/encoders/encoder_epoch_53"  # Pretrained encoder
hidden_sz: 128             # Hidden layer size
num_envs: 10               # Parallel environments
resume: ""                 # Resume from checkpoint
out: ./log                 # Output directory
seed: 10                   # Random seed
```

### Key Hyperparameters Explained

**`ent_coef` (Entropy Coefficient):**
- Controls exploration vs. exploitation
- Higher = more exploration
- Typical range: 0.001 - 0.01

**`clip_range` (PPO Clipping):**
- Limits policy update size
- Smaller = more stable
- Typical range: 0.1 - 0.3 (ours is conservative at 0.015)

**`n_steps` (Rollout Length):**
- Steps collected before update
- Larger = more stable but slower
- Typical: 2048 - 4096

**`num_envs` (Parallel Environments):**
- More = faster data collection
- Limited by memory/CPU
- Typical: 4 - 32

### Creating Your Config

**Customize for Your Needs:**
```yaml
# configs/my_training_config.yaml
algo:
  name: ppo
  ent_coef: 0.002          # Increase exploration
  clip_range: 0.2          # Standard PPO
  learning_rate: 1e-4     # Fixed learning rate
  n_steps: 4096            # Longer rollouts
  batch_sz: 512            # Larger batches

problem:
  terrain_type: "perlin"

total_timesteps: 5e6       # Shorter training
num_envs: 16              # More parallel envs
seed: 42
out: ./my_training_logs
```

---

## 🎨 Step 3: Pretraining (Optional)

### Why Pretrain the Encoder?

**Problem:** Training CNN encoder from scratch during RL is slow and unstable.

**Solution:** Pretrain encoder separately, then freeze during RL.

### Data Collection

**Collect Depth Images:**
```bash
python ballbot_rl/data/collect.py \
    --n_steps 100000 \
    --n_envs 10 \
    --policy ""  # Empty = random policy
```

**What This Does:**
- Runs environment with random actions
- Collects depth images from cameras
- Saves to log directory

### Encoder Pretraining

**Train Autoencoder:**
```bash
python ballbot_rl/encoders/pretrain.py \
    --data_dir <path_to_collected_data> \
    --save_encoder_to ./encoder_pretrained \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3
```

**What This Does:**
- Trains autoencoder on depth images
- Learns compressed representations
- Saves encoder for RL training

**Output:**
- `encoder_pretrained/encoder_epoch_50` - Frozen encoder

---

## 🏋️ Step 4: Training

### Basic Training Command

**Start Training:**
```bash
cd scripts
python train.py --config ../configs/train_ppo_directional.yaml
```

**What Happens:**
1. Loads configuration
2. Creates parallel environments
3. Initializes PPO agent
4. Trains for `total_timesteps`
5. Saves checkpoints and logs

### Training Output

**Console Output:**
```
Using SubprocVecEnv (multi-process)
num_total_params=123456
num_learnable_params=98765
----------------------------------
| rollout/           |          |
|    ep_len_mean     | 1.23e+03 |
|    ep_rew_mean     | 45.6     |
| time/              |          |
|    fps             | 1234     |
|    iterations      | 100      |
|    total_timesteps | 204800   |
----------------------------------
```

**Log Files:**
- `log/progress.csv` - Training metrics
- `log/config.yaml` - Training configuration
- `log/checkpoints/` - Model checkpoints
- `log/best_model/` - Best model (by eval reward)

### Training Stages

**Stage 1: Exploration (0-1M steps)**
- High entropy (exploration)
- Low episode reward
- Learning to balance

**Stage 2: Learning (1M-5M steps)**
- Decreasing entropy
- Increasing reward
- Learning navigation

**Stage 3: Refinement (5M-10M steps)**
- Low entropy (exploitation)
- High reward
- Fine-tuning

---

## 📊 Step 5: Monitoring

### Real-Time Monitoring

**Watch Training Progress:**
```bash
# In another terminal
tail -f scripts/log/progress.csv
```

**Plot Training Curves:**
```bash
ballbot-plot-training \
    --csv log/progress.csv \
    --config log/config.yaml \
    --plot_train
```

**What to Monitor:**
- `rollout/ep_rew_mean` - Should increase
- `rollout/ep_len_mean` - Should increase
- `train/policy_gradient_loss` - Should decrease
- `train/value_loss` - Should decrease
- `train/entropy_loss` - Should decrease (exploration)

### Using Weights & Biases

**Setup W&B:**
```python
# Add to train.py
import wandb
wandb.init(project="ballbot-rl", config=config)
```

**Monitor Remotely:**
- View dashboard at wandb.ai
- Compare experiments
- Track hyperparameters

### Key Metrics to Watch

**Good Training Signs:**
- ✅ Episode reward steadily increasing
- ✅ Episode length increasing
- ✅ Losses decreasing
- ✅ Evaluation reward tracking training

**Warning Signs:**
- ⚠️ Reward not increasing
- ⚠️ Losses oscillating wildly
- ⚠️ Evaluation much lower than training (overfitting)
- ⚠️ Entropy too low (overfitting)

---

## 🧪 Step 6: Evaluation

### Evaluate Trained Policy

**Basic Evaluation:**
```bash
python test.py \
    --algo ppo \
    --path log/best_model/best_model.zip \
    --n_test 10 \
    --seed 42
```

**What This Does:**
- Loads trained policy
- Runs evaluation episodes
- Displays results

### Evaluation Metrics

**Key Metrics:**
- **Success Rate**: Percentage of successful episodes
- **Mean Reward**: Average episode reward
- **Mean Episode Length**: Average episode duration
- **Failure Rate**: Percentage of episodes ending in failure

**Compute Metrics:**
```python
def evaluate_policy(model, env, n_episodes=100):
    """
    Comprehensive policy evaluation.
    """
    results = {
        'rewards': [],
        'lengths': [],
        'successes': 0,
        'failures': 0
    }
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        results['rewards'].append(episode_reward)
        results['lengths'].append(episode_length)
        
        if info.get('failure', False):
            results['failures'] += 1
        else:
            results['successes'] += 1
    
    # Compute statistics
    print(f"Mean Reward: {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}")
    print(f"Mean Length: {np.mean(results['lengths']):.1f} ± {np.std(results['lengths']):.1f}")
    print(f"Success Rate: {100*results['successes']/n_episodes:.1f}%")
    print(f"Failure Rate: {100*results['failures']/n_episodes:.1f}%")
    
    return results
```

### Evaluation on Different Terrains

**Test Generalization:**
```bash
# Test on different terrain types
python test.py \
    --algo ppo \
    --path log/best_model/best_model.zip \
    --override_terrain_type "flat" \
    --n_test 10
```

**Compare Performance:**
- Training terrain vs. test terrain
- Flat vs. Perlin terrain
- Different difficulty levels

---

## 🔧 Step 7: Troubleshooting

### Common Issues and Solutions

#### Issue 1: Training Not Improving

**Symptoms:**
- Reward not increasing
- Episode length not increasing
- Losses not decreasing

**Diagnosis:**
```python
# Check if environment works
env = make_ballbot_env()()
obs, _ = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"Reward: {reward}, Done: {done}")
```

**Solutions:**
1. Check learning rate (may be too low)
2. Check reward function (may be too sparse)
3. Check observation normalization
4. Verify environment is resetting correctly

#### Issue 2: Unstable Training

**Symptoms:**
- Losses oscillating
- Reward jumping around
- Policy performance degrading

**Solutions:**
1. Reduce learning rate (try 10x smaller)
2. Reduce clip range (for PPO)
3. Increase batch size
4. Add gradient clipping

#### Issue 3: Overfitting

**Symptoms:**
- Training reward >> evaluation reward
- Policy works in training but fails in evaluation

**Solutions:**
1. Add domain randomization
2. Reduce policy complexity
3. Add regularization (weight decay)
4. Early stopping

#### Issue 4: Out of Memory

**Symptoms:**
- CUDA out of memory errors
- System crashes

**Solutions:**
1. Reduce `num_envs` (fewer parallel environments)
2. Reduce `n_steps` (shorter rollouts)
3. Reduce batch size
4. Use CPU instead of GPU

#### Issue 5: Slow Training

**Symptoms:**
- Training takes too long
- Low FPS (frames per second)

**Solutions:**
1. Increase `num_envs` (more parallel)
2. Disable cameras if not needed
3. Use GPU for neural network
4. Reduce image resolution

### Debugging Checklist

- [ ] Environment works correctly
- [ ] Config file is valid
- [ ] Checkpoints are saving
- [ ] Logs are being written
- [ ] No errors in console
- [ ] GPU/CPU usage is reasonable
- [ ] Memory usage is reasonable

---

## 🚀 Step 8: Deployment

### Save Final Model

**After Training:**
```python
# Model is automatically saved to:
# - log/best_model/best_model.zip (best by eval reward)
# - log/checkpoints/ppo_agent_<steps>_steps.zip (periodic)
```

### Load and Use Model

**Load Trained Policy:**
```python
from stable_baselines3 import PPO

# Load model
model = PPO.load("log/best_model/best_model.zip")

# Use for inference
env = make_ballbot_env()()
obs, _ = env.reset()

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    
    if done:
        obs, _ = env.reset()
```

### Export for Deployment

**Save Model Components:**
```python
# Save policy only (smaller file)
model.policy.save("policy_only.zip")

# Save full model
model.save("full_model.zip")

# Save with custom objects
model.save("model_with_custom.zip", 
           custom_objects={'learning_rate': 1e-4})
```

---

## 📖 Complete Workflow Example

### End-to-End Training Session

**1. Setup:**
```bash
# Install dependencies
pip install -r requirements.txt
cd ballbot_gym && pip install -e . && cd ..

# Verify installation
cd scripts && python test_pid.py
```

**2. Configure:**
```bash
# Edit config file
vim ../configs/train_ppo_directional.yaml

# Or create custom config
cp ../configs/train_ppo_directional.yaml my_config.yaml
# Edit my_config.yaml
```

**3. Train:**
```bash
# Start training
python train.py --config ../configs/train_ppo_directional.yaml

# Monitor in another terminal
tail -f log/progress.csv
```

**4. Monitor:**
```bash
# Plot training curves
ballbot-plot-training \
    --csv log/progress.csv \
    --config log/config.yaml \
    --plot_train
```

**5. Evaluate:**
```bash
# Evaluate best model
python test.py \
    --algo ppo \
    --path log/best_model/best_model.zip \
    --n_test 20
```

**6. Deploy:**
```python
# Load and use
from stable_baselines3 import PPO
model = PPO.load("log/best_model/best_model.zip")
# Use model for inference...
```

---

## 📊 Summary

### Training Pipeline Checklist

**Setup:**
- [ ] MuJoCo installed with patch
- [ ] Dependencies installed
- [ ] Environment verified

**Configuration:**
- [ ] Config file created/edited
- [ ] Hyperparameters set
- [ ] Output directory specified

**Training:**
- [ ] Training started
- [ ] Monitoring set up
- [ ] Checkpoints saving

**Evaluation:**
- [ ] Policy evaluated
- [ ] Metrics computed
- [ ] Generalization tested

**Deployment:**
- [ ] Model saved
- [ ] Model loaded and tested
- [ ] Ready for use

### Key Takeaways

1. **Setup is critical** - Get MuJoCo and dependencies right
2. **Config matters** - Hyperparameters significantly affect training
3. **Monitor actively** - Watch for issues early
4. **Evaluate thoroughly** - Test on diverse conditions
5. **Troubleshoot systematically** - Use checklist approach

### Next Steps

- Experiment with hyperparameters
- Try different reward functions
- Test on different terrains
- Deploy to real robot (see [Sim-to-Real Transfer](12_sim_to_real_transfer.md))

---

## 📚 Further Reading

### Tutorials

- [Introduction to Gymnasium](01_introduction_to_gymnasium.md) - Environment API
- [Actor-Critic Methods](05_actor_critic_methods.md) - Understanding PPO
- [Debugging & Visualization](11_debugging_visualization.md) - Monitoring training
- [Sim-to-Real Transfer](12_sim_to_real_transfer.md) - Deploying policies

### Documentation

- [Environment & RL Workflow](../03_environment_and_rl.md) - Detailed setup
- [Code Walkthrough](../05_code_walkthrough.md) - Implementation details

### Code References

- `ballbot_rl/training/train.py` - Training script
- `ballbot_rl/evaluation/evaluate.py` - Evaluation script
- `configs/train_ppo_directional.yaml` - Configuration template

---

*Last Updated: 2025*

**Happy Training! 🚀**

