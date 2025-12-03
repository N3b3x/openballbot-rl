# 🎁 Reward Design for Robotics

*A comprehensive guide to designing effective reward functions for robotics reinforcement learning*

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [What is a Reward Function?](#what-is-a-reward-function)
3. [Types of Rewards](#types-of-rewards)
4. [Reward Shaping Principles](#reward-shaping-principles)
5. [Real-World Example: Ballbot Rewards](#real-world-example-ballbot-rewards)
6. [Common Reward Components](#common-reward-components)
7. [Reward Normalization](#reward-normalization)
8. [Multi-Objective Rewards](#multi-objective-rewards)
9. [Common Pitfalls](#common-pitfalls)
10. [Best Practices](#best-practices)
11. [Summary](#summary)

---

## 🎯 Introduction

The reward function is arguably **the most important component** of a reinforcement learning system. It defines what the agent should optimize for, shaping every aspect of learned behavior.

> "The reward function is the most important hyperparameter. Get it wrong, and no amount of algorithmic sophistication will help."  
> — *Pieter Abbeel, UC Berkeley*

In robotics, reward design is particularly challenging because:
- We want **safe, stable, and efficient** behaviors
- Multiple objectives must be balanced (speed, stability, energy)
- Rewards must be **dense enough** to guide learning
- But not so **shaped** that they prevent discovery of better solutions

**Key Questions This Tutorial Answers:**
- What makes a good reward function?
- How do we balance multiple objectives?
- Should rewards be sparse or dense?
- How do we normalize rewards?
- What are common reward components for robotics?

---

## 🔍 What is a Reward Function?

### Mathematical Definition

In a Markov Decision Process (MDP), the reward function *R*: **𝒮** × **𝒜** × **𝒮** → ℝ maps state transitions to scalar rewards:

*r*ₜ = *R*(**s**ₜ, **a**ₜ, **s**ₜ₊₁)

The agent's goal is to maximize the **expected return**:

*G*ₜ = Σₖ₌₀^∞ *γ*ᵏ *r*ₜ₊ₖ₊₁

Where *γ* ∈ [0, 1] is the discount factor.

### Reward as Learning Signal

The reward function serves as the **only learning signal** for the agent:

- **Positive rewards** encourage behaviors
- **Negative rewards** (penalties) discourage behaviors
- **Zero rewards** provide no signal

> "The reward function is the agent's teacher. It must be clear, consistent, and aligned with our true objectives."  
> — *Sergey Levine, UC Berkeley*

### In Gymnasium

```python
def step(self, action):
    # ... apply action, step physics ...
    
    # Compute reward
    reward = self.compute_reward(obs, action)
    
    return obs, reward, terminated, truncated, info
```

---

## 📦 Types of Rewards

### 1. Sparse Rewards

**Sparse rewards** are only given at specific events (success, failure):

```python
def compute_reward(self, obs, action):
    if self.reached_goal():
        return 100.0  # Large positive reward
    elif self.failed():
        return -100.0  # Large negative reward
    else:
        return 0.0  # No signal during task
```

**Characteristics:**
- ✅ Simple to define
- ✅ Aligned with true objective
- ❌ Very hard to learn from (credit assignment problem)
- ❌ Requires extensive exploration

**When to Use:**
- Simple tasks with clear success/failure
- When you have strong exploration strategies
- When dense rewards would be misleading

### 2. Dense Rewards

**Dense rewards** provide signal at every timestep:

```python
def compute_reward(self, obs, action):
    # Reward for making progress
    progress_reward = self.get_progress()
    
    # Penalty for instability
    stability_penalty = -self.get_tilt_angle()
    
    # Penalty for large actions (energy efficiency)
    action_penalty = -0.01 * np.linalg.norm(action)**2
    
    return progress_reward + stability_penalty + action_penalty
```

**Characteristics:**
- ✅ Provides learning signal at every step
- ✅ Easier to learn from
- ❌ Can lead to suboptimal behaviors if poorly designed
- ❌ Requires careful tuning

**When to Use:**
- Complex tasks requiring fine-grained control
- When sparse rewards would be too rare
- When you want to guide learning toward specific behaviors

> "Dense rewards are a double-edged sword. They make learning easier, but they can also lead to reward hacking if not designed carefully."  
> — *John Schulman, OpenAI*

### 3. Shaped Rewards

**Shaped rewards** provide intermediate signals that guide toward the goal:

```python
def compute_reward(self, obs, action):
    # Distance to goal (closer = better)
    distance_to_goal = np.linalg.norm(self.goal_pos - self.robot_pos)
    distance_reward = -distance_to_goal  # Negative = penalty for distance
    
    # Velocity toward goal (faster = better, if moving correctly)
    velocity_toward_goal = np.dot(self.robot_vel, self.goal_direction)
    velocity_reward = velocity_toward_goal
    
    return distance_reward + velocity_reward
```

**Characteristics:**
- ✅ Guides learning effectively
- ✅ Can speed up learning significantly
- ❌ Must be carefully designed to avoid local optima
- ❌ Can prevent discovery of better solutions

---

## 🎨 Reward Shaping Principles

### 1. Alignment with True Objective

The reward function should align with what we **actually want**:

**❌ Misaligned:**
```python
# We want the robot to reach the goal, but reward is for speed
reward = np.linalg.norm(velocity)  # Robot might go in circles!
```

**✅ Aligned:**
```python
# Reward progress toward goal
goal_direction = (self.goal_pos - self.robot_pos) / distance
reward = np.dot(velocity, goal_direction)  # Moving toward goal
```

### 2. Scale Appropriately

Rewards should be in a reasonable range:

```python
# ✅ Good: Rewards in [-1, 1] range
reward = np.clip(reward, -1.0, 1.0)

# ❌ Bad: Rewards can be huge
reward = distance_to_goal * 1000  # Could be thousands!
```

### 3. Balance Components

When combining multiple reward terms, balance their scales:

```python
# ✅ Good: Balanced components
reward = (
    0.5 * progress_reward +      # Scale: ~0.1
    0.3 * stability_reward +    # Scale: ~0.1
    0.2 * efficiency_reward      # Scale: ~0.01
)

# ❌ Bad: One component dominates
reward = (
    100.0 * progress_reward +    # Dominates everything!
    0.01 * stability_reward       # Ignored
)
```

### 4. Avoid Reward Hacking

**Reward hacking** occurs when the agent finds ways to maximize reward that don't align with our true objective:

**Example:**
```python
# ❌ Bad: Agent might learn to "cheat"
def compute_reward(self, obs, action):
    # Reward for being close to goal
    distance = np.linalg.norm(self.goal_pos - self.robot_pos)
    return -distance
    
# Problem: Agent might learn to stay at start if goal is far,
# or find a bug that makes distance negative
```

**Solution:**
```python
# ✅ Good: Multiple constraints prevent hacking
def compute_reward(self, obs, action):
    # Progress reward (but must actually move)
    progress = self.get_progress_toward_goal()
    
    # Stability constraint (must stay upright)
    if self.is_fallen():
        return -100.0  # Large penalty
    
    # Efficiency constraint (don't waste energy)
    energy_penalty = -0.01 * np.linalg.norm(action)**2
    
    return progress + energy_penalty
```

> "Reward hacking is the bane of reward engineering. Always test your reward function with simple policies first."  
> — *Dario Amodei, Anthropic*

---

## 🤖 Real-World Example: Ballbot Rewards

Let's examine the **Ballbot** environment's reward function:

### Reward Components

The Ballbot uses a **multi-component reward** with three terms:

```python
# From bbot_env.py step() method

# 1. Directional reward (encourages movement toward goal)
reward = self.reward_obj(obs) / 100.0
self.reward_term_1_hist.append(reward)

# 2. Action regularization (encourages efficiency)
action_regularization = -0.0001 * (np.linalg.norm(omniwheel_commands)**2)
self.reward_term_2_hist.append(action_regularization)
reward += action_regularization

# 3. Survival bonus (encourages stability)
if not self.is_fallen():
    reward += 0.02
```

### Component 1: Directional Reward

```python
# From Rewards.py
class DirectionalReward:
    def __call__(self, state):
        # Velocity component in target direction
        dir_rew = state["vel"][:2].dot(self.target_direction)
        return dir_rew
```

**Mathematical Form:**
*r*ᵈⁱʳ = **v** · **d**ᵍᵒᵃˡ

Where:
- **v** is the 2D velocity vector [*v*ₓ, *v*ᵧ]
- **d**ᵍᵒᵃˡ is the target direction [*d*ₓ, *d*ᵧ]

**Interpretation:**
- Positive when moving toward goal
- Zero when moving perpendicular
- Negative when moving away

**Why Divide by 100?**
The raw directional reward can be large (velocity in m/s). Dividing by 100 scales it to a reasonable range (~0.01 to 0.1) to balance with other components.

### Component 2: Action Regularization

```python
action_regularization = -0.0001 * (np.linalg.norm(omniwheel_commands)**2)
```

**Mathematical Form:**
*r*ᵃᶜᵗⁱᵒⁿ = -λ ||**a**||²

Where:
- **a** is the action vector
- λ = 0.0001 is the regularization coefficient

**Interpretation:**
- Penalizes large actions (energy efficiency)
- Encourages smooth control
- Prevents excessive motor commands

**Why L2 Norm?**
The L2 norm (squared) provides a smooth penalty that:
- Is differentiable everywhere
- Grows quadratically with action magnitude
- Naturally encourages smaller actions

### Component 3: Survival Bonus

```python
if not self.is_fallen():
    reward += 0.02
```

**Interpretation:**
- Small positive reward for staying upright
- Provides learning signal even when not making progress
- Encourages stability

**Why Small?**
The survival bonus (0.02) is small compared to other components to:
- Not dominate the reward signal
- Still provide guidance for stability
- Allow progress rewards to be the primary signal

### Complete Reward Function

```python
def compute_reward(self, obs, action):
    # Base reward: directional progress
    r_dir = (obs["vel"][:2].dot(self.goal_direction)) / 100.0
    
    # Regularization: energy efficiency
    r_action = -0.0001 * np.linalg.norm(action)**2
    
    # Survival: stability bonus
    r_survival = 0.02 if not self.is_fallen() else 0.0
    
    # Total reward
    reward = r_dir + r_action + r_survival
    
    return reward
```

### Reward Scale Analysis

| Component | Typical Range | Weight | Purpose |
|-----------|---------------|--------|---------|
| Directional | [-0.1, 0.1] | 1.0 | Progress toward goal |
| Action Reg. | [-0.0001, 0] | 1.0 | Energy efficiency |
| Survival | [0, 0.02] | 1.0 | Stability |

**Total Reward Range:** Approximately [-0.1, 0.12]

---

## 🧩 Common Reward Components

### 1. Progress Rewards

Reward for making progress toward a goal:

```python
# Distance-based
progress = -np.linalg.norm(goal_pos - robot_pos)

# Velocity-based
goal_direction = (goal_pos - robot_pos) / distance
progress = np.dot(velocity, goal_direction)
```

### 2. Stability Rewards

Reward for maintaining stability:

```python
# Tilt angle penalty
tilt_angle = self.compute_tilt_angle()
stability = -tilt_angle / max_tilt  # Normalized penalty

# Upright bonus
if tilt_angle < threshold:
    stability += 0.1
```

### 3. Efficiency Rewards

Reward for energy-efficient control:

```python
# Action magnitude penalty
efficiency = -0.01 * np.linalg.norm(action)**2

# Torque penalty
efficiency = -0.001 * np.sum(np.abs(torques))
```

### 4. Safety Rewards

Penalize unsafe behaviors:

```python
# Collision penalty
if self.in_collision():
    safety = -10.0
else:
    safety = 0.0

# Joint limit penalty
if self.joints_at_limits():
    safety = -1.0
```

### 5. Task-Specific Rewards

Rewards specific to the task:

```python
# Manipulation: object distance
if self.gripper_closed():
    object_distance = np.linalg.norm(self.object_pos - self.target_pos)
    task_reward = -object_distance
```

---

## ⚖️ Reward Normalization

### Why Normalize?

Reward normalization helps with:
- **Stable learning**: Prevents gradient explosions
- **Algorithm compatibility**: Many algorithms assume normalized rewards
- **Hyperparameter transfer**: Easier to transfer hyperparameters

### Normalization Strategies

#### 1. Clipping

```python
reward = np.clip(reward, -1.0, 1.0)
```

#### 2. Scaling

```python
# Scale to [-1, 1] range
reward = reward / max_expected_reward
reward = np.clip(reward, -1.0, 1.0)
```

#### 3. Running Statistics

```python
# Normalize using running mean and std
self.reward_mean = 0.99 * self.reward_mean + 0.01 * reward
self.reward_std = 0.99 * self.reward_std + 0.01 * (reward - self.reward_mean)**2
normalized_reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
```

> "Reward normalization is crucial for stable learning, especially when combining multiple reward components."  
> — *Tuomas Haarnoja, Google DeepMind*

---

## 🎯 Multi-Objective Rewards

### Weighted Sum

The most common approach:

```python
reward = (
    w1 * progress_reward +
    w2 * stability_reward +
    w3 * efficiency_reward
)
```

**Challenge:** Choosing weights is often trial-and-error.

### Constrained Optimization

Treat some objectives as constraints:

```python
# Primary objective: progress
reward = progress_reward

# Constraint: must stay stable
if tilt_angle > max_tilt:
    reward = -100.0  # Large penalty (constraint violation)
```

### Pareto Optimization

For truly multi-objective problems, consider Pareto-optimal solutions (advanced topic).

---

## ⚠️ Common Pitfalls

### 1. Reward Hacking

**Problem:** Agent finds ways to maximize reward that don't align with objectives.

**Solution:** Test with simple policies, add constraints.

### 2. Unbalanced Components

**Problem:** One component dominates, others ignored.

**Solution:** Scale components appropriately, use similar magnitudes.

### 3. Sparse Rewards Too Sparse

**Problem:** Agent never receives reward signal.

**Solution:** Add dense shaping rewards or improve exploration.

### 4. Dense Rewards Too Shaped

**Problem:** Agent optimizes for shaped reward, not true objective.

**Solution:** Validate that shaped rewards lead to desired behavior.

---

## ✅ Best Practices

### 1. Start Simple

Begin with a simple reward, then add complexity:

```python
# Start: Just progress
reward = -distance_to_goal

# Add: Stability
reward = -distance_to_goal - 0.1 * tilt_angle

# Add: Efficiency
reward = -distance_to_goal - 0.1 * tilt_angle - 0.01 * ||action||²
```

### 2. Test with Simple Policies

Before training, test reward with:
- Random policy
- Simple heuristic policy
- Hand-designed policy

### 3. Monitor Reward Components

```python
# Log individual components
self.reward_components = {
    "progress": progress_reward,
    "stability": stability_reward,
    "efficiency": efficiency_reward
}
```

### 4. Validate Learned Behavior

After training, check:
- Does the agent actually do what we want?
- Are there unexpected behaviors?
- Is the reward aligned with true objective?

### 5. Document Reward Function

```python
"""
Reward Function:
- Directional reward: velocity component toward goal, scaled by 1/100
- Action regularization: -0.0001 * ||action||²
- Survival bonus: +0.02 if robot hasn't fallen

Total reward range: approximately [-0.1, 0.12]
"""
```

---

## 🤖 Advanced Reward Learning Techniques ⭐⭐

### Inverse Reinforcement Learning (IRL)

**Concept:** Learn reward function from expert demonstrations.

**Why Use IRL?**
- Don't know how to specify reward manually
- Have expert demonstrations available
- Want to learn human preferences

**Basic IRL:**
```python
def learn_reward_from_demos(expert_trajectories):
    """
    Learn reward function that makes expert trajectories optimal.
    """
    # Initialize reward function
    reward_net = nn.Sequential(
        nn.Linear(state_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    # Train to maximize likelihood of expert trajectories
    for trajectory in expert_trajectories:
        # Compute reward for expert trajectory
        expert_reward = sum(reward_net(state) for state in trajectory)
        
        # Sample random trajectories
        random_reward = sum(reward_net(state) for state in random_trajectory)
        
        # Expert should have higher reward
        loss = -torch.log(torch.sigmoid(expert_reward - random_reward))
        loss.backward()
```

**Modern IRL:**
- **GAIL (Generative Adversarial Imitation Learning)**: Uses GANs to learn rewards
- **AIRL (Adversarial Inverse RL)**: More robust IRL with adversarial training

### Preference-Based Reward Learning ⭐

**Concept:** Learn reward from human preferences (better than demonstrations).

**Why Better?**
- Easier for humans (just compare two trajectories)
- More scalable than demonstrations
- Captures true preferences better

**Implementation:**
```python
class PreferenceRewardLearner:
    """
    Learn reward function from pairwise preferences.
    """
    def __init__(self):
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def learn_from_preference(self, trajectory_a, trajectory_b, preference):
        """
        preference: 1 if A > B, 0 if B > A
        """
        # Compute rewards
        reward_a = sum(self.reward_net(state) for state in trajectory_a)
        reward_b = sum(self.reward_net(state) for state in trajectory_b)
        
        # Preference probability (Bradley-Terry model)
        prob_a_preferred = torch.sigmoid(reward_a - reward_b)
        
        # Loss: maximize likelihood of preference
        if preference == 1:
            loss = -torch.log(prob_a_preferred)
        else:
            loss = -torch.log(1 - prob_a_preferred)
        
        loss.backward()
        return loss
```

**Modern Approaches:**
- **PEBBLE (Preference-based RL)**: Active preference querying
- **D-REX (Diverse Reward Extrapolation)**: Learn from suboptimal demonstrations

### Reinforcement Learning from Human Feedback (RLHF) ⭐⭐

**Concept:** Fine-tune policies using human feedback, popularized by ChatGPT.

**Process:**
1. **Collect human comparisons**: Humans rank trajectory pairs
2. **Train reward model**: Learn reward from comparisons
3. **RL fine-tuning**: Optimize policy with learned reward

**Implementation:**
```python
# Step 1: Collect human preferences
preferences = collect_human_preferences(trajectory_pairs)

# Step 2: Train reward model
reward_model = train_reward_from_preferences(preferences)

# Step 3: Fine-tune policy with learned reward
def learned_reward(state, action, next_state):
    return reward_model(state)

env = wrap_with_reward(env, learned_reward)
policy = PPO("MultiInputPolicy", env)
policy.learn(total_timesteps=1_000_000)
```

**Benefits:**
- Aligns with human values
- Can improve safety
- Better generalization
- Used in production systems (ChatGPT, robotics)

### Reward Model Ensembles

**Concept:** Use multiple reward models for robustness.

```python
class EnsembleReward:
    """
    Ensemble of reward models for robust reward learning.
    """
    def __init__(self, n_models=5):
        self.reward_models = [
            RewardModel() for _ in range(n_models)
        ]
    
    def predict(self, state):
        # Average predictions
        rewards = [model(state) for model in self.reward_models]
        return torch.mean(torch.stack(rewards), dim=0)
    
    def uncertainty(self, state):
        # Compute prediction uncertainty
        rewards = [model(state) for model in self.reward_models]
        return torch.std(torch.stack(rewards), dim=0)
```

**Benefits:**
- More robust to reward model errors
- Can detect uncertain states
- Better for safety-critical applications

---

## 📊 Summary

### Key Takeaways

1. **Reward function is critical** - It shapes everything the agent learns

2. **Balance components** - Ensure no single component dominates

3. **Test thoroughly** - Validate with simple policies before training

4. **Normalize appropriately** - Keep rewards in reasonable ranges

5. **Avoid reward hacking** - Add constraints and test edge cases

6. **Start simple** - Add complexity gradually

7. **Consider reward learning** - IRL, preferences, RLHF for complex objectives ⭐

### Reward Design Checklist

- [ ] Reward aligns with true objective
- [ ] Components are balanced (similar scales)
- [ ] Rewards are normalized appropriately
- [ ] Tested with simple policies
- [ ] No obvious reward hacking opportunities
- [ ] Documented clearly
- [ ] Considered reward learning if manual design is difficult ⭐

---

## 📚 Further Reading

### Papers

**Classic Reward Design:**
- **Ng et al. (1999)** - "Policy invariance under reward transformations" - Reward shaping theory
- **Hadfield-Menell et al. (2016)** - "Cooperative inverse reinforcement learning" - Reward alignment
- **Amodei et al. (2016)** - "Concrete problems in AI safety" - Reward hacking

**Modern Reward Learning:**
- **Ho & Ermon (2016)** - "Generative Adversarial Imitation Learning" - GAIL
- **Fu et al. (2018)** - "Learning Robust Rewards with Adversarial Inverse Reinforcement Learning" - AIRL
- **Christiano et al. (2017)** - "Deep Reinforcement Learning from Human Preferences" - Preference learning
- **Lee et al. (2021)** - "PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via Relabeling Experience" - Active preference learning
- **Ouyang et al. (2022)** - "Training language models to follow instructions with human feedback" - RLHF
- **Brown et al. (2020)** - "Language Models are Few-Shot Learners" - GPT-3 (uses RLHF)

### Books
- **Sutton & Barto** - "Reinforcement Learning: An Introduction" - Chapter 3: Rewards

### Code Examples
- Ballbot environment: `ballbotgym/ballbotgym/bbot_env.py` - `step()` method
- Ballbot rewards: `ballbotgym/ballbotgym/Rewards.py`

---

## 🎓 Exercises

1. **Modify Ballbot Rewards**: Change the directional reward weight. How does this affect learned behavior?

2. **Add New Component**: Add a "smoothness" reward that penalizes rapid action changes.

3. **Sparse Rewards**: Convert Ballbot to sparse rewards (only reward at goal). How does learning change?

---

*Next Tutorial: [Environment Design Patterns](05_environment_design_patterns.md)*

---

**Happy Learning! 🚀**

