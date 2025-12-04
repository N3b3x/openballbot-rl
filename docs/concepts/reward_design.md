# Reward Design: From Constraints to Rewards

*How classical control constraints translate into RL reward functions*

---

## Overview

This document explains the principles of reward design for robotics RL, with a focus on how classical control constraints and objectives become reward components. It bridges control theory and RL by showing how to encode desired behaviors in reward functions.

---

## Reward Design Principles

### 1. Encode Objectives as Positive Rewards

**Principle:** What you want the agent to do → positive reward

**For Ballbot:**
- **Navigate toward goal**: Reward velocity in target direction
- **Maintain balance**: Reward staying upright
- **Move efficiently**: Reward smooth control

**Example:**
```python
r_direction = (v_xy · g_target) / 100  # Positive reward for goal-directed motion
r_survival = 0.02 if upright else 0     # Positive reward for balance
```

### 2. Encode Constraints as Penalties

**Principle:** What you want to avoid → negative reward (penalty)

**For Ballbot:**
- **Don't fall**: Penalty for large tilts (or termination)
- **Don't waste energy**: Penalty for large actions
- **Don't violate limits**: Penalty for exceeding constraints

**Example:**
```python
r_action = -0.0001 ||a||²  # Penalty for large actions
r_tilt = -penalty if tilt > threshold  # Penalty for excessive tilt
```

### 3. Balance Trade-offs via Weights

**Principle:** Use weights (α₁, α₂, ...) to balance multiple objectives

**For Ballbot:**
```python
r = α₁·r_direction + α₂·r_action + α₃·r_survival
```

**Typical Values:**
- α₁ = 0.01 (directional reward scale)
- α₂ = 0.0001 (action penalty scale)
- α₃ = 0.02 (survival bonus)

**Why These Values?**
- Directional reward is small (velocity typically 0-1 m/s)
- Action penalty is very small (prevents reward hacking without dominating)
- Survival bonus is moderate (encourages balance without being too strong)

---

## Ballbot Reward Function

### Complete Reward Function

**From Salehi (2025) and implemented in code:**

```python
r = α₁(v_xy · g_target) - α₂||a||² + α₃·𝟙[upright]
```

Where:
- **v_xy**: 2D velocity vector (x, y components)
- **g_target**: Target direction vector (normalized)
- **a**: Action vector (three wheel torques)
- **𝟙[upright]**: Indicator function (1 if upright, 0 otherwise)
- **α₁ = 0.01**: Directional reward scale
- **α₂ = 0.0001**: Action penalty scale
- **α₃ = 0.02**: Survival bonus

### Component Breakdown

#### 1. Directional Reward: `α₁(v_xy · g_target)`

**Purpose:** Encourage goal-directed navigation

**Mathematical Form:**
- Dot product of velocity and target direction
- Positive when moving toward goal
- Zero when moving perpendicular
- Negative when moving away

**Why This Works:**
- Rewards progress toward goal
- Doesn't require reaching exact position
- Works for continuous navigation

**Scaling:**
- Divided by 100 to keep values reasonable
- α₁ = 0.01 further scales it
- Typical values: 0.001 - 0.01 per step

**Connection to Control Theory:**
- Similar to trajectory tracking objective
- But doesn't require exact path
- More flexible than waypoint following

**Code Location:**
- `ballbot_gym/rewards/directional.py` - `DirectionalReward` class

---

#### 2. Action Penalty: `-α₂||a||²`

**Purpose:** Prevent reward hacking and encourage efficiency

**Mathematical Form:**
- L2 norm of action vector squared
- Always negative (penalty)
- Larger actions → larger penalty

**Why This Works:**
- Prevents agent from spinning in place (high velocity but no progress)
- Encourages smooth, efficient control
- Prevents excessive energy use

**Common Pitfall Without This:**
- Agent might learn to oscillate rapidly
- High frequency control → high velocity → high reward
- But inefficient and unstable

**Scaling:**
- α₂ = 0.0001 is very small
- Prevents reward hacking without dominating reward
- Typical penalty: -0.001 to -0.01 per step

**Connection to Control Theory:**
- Similar to control effort minimization
- Encourages energy-efficient control
- Prevents actuator saturation

**Code Location:**
- `ballbot_gym/rewards/directional.py` - Included in reward computation

---

#### 3. Survival Bonus: `α₃·𝟙[upright]`

**Purpose:** Encourage balance maintenance

**Mathematical Form:**
- Indicator function: 1 if upright, 0 otherwise
- Upright = tilt angle below threshold
- Constant reward when balanced

**Why This Works:**
- Provides steady reward for staying balanced
- Prevents agent from learning to fall gracefully
- Encourages stable policies

**Threshold:**
- Typically: tilt angle < 30° or 45°
- Depends on robot design
- Too strict → agent too conservative
- Too loose → agent learns to fall

**Scaling:**
- α₃ = 0.02 is moderate
- Provides steady reward stream
- Typical: 0.01 - 0.05 per step

**Connection to Control Theory:**
- Encodes balance constraint (Carius 2022)
- Similar to constraint satisfaction
- But soft constraint (reward) vs. hard constraint (termination)

**Code Location:**
- `ballbot_gym/rewards/directional.py` - Included in reward computation
- `ballbot_gym/envs/ballbot_env.py` - Checks upright condition

---

## Constraint Handling

### From Hard Constraints to Soft Penalties

**Classical Control:**
- Hard constraints: Must satisfy exactly
- Example: Tilt angle < 30° (hard limit)
- Violation → system failure

**RL Approach:**
- Soft constraints: Penalize violations
- Example: Penalty for large tilts
- Violation → negative reward, but can recover

**Why Soft Constraints?**
- Allows exploration
- Agent can learn from mistakes
- More robust to edge cases

**Implementation:**
```python
# Hard constraint (termination)
if tilt_angle > threshold:
    terminated = True
    reward = -large_penalty

# Soft constraint (penalty)
if tilt_angle > warning_threshold:
    reward -= penalty_proportional_to_tilt
```

**Code Location:**
- `ballbot_gym/envs/ballbot_env.py` - Termination conditions
- `ballbot_gym/rewards/directional.py` - Reward computation

---

## Reward Shaping Strategies

### 1. Sparse vs. Dense Rewards

**Sparse Rewards:**
- Reward only at goal
- Example: +1 if reached goal, 0 otherwise
- **Problem:** Hard to learn (rare rewards)

**Dense Rewards:**
- Reward at every step
- Example: Directional reward + survival bonus
- **Advantage:** Easier to learn (frequent feedback)

**For Ballbot:**
- Dense rewards work better
- Directional reward provides continuous feedback
- Survival bonus encourages exploration

### 2. Reward Normalization

**Principle:** Keep reward magnitudes reasonable

**Why Normalize?**
- Prevents numerical issues
- Makes hyperparameter tuning easier
- Improves learning stability

**For Ballbot:**
- Directional reward: Divide by 100
- Action penalty: Very small coefficient
- Survival bonus: Moderate value

### 3. Reward Hacking Prevention

**Common Hacks:**
- Spinning in place (high velocity, no progress)
- Oscillating rapidly (high frequency control)
- Exploiting reward bugs

**Prevention:**
- Action penalty prevents spinning
- Proper reward design prevents oscillation
- Testing and validation catch bugs

**Example:**
Without action penalty:
- Agent learns: spin fast → high velocity → high reward
- But no actual progress

With action penalty:
- Spinning → high action penalty → net negative reward
- Agent learns efficient control instead

---

## Design Trade-offs

### 1. Exploration vs. Exploitation

**High Survival Bonus:**
- Agent becomes too conservative
- Doesn't explore enough
- Slow learning

**Low Survival Bonus:**
- Agent takes risks
- More exploration
- But more failures

**Solution:** Moderate survival bonus (α₃ = 0.02)

### 2. Speed vs. Stability

**High Directional Reward:**
- Agent prioritizes speed
- May sacrifice stability
- More falls

**Low Directional Reward:**
- Agent prioritizes stability
- Moves slowly
- Less progress

**Solution:** Balance via weights (α₁ = 0.01)

### 3. Efficiency vs. Performance

**High Action Penalty:**
- Agent uses minimal control
- Very efficient
- But may be too slow

**Low Action Penalty:**
- Agent uses more control
- Faster response
- But less efficient

**Solution:** Small penalty (α₂ = 0.0001)

---

## Common Pitfalls

### 1. Reward Scaling Issues

**Problem:** Rewards too large or too small

**Symptoms:**
- Training unstable
- Policy doesn't learn
- Numerical issues

**Solution:** Normalize rewards, use appropriate scales

### 2. Conflicting Rewards

**Problem:** Rewards push agent in opposite directions

**Example:**
- Directional reward: Move fast
- Action penalty: Use less control
- Conflict: Can't move fast without control

**Solution:** Balance weights carefully, test different combinations

### 3. Reward Hacking

**Problem:** Agent exploits reward function

**Example:**
- Spinning in place
- Oscillating rapidly
- Exploiting bugs

**Solution:** Action penalty, proper reward design, testing

### 4. Sparse Rewards

**Problem:** Rewards too rare

**Example:**
- Only reward at goal
- Agent never reaches goal
- No learning signal

**Solution:** Dense rewards, reward shaping, curriculum learning

---

## Best Practices

### 1. Start Simple

**Begin with basic rewards:**
- Survival bonus
- Simple objective (e.g., move forward)

**Then add complexity:**
- Directional reward
- Action penalty
- Additional objectives

### 2. Test Incrementally

**Test each component:**
- Does survival bonus encourage balance? ✓
- Does directional reward encourage navigation? ✓
- Does action penalty prevent hacking? ✓

**Then test together:**
- Do they work together? ✓
- Are weights balanced? ✓

### 3. Monitor Reward Components

**Track individual components:**
- Directional reward over time
- Action penalty over time
- Survival bonus over time

**Why:**
- Understand what agent is learning
- Identify issues early
- Debug reward function

### 4. Validate Behavior

**Check learned behavior:**
- Does agent balance? ✓
- Does agent navigate? ✓
- Does agent avoid reward hacking? ✓

**If not:**
- Adjust reward weights
- Add/remove components
- Re-test

---

## Connection to Research

### Carius 2022: Constraint-Aware Control

**Key Insight:** Constraints can be encoded in control objective

**Translation to RL:**
- Hard constraints → termination conditions
- Soft constraints → reward penalties
- Objectives → positive rewards

**Example:**
- Constraint: Tilt angle < 30°
- RL: Penalty for large tilts + termination if exceeded

### Salehi 2025: RL Navigation

**Key Insight:** Dense rewards + multi-objective optimization

**Implementation:**
- Directional reward (navigation)
- Action penalty (efficiency)
- Survival bonus (balance)

**Result:** Robust, generalizable policies

---

## Summary

**Reward Design Principles:**
1. Encode objectives as positive rewards
2. Encode constraints as penalties
3. Balance trade-offs via weights
4. Prevent reward hacking
5. Use dense rewards for learning

**Ballbot Reward:**
- Directional: Encourages navigation
- Action penalty: Prevents hacking
- Survival: Encourages balance

**Key Takeaways:**
- Rewards encode desired behavior
- Constraints become penalties
- Weights balance trade-offs
- Testing validates design

---

## Next Steps

- Read [RL Fundamentals](rl_fundamentals.md) for MDP formulation
- Read [Observation Design](observation_design.md) for multi-modal observations
- Read [Mechanics to RL Guide](../research/mechanics_to_rl.md) for mathematical details
- Explore [Code Walkthrough](../api/code_walkthrough.md) to see rewards in code
- Try [Custom Reward Example](../../examples/02_custom_reward.py) to experiment

---

*Last Updated: 2025*

