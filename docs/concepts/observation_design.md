# Observation Design: Multi-Modal Fusion for Robust Control

*How to design observation spaces that combine proprioception and vision for robust ballbot control*

---

## Overview

This document explains the principles of observation space design for robotics RL, with a focus on multi-modal fusion. It explains why ballbot needs both proprioception and vision, and how to combine them effectively.

---

## Observation Space Principles

### 1. Include All Necessary Information

**Principle:** Agent needs all information required for control

**For Ballbot:**
- **Balance control**: Need tilt angles and angular velocities
- **Terrain navigation**: Need visual information about terrain ahead
- **Motion control**: Need current velocities

**Missing Information:**
- Agent can't control what it can't observe
- Partial observability makes learning harder
- Need to balance completeness vs. complexity

### 2. Normalize Observations

**Principle:** All observations should be in similar ranges

**Why Normalize?**
- Neural networks train better with normalized inputs
- Prevents some inputs from dominating others
- Improves learning stability

**Typical Ranges:**
- Angles: [-π, π] or normalized to [-1, 1]
- Velocities: Normalized by typical max values
- Images: [0, 1] or normalized

### 3. Handle Multi-Modal Inputs

**Principle:** Different modalities need different processing

**For Ballbot:**
- **Proprioception**: Low-dimensional, already structured
- **Vision**: High-dimensional, needs feature extraction

**Solution:** Separate extractors + fusion

---

## Ballbot Observation Space

### Proprioceptive Observations

**What:** Internal state information from robot's own sensors

**Components:**
1. **Orientation**: Roll (φ), pitch (θ), yaw (ψ) angles
2. **Angular velocities**: (φ̇, θ̇, ψ̇)
3. **Linear velocities**: (ẋ, ẏ, ż)
4. **Position**: (x, y, z) - though z is usually fixed

**Why Needed:**
- **Balance control**: Must know tilt angles to correct
- **Motion control**: Must know velocities to plan
- **State estimation**: Needed for closed-loop control

**Connection to Mechanics:**
- These are the state variables (q, q̇) from Lagrangian dynamics
- Directly measurable from IMU and encoders
- Essential for balance control

**Code Location:**
- `ballbot_gym/envs/ballbot_env.py` - Extracts from MuJoCo state
- `ballbot_gym/envs/observation_spaces.py` - Defines space

**Normalization:**
- Angles: Typically in radians, may normalize to [-1, 1]
- Velocities: Normalized by typical max values
- Position: May normalize by workspace size

---

### Exteroceptive Observations (Vision)

**What:** External information about environment

**Components:**
1. **Depth images**: From RGB-D cameras
2. **Encoded features**: Processed by CNN encoder

**Why Needed:**
- **Terrain awareness**: Must see terrain ahead to navigate
- **Slope detection**: Need to anticipate slopes
- **Obstacle avoidance**: Need to see obstacles

**Connection to Problem:**
- Uneven terrain requires visual perception
- Proprioception alone causes state aliasing
- Vision disambiguates similar proprioceptive states

**Code Location:**
- `ballbot_gym/sensors/rgbd.py` - Camera rendering
- `ballbot_rl/encoders/models.py` - Depth encoder
- `ballbot_rl/policies/mlp_policy.py` - Feature extraction

**Processing:**
- Raw depth images: 128×128 pixels
- CNN encoder: Reduces to 20-dimensional features
- Normalized: Typically [0, 1] or tanh output

---

## Multi-Modal Fusion

### Why Multi-Modal?

**Problem:** Single modality insufficient

**Proprioception Alone:**
- Can balance on flat terrain
- But can't anticipate slopes
- State aliasing: Same proprioceptive state, different terrains

**Vision Alone:**
- Can see terrain
- But can't sense current balance state
- Missing critical control information

**Solution:** Combine both modalities

**Benefits:**
- Proprioception: Balance control
- Vision: Terrain navigation
- Together: Robust performance

---

### Fusion Architecture

**Early Fusion:**
- Concatenate raw observations
- Single network processes all inputs
- **Problem:** Different modalities need different processing

**Late Fusion:**
- Separate extractors for each modality
- Concatenate extracted features
- Single network processes fused features
- **Advantage:** Each modality processed appropriately

**For Ballbot:**
- Uses late fusion
- Proprioceptive: Flatten and pass through
- Visual: CNN encoder extracts features
- Fusion: Concatenate + MLP

**Code Location:**
- `ballbot_rl/policies/mlp_policy.py` - `Extractor` class
- Separate extractors for each observation key
- Concatenation in forward pass

---

## Observation Processing Pipeline

### 1. Raw Observations

**From MuJoCo:**
- Orientation quaternion
- Angular velocities
- Linear velocities
- Positions
- Depth images (from cameras)

**Extraction:**
- Convert quaternion to Euler angles
- Extract relevant components
- Render depth images

**Code:**
- `ballbot_gym/envs/ballbot_env.py` - `_get_obs()` method

---

### 2. Normalization

**Proprioceptive:**
- Angles: Convert to radians, normalize
- Velocities: Normalize by max values
- Positions: Normalize by workspace

**Visual:**
- Depth images: Normalize to [0, 1]
- Encoded features: Typically tanh output [-1, 1]

**Code:**
- `ballbot_gym/envs/ballbot_env.py` - Normalization in observation extraction

---

### 3. Feature Extraction

**Proprioceptive:**
- Flatten and concatenate
- May pass through small MLP
- Output: Feature vector

**Visual:**
- CNN encoder processes depth images
- Reduces 128×128 → 20 dimensions
- Output: Encoded features

**Code:**
- `ballbot_rl/policies/mlp_policy.py` - Feature extractors
- `ballbot_rl/encoders/models.py` - CNN encoder

---

### 4. Fusion

**Concatenation:**
- Combine proprioceptive and visual features
- Single feature vector
- Pass to policy network

**Code:**
- `ballbot_rl/policies/mlp_policy.py` - Concatenation in forward pass

---

## Design Considerations

### 1. What Information to Include?

**Essential:**
- Orientation (balance control)
- Angular velocities (balance control)
- Linear velocities (motion control)
- Vision (terrain navigation)

**Optional:**
- Position (if needed for navigation)
- Goal position (if goal-directed)
- Previous actions (if using history)

**Not Needed:**
- Perfect state information (if using vision)
- Future terrain (agent learns to anticipate)

---

### 2. How Much Vision?

**Too Little:**
- Small field of view
- Low resolution
- **Problem:** Can't see terrain ahead

**Too Much:**
- Large field of view
- High resolution
- **Problem:** High-dimensional, slow training

**Sweet Spot:**
- 128×128 depth images
- Two cameras (forward-looking)
- Encoded to 20 dimensions
- **Result:** Good terrain awareness, manageable dimensionality

---

### 3. Normalization Strategy

**Why Normalize:**
- Neural networks train better
- Prevents some inputs from dominating
- Improves stability

**How to Normalize:**
- **Angles**: Normalize to [-1, 1] or keep in radians
- **Velocities**: Divide by typical max values
- **Images**: Normalize to [0, 1]
- **Features**: Use tanh or normalize

**Code:**
- `ballbot_gym/envs/ballbot_env.py` - Normalization logic

---

## Partial Observability

### What is Partial Observability?

**Definition:** Agent doesn't observe full state

**For Ballbot:**
- Doesn't observe future terrain
- Doesn't observe exact friction coefficients
- Doesn't observe all forces

**Why It Matters:**
- Agent must learn to anticipate
- Must use history or vision
- Makes learning harder

**Solutions:**
- **Vision**: See terrain ahead
- **History**: Include previous observations
- **Recurrent networks**: Process sequences

**For Ballbot:**
- Vision provides terrain awareness
- No explicit history (high-frequency control)
- Works because vision disambiguates states

---

## Observation Space Design Process

### Step 1: Identify Required Information

**Questions:**
- What does agent need to know?
- What can be measured?
- What's essential vs. optional?

**For Ballbot:**
- Essential: Orientation, velocities, vision
- Optional: Position, goal

---

### Step 2: Design Observation Structure

**Questions:**
- How to structure observations?
- Dict vs. Box space?
- What keys to use?

**For Ballbot:**
- Dict space with keys: "proprioceptive", "depth"
- Allows different processing per modality
- Flexible and extensible

---

### Step 3: Normalize Observations

**Questions:**
- What ranges are reasonable?
- How to normalize?
- Test normalization?

**For Ballbot:**
- Angles: Radians or normalized
- Velocities: Normalized by max
- Images: [0, 1]

---

### Step 4: Design Feature Extractors

**Questions:**
- How to process each modality?
- What architecture?
- Pretrained or trainable?

**For Ballbot:**
- Proprioceptive: Flatten
- Visual: CNN encoder (pretrained)
- Fusion: Concatenate

---

### Step 5: Test and Iterate

**Questions:**
- Does agent learn?
- Are observations informative?
- Can improve design?

**For Ballbot:**
- Test on flat terrain first
- Then test on uneven terrain
- Validate learned behavior

---

## Common Pitfalls

### 1. Missing Critical Information

**Problem:** Observation space missing essential information

**Example:**
- No vision → can't navigate terrain
- No orientation → can't balance

**Solution:** Include all necessary information

---

### 2. Poor Normalization

**Problem:** Observations in different ranges

**Example:**
- Angles: [-π, π]
- Velocities: [0, 10]
- **Problem:** Velocities dominate

**Solution:** Normalize to similar ranges

---

### 3. Too High Dimensional

**Problem:** Observation space too large

**Example:**
- Raw depth images: 128×128 = 16,384 dimensions
- **Problem:** Slow training, overfitting

**Solution:** Use feature extraction (CNN encoder)

---

### 4. State Aliasing

**Problem:** Different states look the same

**Example:**
- Same tilt angle on different terrains
- **Problem:** Agent can't distinguish

**Solution:** Add vision to disambiguate

---

## Best Practices

### 1. Start Simple

**Begin with:**
- Proprioception only
- Test on simple task (flat terrain)

**Then add:**
- Vision
- Test on complex task (uneven terrain)

---

### 2. Validate Observations

**Check:**
- Are observations informative?
- Can agent learn from them?
- Do they contain necessary information?

**Test:**
- Train simple policy
- Check if it learns
- Analyze learned behavior

---

### 3. Monitor Feature Extraction

**Track:**
- Feature magnitudes
- Feature distributions
- Learning progress

**Why:**
- Identify issues early
- Debug observation processing
- Validate architecture

---

## Connection to Research

### Salehi 2025: Multi-Modal RL

**Key Insight:** Proprioception + vision needed for robust navigation

**Implementation:**
- Proprioceptive: Orientation, velocities
- Visual: Depth images encoded by CNN
- Fusion: Concatenation + MLP

**Result:** Robust policies that generalize to new terrains

---

## Summary

**Observation Design Principles:**
1. Include all necessary information
2. Normalize observations
3. Handle multi-modal inputs appropriately
4. Use feature extraction for high-dimensional inputs
5. Test and iterate

**Ballbot Observations:**
- Proprioceptive: Orientation, velocities
- Visual: Depth images (encoded)
- Fusion: Late fusion with separate extractors

**Key Takeaways:**
- Multi-modal observations enable robust control
- Proper normalization improves learning
- Feature extraction reduces dimensionality
- Testing validates design

---

## Next Steps

- Read [RL Fundamentals](rl_fundamentals.md) for MDP formulation
- Read [Reward Design](reward_design.md) for reward engineering
- Read [Multi-Modal Fusion Tutorial](../tutorials/10_multimodal_fusion.md) for implementation details
- Explore [Code Walkthrough](../api/code_walkthrough.md) to see observations in code
- Try [Custom Policy Example](../../examples/04_custom_policy.py) to experiment

---

*Last Updated: 2025*

