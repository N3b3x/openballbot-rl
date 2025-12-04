# Tutorial Dependencies and Reading Order

*A guide to understanding tutorial prerequisites and recommended reading order*

---

## Overview

This document provides a dependency graph showing which tutorials depend on which others, prerequisites for each tutorial, and recommended reading orders for different learning paths.

---

## Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                    Tutorial Dependencies                     │
└─────────────────────────────────────────────────────────────┘

00_ballbot_introduction.md
  │
  ├─→ 01_introduction_to_gymnasium.md
  │     │
  │     ├─→ 02_action_spaces_in_rl.md
  │     │     │
  │     │     ├─→ 03_observation_spaces_in_rl.md
  │     │     │     │
  │     │     │     ├─→ 04_reward_design_for_robotics.md
  │     │     │     │     │
  │     │     │     │     └─→ 05_actor_critic_methods.md
  │     │     │     │           │
  │     │     │     │           ├─→ 06_environment_wrappers.md
  │     │     │     │           │     │
  │     │     │     │           │     └─→ 13_complete_training_guide.md
  │     │     │     │           │
  │     │     │     │           └─→ 09_hyperparameter_tuning.md
  │     │     │     │                 │
  │     │     │     │                 └─→ 13_complete_training_guide.md
  │     │     │     │
  │     │     │     └─→ 10_multimodal_fusion.md
  │     │     │           │
  │     │     │           └─→ 13_complete_training_guide.md
  │     │     │
  │     │     └─→ 05_actor_critic_methods.md
  │     │
  │     └─→ 07_camera_rendering_in_mujoco.md
  │           │
  │           └─→ 08_working_with_mujoco_simulation_state.md
  │                 │
  │                 └─→ 10_multimodal_fusion.md
  │
  └─→ 15_terrain_generation.md
        │
        └─→ 13_complete_training_guide.md

11_debugging_visualization.md
  └─→ (depends on: 13_complete_training_guide.md)

12_sim_to_real_transfer.md
  └─→ (depends on: 13_complete_training_guide.md)

14_curriculum_learning.md
  └─→ (depends on: 13_complete_training_guide.md)
```

---

## Tutorial Prerequisites

### Core Tutorials (Foundation)

#### 00_ballbot_introduction.md
**Prerequisites:** None
**Dependencies:** None
**What you'll learn:** What a ballbot is, why it's challenging, why RL helps

#### 01_introduction_to_gymnasium.md
**Prerequisites:** 
- Basic Python knowledge
- Understanding of what RL is (high-level)

**Dependencies:** 
- None (but 00_ballbot_introduction.md provides context)

**What you'll learn:** Gymnasium API, environment interface, basic usage

---

#### 02_action_spaces_in_rl.md
**Prerequisites:**
- Understanding of Gymnasium API (01_introduction_to_gymnasium.md)
- Basic RL concepts

**Dependencies:**
- 01_introduction_to_gymnasium.md

**What you'll learn:** Action space types, normalization, continuous actions

---

#### 03_observation_spaces_in_rl.md
**Prerequisites:**
- Understanding of Gymnasium API (01_introduction_to_gymnasium.md)
- Understanding of action spaces (02_action_spaces_in_rl.md)
- Basic RL concepts

**Dependencies:**
- 01_introduction_to_gymnasium.md
- 02_action_spaces_in_rl.md (recommended)

**What you'll learn:** Observation space types, normalization, multi-modal observations

---

#### 04_reward_design_for_robotics.md
**Prerequisites:**
- Understanding of Gymnasium API (01_introduction_to_gymnasium.md)
- Understanding of action spaces (02_action_spaces_in_rl.md)
- Understanding of observation spaces (03_observation_spaces_in_rl.md)
- Basic RL concepts

**Dependencies:**
- 01_introduction_to_gymnasium.md
- 02_action_spaces_in_rl.md (recommended)
- 03_observation_spaces_in_rl.md (recommended)

**What you'll learn:** Reward function design, shaping, multi-objective rewards

---

#### 05_actor_critic_methods.md
**Prerequisites:**
- Understanding of Gymnasium API (01_introduction_to_gymnasium.md)
- Understanding of action spaces (02_action_spaces_in_rl.md)
- Understanding of observation spaces (03_observation_spaces_in_rl.md)
- Understanding of reward design (04_reward_design_for_robotics.md)
- Basic neural network knowledge
- Calculus (derivatives, gradients)

**Dependencies:**
- 01_introduction_to_gymnasium.md
- 02_action_spaces_in_rl.md
- 03_observation_spaces_in_rl.md
- 04_reward_design_for_robotics.md

**What you'll learn:** Actor-Critic architecture, PPO, policy gradients, value functions

---

#### 06_environment_wrappers.md
**Prerequisites:**
- Understanding of Gymnasium API (01_introduction_to_gymnasium.md)
- Understanding of Actor-Critic methods (05_actor_critic_methods.md)
- Basic understanding of parallel processing

**Dependencies:**
- 01_introduction_to_gymnasium.md
- 05_actor_critic_methods.md (recommended)

**What you'll learn:** Environment wrappers, vectorization, monitoring, logging

---

#### 09_hyperparameter_tuning.md
**Prerequisites:**
- Understanding of Actor-Critic methods (05_actor_critic_methods.md)
- Understanding of PPO algorithm
- Basic understanding of training workflow

**Dependencies:**
- 05_actor_critic_methods.md

**What you'll learn:** PPO hyperparameters, tuning strategies, monitoring metrics

---

### Specialized Tutorials

#### 07_camera_rendering_in_mujoco.md
**Prerequisites:**
- Understanding of Gymnasium API (01_introduction_to_gymnasium.md)
- Basic MuJoCo knowledge
- Understanding of computer vision basics

**Dependencies:**
- 01_introduction_to_gymnasium.md

**What you'll learn:** Camera setup, RGB-D rendering, image processing

---

#### 08_working_with_mujoco_simulation_state.md
**Prerequisites:**
- Understanding of Gymnasium API (01_introduction_to_gymnasium.md)
- Understanding of MuJoCo basics (07_camera_rendering_in_mujoco.md)
- Understanding of physics simulation

**Dependencies:**
- 01_introduction_to_gymnasium.md
- 07_camera_rendering_in_mujoco.md (recommended)

**What you'll learn:** Accessing MuJoCo state, extracting observations, physics queries

---

#### 10_multimodal_fusion.md
**Prerequisites:**
- Understanding of observation spaces (03_observation_spaces_in_rl.md)
- Understanding of camera rendering (07_camera_rendering_in_mujoco.md)
- Understanding of MuJoCo state (08_working_with_mujoco_simulation_state.md)
- Basic neural network knowledge

**Dependencies:**
- 03_observation_spaces_in_rl.md
- 07_camera_rendering_in_mujoco.md (recommended)
- 08_working_with_mujoco_simulation_state.md (recommended)

**What you'll learn:** Combining proprioception and vision, feature fusion, encoder design

---

#### 11_debugging_visualization.md
**Prerequisites:**
- Understanding of training workflow (13_complete_training_guide.md)
- Basic Python knowledge
- Understanding of visualization tools (matplotlib, tensorboard)

**Dependencies:**
- 13_complete_training_guide.md

**What you'll learn:** Debugging RL training, visualization tools, common issues

---

#### 12_sim_to_real_transfer.md
**Prerequisites:**
- Understanding of training workflow (13_complete_training_guide.md)
- Understanding of domain adaptation
- Basic robotics knowledge

**Dependencies:**
- 13_complete_training_guide.md

**What you'll learn:** Sim-to-real challenges, domain randomization, transfer strategies

---

#### 13_complete_training_guide.md
**Prerequisites:**
- Understanding of Actor-Critic methods (05_actor_critic_methods.md)
- Understanding of environment wrappers (06_environment_wrappers.md)
- Understanding of hyperparameter tuning (09_hyperparameter_tuning.md)
- Understanding of reward design (04_reward_design_for_robotics.md)
- Understanding of observation spaces (03_observation_spaces_in_rl.md)
- Understanding of action spaces (02_action_spaces_in_rl.md)
- Basic Python knowledge
- Understanding of Stable-Baselines3

**Dependencies:**
- 05_actor_critic_methods.md
- 06_environment_wrappers.md (recommended)
- 09_hyperparameter_tuning.md (recommended)
- 04_reward_design_for_robotics.md
- 03_observation_spaces_in_rl.md
- 02_action_spaces_in_rl.md

**What you'll learn:** Complete training workflow, hyperparameter tuning, evaluation

---

#### 14_curriculum_learning.md
**Prerequisites:**
- Understanding of training workflow (13_complete_training_guide.md)
- Understanding of curriculum learning concepts

**Dependencies:**
- 13_complete_training_guide.md

**What you'll learn:** Curriculum design, difficulty scheduling, progressive training

---

#### 15_terrain_generation.md
**Prerequisites:**
- Understanding of ballbot basics (00_ballbot_introduction.md)
- Basic Python knowledge
- Understanding of procedural generation

**Dependencies:**
- 00_ballbot_introduction.md

**What you'll learn:** Terrain generation, Perlin noise, procedural content

---

## Recommended Reading Orders

### Path 1: Complete Beginner (0 → Production)

**Goal:** Understand everything from scratch

1. **00_ballbot_introduction.md** - What is a ballbot?
2. **01_introduction_to_gymnasium.md** - Gymnasium API
3. **02_action_spaces_in_rl.md** - Actions
4. **03_observation_spaces_in_rl.md** - Observations
5. **04_reward_design_for_robotics.md** - Rewards
6. **05_actor_critic_methods.md** - Algorithms
7. **13_complete_training_guide.md** - Training
8. **11_debugging_visualization.md** - Debugging
9. **12_sim_to_real_transfer.md** - Transfer (optional)
10. **14_curriculum_learning.md** - Advanced (optional)

**Specialized Topics (as needed):**
- **07_camera_rendering_in_mujoco.md** - If using vision
- **08_working_with_mujoco_simulation_state.md** - If accessing state
- **10_multimodal_fusion.md** - If using multi-modal observations
- **15_terrain_generation.md** - If generating terrains

---

### Path 2: RL Practitioner (Skip Basics)

**Goal:** Quickly understand ballbot-specific aspects

1. **00_ballbot_introduction.md** - Ballbot basics
2. **02_action_spaces_in_rl.md** - Ballbot actions
3. **03_observation_spaces_in_rl.md** - Ballbot observations
4. **04_reward_design_for_robotics.md** - Ballbot rewards
5. **13_complete_training_guide.md** - Training workflow
6. **10_multimodal_fusion.md** - Multi-modal fusion (if using vision)
7. **11_debugging_visualization.md** - Debugging

**Specialized Topics (as needed):**
- **07_camera_rendering_in_mujoco.md** - Camera setup
- **08_working_with_mujoco_simulation_state.md** - State access
- **15_terrain_generation.md** - Terrain generation

---

### Path 3: Control Theory Expert

**Goal:** Understand how classical control translates to RL

1. **00_ballbot_introduction.md** - Ballbot mechanics
2. **02_action_spaces_in_rl.md** - Control inputs
3. **03_observation_spaces_in_rl.md** - State variables
4. **04_reward_design_for_robotics.md** - Constraints → rewards
5. **05_actor_critic_methods.md** - RL algorithms
6. **13_complete_training_guide.md** - Training workflow

**Specialized Topics (as needed):**
- **10_multimodal_fusion.md** - Multi-modal observations
- **12_sim_to_real_transfer.md** - Transfer learning

---

### Path 4: Researcher (Paper → Code)

**Goal:** Map papers to implementation

1. **00_ballbot_introduction.md** - Physical design
2. **02_action_spaces_in_rl.md** - Action implementation
3. **03_observation_spaces_in_rl.md** - Observation implementation
4. **04_reward_design_for_robotics.md** - Reward implementation
5. **05_actor_critic_methods.md** - Algorithm implementation
6. **13_complete_training_guide.md** - Training pipeline
7. **10_multimodal_fusion.md** - Multi-modal architecture
8. **11_debugging_visualization.md** - Evaluation

**Specialized Topics (as needed):**
- **07_camera_rendering_in_mujoco.md** - Vision pipeline
- **08_working_with_mujoco_simulation_state.md** - State extraction
- **15_terrain_generation.md** - Terrain generation

---

## Quick Reference

### Minimal Path (Fastest)

**For:** Quick start, just want to train

1. **00_ballbot_introduction.md** - Context
2. **13_complete_training_guide.md** - Training

**Time:** ~1 hour

---

### Standard Path (Recommended)

**For:** Most users

1. **00_ballbot_introduction.md**
2. **01_introduction_to_gymnasium.md**
3. **02_action_spaces_in_rl.md**
4. **03_observation_spaces_in_rl.md**
5. **04_reward_design_for_robotics.md**
6. **05_actor_critic_methods.md**
7. **13_complete_training_guide.md**

**Time:** ~1 day

---

### Complete Path (Comprehensive)

**For:** Deep understanding

1. All core tutorials (00-05)
2. **13_complete_training_guide.md**
3. All specialized tutorials (07-15)

**Time:** ~1 week

---

## Dependency Summary Table

| Tutorial | Prerequisites | Dependencies | Difficulty |
|----------|--------------|--------------|------------|
| 00_ballbot_introduction | None | None | Beginner |
| 01_introduction_to_gymnasium | Python basics | None | Beginner |
| 02_action_spaces_in_rl | Gymnasium API | 01 | Beginner |
| 03_observation_spaces_in_rl | Gymnasium API | 01, 02 | Beginner |
| 04_reward_design_for_robotics | RL basics | 01, 02, 03 | Intermediate |
| 05_actor_critic_methods | Neural networks | 01-04 | Intermediate |
| 07_camera_rendering_in_mujoco | MuJoCo basics | 01 | Intermediate |
| 08_working_with_mujoco_simulation_state | MuJoCo | 01, 07 | Intermediate |
| 10_multimodal_fusion | Neural networks | 03, 07, 08 | Advanced |
| 11_debugging_visualization | Training experience | 13 | Intermediate |
| 12_sim_to_real_transfer | Training experience | 13 | Advanced |
| 13_complete_training_guide | RL basics | 02-05 | Intermediate |
| 14_curriculum_learning | Training experience | 13 | Advanced |
| 15_terrain_generation | Python basics | 00 | Beginner |

---

## Tips

1. **Follow dependencies**: Don't skip prerequisites
2. **Read in order**: Tutorials build on each other
3. **Practice**: Run examples as you read
4. **Reference**: Use this document to plan your path
5. **Flexible**: Adjust based on your background

---

## Next Steps

- Choose your learning path above
- Start with the first tutorial in your path
- Follow dependencies as you progress
- Reference this document when unsure

---

*Last Updated: 2025*

