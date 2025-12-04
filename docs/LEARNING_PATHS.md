# Learning Paths

*Choose your path based on your background and goals*

---

## Overview

openballbot-rl is a complex project that combines classical control theory, reinforcement learning, robotics simulation, and software engineering. This guide provides four learning paths tailored to different backgrounds and goals.

Each path includes:
- Step-by-step reading order
- File references
- Estimated time
- Prerequisites
- Learning objectives

---

## Path 1: Complete Beginner (0 → Production)

**For:** Someone new to RL, robotics, or both

**Goal:** Understand everything from scratch and be able to train and deploy a policy

**Estimated Time:** 2-3 weeks (part-time)

### Step 1: Installation & Setup (Day 1)

**Objective:** Get the environment running

1. Read [Installation Guide](getting_started/installation.md)
2. Follow installation steps
3. Run verification: `python scripts/test_pid.py`
4. Read [Quick Start](getting_started/quick_start.md)
5. Run basic example: `python examples/01_basic_usage.py`

**Files:**
- `docs/getting_started/installation.md`
- `docs/getting_started/quick_start.md`
- `examples/01_basic_usage.py`

**Success Criteria:** Environment runs without errors

---

### Step 2: Understand the Robot (Day 2-3)

**Objective:** Understand what a ballbot is and why it's challenging

1. Read [Ballbot Introduction](tutorials/00_ballbot_introduction.md)
2. Read [Research Timeline](research/timeline.md) - Focus on Lauwers 2006 and Nagarajan 2014
3. Read [Ballbot Mechanics](concepts/ballbot_mechanics.md) (if exists, otherwise skip)

**Files:**
- `docs/tutorials/00_ballbot_introduction.md`
- `docs/research/timeline.md`
- `docs/concepts/ballbot_mechanics.md` (if exists)

**Success Criteria:** Can explain what a ballbot is and why RL helps

---

### Step 3: Learn RL Fundamentals (Day 4-7)

**Objective:** Understand RL concepts in the context of this project

1. Read [Introduction to Gymnasium](tutorials/01_introduction_to_gymnasium.md)
2. Read [Action Spaces](tutorials/02_action_spaces_in_rl.md)
3. Read [Observation Spaces](tutorials/03_observation_spaces_in_rl.md)
4. Read [Reward Design](tutorials/04_reward_design_for_robotics.md)
5. Read [RL Fundamentals](concepts/rl_fundamentals.md) (if exists)
6. Read [Actor-Critic Methods](tutorials/05_actor_critic_methods.md)

**Files:**
- `docs/tutorials/01_introduction_to_gymnasium.md`
- `docs/tutorials/02_action_spaces_in_rl.md`
- `docs/tutorials/03_observation_spaces_in_rl.md`
- `docs/tutorials/04_reward_design_for_robotics.md`
- `docs/tutorials/05_actor_critic_methods.md`
- `docs/concepts/rl_fundamentals.md` (if exists)

**Success Criteria:** Understand MDP formulation, observation/action/reward spaces

---

### Step 4: Understand the Math (Day 8-10)

**Objective:** Connect physics to RL

1. Read [Mechanics to RL Guide](research/mechanics_to_rl.md)
2. Read [Research Timeline](research/timeline.md) - Full read
3. Read [Code Mapping](research/code_mapping.md) (if exists)

**Files:**
- `docs/research/mechanics_to_rl.md`
- `docs/research/timeline.md`
- `docs/research/code_mapping.md` (if exists)

**Success Criteria:** Understand how Lagrangian dynamics map to observation spaces

---

### Step 5: Explore Examples (Day 11-12)

**Objective:** See how components work

1. Run [Custom Reward Example](examples/02_custom_reward.py)
2. Run [Custom Terrain Example](examples/03_custom_terrain.py)
3. Run [Custom Policy Example](examples/04_custom_policy.py)
4. Read [Extension Guide](architecture/extension_guide.md)

**Files:**
- `examples/02_custom_reward.py`
- `examples/03_custom_terrain.py`
- `examples/04_custom_policy.py`
- `docs/architecture/extension_guide.md`

**Success Criteria:** Can modify and run examples

---

### Step 6: Training Workflow (Day 13-15)

**Objective:** Train your first policy

1. Read [Complete Training Guide](tutorials/13_complete_training_guide.md)
2. Read [Training User Guide](user_guides/training.md)
3. Run [Training Workflow Example](examples/05_training_workflow.py)
4. Train a policy: `ballbot-train --config configs/train/ppo_directional.yaml`

**Files:**
- `docs/tutorials/13_complete_training_guide.md`
- `docs/user_guides/training.md`
- `examples/05_training_workflow.py`
- `configs/train/ppo_directional.yaml`

**Success Criteria:** Successfully train a policy

---

### Step 7: Advanced Topics (Day 16-18)

**Objective:** Deepen understanding

1. Read [Multi-Modal Fusion](tutorials/10_multimodal_fusion.md)
2. Read [Debugging & Visualization](tutorials/11_debugging_visualization.md)
3. Read [Terrain Generation](tutorials/15_terrain_generation.md)
4. Read [Advanced Topics](user_guides/advanced_topics.md)

**Files:**
- `docs/tutorials/10_multimodal_fusion.md`
- `docs/tutorials/11_debugging_visualization.md`
- `docs/tutorials/15_terrain_generation.md`
- `docs/user_guides/advanced_topics.md`

**Success Criteria:** Understand advanced concepts

---

## Path 2: RL Practitioner (Skip Basics)

**For:** Experienced with RL, new to ballbots/robotics

**Goal:** Quickly understand ballbot-specific aspects and start training

**Estimated Time:** 3-5 days (part-time)

### Step 1: Quick Setup (Day 1)

1. Read [Installation Guide](getting_started/installation.md) - Skip basics
2. Install and verify: `python scripts/test_pid.py`
3. Read [Quick Start](getting_started/quick_start.md)

**Files:**
- `docs/getting_started/installation.md`
- `docs/getting_started/quick_start.md`

---

### Step 2: Ballbot-Specific Concepts (Day 1-2)

1. Read [Ballbot Introduction](tutorials/00_ballbot_introduction.md)
2. Read [Research Timeline](research/timeline.md) - Skim for context
3. Read [Mechanics to RL Guide](research/mechanics_to_rl.md) - Focus on reward design

**Files:**
- `docs/tutorials/00_ballbot_introduction.md`
- `docs/research/timeline.md`
- `docs/research/mechanics_to_rl.md`

---

### Step 3: Architecture & Extension (Day 2-3)

1. Read [Architecture Overview](architecture/README.md)
2. Read [Component System](architecture/component_system.md)
3. Read [Extension Guide](architecture/extension_guide.md)
4. Run examples: `examples/02_custom_reward.py`, `examples/03_custom_terrain.py`

**Files:**
- `docs/architecture/README.md`
- `docs/architecture/component_system.md`
- `docs/architecture/extension_guide.md`
- `examples/02_custom_reward.py`
- `examples/03_custom_terrain.py`

---

### Step 4: Training (Day 3-4)

1. Read [Complete Training Guide](tutorials/13_complete_training_guide.md)
2. Read [Training User Guide](user_guides/training.md)
3. Train a policy
4. Read [Debugging & Visualization](tutorials/11_debugging_visualization.md)

**Files:**
- `docs/tutorials/13_complete_training_guide.md`
- `docs/user_guides/training.md`
- `docs/tutorials/11_debugging_visualization.md`

---

### Step 5: Advanced Topics (Day 4-5)

1. Read [Multi-Modal Fusion](tutorials/10_multimodal_fusion.md)
2. Read [Terrain Generation](tutorials/15_terrain_generation.md)
3. Read [Advanced Topics](user_guides/advanced_topics.md)

**Files:**
- `docs/tutorials/10_multimodal_fusion.md`
- `docs/tutorials/15_terrain_generation.md`
- `docs/user_guides/advanced_topics.md`

---

## Path 3: Control Theory Expert

**For:** Strong control theory background, new to RL

**Goal:** Understand how classical control translates to RL

**Estimated Time:** 1-2 weeks (part-time)

### Step 1: Research Foundation (Day 1-2)

1. Read [Research Timeline](research/timeline.md) - Full read
2. Read [Mechanics to RL Guide](research/mechanics_to_rl.md) - Focus on math
3. Read [Code Mapping](research/code_mapping.md) (if exists)

**Files:**
- `docs/research/timeline.md`
- `docs/research/mechanics_to_rl.md`
- `docs/research/code_mapping.md` (if exists)

---

### Step 2: RL Concepts (Day 3-5)

1. Read [Introduction to Gymnasium](tutorials/01_introduction_to_gymnasium.md)
2. Read [Reward Design](tutorials/04_reward_design_for_robotics.md)
3. Read [RL Fundamentals](concepts/rl_fundamentals.md) (if exists)
4. Read [Actor-Critic Methods](tutorials/05_actor_critic_methods.md)

**Files:**
- `docs/tutorials/01_introduction_to_gymnasium.md`
- `docs/tutorials/04_reward_design_for_robotics.md`
- `docs/tutorials/05_actor_critic_methods.md`
- `docs/concepts/rl_fundamentals.md` (if exists)

---

### Step 3: Architecture (Day 5-6)

1. Read [Architecture Overview](architecture/README.md)
2. Read [Component System](architecture/component_system.md)
3. Read [Design Decisions](architecture/design_decisions.md) (if exists)
4. Read [Code Walkthrough](api/code_walkthrough.md)

**Files:**
- `docs/architecture/README.md`
- `docs/architecture/component_system.md`
- `docs/architecture/design_decisions.md` (if exists)
- `docs/api/code_walkthrough.md`

---

### Step 4: Implementation (Day 7-10)

1. Read [Complete Training Guide](tutorials/13_complete_training_guide.md)
2. Run examples
3. Train a policy
4. Read [Advanced Topics](user_guides/advanced_topics.md)

**Files:**
- `docs/tutorials/13_complete_training_guide.md`
- `docs/user_guides/advanced_topics.md`
- `examples/`

---

## Path 4: Researcher (Paper → Code)

**For:** Want to understand research papers and their implementation

**Goal:** Map papers to code, understand implementation details

**Estimated Time:** 1 week (part-time)

### Step 1: Papers First (Day 1-2)

1. Read papers in `docs/research/papers/`:
   - Lauwers 2006
   - Nagarajan 2014
   - Carius 2022
   - Salehi 2025
2. Read [Research Timeline](research/timeline.md)
3. Read [Mechanics to RL Guide](research/mechanics_to_rl.md)

**Files:**
- `docs/research/papers/`
- `docs/research/timeline.md`
- `docs/research/mechanics_to_rl.md`

---

### Step 2: Code Mapping (Day 2-3)

1. Read [Code Mapping](research/code_mapping.md) (if exists)
2. Read [Code Walkthrough](api/code_walkthrough.md)
3. Read [Architecture Overview](architecture/README.md)

**Files:**
- `docs/research/code_mapping.md` (if exists)
- `docs/api/code_walkthrough.md`
- `docs/architecture/README.md`

---

### Step 3: Implementation Details (Day 3-5)

1. Read [Component System](architecture/component_system.md)
2. Read [Extension Guide](architecture/extension_guide.md)
3. Read [Advanced Topics](user_guides/advanced_topics.md)
4. Explore codebase: `ballbot_gym/`, `ballbot_rl/`

**Files:**
- `docs/architecture/component_system.md`
- `docs/architecture/extension_guide.md`
- `docs/user_guides/advanced_topics.md`
- Code files

---

### Step 4: Training & Evaluation (Day 5-7)

1. Read [Complete Training Guide](tutorials/13_complete_training_guide.md)
2. Read [Training User Guide](user_guides/training.md)
3. Run training workflow
4. Analyze results

**Files:**
- `docs/tutorials/13_complete_training_guide.md`
- `docs/user_guides/training.md`
- `examples/05_training_workflow.py`

---

## Quick Reference

### Essential Files by Topic

**Installation:**
- `docs/getting_started/installation.md`
- `docs/getting_started/quick_start.md`

**Theory:**
- `docs/research/timeline.md`
- `docs/research/mechanics_to_rl.md`

**Architecture:**
- `docs/architecture/README.md`
- `docs/architecture/component_system.md`

**Training:**
- `docs/tutorials/13_complete_training_guide.md`
- `docs/user_guides/training.md`

**Code:**
- `docs/api/code_walkthrough.md`
- `examples/`

**Reference:**
- `docs/user_guides/glossary.md`
- `docs/user_guides/faq.md`
- `docs/user_guides/quick_reference.md`

---

## Tips

1. **Don't skip examples** - They reinforce concepts
2. **Read code alongside docs** - See theory in practice
3. **Use glossary** - Look up unfamiliar terms
4. **Check FAQ** - Common questions answered
5. **Experiment** - Modify examples to learn

---

## Next Steps

After completing your path:
- Contribute improvements
- Experiment with new components
- Read advanced tutorials
- Explore research papers deeper

---

*Last Updated: 2025*

