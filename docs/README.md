# 📚 OpenBallBot-RL Documentation

*Comprehensive guides connecting research papers to implementation*

---

## 🎯 Overview

This documentation provides a complete understanding of the OpenBallBot-RL project, from the foundational research papers to the practical implementation details. Each document is designed to be self-contained while building on previous concepts.

---

## 📖 Documentation Index

### 1. [Research Timeline](01_research_timeline.md) 📅

**Start here!** This document traces the evolution of ballbot research from 2006 to 2025, showing how each paper contributed essential pieces to the final RL system.

**What you'll learn:**
- The history of ballbot development
- Key contributions from each research paper
- How classical mechanics led to RL-based control
- Why certain design choices were made

**Reading time:** ~30 minutes

---

### 2. [Mechanics to RL Guide](02_mechanics_to_rl.md) 🔬

This tutorial bridges classical control theory and modern reinforcement learning, showing how Lagrangian dynamics translate into RL reward functions.

**What you'll learn:**
- How Lagrangian equations map to observation spaces
- Why constraints become reward penalties
- The complete mathematical formulation
- How physics informs reward design

**Reading time:** ~45 minutes

**Prerequisites:** Basic understanding of Lagrangian mechanics (covered in the document)

---

### 3. [Environment & RL Workflow](03_environment_and_rl.md) 🏗️

A step-by-step implementation guide covering setup, training, and evaluation of the OpenBallBot-RL system.

**What you'll learn:**
- How to build MuJoCo with anisotropic friction
- Environment architecture and design
- Reward system implementation
- Training PPO policies effectively
- Evaluation and troubleshooting

**Reading time:** ~60 minutes

**Prerequisites:** Understanding of RL basics (PPO, Gymnasium API)

---

### 4. [Visual Diagrams](04_visual_diagrams.md) 📊

Visual representations of system architecture, data flow, and key concepts using ASCII diagrams.

**What you'll learn:**
- System architecture overview
- Complete data flow through the system
- Training pipeline visualization
- Observation and reward computation flows
- File dependencies and relationships

**Reading time:** ~20 minutes

**Use when:** You need a visual understanding of how components connect

---

### 5. [Code Walkthrough](05_code_walkthrough.md) 💻

A detailed guide to the key code files and their connections in OpenBallBot-RL.

**What you'll learn:**
- How each major file works
- Code structure and organization
- Key functions and their purposes
- How components interact
- Real-world code examples

**Reading time:** ~45 minutes

**Use when:** You want to understand the implementation details or modify the code

---

### 6. [Glossary](06_glossary.md) 📖

A comprehensive glossary of technical terms used throughout the documentation.

**What you'll find:**
- Physics and dynamics terms
- Reinforcement learning concepts
- Robotics terminology
- Simulation and control theory terms
- Mathematical notation explanations

**Reading time:** Reference document (browse as needed)

**Use when:** You encounter unfamiliar terms or need quick definitions

---

### 7. [FAQ](07_faq.md) ❓

Frequently asked questions with detailed answers covering common topics and issues.

**What you'll find:**
- General questions about ballbots and RL
- Physics and simulation questions
- Implementation and training questions
- Troubleshooting common issues
- Tips and best practices

**Reading time:** ~30 minutes (or browse specific questions)

**Use when:** You have a specific question or encounter a problem

---

### 8. [Quick Reference](08_quick_reference.md) ⚡

Essential formulas, commands, and information at a glance.

**What you'll find:**
- Key mathematical formulas
- Common commands and scripts
- Hyperparameter defaults
- File structure overview
- Python code snippets
- Debugging tips

**Reading time:** Reference document (print for quick access)

**Use when:** You need quick access to formulas or commands

---

### 9. [Advanced Topics](09_advanced_topics.md) 🔬

In-depth analysis of advanced concepts and implementation details.

**What you'll learn:**
- Depth encoder pretraining process and architecture
- MuJoCo anisotropic friction patch technical details
- Terrain generation algorithms and robot placement
- Camera rendering and RGB-D processing
- Observation normalization strategies
- Reward component analysis and design choices
- Training callbacks and evaluation metrics
- Neural network architecture details

**Reading time:** ~60 minutes

**Prerequisites:** Understanding of basic concepts from other docs

**Use when:** You want to understand implementation details or modify the system

---

## 🗺️ Recommended Reading Path

### For Beginners

1. **Start with Research Timeline** - Understand the big picture
2. **Read Mechanics to RL Guide** - Learn the mathematical foundations
3. **Follow Environment & RL Workflow** - Implement the system
4. **Reference Visual Diagrams** - See how everything connects
5. **Check FAQ** - When you have questions

### For Experienced RL Practitioners

1. **Skim Research Timeline** - Get context on papers
2. **Focus on Mechanics to RL Guide** - Understand reward design
3. **Jump to Environment & RL Workflow** - Start implementing
4. **Use Code Walkthrough** - Understand implementation details
5. **Keep Quick Reference handy** - For formulas and commands

### For Control Theory Experts

1. **Read Mechanics to RL Guide** - See how classical control translates to RL
2. **Review Research Timeline** - Understand the evolution
3. **Reference Environment & RL Workflow** - For implementation details
4. **Check Glossary** - For RL-specific terminology
5. **Use Code Walkthrough** - To see how theory maps to code

---

## 📚 Research Papers Reference

All papers referenced in the documentation are stored in `../research_papers/`:

- **Lauwers2006_inverse_mouse_ball.pdf** - Original ballbot prototype
- **Nagarajan2014_ballbot.pdf** - Lagrangian dynamics and control
- **Carius2022_constrained_path_integral.pdf** - Constraint-aware control
- **Salehi2025_ballbot_rl.pdf** - Complete RL navigation system

External links:
- **Raffin2021 Stable-Baselines3** - [JMLR Paper](https://jmlr.org/papers/v22/20-1364.html)
- **Zakka2025 MuJoCo Playground** - [arXiv:2502.08844](https://arxiv.org/abs/2502.08844)

---

## 🔗 Related Resources

### Codebase Files

- `ballbotgym/ballbotgym/bbot_env.py` - Main environment implementation
- `ballbotgym/ballbotgym/Rewards.py` - Reward function definitions
- `ballbotgym/ballbotgym/terrain.py` - Terrain generation
- `scripts/train.py` - Training script
- `scripts/test.py` - Evaluation script
- `config/train_ppo_directional.yaml` - Training configuration

### External Tutorials

The project includes additional tutorials in `../Tutorials/`:
- Introduction to Gymnasium
- Action Spaces in RL
- Observation Spaces in RL
- Reward Design for Robotics
- Actor-Critic Methods
- Camera Rendering in MuJoCo
- Working with MuJoCo Simulation State

---

## 💡 Key Concepts Quick Reference

### Lagrangian Dynamics
\[
L = T - V = \frac{1}{2}\dot{\mathbf{q}}^T \mathbf{M}(\mathbf{q}) \dot{\mathbf{q}} - mgh(\mathbf{q})
\]

### Reward Function
\[
r = \frac{\mathbf{v}_{xy} \cdot \mathbf{g}}{100} - 0.0001\|\mathbf{a}\|^2 + 0.02 \cdot \mathbb{1}[\text{upright}]
\]

### Observation Space
- Orientation: \((\phi, \theta, \psi)\)
- Angular velocities: \((\dot{\phi}, \dot{\theta}, \dot{\psi})\)
- Linear velocities: \((\dot{x}, \dot{y}, \dot{z})\)
- Depth images: Encoded camera observations

### Action Space
- Three motor torques: \([\tau_0, \tau_1, \tau_2]^T\)
- Normalized to \([-1, 1]\)
- Scaled to \([-10, 10]\) rad/s internally

---

## 🎓 Learning Objectives

After reading this documentation, you should be able to:

1. **Understand the research foundation**
   - Explain how each paper contributed to OpenBallBot-RL
   - Understand the evolution from classical control to RL

2. **Grasp the mathematical foundations**
   - Derive the Lagrangian dynamics
   - Understand how constraints become rewards
   - See the connection between physics and RL

3. **Implement the system**
   - Build MuJoCo with required patches
   - Train PPO policies effectively
   - Evaluate and analyze results

4. **Extend the system**
   - Modify reward functions
   - Add new observations
   - Experiment with different algorithms

---

## 🤝 Contributing

Found an error or want to improve the documentation?

1. Check existing issues
2. Create a new issue with details
3. Submit a pull request with improvements

---

## 📝 Document Structure

Each document follows this structure:

1. **Introduction** - Overview and motivation
2. **Core Concepts** - Fundamental ideas with math
3. **Real-World Examples** - Implementation details
4. **Step-by-Step Guides** - Practical instructions
5. **Summary** - Key takeaways

---

## 🔍 Finding Information

### By Topic

- **Physics & Dynamics:** [Mechanics to RL Guide](02_mechanics_to_rl.md), Part 1; [Glossary](06_glossary.md), Physics section
- **Reward Design:** [Mechanics to RL Guide](02_mechanics_to_rl.md), Part 3; [Code Walkthrough](05_code_walkthrough.md), Reward System
- **Training:** [Environment & RL Workflow](03_environment_and_rl.md), Step 4; [Code Walkthrough](05_code_walkthrough.md), Training Script
- **Evaluation:** [Environment & RL Workflow](03_environment_and_rl.md), Step 5; [Code Walkthrough](05_code_walkthrough.md), Evaluation Script
- **Research History:** [Research Timeline](01_research_timeline.md)
- **System Architecture:** [Visual Diagrams](04_visual_diagrams.md)
- **Code Implementation:** [Code Walkthrough](05_code_walkthrough.md)
- **Troubleshooting:** [FAQ](07_faq.md), Troubleshooting section; [Environment & RL Workflow](03_environment_and_rl.md), Step 6
- **Quick Commands:** [Quick Reference](08_quick_reference.md)

### By Paper

- **Lauwers 2006:** [Research Timeline](01_research_timeline.md), Section 2006
- **Nagarajan 2014:** [Research Timeline](01_research_timeline.md), Section 2014; [Mechanics to RL Guide](02_mechanics_to_rl.md), Part 1
- **Carius 2022:** [Research Timeline](01_research_timeline.md), Section 2022; [Mechanics to RL Guide](02_mechanics_to_rl.md), Part 2
- **Salehi 2025:** [Research Timeline](01_research_timeline.md), Section 2025; [Environment & RL Workflow](03_environment_and_rl.md), throughout
- **Zakka 2025:** [Research Timeline](01_research_timeline.md), Section 2025 (MuJoCo Physics Fixes); [FAQ](07_faq.md), Q4-Q5

---

## 🚀 Quick Start

New to the project? Follow this path:

1. **Read** [Research Timeline](01_research_timeline.md) (30 min)
2. **Skim** [Mechanics to RL Guide](02_mechanics_to_rl.md) (20 min)
3. **Follow** [Environment & RL Workflow](03_environment_and_rl.md), Steps 1-4 (2-3 hours)
4. **Train** your first policy!
5. **Evaluate** and analyze results

**Pro Tips:**
- Keep [Quick Reference](08_quick_reference.md) open while working
- Check [FAQ](07_faq.md) if you encounter issues
- Use [Visual Diagrams](04_visual_diagrams.md) to understand system flow
- Reference [Code Walkthrough](05_code_walkthrough.md) when modifying code

---

## 📚 Complete Documentation List

1. [Research Timeline](01_research_timeline.md) - Historical context and paper summaries
2. [Mechanics to RL Guide](02_mechanics_to_rl.md) - Mathematical foundations
3. [Environment & RL Workflow](03_environment_and_rl.md) - Implementation guide
4. [Visual Diagrams](04_visual_diagrams.md) - System architecture and data flow
5. [Code Walkthrough](05_code_walkthrough.md) - Detailed code explanations
6. [Glossary](06_glossary.md) - Technical term definitions
7. [FAQ](07_faq.md) - Common questions and answers
8. [Quick Reference](08_quick_reference.md) - Formulas, commands, and snippets
9. [Advanced Topics](09_advanced_topics.md) - Deep dive into key components

---

**Happy Learning! 🚀**

*Last Updated: 2025*

