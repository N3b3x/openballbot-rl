# 📚 openballbot-rl Documentation

*Comprehensive guides connecting research papers to implementation*

---

## 🎯 Overview

This documentation provides a complete understanding of the openballbot-rl project, from the foundational research papers to the practical implementation details. Each document is designed to be self-contained while building on previous concepts.

**New to the project?** Start with [Learning Paths](LEARNING_PATHS.md) to find your personalized learning journey!

---

## 🗺️ Learning Paths

### [Learning Paths Guide](LEARNING_PATHS.md) 🎓

**Choose your path based on your background:**
- **Complete Beginner** (0 → Production) - Start from scratch
- **RL Practitioner** (Skip Basics) - Quick start for RL experts
- **Control Theory Expert** - Bridge classical control to RL
- **Researcher** (Paper → Code) - Map research to implementation

Each path includes step-by-step guides, file references, and estimated time.

---

## 📖 Documentation Index

### 1. [Research Timeline](research/timeline.md) 📅

**Start here!** This document traces the evolution of ballbot research from 2006 to 2025, showing how each paper contributed essential pieces to the final RL system.

**What you'll learn:**
- The history of ballbot development
- Key contributions from each research paper
- How classical mechanics led to RL-based control
- Why certain design choices were made

**Reading time:** ~30 minutes

---

### 2. [Mechanics to RL Guide](research/mechanics_to_rl.md) 🔬

This tutorial bridges classical control theory and modern reinforcement learning, showing how Lagrangian dynamics translate into RL reward functions.

**What you'll learn:**
- How Lagrangian equations map to observation spaces
- Why constraints become reward penalties
- The complete mathematical formulation
- How physics informs reward design

**Reading time:** ~45 minutes

**Prerequisites:** Basic understanding of Lagrangian mechanics (covered in the document)

**See also:**
- [Code Mapping](research/code_mapping.md) - Maps papers to code files
- [Ballbot Mechanics](concepts/ballbot_mechanics.md) - Physics fundamentals
- [RL Fundamentals](concepts/rl_fundamentals.md) - MDP formulation

---

### 3. [Getting Started Guide](getting_started/README.md) 🏗️

A step-by-step implementation guide covering setup, training, and evaluation of the openballbot-rl system.

**What you'll learn:**
- How to build MuJoCo with anisotropic friction
- Environment architecture and design
- Reward system implementation
- Training PPO policies effectively
- Evaluation and troubleshooting

**Reading time:** ~60 minutes

**Prerequisites:** Understanding of RL basics (PPO, Gymnasium API)

**See also:**
- [Installation Guide](getting_started/installation.md) - Detailed installation steps
- [Quick Start](getting_started/quick_start.md) - 5-minute quick start
- [First Steps](getting_started/first_steps.md) - Your first environment run
- [Complete Training Guide](tutorials/13_complete_training_guide.md) - Full training workflow
- [Tutorial Dependencies](tutorials/DEPENDENCIES.md) - Prerequisites and reading order

---

### 4. [Visual Diagrams](design/diagrams.md) 📊

Visual representations of system architecture, data flow, and key concepts using ASCII diagrams.

**What you'll learn:**
- System architecture overview
- Complete data flow through the system
- Training pipeline visualization
- Observation and reward computation flows
- File dependencies and relationships

**Reading time:** ~20 minutes

**Use when:** You need a visual understanding of how components connect

**See also:**
- [Architecture Overview](architecture/README.md) - Detailed architecture documentation
- [Design Decisions](architecture/design_decisions.md) - Why design choices were made

---

### 5. [Code Walkthrough](api/code_walkthrough.md) 💻

A detailed guide to the key code files and their connections in openballbot-rl.

**What you'll learn:**
- How each major file works
- Code structure and organization
- Key functions and their purposes
- How components interact
- Real-world code examples

**Reading time:** ~45 minutes

**Use when:** You want to understand the implementation details or modify the code

**See also:**
- [Code Mapping](research/code_mapping.md) - Maps research papers to code files
- [Architecture Overview](architecture/README.md) - System design

---

### 6. [Glossary](user_guides/glossary.md) 📖

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

### 7. [FAQ](user_guides/faq.md) ❓

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

### 8. [Quick Reference](user_guides/quick_reference.md) ⚡

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

### 9. [Visualization Guide](user_guides/visualization.md) 🎨

Complete guide to visualizing environments, models, and training progress.

**What you'll learn:**
- How to visualize environments interactively
- Visualizing single trained models
- Batch visualization of archived models
- Plotting training progress
- Window titles and identification
- Troubleshooting visualization issues

**Reading time:** ~20 minutes

**Use when:** You want to visualize environments or trained models

**See also:**
- [Training Guide](user_guides/training.md) - How to train models
- [Experiment Organization](user_guides/experiment_organization.md) - Managing archived models

---

### 10. [Advanced Topics](user_guides/advanced_topics.md) 🔬

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

## 🧠 Conceptual Guides

### [Ballbot Mechanics](concepts/ballbot_mechanics.md) 🤖

**What:** Physics, dynamics, and why RL helps

**Topics:**
- What a ballbot is physically
- Why it's challenging to control
- How RL addresses these challenges
- Connection to research papers

**Use when:** You want to understand the physics foundation

---

### [RL Fundamentals](concepts/rl_fundamentals.md) 🎯

**What:** MDP formulation for ballbot

**Topics:**
- State space design
- Action space design
- Reward function design
- Policy architecture
- Training process

**Use when:** You want to understand how mechanics translate to RL

---

### [Reward Design](concepts/reward_design.md) 🎁

**What:** From constraints to rewards

**Topics:**
- Reward design principles
- Encoding objectives and constraints
- Reward shaping strategies
- Design trade-offs
- Common pitfalls

**Use when:** You want to design or modify reward functions

---

### [Observation Design](concepts/observation_design.md) 👁️

**What:** Multi-modal fusion for robust control

**Topics:**
- Observation space principles
- Proprioceptive vs. exteroceptive observations
- Multi-modal fusion
- Feature extraction
- Design considerations

**Use when:** You want to design or modify observation spaces

---

## 🏗️ Architecture Documentation

### [Architecture Overview](architecture/README.md) 🏛️

**What:** System architecture and design

**Topics:**
- Component registry pattern
- Factory functions
- Configuration system
- File structure
- Extension guide

**Use when:** You want to understand the system design or extend it

---

### [Design Decisions](architecture/design_decisions.md) 💡

**What:** Why design choices were made

**Topics:**
- Reward design decisions
- Terrain design decisions
- Observation design decisions
- Policy architecture decisions
- Algorithm decisions

**Use when:** You want to understand trade-offs and rationale

---

### [Component System](architecture/component_system.md) 🔧

**What:** Component registry and extensibility

**Topics:**
- Registry pattern
- Component registration
- Factory functions
- Configuration-driven design

**Use when:** You want to add custom components

---

### [Extension Guide](architecture/extension_guide.md) ➕

**What:** How to extend the system

**Topics:**
- Adding custom rewards
- Adding custom terrains
- Adding custom policies
- Best practices

**Use when:** You want to add new components

---

## 📚 Research Documentation

### [Research Timeline](research/timeline.md) 📅

**What:** Evolution from 2006 to 2025

**Topics:**
- Lauwers 2006: First ballbot
- Nagarajan 2014: Dynamics foundation
- Carius 2022: Constraint-aware control
- Salehi 2025: RL implementation

**Use when:** You want historical context

---

### [Code Mapping](research/code_mapping.md) 🔗

**What:** Maps research papers to code files

**Topics:**
- Paper contributions → code files
- Design decisions → implementations
- Mathematical connections → code

**Use when:** You want to see how theory maps to practice

---

## 📖 Tutorials

### [Tutorial Dependencies](tutorials/DEPENDENCIES.md) 🔗

**What:** Prerequisites and reading order

**Topics:**
- Dependency graph
- Tutorial prerequisites
- Recommended reading orders
- Learning paths

**Use when:** You want to plan your learning path

**See also:** [Learning Paths](LEARNING_PATHS.md) for personalized guides

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

All papers referenced in the documentation are stored in `research/papers/`:

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

- `ballbot_gym/bbot_env.py` - Main environment implementation
- `ballbot_gym/Rewards.py` - Reward function definitions
- `ballbot_gym/terrain.py` - Terrain generation
- `ballbot_rl/training/train.py` - Training script
- `ballbot_rl/evaluation/evaluate.py` - Evaluation script
- `configs/train_ppo_directional.yaml` - Training configuration

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
   - Explain how each paper contributed to openballbot-rl
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

- **Physics & Dynamics:** [Mechanics to RL Guide](research/mechanics_to_rl.md), Part 1; [Glossary](user_guides/glossary.md), Physics section
- **Reward Design:** [Mechanics to RL Guide](research/mechanics_to_rl.md), Part 3; [Code Walkthrough](api/code_walkthrough.md), Reward System
- **Training:** [Complete Training Guide](tutorials/13_complete_training_guide.md); [Code Walkthrough](api/code_walkthrough.md), Training Script
- **Evaluation:** [Complete Training Guide](tutorials/13_complete_training_guide.md); [Code Walkthrough](api/code_walkthrough.md), Evaluation Script
- **Research History:** [Research Timeline](research/timeline.md)
- **System Architecture:** [Visual Diagrams](design/diagrams.md); [Architecture Overview](architecture/README.md)
- **Code Implementation:** [Code Walkthrough](api/code_walkthrough.md)
- **Troubleshooting:** [FAQ](user_guides/faq.md), Troubleshooting section; [Getting Started](getting_started/README.md)
- **Quick Commands:** [Quick Reference](user_guides/quick_reference.md)

### By Paper

- **Lauwers 2006:** [Research Timeline](research/timeline.md), Section 2006
- **Nagarajan 2014:** [Research Timeline](research/timeline.md), Section 2014; [Mechanics to RL Guide](research/mechanics_to_rl.md), Part 1
- **Carius 2022:** [Research Timeline](research/timeline.md), Section 2022; [Mechanics to RL Guide](research/mechanics_to_rl.md), Part 2
- **Salehi 2025:** [Research Timeline](research/timeline.md), Section 2025; [Complete Training Guide](tutorials/13_complete_training_guide.md), throughout
- **Zakka 2025:** [Research Timeline](research/timeline.md), Section 2025 (MuJoCo Physics Fixes); [FAQ](user_guides/faq.md), Q4-Q5

---

## 🚀 Quick Start

New to the project? Choose your path:

### Option 1: Personalized Learning Path
1. **Read** [Learning Paths](LEARNING_PATHS.md) - Choose your path based on background
2. **Follow** your chosen path step-by-step
3. **Reference** [Tutorial Dependencies](tutorials/DEPENDENCIES.md) for prerequisites

### Option 2: Quick Start (Fastest)
1. **Read** [Research Timeline](research/timeline.md) (30 min) - Context
2. **Follow** [Getting Started Guide](getting_started/README.md), Steps 1-4 (2-3 hours)
3. **Train** your first policy!
4. **Evaluate** and analyze results

**Pro Tips:**
- Keep [Quick Reference](user_guides/quick_reference.md) open while working
- Check [FAQ](user_guides/faq.md) if you encounter issues
- Use [Visual Diagrams](design/diagrams.md) to understand system flow
- Reference [Code Walkthrough](api/code_walkthrough.md) when modifying code
- Check [Design Decisions](architecture/design_decisions.md) to understand trade-offs

---

## 📚 Complete Documentation List

### Getting Started
- [Learning Paths](LEARNING_PATHS.md) - Personalized learning guides
- [Getting Started Guide](getting_started/README.md) - Implementation guide
- [Installation Guide](getting_started/installation.md) - Setup instructions
- [Quick Start](getting_started/quick_start.md) - 5-minute quick start

### Research & Theory
- [Research Timeline](research/timeline.md) - Historical context and paper summaries
- [Mechanics to RL Guide](research/mechanics_to_rl.md) - Mathematical foundations
- [Code Mapping](research/code_mapping.md) - Papers → code connections

### Concepts
- [Ballbot Mechanics](concepts/ballbot_mechanics.md) - Physics and dynamics
- [RL Fundamentals](concepts/rl_fundamentals.md) - MDP formulation
- [Reward Design](concepts/reward_design.md) - Reward engineering
- [Observation Design](concepts/observation_design.md) - Multi-modal fusion

### Architecture
- [Architecture Overview](architecture/README.md) - System design
- [Design Decisions](architecture/design_decisions.md) - Design rationale
- [Component System](architecture/component_system.md) - Registry pattern
- [Extension Guide](architecture/extension_guide.md) - How to extend

### Tutorials
- [Tutorial Dependencies](tutorials/DEPENDENCIES.md) - Prerequisites and reading order
- [Complete Training Guide](tutorials/13_complete_training_guide.md) - Full training workflow
- See [tutorials/](tutorials/) for all tutorials

### Reference
- [Visual Diagrams](design/diagrams.md) - System architecture and data flow
- [Code Walkthrough](api/code_walkthrough.md) - Detailed code explanations
- [Glossary](user_guides/glossary.md) - Technical term definitions
- [FAQ](user_guides/faq.md) - Common questions and answers
- [Quick Reference](user_guides/quick_reference.md) - Formulas, commands, and snippets
- [Advanced Topics](user_guides/advanced_topics.md) - Deep dive into key components

---

**Happy Learning! 🚀**

*Last Updated: 2025*

