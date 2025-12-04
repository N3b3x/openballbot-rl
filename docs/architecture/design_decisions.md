# Design Decisions: Why We Made These Choices

*Documenting the rationale behind key architectural and design decisions*

---

## Overview

This document explains why specific design choices were made in openballbot-rl. Understanding these decisions helps you:
- Appreciate the trade-offs involved
- Make informed modifications
- Avoid common pitfalls
- Extend the system appropriately

---

## Reward Design Decisions

### Why Directional Reward?

**Decision:** Use directional reward (velocity toward goal) instead of waypoint following or position-based rewards.

**Rationale:**
1. **Flexibility**: Works for continuous navigation without requiring exact waypoints
2. **Simplicity**: Single target direction vector, easy to specify
3. **Generalization**: Agent learns to navigate in any direction, not just to specific points
4. **Efficiency**: Doesn't require path planning or waypoint management

**Alternatives Considered:**
- **Waypoint following**: More complex, requires path planning
- **Position-based**: Requires exact goal positions, less flexible
- **Sparse rewards**: Too rare, hard to learn from

**Code Location:**
- `ballbot_gym/rewards/directional.py`

**Trade-offs:**
- ✅ Simple and flexible
- ✅ Enables continuous navigation
- ❌ Doesn't enforce exact paths
- ❌ Requires careful scaling

---

### Why Action Penalty?

**Decision:** Include action penalty (-α₂||a||²) in reward function.

**Rationale:**
1. **Prevents reward hacking**: Without penalty, agent might spin in place (high velocity, no progress)
2. **Encourages efficiency**: Prevents excessive control effort
3. **Energy efficiency**: Encourages minimal control for given task
4. **Stability**: Prevents high-frequency oscillations

**Alternatives Considered:**
- **No penalty**: Agent learns inefficient behaviors
- **Hard limits**: Prevents exploration, less flexible
- **Frequency penalty**: More complex, harder to tune

**Code Location:**
- `ballbot_gym/rewards/directional.py` - Included in reward computation

**Trade-offs:**
- ✅ Prevents reward hacking
- ✅ Encourages efficiency
- ❌ Must be carefully scaled (too large → agent too conservative)
- ❌ Adds hyperparameter to tune

---

### Why Survival Bonus?

**Decision:** Include survival bonus (α₃·𝟙[upright]) in reward function.

**Rationale:**
1. **Encourages balance**: Provides steady reward for staying upright
2. **Prevents falls**: Agent learns to avoid falling
3. **Exploration**: Allows agent to explore while maintaining balance
4. **Dense rewards**: Provides learning signal at every step

**Alternatives Considered:**
- **Sparse reward**: Only reward at episode end → too rare
- **Tilt penalty only**: Negative signal → harder to learn
- **No survival reward**: Agent might learn to fall gracefully

**Code Location:**
- `ballbot_gym/rewards/directional.py` - Included in reward computation
- `ballbot_gym/envs/ballbot_env.py` - Checks upright condition

**Trade-offs:**
- ✅ Encourages balance
- ✅ Provides dense rewards
- ❌ Must balance with other rewards (too large → agent too conservative)
- ❌ Requires threshold tuning

---

## Terrain Design Decisions

### Why Perlin Noise Terrain?

**Decision:** Use Perlin noise for procedural terrain generation.

**Rationale:**
1. **Infinite variety**: Procedural generation creates unlimited terrain variations
2. **Realistic**: Smooth gradients match real-world terrain
3. **Controllable**: Parameters (scale, octaves) control difficulty
4. **Efficient**: Fast generation, no need to store terrain files
5. **Generalization**: Training on diverse terrains → robust policies

**Alternatives Considered:**
- **Fixed terrains**: Limited variety, doesn't generalize
- **Random noise**: Too chaotic, unrealistic
- **Heightmap imports**: Requires external files, less flexible

**Code Location:**
- `ballbot_gym/terrain/perlin.py`

**Trade-offs:**
- ✅ Infinite variety
- ✅ Realistic smooth gradients
- ✅ Controllable difficulty
- ❌ May not match specific real-world terrains exactly
- ❌ Requires parameter tuning

---

### Why Perlin Noise Parameters?

**Decision:** Default parameters: scale=25.0, octaves=4, persistence=0.2, lacunarity=2.0

**Rationale:**
1. **Scale (25.0)**: Creates terrain features appropriate for ballbot size
2. **Octaves (4)**: Good balance between detail and smoothness
3. **Persistence (0.2)**: Prevents high-frequency noise from dominating
4. **Lacunarity (2.0)**: Standard value for Perlin noise

**Alternatives Considered:**
- **Higher scale**: Larger features → easier navigation
- **Lower scale**: Smaller features → harder navigation
- **More octaves**: More detail → potentially harder
- **Less octaves**: Less detail → potentially easier

**Code Location:**
- `ballbot_gym/terrain/perlin.py` - Default parameters
- `configs/train/ppo_directional.yaml` - Configurable via config

**Trade-offs:**
- ✅ Good default for learning
- ✅ Can be tuned for difficulty
- ❌ May need adjustment for specific tasks

---

## Observation Design Decisions

### Why Multi-Modal Observations?

**Decision:** Combine proprioception (internal state) and vision (depth images).

**Rationale:**
1. **Balance control**: Proprioception needed for balance (tilt angles, velocities)
2. **Terrain navigation**: Vision needed to see terrain ahead
3. **State disambiguation**: Vision prevents state aliasing (same proprioceptive state, different terrains)
4. **Robustness**: Both modalities provide complementary information

**Alternatives Considered:**
- **Proprioception only**: Works on flat terrain, fails on uneven terrain
- **Vision only**: Missing balance information, can't control
- **Other sensors**: More complex, may not be necessary

**Code Location:**
- `ballbot_gym/envs/observation_spaces.py` - Observation space definition
- `ballbot_rl/policies/mlp_policy.py` - Multi-modal fusion

**Trade-offs:**
- ✅ Robust performance
- ✅ Handles diverse terrains
- ❌ More complex architecture
- ❌ Requires vision encoder training

---

### Why Depth Images (Not RGB)?

**Decision:** Use depth images instead of RGB images.

**Rationale:**
1. **Terrain navigation**: Depth provides direct distance information
2. **Simplicity**: Easier to process than RGB
3. **Efficiency**: Lower dimensionality than RGB
4. **Sufficiency**: Depth is sufficient for terrain navigation

**Alternatives Considered:**
- **RGB images**: More information, but harder to process
- **RGB-D**: Both RGB and depth → more complex, may be overkill
- **LiDAR**: More accurate, but not available in simulation

**Code Location:**
- `ballbot_gym/sensors/rgbd.py` - Depth camera rendering
- `ballbot_rl/encoders/models.py` - Depth encoder

**Trade-offs:**
- ✅ Direct distance information
- ✅ Simpler than RGB
- ✅ Efficient processing
- ❌ Less information than RGB-D
- ❌ May miss color/texture cues

---

### Why Depth Encoder Pretraining?

**Decision:** Pretrain depth encoder on diverse terrain data before RL training.

**Rationale:**
1. **Feature extraction**: Encoder learns useful terrain features
2. **Dimensionality reduction**: Reduces 128×128 → 20 dimensions
3. **Stability**: Frozen encoder prevents feature drift during RL
4. **Efficiency**: Faster RL training with pretrained features

**Alternatives Considered:**
- **End-to-end training**: Encoder and policy train together → slower, less stable
- **No encoder**: Raw pixels → too high-dimensional, slow training
- **Hand-crafted features**: Less flexible, requires domain knowledge

**Code Location:**
- `ballbot_rl/encoders/pretrain.py` - Pretraining script
- `ballbot_rl/encoders/models.py` - Encoder architecture

**Trade-offs:**
- ✅ Faster RL training
- ✅ Stable features
- ✅ Good dimensionality reduction
- ❌ Requires pretraining step
- ❌ Encoder architecture fixed

---

## Policy Architecture Decisions

### Why MLP (Not RNN)?

**Decision:** Use MLP policy instead of RNN (LSTM/GRU).

**Rationale:**
1. **High-frequency control**: Ballbot requires high-frequency control (500 Hz), RNN unnecessary
2. **Vision provides history**: Depth encoder captures spatial context
3. **Simplicity**: MLP is simpler, faster, easier to train
4. **Sufficiency**: MLP sufficient for this task

**Alternatives Considered:**
- **RNN**: Adds temporal modeling, but may be unnecessary
- **Transformer**: More complex, may be overkill
- **CNN-only**: Missing proprioception, can't balance

**Code Location:**
- `ballbot_rl/policies/mlp_policy.py` - MLP architecture

**Trade-offs:**
- ✅ Simple and fast
- ✅ Sufficient for task
- ❌ No explicit temporal modeling
- ❌ May struggle with very long-term dependencies

---

### Why Late Fusion?

**Decision:** Use late fusion (separate extractors → concatenate → MLP) instead of early fusion.

**Rationale:**
1. **Different modalities**: Proprioception and vision need different processing
2. **Flexibility**: Can use different architectures for each modality
3. **Efficiency**: Each modality processed appropriately
4. **Modularity**: Easy to modify one modality without affecting others

**Alternatives Considered:**
- **Early fusion**: Concatenate raw observations → single network
- **Attention**: More complex, may be unnecessary
- **Separate policies**: Too complex, hard to coordinate

**Code Location:**
- `ballbot_rl/policies/mlp_policy.py` - Late fusion architecture

**Trade-offs:**
- ✅ Handles multi-modal inputs well
- ✅ Flexible architecture
- ✅ Modular design
- ❌ More complex than early fusion
- ❌ Requires careful feature dimension matching

---

## Algorithm Decisions

### Why PPO?

**Decision:** Use Proximal Policy Optimization (PPO) instead of other RL algorithms.

**Rationale:**
1. **Stability**: Clipped objective prevents large policy updates
2. **Sample efficiency**: Good balance between sample efficiency and stability
3. **Simplicity**: Easier to implement and tune than TRPO
4. **Proven**: Works well for continuous control tasks
5. **Infrastructure**: Stable-Baselines3 provides robust implementation

**Alternatives Considered:**
- **SAC**: Off-policy, may be more sample efficient, but more complex
- **TD3**: Good for continuous control, but more hyperparameters
- **A2C**: Simpler, but less stable than PPO
- **TRPO**: More stable, but more complex

**Code Location:**
- `ballbot_rl/training/train.py` - Uses Stable-Baselines3 PPO

**Trade-offs:**
- ✅ Stable training
- ✅ Good sample efficiency
- ✅ Well-supported
- ❌ May not be optimal for all tasks
- ❌ Requires hyperparameter tuning

---

## Architecture Decisions

### Why Component Registry Pattern?

**Decision:** Use registry pattern for extensible components (rewards, terrains, policies).

**Rationale:**
1. **Extensibility**: Add components without modifying core code
2. **Discoverability**: Easy to see available components
3. **Testability**: Can mock registry for testing
4. **Configuration-driven**: Switch components via config files
5. **Modularity**: Clear separation of concerns

**Alternatives Considered:**
- **Hard-coded components**: Simple, but not extensible
- **Plugin system**: More complex, may be overkill
- **Factory pattern only**: Less discoverable

**Code Location:**
- `ballbot_gym/core/registry.py` - Component registry
- `ballbot_gym/core/factories.py` - Factory functions

**Trade-offs:**
- ✅ Highly extensible
- ✅ Easy to discover components
- ✅ Configuration-driven
- ❌ More complex than hard-coding
- ❌ Requires registration boilerplate

---

### Why Configuration-Driven?

**Decision:** Use YAML configuration files to drive component selection and parameters.

**Rationale:**
1. **Usability**: Non-programmers can change components
2. **Reproducibility**: Config files are version-controlled
3. **Experimentation**: Easy to try different combinations
4. **Separation**: Separates code from configuration
5. **Flexibility**: Switch components without code changes

**Alternatives Considered:**
- **Hard-coded**: Simple, but not flexible
- **Command-line args**: Flexible, but verbose
- **Code-based config**: Flexible, but requires code changes

**Code Location:**
- `configs/train/ppo_directional.yaml` - Training config
- `configs/env/{terrain}_{reward}.yaml` - Environment configs (e.g., `perlin_directional.yaml`)
- `ballbot_gym/core/config.py` - Config utilities

**Trade-offs:**
- ✅ Easy to experiment
- ✅ Reproducible
- ✅ User-friendly
- ❌ Requires config parsing
- ❌ May need validation

---

### Why Separate Packages (ballbot_gym vs ballbot_rl)?

**Decision:** Split into `ballbot_gym` (environment) and `ballbot_rl` (training).

**Rationale:**
1. **Separation of concerns**: Environment vs. training logic
2. **Reusability**: Environment can be used without RL
3. **Clarity**: Clear boundaries between components
4. **Modularity**: Can swap RL implementations
5. **Installation**: Can install environment separately

**Alternatives Considered:**
- **Single package**: Simpler, but less modular
- **More packages**: More modular, but more complex
- **Different split**: Could split differently, but current split is logical

**Code Location:**
- `ballbot_gym/` - Environment package
- `ballbot_rl/` - RL package

**Trade-offs:**
- ✅ Clear separation
- ✅ Modular design
- ✅ Reusable environment
- ❌ More packages to manage
- ❌ Requires careful dependency management

---

## Implementation Decisions

### Why MuJoCo?

**Decision:** Use MuJoCo physics engine for simulation.

**Rationale:**
1. **Accuracy**: Accurate physics simulation
2. **Speed**: Fast enough for RL training
3. **Features**: Supports complex contact dynamics
4. **Anisotropic friction**: Can be patched for omniwheels
5. **Industry standard**: Widely used in robotics RL

**Alternatives Considered:**
- **PyBullet**: Good alternative, but less accurate
- **Gazebo**: More features, but slower
- **Custom physics**: More control, but much more work

**Code Location:**
- `ballbot_gym/models/ballbot.xml` - MuJoCo model
- `tools/mujoco_fix.patch` - Anisotropic friction patch

**Trade-offs:**
- ✅ Accurate physics
- ✅ Fast simulation
- ✅ Industry standard
- ❌ Requires patching for omniwheels
- ❌ Must build from source

---

### Why Anisotropic Friction Patch?

**Decision:** Patch MuJoCo to support anisotropic friction for omniwheel simulation.

**Rationale:**
1. **Physical accuracy**: Omniwheels have directional friction
2. **Necessity**: Without patch, omniwheels don't work correctly
3. **Simplicity**: Patch is simpler than full wheel model
4. **Efficiency**: Avoids complex wheel physics

**Alternatives Considered:**
- **Full wheel model**: More accurate, but much more complex
- **Isotropic friction**: Simpler, but incorrect physics
- **Different simulation**: Would require different engine

**Code Location:**
- `tools/mujoco_fix.patch` - MuJoCo patch

**Trade-offs:**
- ✅ Enables omniwheel simulation
- ✅ Simpler than full wheel model
- ❌ Requires patching MuJoCo
- ❌ Must build from source

---

## Summary

**Key Design Principles:**
1. **Simplicity**: Prefer simpler solutions when they work
2. **Extensibility**: Enable easy extension without core changes
3. **Modularity**: Clear separation of concerns
4. **Robustness**: Design for diverse scenarios
5. **Usability**: Make it easy to use and experiment

**Common Trade-offs:**
- Simplicity vs. Flexibility
- Accuracy vs. Speed
- Modularity vs. Simplicity
- Generalization vs. Specificity

**When to Reconsider:**
- If requirements change significantly
- If better alternatives emerge
- If current design causes problems
- If performance is insufficient

---

## Next Steps

- Read [Architecture Overview](README.md) for system design
- Read [Component System](component_system.md) for registry details
- Read [Extension Guide](extension_guide.md) to add components
- Explore [Code Walkthrough](../api/code_walkthrough.md) to see implementations

---

*Last Updated: 2025*

