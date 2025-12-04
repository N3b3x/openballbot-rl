# Architecture Documentation

This directory contains detailed documentation about the openballbot-rl architecture.

## Overview

openballbot-rl uses a **plugin-based, configuration-driven architecture** that enables easy extension without modifying core code. The system is designed for:

- **Extensibility**: Add new components without touching core code
- **Maintainability**: Clear separation of concerns
- **Testability**: Dependency injection and factory patterns
- **Usability**: Configuration-driven component selection

The system is built around:

1. **Component Registry**: Central registry for all extensible components
2. **Factory Pattern**: Creates components from configuration
3. **Configuration System**: YAML-based configuration drives component selection
4. **Type Safety**: Comprehensive type hints and protocols

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Configuration (YAML)                │
│  - Component selection (reward, terrain, policy)             │
│  - Component parameters                                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Configuration System                            │
│  - Loads and validates YAML configs                         │
│  - Extracts component configs                               │
│  - Provides defaults and backward compatibility             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Factory Functions                           │
│  create_reward()  create_terrain()  create_policy()          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Component Registry                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Rewards    │  │   Terrains   │  │   Policies   │     │
│  │  Dictionary  │  │  Dictionary  │  │  Dictionary  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Environment (BBotSimulation)                     │
│  - Uses factories to create components                       │
│  - Runs simulation with configured components                │
└─────────────────────────────────────────────────────────────┘
```

## Component System

### Registry Pattern

All extensible components are registered in a central `ComponentRegistry`:

- **Rewards**: Reward functions that compute rewards from state
- **Terrains**: Terrain generators that create heightfields
- **Policies**: Policy architectures (feature extractors)
- **Sensors**: Sensor implementations (future)

### Auto-Registration

Components register themselves when their module is imported:

```python
# ballbot_gym/rewards/__init__.py
from ballbot_gym.core.registry import ComponentRegistry
from ballbot_gym.rewards.directional import DirectionalReward

ComponentRegistry.register_reward("directional", DirectionalReward)
```

### Factory Pattern

Components are created from configuration using factory functions:

```python
from ballbot_gym.core.factories import create_reward

config = {"type": "directional", "config": {"target_direction": [0, 1]}}
reward = create_reward(config)
```

## Extension Points

### 1. Reward Functions

**Location**: `ballbot_gym/rewards/`

**Base Class**: `BaseReward`

**Interface**: `__call__(state: Dict) -> float`

**Example**: See `examples/02_custom_reward.py`

### 2. Terrain Generators

**Location**: `ballbot_gym/terrain/`

**Interface**: `(n: int, **kwargs) -> np.ndarray`

**Example**: See `examples/03_custom_terrain.py`

### 3. Policy Architectures

**Location**: `ballbot_rl/policies/`

**Base Class**: `BaseFeaturesExtractor` (from Stable-Baselines3)

**Example**: See `examples/04_custom_policy.py`

### 4. Classical Controllers

**Location**: `ballbot_gym/controllers/`

**Purpose**: Classical control methods (e.g., PID) for comparison/testing

**Note**: Controllers are separate from RL policies. They're located in `ballbot_gym` because they interact with the environment directly, not through RL training.

**Example**: `ballbot_gym/controllers/pid.py` - PID controller for balance testing

## Configuration System

### Configuration Structure

```yaml
problem:
  terrain:
    type: "perlin"  # Component name
    config:         # Component-specific parameters
      scale: 25.0
      octaves: 4
      seed: null    # null = random
  
  reward:
    type: "directional"
    config:
      target_direction: [0.0, 1.0]
  
  policy:
    type: "mlp"
    config:
      hidden_sizes: [128, 128, 128, 128]
```

### Backward Compatibility

The system maintains backward compatibility with old config format:

```yaml
# Old format (still works)
problem:
  terrain_type: "perlin"

# New format (preferred)
problem:
  terrain:
    type: "perlin"
    config: {}
```

## Data Flow

### Environment Initialization

1. Config loaded from YAML
2. Component configs extracted
3. Factories create components from configs
4. Environment uses components

### Episode Execution

1. Environment resets
2. Terrain generator creates new terrain
3. Reward function computes rewards
4. Policy selects actions
5. MuJoCo simulates physics

## Testing Strategy

### Unit Tests

- Test components independently
- Test registry registration/retrieval
- Test factory functions
- Test configuration loading

### Integration Tests

- Test components in environment
- Test configuration-driven setup
- Test backward compatibility

### Example Tests

- Run example scripts
- Verify examples work end-to-end

## Development Workflow

### Adding a Component

1. Create component class/function
2. Register in `__init__.py`
3. Add tests
4. Add example usage
5. Update documentation

### Modifying Core

1. Maintain backward compatibility
2. Update tests
3. Update documentation
4. Update examples if needed

## Design Decisions

### Why Registry Pattern?

- **Discoverability**: Easy to see available components
- **Extensibility**: Add components without modifying core
- **Testability**: Mock registry for testing

### Why Factory Pattern?

- **Decoupling**: Environment doesn't know component details
- **Flexibility**: Switch components via config
- **Testability**: Inject test components

### Why Configuration-Driven?

- **Usability**: Non-programmers can change components
- **Reproducibility**: Config files are version-controlled
- **Experimentation**: Easy to try different combinations

## File Organization

```
ballbot_gym/
├── core/              # Core infrastructure
│   ├── registry.py    # Component registry
│   ├── factories.py   # Factory functions
│   └── config.py      # Config utilities
├── rewards/           # Reward functions
│   └── base.py        # BaseReward interface
├── terrain/           # Terrain generators (functions)
├── controllers/       # Classical controllers
│   └── pid.py         # PID controller
└── envs/              # Environment implementation

ballbot_rl/
├── policies/          # Policy architectures (uses BaseFeaturesExtractor from SB3)
└── training/          # Training utilities

examples/              # Usage examples
docs/architecture/     # Architecture documentation
```

## Documentation Structure

- **[Component System](component_system.md)** - How the registry works
- **[Extension Guide](extension_guide.md)** - How to add new components
- **[Configuration](configuration.md)** - Config system explained (if exists)
- **[Design Decisions](design_decisions.md)** - Why we made these choices (if exists)
- **[Data Flow](data_flow.md)** - How data flows through the system (if exists)

## Quick Links

- [Adding a Custom Reward](../examples/02_custom_reward.py)
- [Adding a Custom Terrain](../examples/03_custom_terrain.py)
- [Complete Training Example](../examples/05_training_workflow.py)

## Benefits

1. **Extensibility**: Add components without modifying core code
2. **Discoverability**: Registry shows all available components
3. **Testability**: Dependency injection enables easy testing
4. **Maintainability**: Clear separation of concerns
5. **Configuration-Driven**: Switch components via config files

## Next Steps

- Read [Component System](component_system.md) for details
- Follow [Extension Guide](extension_guide.md) to add components
- See [Examples](../examples/) for code examples
