# Component System Architecture

## Overview

The component system is the foundation of openballbot-rl's extensibility. It uses a **registry pattern** to manage all extensible components (rewards, terrains, policies, sensors).

## Registry Pattern

### How It Works

1. **Registration**: Components register themselves on import
2. **Discovery**: Registry maintains a dictionary of available components
3. **Retrieval**: Components are retrieved by name and instantiated with config

### ComponentRegistry Class

The `ComponentRegistry` is a class-level singleton that stores registered components:

```python
class ComponentRegistry:
    _rewards: Dict[str, Type[BaseReward]] = {}
    _terrains: Dict[str, Callable] = {}
    _policies: Dict[str, Type] = {}
    _sensors: Dict[str, Type] = {}
```

### Registration Lifecycle

1. **Module Import**: When a module is imported, it registers its components
2. **Auto-Registration**: Components register themselves in `__init__.py`
3. **Lazy Loading**: Components are only instantiated when needed

### Example: Reward Registration

```python
# ballbot_gym/rewards/__init__.py
from ballbot_gym.core.registry import ComponentRegistry
from ballbot_gym.rewards.directional import DirectionalReward

# Auto-register on import
ComponentRegistry.register_reward("directional", DirectionalReward)
```

When `ballbot_gym.rewards` is imported, `DirectionalReward` is automatically registered.

## Component Types

### Rewards

Rewards inherit from `BaseReward` and implement `__call__(state: Dict) -> float`:

```python
class MyReward(BaseReward):
    def __call__(self, state: Dict) -> float:
        # Compute reward
        return reward_value
```

### Terrains

Terrain generators are callable functions that take `n: int, **kwargs` and return `np.ndarray`:

```python
def generate_my_terrain(n: int, **kwargs) -> np.ndarray:
    # Generate terrain
    return terrain_array
```

**Note**: Terrain generators are functions, not classes. There is no `BaseTerrain` base class.

### Policies

Policies are classes (typically feature extractors) that can be instantiated:

```python
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MyPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space, **kwargs):
        # Initialize policy
        ...
```

**Note**: RL policies inherit from `BaseFeaturesExtractor` (Stable-Baselines3), not a custom `Policy` base class.

### Controllers (Classical Control)

Classical controllers (e.g., PID) are located in `ballbot_gym/controllers/`:

```python
from ballbot_gym.controllers import PID

pid = PID(dt=0.002, k_p=20, k_i=15, k_d=2)
```

**Note**: Controllers are separate from RL policies and don't inherit from any base class.

## Factory Pattern

Factories create components from configuration dictionaries:

```python
from ballbot_gym.core.factories import create_reward

config = {
    "type": "directional",
    "config": {"target_direction": [0.0, 1.0]}
}
reward = create_reward(config)
```

### Factory Functions

- `create_reward(config)` - Creates reward function
- `create_terrain(config)` - Creates terrain generator
- `create_policy(config)` - Creates policy class
- `validate_config(config, component_type)` - Validates configuration

## Best Practices

### 1. Always Inherit from Base Classes

```python
# ✅ Good
class MyReward(BaseReward):
    ...

# ❌ Bad
class MyReward:  # Doesn't inherit from BaseReward
    ...
```

### 2. Register in `__init__.py`

```python
# ✅ Good: Auto-registers on import
# rewards/__init__.py
ComponentRegistry.register_reward("my_reward", MyReward)

# ❌ Bad: Manual registration required
# User must call: ComponentRegistry.register_reward("my_reward", MyReward)
```

### 3. Use Type Hints

```python
# ✅ Good
def __call__(self, state: Dict[str, Any]) -> float:
    ...

# ❌ Bad
def __call__(self, state):
    ...
```

### 4. Validate Configuration

```python
# ✅ Good: Validate in factory
def create_reward(config: Dict[str, Any]) -> BaseReward:
    validate_config(config, "reward")
    ...

# ❌ Bad: No validation
def create_reward(config):
    ...
```

## Error Handling

The registry provides helpful error messages:

```python
# Unknown component
ComponentRegistry.get_reward("unknown")
# Raises: ValueError: Unknown reward: 'unknown'. Available rewards: ['directional', 'distance']

# Invalid config
create_reward({"type": "directional"})  # Missing config
# Raises: ValueError: Failed to create reward 'directional': ...
```

## Testing

Components can be tested independently:

```python
def test_my_reward():
    reward = MyReward(scale=0.1)
    state = {"vel": np.array([0.5, 0.3, 0.0])}
    result = reward(state)
    assert isinstance(result, float)
```

## Extension Points

The registry supports:

1. **Rewards**: Add new reward functions
2. **Terrains**: Add new terrain generators
3. **Policies**: Add new policy architectures
4. **Sensors**: Add new sensor types (future)

## See Also

- [Extension Guide](extension_guide.md) - Step-by-step guide
- [Configuration](configuration.md) - Config system
- [Examples](../../examples/) - Code examples

