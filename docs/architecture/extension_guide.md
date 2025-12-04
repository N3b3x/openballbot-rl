# Extension Guide

This guide shows you how to add new components to openballbot-rl without modifying core code.

## Overview

Adding a new component involves:
1. Create the component class/function
2. Register it in the appropriate `__init__.py`
3. Use it via configuration

## Adding a Custom Reward Function

### Step 1: Create Reward Class

Create `ballbot_gym/rewards/my_reward.py`:

```python
"""My custom reward function."""
import numpy as np
from typing import Dict
from ballbot_gym.rewards.base import BaseReward

class MyReward(BaseReward):
    """Custom reward that does X."""
    
    def __init__(self, param1: float, param2: float = 1.0):
        self.param1 = param1
        self.param2 = param2
    
    def __call__(self, state: Dict) -> float:
        """Compute reward."""
        # Your reward logic here
        return reward_value
```

### Step 2: Register Reward

Update `ballbot_gym/rewards/__init__.py`:

```python
from ballbot_gym.core.registry import ComponentRegistry
from ballbot_gym.rewards.my_reward import MyReward

# Auto-register
ComponentRegistry.register_reward("my_reward", MyReward)

__all__ = [..., "MyReward"]
```

### Step 3: Use in Configuration

```yaml
problem:
  reward:
    type: "my_reward"
    config:
      param1: 0.5
      param2: 1.0
```

### Step 4: Test

```python
from ballbot_gym.core.factories import create_reward

config = {"type": "my_reward", "config": {"param1": 0.5}}
reward = create_reward(config)
assert isinstance(reward, MyReward)
```

## Adding a Custom Terrain Generator

### Step 1: Create Terrain Function

Create `ballbot_gym/terrain/my_terrain.py`:

```python
"""My custom terrain generator."""
import numpy as np
from typing import Optional

def generate_my_terrain(
    n: int,
    param1: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate custom terrain.
    
    Args:
        n: Grid size (should be odd)
        param1: Terrain parameter
        seed: Random seed
        
    Returns:
        1D array of height values, shape (n*n,), normalized to [0, 1]
    """
    assert n % 2 == 1, "n should be odd"
    
    # Generate terrain
    terrain = np.zeros((n, n))
    # ... your terrain generation logic ...
    
    # Normalize to [0, 1]
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 1e-8)
    
    return terrain.flatten()
```

### Step 2: Register Terrain

Update `ballbot_gym/terrain/__init__.py`:

```python
from ballbot_gym.core.registry import ComponentRegistry
from ballbot_gym.terrain.my_terrain import generate_my_terrain

# Auto-register
ComponentRegistry.register_terrain("my_terrain", generate_my_terrain)

__all__ = [..., "generate_my_terrain"]
```

### Step 3: Use in Configuration

```yaml
problem:
  terrain:
    type: "my_terrain"
    config:
      param1: 1.5
      seed: null  # null = random
```

## Adding a Custom Policy Architecture

### Step 1: Create Policy Class

Create `ballbot_rl/policies/my_policy.py`:

```python
"""My custom policy architecture."""
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class MyPolicy(BaseFeaturesExtractor):
    """Custom policy feature extractor."""
    
    def __init__(self, observation_space: gym.spaces.Dict, **kwargs):
        super().__init__(observation_space, features_dim=1)
        # Your policy architecture here
        ...
    
    def forward(self, observations) -> torch.Tensor:
        # Your forward pass
        ...
```

### Step 2: Register Policy

Update `ballbot_rl/policies/__init__.py`:

```python
from ballbot_gym.core.registry import ComponentRegistry
from ballbot_rl.policies.my_policy import MyPolicy

# Auto-register
ComponentRegistry.register_policy("my_policy", MyPolicy)

__all__ = [..., "MyPolicy"]
```

### Step 3: Use in Configuration

```yaml
problem:
  policy:
    type: "my_policy"
    config:
      hidden_size: 256
```

## Testing Your Component

### Unit Tests

Create `tests/unit/test_my_component.py`:

```python
import pytest
from ballbot_gym.core.factories import create_reward
from ballbot_gym.rewards.my_reward import MyReward

def test_my_reward_creation():
    config = {"type": "my_reward", "config": {"param1": 0.5}}
    reward = create_reward(config)
    assert isinstance(reward, MyReward)

def test_my_reward_computation():
    reward = MyReward(param1=0.5)
    state = {"vel": np.array([0.5, 0.3, 0.0])}
    result = reward(state)
    assert isinstance(result, float)
```

### Integration Tests

Test your component in the environment:

```python
import gymnasium as gym
import ballbot_gym

def test_my_reward_in_env():
    env = gym.make(
        "ballbot-v0.1",
        reward_config={"type": "my_reward", "config": {"param1": 0.5}},
        terrain_type="flat"
    )
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    env.close()
```

## Checklist

When adding a new component:

- [ ] Component inherits from base class (if applicable)
- [ ] Component is registered in `__init__.py`
- [ ] Component has type hints
- [ ] Component has docstrings
- [ ] Component is tested
- [ ] Example usage is provided
- [ ] Documentation is updated

## Common Pitfalls

### 1. Forgetting to Register

```python
# ❌ Bad: Component exists but isn't registered
class MyReward(BaseReward):
    ...

# ✅ Good: Registered in __init__.py
ComponentRegistry.register_reward("my_reward", MyReward)
```

### 2. Wrong Base Class

```python
# ❌ Bad: Doesn't inherit from BaseReward
class MyReward:
    ...

# ✅ Good: Inherits from BaseReward
class MyReward(BaseReward):
    ...
```

### 3. Missing Configuration Keys

```python
# ❌ Bad: Config missing "type" key
config = {"config": {"param1": 0.5}}

# ✅ Good: Config has "type" key
config = {"type": "my_reward", "config": {"param1": 0.5}}
```

## Examples

See the `examples/` directory for complete examples:
- `examples/02_custom_reward.py` - Custom reward example
- `examples/03_custom_terrain.py` - Custom terrain example
- `examples/04_custom_policy.py` - Custom policy example

## Contributing

When contributing a new component:

1. Follow the extension guide
2. Add tests
3. Add example usage
4. Update documentation
5. Submit a pull request

## See Also

- [Component System](component_system.md) - How registry works
- [Configuration](configuration.md) - Config system
- [Examples](../../examples/) - Code examples

