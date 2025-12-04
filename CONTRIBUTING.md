# Contributing to openballbot-rl

Thank you for your interest in contributing! This guide will help you get started.

## Quick Start

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Code Style

We follow Python best practices:

- **Formatting**: Black (line length 88)
- **Linting**: Ruff
- **Type Checking**: MyPy
- **Docstrings**: Google style

Run before committing:

```bash
make format      # Format code
make lint        # Check linting
make type-check  # Check types
```

## Adding Components

### Adding a Reward Function

1. Create `ballbot_gym/rewards/my_reward.py`:

```python
from ballbot_gym.rewards.base import BaseReward

class MyReward(BaseReward):
    def __call__(self, state: dict) -> float:
        # Your implementation
        return reward_value
```

2. Register in `ballbot_gym/rewards/__init__.py`:

```python
from ballbot_gym.core.registry import ComponentRegistry
from ballbot_gym.rewards.my_reward import MyReward

ComponentRegistry.register_reward("my_reward", MyReward)
```

3. Add tests in `tests/unit/test_rewards.py`
4. Add example in `examples/`

See [Extension Guide](docs/architecture/extension_guide.md) for details.

### Adding a Terrain Generator

1. Create `ballbot_gym/terrain/my_terrain.py`:

```python
import numpy as np

def generate_my_terrain(n: int, **kwargs) -> np.ndarray:
    # Your implementation
    return terrain_array
```

2. Register in `ballbot_gym/terrain/__init__.py`
3. Add tests
4. Add example

### Adding a Policy Architecture

1. Create `ballbot_rl/policies/my_policy.py`
2. Inherit from `BaseFeaturesExtractor`
3. Register in `ballbot_rl/policies/__init__.py`
4. Add tests
5. Add example

## Testing

### Running Tests

```bash
make test           # Run all tests
make test-cov       # Run with coverage
pytest tests/ -v    # Verbose output
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use pytest fixtures from `tests/conftest.py`

Example:

```python
def test_my_reward():
    from ballbot_gym.core.factories import create_reward
    
    config = {"type": "my_reward", "config": {}}
    reward = create_reward(config)
    assert reward is not None
```

## Documentation

### Updating Documentation

- Code changes should update relevant docs
- Add examples for new features
- Update architecture docs if needed

### Documentation Structure

- `docs/architecture/` - System architecture
- `docs/user_guides/` - User-facing guides
- `docs/tutorials/` - Tutorials
- `examples/` - Code examples

## Pull Request Process

1. **Update Documentation**: If adding features, update docs
2. **Add Tests**: New code should have tests
3. **Run Checks**: Ensure `make lint` and `make test` pass
4. **Write Clear PR Description**: Explain what and why
5. **Link Issues**: Reference related issues

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Examples added (if applicable)
- [ ] All checks pass
- [ ] Backward compatibility maintained (if applicable)

## Component Addition Checklist

When adding a new component:

- [ ] Component inherits from base class
- [ ] Component registered in `__init__.py`
- [ ] Type hints added
- [ ] Docstrings added
- [ ] Unit tests added
- [ ] Integration test added
- [ ] Example usage provided
- [ ] Documentation updated

## Questions?

- Check [Architecture Docs](docs/architecture/)
- See [Examples](examples/)
- Open an issue for questions

## Code of Conduct

- Be respectful
- Be constructive
- Help others learn

Thank you for contributing!

