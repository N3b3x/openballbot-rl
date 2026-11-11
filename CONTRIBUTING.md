# Contributing to openballbot-rl

Thank you for your interest in contributing. This guide will help you get started.

## Which part are you changing?

| Path | Where | Notes |
|---|---|---|
| Current pipeline | `experiments/warp_ppo/`, `experiments/mjx_port/`, `scripts/study/`, `specs/` | GPU (NVIDIA, CUDA) and the study venv; see [docs/pipeline.md](docs/pipeline.md) for every stage, flag and environment variable |
| ROS 2 package | `ros2/ballbot_control/` | The tests need neither ROS nor a GPU |
| Documentation and website | `docs/`, `site/`, `scripts/docs/`, `scripts/site/` | Numbers come from the study log, with their S-number |
| Legacy SB3 path | `ballbot_gym/`, `ballbot_rl/`, `configs/`, `examples/`, `tests/` | Inherited from the upstream OpenBallBot-RL; the sections below on components, code style and `make` targets describe this path |

This project is simulation only. Please do not add text or numbers that suggest a physical robot or hardware results, and
put every new number in the study log ([docs/research/tracking_study_log.md](docs/research/tracking_study_log.md)).
Do not commit checkpoints, rollouts or recordings (they are git-ignored; see `.gitignore`).

## Checks to run

```bash
# ROS 2 package (pure-Python core, no ROS and no GPU needed)
cd ros2/ballbot_control && python -m pytest -q

# Documentation: links, image paths and backticked file paths in Readme.md, docs/ and ros2/README.md
python scripts/docs/check_docs.py

# Website: internal links, fragments, titles and dashes in site/
python scripts/site/check_links.py

# Legacy SB3 path (see "Code Style" and "Testing" below)
make lint && make test
```

Regenerate the diagrams and clip previews with `python scripts/docs/make_diagrams.py` and
`scripts/docs/make_previews.py` (the latter needs the study venv and the source clips); the social preview image with
`scripts/docs/make_social_preview.py`.

## Quick Start

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add or run the relevant tests above
5. Submit a pull request (the template lists the checks)

## Code Style (legacy SB3 path)

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

## Adding Components (legacy SB3 path)

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

## Testing (legacy SB3 path)

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

- Read [docs/pipeline.md](docs/pipeline.md) and the [documentation index](docs/README.md)
- Check [Architecture Docs](docs/architecture/) (legacy path)
- See [Examples](examples/)
- Open an issue for questions

## Code of Conduct

- Be respectful
- Be constructive
- Help others learn

Thank you for contributing.

