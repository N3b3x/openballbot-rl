# Examples

This directory contains example scripts demonstrating how to use openballbot-rl.

## Quick Start

1. **Basic Usage** (`01_basic_usage.py`) - Minimal environment usage
2. **Custom Reward** (`02_custom_reward.py`) - How to add a custom reward function
3. **Custom Terrain** (`03_custom_terrain.py`) - How to add a custom terrain generator
4. **Custom Policy** (`04_custom_policy.py`) - How to add a custom policy architecture
5. **Training Workflow** (`05_training_workflow.py`) - Complete training example
6. **Configuration Examples** (`06_configuration_examples.py`) - Config file examples

## Advanced Examples

See `advanced/` directory for:
- Curriculum learning
- Multi-goal navigation
- Sim-to-real transfer

## Running Examples

```bash
# Basic usage
python examples/01_basic_usage.py

# Custom component
python examples/02_custom_reward.py

# Training
python examples/05_training_workflow.py
```

## Contributing

When adding new examples:
1. Follow the naming convention: `NN_description.py`
2. Include comprehensive docstrings
3. Add comments explaining key concepts
4. Test that examples run successfully
5. Update this README

