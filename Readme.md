# Reinforcement Learning for Ballbot control in uneven terrain

This repo contains the MuJoCo-based ballbot simulation as well as the RL code from the paper *Salehi, Achkan. "Reinforcement Learning for Ballbot Navigation in Uneven Terrain." arXiv preprint arXiv:2505.18417 (2025)* ([link](https://arxiv.org/abs/2505.18417)).

Here are some navigation examples from a trained policy on four different, *randomly sampled* uneven terrains:


<p float="left">
  <img src="/outputs/visualizations/images/episode_1.gif" width="40.0%" />
  <img src="/outputs/visualizations/images/episode_2.gif" width="40.0%" />
  <img src="/outputs/visualizations/images/episode_3.gif" width="40.0%" />
  <img src="/outputs/visualizations/images/episode_4.gif" width="29.0%" />
</p>

Note that in the above, eval episodes `1, 2, 3` are `4000` timesteps long, while episode `4` is `10000` steps long. The policy has been trained with a maximum of `4000` steps, therefore, this last evaluation can bee seen as a demonstration of generalization capability. 

## 🏗️ Extensible Architecture

openballbot-rl features a **plugin-based, configuration-driven architecture** that makes it easy to add new components without modifying core code.

### Key Features

- **Component Registry**: Central registry for rewards, terrains, and policies
- **Factory Pattern**: Create components from configuration
- **Configuration-Driven**: Switch components via YAML configs
- **Easy Extension**: Add new components in minutes

### Quick Example: Adding a Custom Reward

```python
from ballbot_gym.rewards.base import BaseReward
from ballbot_gym.core.registry import ComponentRegistry

class MyReward(BaseReward):
    def __call__(self, state: dict) -> float:
        return reward_value

# Register it
ComponentRegistry.register_reward("my_reward", MyReward)

# Use in config.yaml
# problem:
#   reward:
#     type: "my_reward"
#     config: {}
```

See [Extension Guide](docs/architecture/extension_guide.md) for details.

## 📚 Comprehensive Documentation

**New!** We've created extensive documentation that connects the research papers to the implementation. Start with the [Documentation README](docs/README.md) for a complete guide.

### Quick Links

- **[Architecture Overview](docs/architecture/README.md)** 🏗️ - System architecture and design
- **[Extension Guide](docs/architecture/extension_guide.md)** 🔧 - How to add custom components
- **[Getting Started](docs/getting_started/quick_start.md)** 🚀 - 5-minute quick start
- **[Examples](examples/)** 💡 - Code examples and tutorials
- **[Research Timeline](docs/research/timeline.md)** 📅 - Evolution from 2006 to 2025
- **[Mechanics to RL Guide](docs/research/mechanics_to_rl.md)** 🔬 - Mathematical derivations
- **[Training Guide](docs/user_guides/training.md)** 🏗️ - Step-by-step training workflow

### Research Papers

All referenced papers are stored in `docs/research/papers/`:
- Lauwers et al. (2006) - Original ballbot prototype
- Nagarajan et al. (2014) - Lagrangian dynamics and control
- Carius et al. (2022) - Constraint-aware control theory
- Salehi (2025) - Complete RL navigation system (this repository)

See the [Documentation README](docs/README.md) for recommended reading paths and learning objectives.

## Warning!

Omniwheels are simulated using capsules with anisotropic friction. This requires a fix that is not (yet) part of the official MuJoCo release. Therefore, you **must** apply the provided patch
to your clone of MuJoCo and build both MuJoCo and the python bindings from source.

For more information, see [This discussion](https://github.com/google-deepmind/mujoco/discussions/2517).

## Installation

### Quick Start

```bash
# Install packages
make install

# Or with dev dependencies
make install-dev

# Verify installation
python setup/verify_installation.py
```

### Building MuJoCo from source (python bindings will be built separately)

Make sure you have CMake and a C++17 compiler installed.

    1. Clone the MuJoCo repository: `git clone https://github.com/deepmind/mujoco.git` and cd into it.
    2. This step is optional but **recommended** due to the patching issue mentioned above: `git checkout 99490163df46f65a0cabcf8efef61b3164faa620`
    3. Copy the patch `tools/mujoco_fix.patch` provided in our repository to `<your_mujoco_repo>`, then `cd` into the latter and apply the patch: `patch -p1 < mujoco_fix.patch`

The rest of the instructions are identical to the [official MuJoCo guide](https://mujoco.readthedocs.io/en/latest/programming/#building-mujoco-from-source) for building from source:
    
    4. Create a new build directory and cd into it.
    5. Run `cmake $PATH_TO_CLONED_REPO` to configure the build.
    6. Run `cmake --build .` to build.
    7. Select the directory: `cmake $PATH_TO_CLONED_REPO -DCMAKE_INSTALL_PREFIX=<my_install_dir>`
    8. After building, install with `cmake --install . `

### Building the python bindings 

Once you have built the patched MuJoCo version from above, the steps for building the python bindings are almost identical to those from the [official MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/python.html#python-bindings):

1. Change to this directory:

```
cd <the_mujoco_repo_from_above>/mujoco/python
```

2. Create a virtual environment and activate it (I use conda, but whatever floats your boat)
3. Generate a source distribution tarball:

```
bash make_sdist.sh
```
This will generate many files in `<repo_clone_path>/mujoco/python/`, among which you'll find `mujoco-x.y.z.tar.gz`.
4. Run this:

```
cd dist
export MUJOCO_PATH=/PATH/TO/MUJOCO \
export MUJOCO_PLUGIN_PATH=/PATH/TO/MUJOCO_PLUGIN \
pip install mujoco-x.y.z.tar.gz #replace x.y.z with the appropriate integers
```

NOTE: If you're using conda, you might need `conda install -c conda-forge libstdcxx` to avoid some gxx related issues.

### Other requirements

Make sure that you have a recent version of `pytorch` as well as a recent version of `stable_baselines3` installed. This code has been tested with torch version `'2.7.0+cu126'`.

Other requirements can be found in `requirements.txt`.


### Install the Ballbot Environment

The installation script handles this automatically, but you can also install manually:

```bash
pip install -e ballbot_gym/
pip install -e ballbot_rl/
```

### Sanity Check

To test that everything works well, run

```bash
python scripts/test_pid.py
```

This uses a simple PID controller to balance the robot on flat terrain.

## Training an agent

Edit the `configs/train/ppo_directional.yaml` file if necessary, and then

```bash
# Using CLI entry point (after installation)
ballbot-train --config configs/train/ppo_directional.yaml

# Or using Python module
python -m ballbot_rl.training.train --config configs/train/ppo_directional.yaml
```

To see the progress of your training, you can use

```bash
python tools/visualization/plot_training.py --csv experiments/runs/*/logs/progress.csv --config experiments/runs/*/config.yaml --plot_train
```

The default yaml config file should result in something that looks like 
<p float="left">
  <img src="/outputs/visualizations/images/a.png" width="49.0%" />
  <img src="/outputs/visualizations/images/b.png" width="49.0%" />
</p>

**Note**: The training process uses a pretrained depth-encoder, which is provided in `outputs/encoders/encoder_epoch_53`. If for some reason you prefer to train your own, you can use:

```bash
# Collect data
ballbot-collect --policy <path_to_policy> --n_steps <steps> --n_envs <envs>

# Pretrain encoder
ballbot-pretrain --data_path <data_path> --save_encoder_to <output_path>
```

## Evaluating an agent

You can see how the agent behaves using:

```bash
# Using CLI entry point
ballbot-eval --algo ppo --n_test <number_of_tests> --path <path_to_your_model>

# Or using Python module
python -m ballbot_rl.evaluation.evaluate --algo ppo --n_test <number_of_tests> --path <path_to_your_model>
```

## Trained policies

A trained policy is provided in the `outputs/models/` directory, and can be tested using the commands above.

## Code Quality

This project uses modern Python tooling for code quality:

- **Black**: Code formatting (run `make format`)
- **Ruff**: Linting (run `make lint`)
- **mypy**: Type checking (run `make type-check`)
- **pytest**: Testing (run `make test`)

Pre-commit hooks are configured to run these checks automatically. Install them with:

```bash
make install-dev
pre-commit install
```

## Citation

If you use this code or refer to our work, please cite:

```
@misc{salehi2025reinforcementlearningballbotnavigation,
      title={Reinforcement Learning for Ballbot Navigation in Uneven Terrain},
      author={Achkan Salehi},
      year={2025},
      eprint={2505.18417},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.18417},
}
```
