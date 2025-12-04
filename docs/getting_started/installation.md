# Installation Guide

This guide covers installing openballbot-rl and its dependencies.

## Prerequisites

- Python 3.9+
- CMake
- C++17 compiler
- Git

## Step 1: Install MuJoCo with Anisotropic Friction Patch

**Important**: openballbot-rl requires a patched version of MuJoCo for anisotropic friction support.

### 1.1 Clone MuJoCo

```bash
git clone https://github.com/google-deepmind/mujoco.git
cd mujoco
```

### 1.2 Apply Patch

```bash
# Copy patch file to mujoco directory
cp <path-to-openballbot-rl>/tools/mujoco_fix.patch .

# Apply patch
patch -p1 < mujoco_fix.patch
```

### 1.3 Build MuJoCo

```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/mujoco_install
cmake --build . && cmake --install .
```

### 1.4 Set Environment Variables

```bash
export MUJOCO_PATH=$HOME/mujoco_install
export MUJOCO_PLUGIN_PATH=$HOME/mujoco_install/lib/mujoco_plugin
```

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export MUJOCO_PATH=$HOME/mujoco_install' >> ~/.bashrc
echo 'export MUJOCO_PLUGIN_PATH=$HOME/mujoco_install/lib/mujoco_plugin' >> ~/.bashrc
source ~/.bashrc
```

### 1.5 Build Python Bindings

```bash
cd mujoco/python
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
bash make_sdist.sh
cd dist
pip install mujoco-*.tar.gz
```

## Step 2: Install openballbot-rl

### Option A: Using Make (Recommended)

```bash
cd openballbot-rl
make install
```

### Option B: Manual Installation

```bash
cd openballbot-rl
pip install -e ballbot_gym/
pip install -e ballbot_rl/
pip install -r requirements.txt
```

### Option C: Development Installation

```bash
make install-dev
```

This installs:
- ballbot_gym package
- ballbot_rl package
- Development dependencies (pytest, black, ruff, mypy)
- Pre-commit hooks

## Step 3: Verify Installation

```bash
python scripts/setup/verify_installation.py
```

Or test manually:

```bash
python scripts/test_pid.py
```

This runs a PID controller to balance the robot on flat terrain.

## Troubleshooting

### MuJoCo Import Error

**Problem**: `ImportError: cannot import name 'MjModel'`

**Solutions**:
- Verify `MUJOCO_PATH` is set correctly
- Rebuild Python bindings
- Check that patch was applied: `grep -r "condim" mujoco/src/engine/`

### Anisotropic Friction Not Working

**Problem**: Robot wheels don't behave correctly

**Solutions**:
- Verify patch was applied correctly
- Check MuJoCo version matches patch
- Rebuild MuJoCo from source

### macOS GUI Issues

**Problem**: GUI doesn't work on macOS

**Solutions**:
- Use `mjpython` instead of `python` for GUI
- Or set `GUI=False` in environment creation

## Next Steps

- [First Steps](first_steps.md) - Run your first environment
- [Quick Start](quick_start.md) - 5-minute guide
- [Examples](../../examples/) - Code examples

