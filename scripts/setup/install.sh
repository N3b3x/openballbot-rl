#!/bin/bash
set -e

echo "Installing openballbot-rl..."

# Install MuJoCo (with patch)
bash scripts/setup/install_mujoco.sh

# Install Python packages
pip install -e ballbot_gym/
pip install -e ballbot_rl/

# Install dev dependencies (optional)
if [ "$1" == "--dev" ]; then
    pip install -e "ballbot_gym[dev]"
    pip install -e "ballbot_rl[dev]"
    pip install pre-commit
    pre-commit install
fi

# Verify installation
python scripts/setup/verify_installation.py

echo "Installation complete!"

