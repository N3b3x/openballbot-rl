#!/bin/bash
set -e

echo "Installing MuJoCo with anisotropic friction patch..."

# Check if MuJoCo is already installed
if [ -z "$MUJOCO_PATH" ]; then
    echo "Please set MUJOCO_PATH environment variable"
    exit 1
fi

# Apply patch
cd "$MUJOCO_PATH"
if [ -f "../tools/mujoco_fix.patch" ]; then
    patch -p1 < ../tools/mujoco_fix.patch
    echo "Patch applied successfully"
else
    echo "Warning: mujoco_fix.patch not found"
fi

echo "MuJoCo installation complete"

