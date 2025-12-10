#!/bin/bash
set -e

echo "=========================================="
echo "Cleaning all caches and reinstalling..."
echo "=========================================="
echo

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${VIRTUAL_ENV:-}"

if [ -z "$VENV_PATH" ]; then
    echo "⚠️  Warning: No virtual environment detected."
    echo "   Please activate your virtual environment first:"
    echo "   source rl_lab_env/bin/activate"
    exit 1
fi

echo "Repository root: $REPO_ROOT"
echo "Virtual environment: $VENV_PATH"
echo

# Step 1: Uninstall packages
echo "Step 1: Uninstalling packages..."
pip uninstall -y ballbot-rl ballbot-gym 2>/dev/null || true
echo "✓ Packages uninstalled"
echo

# Step 2: Clear Python bytecode cache
echo "Step 2: Clearing Python bytecode cache..."
find "$REPO_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$REPO_ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$REPO_ROOT" -type f -name "*.pyo" -delete 2>/dev/null || true
echo "✓ Bytecode cache cleared"
echo

# Step 3: Clear build artifacts
echo "Step 3: Clearing build artifacts..."
rm -rf "$REPO_ROOT"/ballbot_gym/*.egg-info 2>/dev/null || true
rm -rf "$REPO_ROOT"/ballbot_rl/*.egg-info 2>/dev/null || true
rm -rf "$REPO_ROOT"/build 2>/dev/null || true
rm -rf "$REPO_ROOT"/dist 2>/dev/null || true
rm -rf "$REPO_ROOT"/.eggs 2>/dev/null || true
echo "✓ Build artifacts cleared"
echo

# Step 4: Clear .pth files
echo "Step 4: Clearing .pth files..."
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
if [ -n "$SITE_PACKAGES" ] && [ -d "$SITE_PACKAGES" ]; then
    rm -f "$SITE_PACKAGES"/_ballbot*.pth 2>/dev/null || true
    echo "✓ .pth files cleared from $SITE_PACKAGES"
else
    echo "⚠️  Could not find site-packages directory"
fi
echo

# Step 5: Clear pip cache (optional, can be slow)
if [ "$1" == "--clear-pip-cache" ]; then
    echo "Step 5: Clearing pip cache..."
    pip cache purge 2>/dev/null || true
    echo "✓ Pip cache cleared"
    echo
else
    echo "Step 5: Skipping pip cache (use --clear-pip-cache to clear)"
    echo
fi

# Step 6: Clear other caches
echo "Step 6: Clearing other caches..."
rm -rf "$REPO_ROOT"/.pytest_cache 2>/dev/null || true
rm -rf "$REPO_ROOT"/.mypy_cache 2>/dev/null || true
rm -rf "$REPO_ROOT"/.ruff_cache 2>/dev/null || true
rm -rf "$REPO_ROOT"/htmlcov 2>/dev/null || true
rm -rf "$REPO_ROOT"/.coverage 2>/dev/null || true
rm -rf "$REPO_ROOT"/.coverage.* 2>/dev/null || true
echo "✓ Other caches cleared"
echo

# Step 7: Reinstall packages
echo "Step 7: Reinstalling packages..."
cd "$REPO_ROOT"
pip install -e ballbot_gym/
pip install -e ballbot_rl/
echo "✓ Packages reinstalled"
echo

# Step 8: Fix .pth files
echo "Step 8: Fixing .pth files..."
if [ -f "$REPO_ROOT/scripts/setup/fix_pth_files.py" ]; then
    python "$REPO_ROOT/scripts/setup/fix_pth_files.py"
else
    # Manual fix if script doesn't exist
    if [ -n "$SITE_PACKAGES" ] && [ -d "$SITE_PACKAGES" ]; then
        echo "$REPO_ROOT" > "$SITE_PACKAGES/_ballbot_rl.pth" 2>/dev/null || true
        echo "$REPO_ROOT" > "$SITE_PACKAGES/_ballbot_gym.pth" 2>/dev/null || true
        echo "✓ .pth files fixed manually"
    fi
fi
echo

# Step 9: Verify installation
echo "Step 9: Verifying installation..."
if [ -f "$REPO_ROOT/scripts/setup/verify_installation.py" ]; then
    python "$REPO_ROOT/scripts/setup/verify_installation.py"
else
    python -c "import ballbot_gym; import ballbot_rl; print('✓ Imports successful')"
fi
echo

echo "=========================================="
echo "✓ Clean installation complete!"
echo "=========================================="

