"""Pytest configuration and shared fixtures."""
import pytest
from pathlib import Path
import sys
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture
def project_root():
    """Return project root directory."""
    return PROJECT_ROOT

@pytest.fixture
def config_dir(project_root):
    """Return config directory."""
    return project_root / "configs"

@pytest.fixture
def outputs_dir(project_root):
    """Return outputs directory."""
    return project_root / "outputs"

@pytest.fixture
def test_config():
    """Return test configuration (old format for backward compatibility)."""
    return {
        "algo": {
            "name": "ppo",
            "n_steps": 100,
            "n_epochs": 2,
        },
        "problem": {
            "terrain_type": "flat",
        },
        "total_timesteps": 1000,
        "frozen_cnn": "",
        "hidden_sz": 64,
        "num_envs": 1,
        "seed": 42,
    }

@pytest.fixture
def test_config_new_format():
    """Return test configuration (new format)."""
    return {
        "algo": {
            "name": "ppo",
            "n_steps": 100,
            "n_epochs": 2,
        },
        "problem": {
            "terrain": {
                "type": "flat",
                "config": {}
            },
            "reward": {
                "type": "directional",
                "config": {
                    "target_direction": [0.0, 1.0]
                }
            }
        },
        "total_timesteps": 1000,
        "frozen_cnn": "",
        "hidden_sz": 64,
        "num_envs": 1,
        "seed": 42,
    }

@pytest.fixture
def reward_config():
    """Return reward configuration."""
    return {
        "type": "directional",
        "config": {
            "target_direction": [0.0, 1.0]
        }
    }

@pytest.fixture
def terrain_config():
    """Return terrain configuration."""
    return {
        "type": "flat",
        "config": {}
    }

@pytest.fixture
def test_state():
    """Return test state dictionary."""
    return {
        "vel": np.array([0.5, 0.3, 0.0]),
        "orientation": np.array([0.1, 0.2, 0.3]),
        "pos2d": np.array([0.0, 0.0])
    }

