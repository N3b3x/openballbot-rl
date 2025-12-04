"""Unit tests for factory functions."""
import pytest
import numpy as np
from ballbot_gym.core.factories import create_reward, create_terrain, validate_config
from ballbot_gym.core.registry import ComponentRegistry
from ballbot_gym.rewards.base import BaseReward
from ballbot_gym.rewards.directional import DirectionalReward
from ballbot_gym.rewards.distance import DistanceReward


class TestFactories:
    """Test factory function functionality."""
    
    def setup_method(self):
        """Setup registry before each test."""
        ComponentRegistry.clear()
        ComponentRegistry.register_reward("directional", DirectionalReward)
        ComponentRegistry.register_reward("distance", DistanceReward)
        ComponentRegistry.register_terrain("flat", lambda n, **kwargs: np.zeros(n * n))
    
    def test_create_reward(self):
        """Test reward creation from config."""
        config = {
            "type": "directional",
            "config": {"target_direction": [1.0, 0.0]}
        }
        reward = create_reward(config)
        assert isinstance(reward, DirectionalReward)
        assert np.allclose(reward.target_direction, [1.0, 0.0])
    
    def test_create_reward_missing_type(self):
        """Test error handling for missing type."""
        config = {"config": {}}
        with pytest.raises(ValueError, match="must have 'type' key"):
            create_reward(config)
    
    def test_create_reward_invalid_type(self):
        """Test error handling for invalid reward type."""
        config = {"type": "nonexistent", "config": {}}
        with pytest.raises(ValueError, match="Failed to create reward"):
            create_reward(config)
    
    def test_create_terrain(self):
        """Test terrain creation from config."""
        config = {
            "type": "flat",
            "config": {}
        }
        terrain_gen = create_terrain(config)
        assert callable(terrain_gen)
        
        terrain = terrain_gen(5)
        assert terrain.shape == (25,)
        assert np.allclose(terrain, 0.0)
    
    def test_create_terrain_with_config(self):
        """Test terrain creation with config parameters."""
        def test_terrain(n: int, param: float = 1.0, **kwargs):
            return np.ones(n * n) * param
        
        ComponentRegistry.register_terrain("test", test_terrain)
        
        config = {
            "type": "test",
            "config": {"param": 0.5}
        }
        terrain_gen = create_terrain(config)
        terrain = terrain_gen(5)
        assert np.allclose(terrain, 0.5)
    
    def test_create_terrain_seed_override(self):
        """Test that seed can be overridden at runtime."""
        def seeded_terrain(n: int, seed: int = 0, **kwargs):
            rng = np.random.RandomState(seed)
            return rng.rand(n * n)
        
        ComponentRegistry.register_terrain("seeded", seeded_terrain)
        
        config = {
            "type": "seeded",
            "config": {"seed": 42}
        }
        terrain_gen = create_terrain(config)
        
        # Override seed at runtime
        terrain1 = terrain_gen(5, seed=100)
        terrain2 = terrain_gen(5, seed=100)
        assert np.allclose(terrain1, terrain2)  # Same seed = same result
    
    def test_validate_config_reward(self):
        """Test config validation for rewards."""
        config = {"type": "directional", "config": {}}
        assert validate_config(config, "reward") is True
    
    def test_validate_config_invalid(self):
        """Test config validation with invalid config."""
        config = {"type": "nonexistent", "config": {}}
        with pytest.raises(ValueError, match="Unknown reward type"):
            validate_config(config, "reward")
    
    def test_validate_config_missing_type(self):
        """Test config validation with missing type."""
        config = {"config": {}}
        with pytest.raises(ValueError, match="must have 'type' key"):
            validate_config(config, "reward")
    
    def test_validate_config_wrong_component_type(self):
        """Test config validation with wrong component type."""
        config = {"type": "directional", "config": {}}
        with pytest.raises(ValueError, match="Unknown component_type"):
            validate_config(config, "invalid_type")

