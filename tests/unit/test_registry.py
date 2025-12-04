"""Unit tests for component registry."""
import pytest
import numpy as np
from ballbot_gym.core.registry import ComponentRegistry
from ballbot_gym.rewards.base import BaseReward
from ballbot_gym.rewards.directional import DirectionalReward
from ballbot_gym.rewards.distance import DistanceReward


class TestComponentRegistry:
    """Test ComponentRegistry functionality."""
    
    def setup_method(self):
        """Clear registry before each test."""
        ComponentRegistry.clear()
        # Re-register default components
        ComponentRegistry.register_reward("directional", DirectionalReward)
        ComponentRegistry.register_reward("distance", DistanceReward)
    
    def test_register_reward(self):
        """Test reward registration."""
        assert "directional" in ComponentRegistry.list_rewards()
        assert "distance" in ComponentRegistry.list_rewards()
    
    def test_get_reward(self):
        """Test reward retrieval."""
        reward = ComponentRegistry.get_reward(
            "directional",
            target_direction=np.array([0.0, 1.0])
        )
        assert isinstance(reward, DirectionalReward)
        assert np.allclose(reward.target_direction, [0.0, 1.0])
    
    def test_get_reward_invalid(self):
        """Test error handling for invalid reward name."""
        with pytest.raises(ValueError, match="Unknown reward"):
            ComponentRegistry.get_reward("nonexistent")
    
    def test_list_rewards(self):
        """Test listing registered rewards."""
        rewards = ComponentRegistry.list_rewards()
        assert isinstance(rewards, list)
        assert "directional" in rewards
        assert "distance" in rewards
    
    def test_register_duplicate_reward(self):
        """Test that duplicate registration raises error."""
        with pytest.raises(ValueError, match="already registered"):
            ComponentRegistry.register_reward("directional", DirectionalReward)
    
    def test_register_invalid_reward(self):
        """Test that non-BaseReward classes raise error."""
        class NotAReward:
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseReward"):
            ComponentRegistry.register_reward("invalid", NotAReward)
    
    def test_register_terrain(self):
        """Test terrain registration."""
        def dummy_terrain(n: int, **kwargs):
            return np.zeros(n * n)
        
        ComponentRegistry.register_terrain("dummy", dummy_terrain)
        assert "dummy" in ComponentRegistry.list_terrains()
        
        terrain_fn = ComponentRegistry.get_terrain("dummy")
        assert callable(terrain_fn)
        result = terrain_fn(5)
        assert result.shape == (25,)
    
    def test_get_terrain_invalid(self):
        """Test error handling for invalid terrain name."""
        with pytest.raises(ValueError, match="Unknown terrain"):
            ComponentRegistry.get_terrain("nonexistent")
    
    def test_clear(self):
        """Test clearing registry."""
        assert len(ComponentRegistry.list_rewards()) > 0
        ComponentRegistry.clear()
        assert len(ComponentRegistry.list_rewards()) == 0
        assert len(ComponentRegistry.list_terrains()) == 0

