"""Integration tests for component system."""
import pytest
import gymnasium as gym
import ballbot_gym
from ballbot_gym.core.registry import ComponentRegistry
from ballbot_gym.core.factories import create_reward, create_terrain
from ballbot_gym.core.config import get_component_config


class TestComponentSystemIntegration:
    """Integration tests for component system."""
    
    def setup_method(self):
        """Ensure components are registered."""
        # Import modules to trigger auto-registration
        import ballbot_gym.rewards
        import ballbot_gym.terrain
    
    def test_environment_with_custom_reward_config(self):
        """Test environment creation with custom reward config."""
        reward_config = {
            "type": "directional",
            "config": {"target_direction": [1.0, 0.0]}
        }
        
        env = gym.make(
            "ballbot-v0.1",
            GUI=False,
            terrain_type="flat",
            reward_config=reward_config,
            max_ep_steps=10
        )
        
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(reward, float)
        assert "pos2d" in info
        
        env.close()
    
    def test_environment_with_custom_terrain_config(self):
        """Test environment creation with custom terrain config."""
        terrain_config = {
            "type": "flat",
            "config": {}
        }
        
        env = gym.make(
            "ballbot-v0.1",
            GUI=False,
            terrain_config=terrain_config,
            max_ep_steps=10
        )
        
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(reward, float)
        
        env.close()
    
    def test_environment_with_both_configs(self):
        """Test environment creation with both reward and terrain configs."""
        reward_config = {
            "type": "directional",
            "config": {"target_direction": [0.0, 1.0]}
        }
        terrain_config = {
            "type": "perlin",
            "config": {"seed": 42}
        }
        
        env = gym.make(
            "ballbot-v0.1",
            GUI=False,
            reward_config=reward_config,
            terrain_config=terrain_config,
            max_ep_steps=10
        )
        
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(reward, float)
        
        env.close()
    
    def test_backward_compatibility_terrain_type(self):
        """Test backward compatibility with terrain_type string."""
        env = gym.make(
            "ballbot-v0.1",
            GUI=False,
            terrain_type="flat",  # Old format
            max_ep_steps=10
        )
        
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(reward, float)
        
        env.close()
    
    def test_get_component_config(self):
        """Test get_component_config utility."""
        config = {
            "problem": {
                "terrain": {
                    "type": "perlin",
                    "config": {"scale": 25.0}
                },
                "reward": {
                    "type": "directional",
                    "config": {"target_direction": [0.0, 1.0]}
                }
            }
        }
        
        terrain_config = get_component_config(config, "terrain")
        assert terrain_config["type"] == "perlin"
        assert terrain_config["config"]["scale"] == 25.0
        
        reward_config = get_component_config(config, "reward")
        assert reward_config["type"] == "directional"
        assert reward_config["config"]["target_direction"] == [0.0, 1.0]
    
    def test_get_component_config_backward_compat(self):
        """Test get_component_config with old format."""
        config = {
            "problem": {
                "terrain_type": "perlin"  # Old format
            }
        }
        
        terrain_config = get_component_config(config, "terrain", default_type="perlin")
        assert terrain_config["type"] == "perlin"
    
    def test_factory_creates_working_components(self):
        """Test that factory-created components work in environment."""
        reward_config = {
            "type": "directional",
            "config": {"target_direction": [0.0, 1.0]}
        }
        reward = create_reward(reward_config)
        
        # Test reward computation
        state = {"vel": np.array([0.0, 0.5, 0.0])}
        reward_val = reward(state)
        assert isinstance(reward_val, float)
        assert reward_val > 0  # Moving in target direction
        
        terrain_config = {
            "type": "flat",
            "config": {}
        }
        terrain_gen = create_terrain(terrain_config)
        terrain = terrain_gen(129)
        assert terrain.shape == (129 * 129,)

