"""Unit tests for reward functions."""
import pytest
import numpy as np
from ballbot_gym.core.factories import create_reward
from ballbot_gym.rewards.directional import DirectionalReward
from ballbot_gym.rewards.distance import DistanceReward


class TestRewardFunctions:
    """Test reward function implementations."""
    
    def test_directional_reward(self):
        """Test DirectionalReward computation."""
        reward = DirectionalReward(target_direction=np.array([1.0, 0.0]))
        
        # Test with velocity in target direction
        state = {"vel": np.array([0.5, 0.0, 0.0])}
        reward_val = reward(state)
        assert reward_val > 0  # Positive reward for moving in target direction
        
        # Test with velocity opposite to target direction
        state = {"vel": np.array([-0.5, 0.0, 0.0])}
        reward_val = reward(state)
        assert reward_val < 0  # Negative reward for moving away
    
    def test_directional_reward_factory(self):
        """Test creating DirectionalReward via factory."""
        config = {
            "type": "directional",
            "config": {"target_direction": [0.0, 1.0]}
        }
        reward = create_reward(config)
        assert isinstance(reward, DirectionalReward)
        assert np.allclose(reward.target_direction, [0.0, 1.0])
    
    def test_distance_reward(self):
        """Test DistanceReward computation."""
        goal = np.array([5.0, 3.0])
        reward = DistanceReward(goal_position=goal, scale=0.1)
        
        # Test with position at goal
        state = {"pos2d": goal}
        reward_val = reward(state)
        assert reward_val == 0.0  # Zero distance = zero penalty
        
        # Test with position away from goal
        state = {"pos2d": np.array([0.0, 0.0])}
        reward_val = reward(state)
        assert reward_val < 0  # Negative reward (penalty) for distance
    
    def test_distance_reward_factory(self):
        """Test creating DistanceReward via factory."""
        config = {
            "type": "distance",
            "config": {
                "goal_position": [5.0, 3.0],
                "scale": 0.1
            }
        }
        reward = create_reward(config)
        assert isinstance(reward, DistanceReward)
        assert np.allclose(reward.goal_position, [5.0, 3.0])
        assert reward.scale == 0.1
    
    def test_distance_reward_missing_pos2d(self):
        """Test DistanceReward error handling."""
        reward = DistanceReward(goal_position=np.array([5.0, 3.0]))
        state = {"vel": np.array([0.5, 0.3, 0.0])}  # Missing pos2d
        
        with pytest.raises(ValueError, match="requires 'pos2d'"):
            reward(state)

