"""Distance-based reward function for ballbot navigation."""
import numpy as np
from typing import Dict

from ballbot_gym.rewards.base import BaseReward


class DistanceReward(BaseReward):
    """
    Reward function based on distance to a goal position.
    
    This reward encourages the agent to minimize distance to a target position
    in the x-y plane. The reward is negative (penalty) proportional to distance.
    
    Attributes:
        goal_position: Target position [x, y] in world coordinates
        scale: Scaling factor for distance penalty
    """
    
    def __init__(self, goal_position: np.ndarray, scale: float = 1.0):
        """
        Initialize DistanceReward.
        
        Args:
            goal_position: Target position as numpy array [x, y]
            scale: Scaling factor for distance penalty (default: 1.0)
        """
        self.goal_position = np.array(goal_position, dtype=np.float32)
        if self.goal_position.shape != (2,):
            raise ValueError(f"goal_position must be shape (2,), got {self.goal_position.shape}")
        self.scale = float(scale)
    
    def __call__(self, state: Dict) -> float:
        """
        Compute distance-based reward.
        
        Args:
            state: Observation dictionary containing "pos2d" key with [x, y] position
            
        Returns:
            Negative reward proportional to distance (closer = less negative)
        """
        if "pos2d" not in state:
            raise ValueError("DistanceReward requires 'pos2d' in state dictionary")
        
        current_pos = np.array(state["pos2d"], dtype=np.float32)
        distance = np.linalg.norm(self.goal_position - current_pos)
        
        # Negative reward (penalty) proportional to distance
        return -self.scale * distance

