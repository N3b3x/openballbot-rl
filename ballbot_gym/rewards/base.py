"""Base reward function interface."""

from abc import ABC, abstractmethod
from typing import Dict


class BaseReward(ABC):
    """Abstract base class for reward functions."""
    
    @abstractmethod
    def __call__(self, state: Dict) -> float:
        """Compute reward for given state.
        
        Args:
            state: Dictionary containing observation/state information
            
        Returns:
            Reward value as float
        """
        pass

