"""Directional reward function for ballbot navigation."""
import numpy as np
from typing import Dict

from ballbot_gym.rewards.base import BaseReward


class DirectionalReward(BaseReward):
    """
    Reward function that encourages the agent to move in a specified target direction.

    The reward is computed as the dot product between the agent's ground-plane velocity 
    (x-y) and a target direction vector. This incentivizes the agent to align its movement 
    with the desired direction.

    Attributes
    ----------
    target_direction : np.ndarray of shape (2,)
        The target direction vector in the x-y plane (should be unit norm for pure alignment reward).
    """

    def __init__(self, target_direction: np.ndarray):
        """
        Initialize the DirectionalReward function.

        Parameters
        ----------
        target_direction : np.ndarray of shape (2,)
            The target direction in the x-y plane (should be normalized to unit vector for standard usage).
        """
        self.target_direction = target_direction

    def __call__(self, state: dict) -> float:
        """
        Compute the directional reward for the given state.

        Parameters
        ----------
        state : dict
            The agent's observation/state dictionary.
            Expected to contain:
                - 'vel': np.ndarray, full 3D velocity (length at least 3),
                          with x, y, z components (order may be platform-dependent).

        Returns
        -------
        float
            The reward, equal to the projection of the x-y velocity onto the target direction.
        """
        # Extract the agent's velocity in x-y plane (assumes last three are x, y, z)
        xy_velocity = state["vel"][-3:-1]
        # Compute directional reward as dot product with target direction
        dir_rew = xy_velocity.dot(self.target_direction)
        return dir_rew
