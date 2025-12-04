"""Helper utilities for ballbot_gym."""
import os
import sys
from contextlib import contextmanager
import numpy as np
import random


@contextmanager
def warnings_stdout_off():
    """
    Context manager to suppress stderr output temporarily.
    
    This is used to suppress benign MuJoCo warnings (e.g., about objective
    convexity) that can clutter the console output during training.
    
    Usage:
        ```python
        with warnings_stdout_off():
            # Code that might produce warnings
            mujoco.mj_step(model, data)
        ```
    
    Note:
        This redirects stderr to /dev/null, so all stderr output (not just
        warnings) will be suppressed. Use with caution.
    """
    devnull = os.open(os.devnull, os.O_RDWR)
    old_stderr = os.dup(sys.stderr.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    try:
        yield
    finally:
        # Restore stderr
        os.dup2(old_stderr, sys.stderr.fileno())
        os.close(devnull)
        os.close(old_stderr)


def sample_direction_uniform(num=1):
    """
    Sample random 2D directions uniformly on the unit circle.
    
    This function samples angles uniformly from [0, 2π] and converts them
    to unit vectors on the circle. Useful for random goal generation.
    
    Args:
        num (int, optional): Number of directions to sample. Defaults to 1.
    
    Returns:
        np.ndarray: Array of shape (num, 2) where each row is a unit vector
            [cos(θ), sin(θ)] for a random angle θ.
    
    Example:
        >>> directions = sample_direction_uniform(5)
        >>> directions.shape
        (5, 2)
        >>> np.linalg.norm(directions, axis=1)  # All should be ~1.0
        array([1., 1., 1., 1., 1.])
    """
    # Sample random angles uniformly from [0, 2π]
    t = np.random.rand(num).reshape(num, 1) * 2 * np.pi
    # Convert to unit vectors: [cos(θ), sin(θ)]
    return np.concatenate([np.cos(t), np.sin(t)], 1)

