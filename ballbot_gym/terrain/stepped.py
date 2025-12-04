"""Stepped terrain generator for ballbot environment."""
import numpy as np
from typing import Optional


def generate_stepped_terrain(
    n: int,
    num_steps: int = 5,
    step_height: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate stepped terrain with discrete height levels.
    
    This creates terrain with distinct height steps, useful for testing
    the robot's ability to navigate discrete obstacles.
    
    Args:
        n: Grid size (number of rows/columns, should be odd)
        num_steps: Number of discrete height steps
        step_height: Height difference between steps (in normalized units)
        seed: Random seed for step placement (None = deterministic)
        
    Returns:
        1D array of height values, shape (n*n,), normalized to [0, 1]
    """
    assert n % 2 == 1, "n should be odd for heightfield symmetry"
    assert num_steps > 0, "num_steps must be positive"
    assert step_height > 0, "step_height must be positive"
    
    # Initialize terrain
    terrain = np.zeros((n, n))
    
    # Create a grid of step assignments
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(0)
    
    # Divide terrain into regions and assign step heights
    step_size = n // num_steps
    for i in range(n):
        for j in range(n):
            # Determine which step this cell belongs to
            step_idx = min((i // step_size) + (j // step_size), num_steps - 1)
            terrain[i, j] = step_idx * step_height
    
    # Add some randomness to step boundaries for smoother transitions
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            # Average with neighbors to smooth boundaries
            neighbors = [
                terrain[i-1, j], terrain[i+1, j],
                terrain[i, j-1], terrain[i, j+1]
            ]
            terrain[i, j] = 0.7 * terrain[i, j] + 0.3 * np.mean(neighbors)
    
    # Normalize to [0, 1]
    terrain_min = terrain.min()
    terrain_max = terrain.max()
    if terrain_max > terrain_min:
        terrain = (terrain - terrain_min) / (terrain_max - terrain_min)
    else:
        terrain = np.zeros_like(terrain)
    
    return terrain.flatten()

