"""Sinusoidal wave terrain generator for ballbot environment."""
import numpy as np
from typing import Optional


def generate_sinusoidal_terrain(
    n: int,
    amplitude: float = 0.5,
    frequency: float = 0.1,
    direction: str = "both",
    phase: float = 0.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate terrain with sinusoidal wave pattern.
    
    Creates smooth, periodic wave patterns useful for testing the robot's
    ability to navigate periodic obstacles and undulating terrain.
    
    Args:
        n: Grid size (number of rows/columns, should be odd)
        amplitude: Amplitude of sine wave (in normalized units, 0-1)
        frequency: Frequency of sine wave (cycles per grid unit)
        direction: Wave direction ("x", "y", or "both")
        phase: Phase offset in radians
        seed: Random seed (unused, for API compatibility)
        
    Returns:
        1D array of height values, shape (n*n,), normalized to [0, 1]
        
    Raises:
        AssertionError: If n is not odd or parameters are invalid
    """
    assert n % 2 == 1, "n should be odd for heightfield symmetry"
    assert 0 <= amplitude <= 1.0, "amplitude should be between 0 and 1"
    assert frequency > 0, "frequency must be positive"
    assert direction in ["x", "y", "both"], "direction must be 'x', 'y', or 'both'"
    
    # Create coordinate grids
    x = np.linspace(0, 2 * np.pi * frequency * n, n)
    y = np.linspace(0, 2 * np.pi * frequency * n, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Generate sine wave pattern
    if direction == "x":
        terrain = amplitude * np.sin(X + phase)
    elif direction == "y":
        terrain = amplitude * np.sin(Y + phase)
    else:  # both
        terrain = amplitude * (np.sin(X + phase) + np.sin(Y + phase)) / 2.0
    
    # Normalize to [0, 1]
    terrain_min = terrain.min()
    terrain_max = terrain.max()
    if terrain_max > terrain_min:
        terrain = (terrain - terrain_min) / (terrain_max - terrain_min)
    else:
        terrain = np.zeros_like(terrain)
    
    return terrain.flatten()

