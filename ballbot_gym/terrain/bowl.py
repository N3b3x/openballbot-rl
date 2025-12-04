"""Bowl/depression terrain generator for ballbot environment."""
import numpy as np
from typing import Optional


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Smooth interpolation function for smooth transitions."""
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def generate_bowl_terrain(
    n: int,
    depth: float = 0.6,
    radius: float = 0.4,
    center_x: float = 0.5,
    center_y: float = 0.5,
    smoothness: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate smooth bowl-shaped depression or elevation terrain.
    
    Creates radial gradients with smooth edges, useful for testing recovery
    from depressions and radial control strategies.
    
    Args:
        n: Grid size (number of rows/columns, should be odd)
        depth: Depth/height of bowl (normalized, 0-1)
        radius: Bowl radius as fraction of grid size (0-1)
        center_x: X position of bowl center (0-1)
        center_y: Y position of bowl center (0-1)
        smoothness: Edge smoothness factor (0.0-1.0)
        seed: Random seed (unused, for API compatibility)
        
    Returns:
        1D array of height values, shape (n*n,), normalized to [0, 1]
        
    Raises:
        AssertionError: If n is not odd or parameters are invalid
    """
    assert n % 2 == 1, "n should be odd for heightfield symmetry"
    assert 0 <= depth <= 1.0, "depth should be between 0 and 1"
    assert 0 < radius <= 1.0, "radius should be between 0 and 1"
    assert 0 <= center_x <= 1.0, "center_x should be between 0 and 1"
    assert 0 <= center_y <= 1.0, "center_y should be between 0 and 1"
    
    # Create coordinate grids normalized to [0, 1]
    x_coords = np.linspace(0, 1, n)
    y_coords = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Calculate radial distance from center
    dx = X - center_x
    dy = Y - center_y
    r = np.sqrt(dx**2 + dy**2)
    
    # Normalize radius to [0, 1] relative to bowl radius
    r_norm = np.clip(r / radius, 0.0, 1.0)
    
    # Create bowl shape using smoothstep for smooth edges
    # For depression: height decreases from center
    # For elevation: height increases from center
    # We'll create a depression (lower in center)
    bowl_shape = depth * (1.0 - smoothstep(0.0, 1.0, r_norm))
    
    # Start with flat terrain at height 1.0 and subtract bowl
    # This puts the "ground" at max height and the bowl as a depression
    terrain = np.ones((n, n)) - bowl_shape
    
    # Clip to [0, 1] to ensure valid hfield data
    # We do NOT re-normalize min/max because depth should control physical depth
    terrain = np.clip(terrain, 0.0, 1.0)
    
    return terrain.flatten()

