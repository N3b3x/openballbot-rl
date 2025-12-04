"""Spiral/radial terrain generator for ballbot environment."""
import numpy as np
from typing import Optional


def generate_spiral_terrain(
    n: int,
    spiral_tightness: float = 0.1,
    height_variation: float = 0.5,
    direction: str = "cw",
    center_x: float = 0.5,
    center_y: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate spiral/radial pattern terrain with smooth gradients.
    
    Creates spiral patterns with continuous curvature, useful for testing
    rotational control and directional challenges.
    
    Args:
        n: Grid size (number of rows/columns, should be odd)
        spiral_tightness: How tight the spiral (higher = tighter)
        height_variation: Height variation along spiral (normalized, 0-1)
        direction: Spiral direction ("cw" clockwise or "ccw" counter-clockwise)
        center_x: X position of spiral center (0-1)
        center_y: Y position of spiral center (0-1)
        seed: Random seed (unused, for API compatibility)
        
    Returns:
        1D array of height values, shape (n*n,), normalized to [0, 1]
        
    Raises:
        AssertionError: If n is not odd or parameters are invalid
    """
    assert n % 2 == 1, "n should be odd for heightfield symmetry"
    assert spiral_tightness > 0, "spiral_tightness must be positive"
    assert 0 <= height_variation <= 1.0, "height_variation should be between 0 and 1"
    assert direction in ["cw", "ccw"], "direction must be 'cw' or 'ccw'"
    
    # Create coordinate grids normalized to [0, 1]
    x_coords = np.linspace(0, 1, n)
    y_coords = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Calculate polar coordinates relative to center
    dx = X - center_x
    dy = Y - center_y
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    
    # Normalize theta to [0, 2π]
    theta = (theta + 2 * np.pi) % (2 * np.pi)
    
    # Reverse direction if clockwise
    if direction == "cw":
        theta = 2 * np.pi - theta
    
    # Create spiral pattern
    # Height varies with both angle and radius
    spiral_phase = spiral_tightness * theta + r
    terrain = height_variation * np.sin(spiral_phase)
    
    # Add radial falloff to create smoother pattern
    max_radius = np.sqrt(2.0) / 2.0  # Maximum distance from center
    r_norm = np.clip(r / max_radius, 0.0, 1.0)
    radial_falloff = 1.0 - r_norm * 0.3  # Slight falloff at edges
    terrain = terrain * radial_falloff
    
    # Shift to be positive (sin goes -1 to 1)
    # Center around 0.5
    terrain = 0.5 + terrain * 0.5
    
    # Clip to [0, 1]
    terrain = np.clip(terrain, 0.0, 1.0)
    
    return terrain.flatten()

