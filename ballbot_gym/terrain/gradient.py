"""Gradient field terrain generator for ballbot environment."""
import numpy as np
from typing import Optional


def generate_gradient_terrain(
    n: int,
    max_slope: float = 20.0,
    gradient_type: str = "linear",
    smoothness: float = 0.5,
    direction: str = "x",
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate terrain with smooth gradient field and varying slopes.
    
    Creates continuous gradients with varying slope angles, useful for
    testing slope compensation and adaptive control strategies.
    
    Args:
        n: Grid size (number of rows/columns, should be odd)
        max_slope: Maximum slope angle in degrees (0-30 recommended)
        gradient_type: Type of gradient ("linear", "radial", "perlin")
        smoothness: Smoothness factor for transitions (0.0-1.0)
        direction: Primary gradient direction for linear type ("x" or "y")
        seed: Random seed for perlin-based gradients
        
    Returns:
        1D array of height values, shape (n*n,), normalized to [0, 1]
        
    Raises:
        AssertionError: If n is not odd or parameters are invalid
    """
    assert n % 2 == 1, "n should be odd for heightfield symmetry"
    assert 0 <= max_slope <= 45, "max_slope should be between 0 and 45 degrees"
    assert gradient_type in ["linear", "radial", "perlin"], "gradient_type must be 'linear', 'radial', or 'perlin'"
    assert direction in ["x", "y"], "direction must be 'x' or 'y'"
    
    # Convert slope angle to normalized height gradient
    # Assuming terrain spans from -1 to 1 in normalized coordinates
    max_height_gradient = np.tan(np.radians(max_slope)) * 2.0
    
    # Create coordinate grids normalized to [-1, 1]
    center = n // 2
    x = (np.arange(n) - center) / center
    y = (np.arange(n) - center) / center
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    if gradient_type == "linear":
        # Linear gradient in specified direction
        if direction == "x":
            terrain = max_height_gradient * (X + 1.0) / 2.0
        else:  # y
            terrain = max_height_gradient * (Y + 1.0) / 2.0
            
    elif gradient_type == "radial":
        # Radial gradient from center
        R = np.sqrt(X**2 + Y**2)
        max_radius = np.sqrt(2.0)
        R_norm = np.clip(R / max_radius, 0.0, 1.0)
        terrain = max_height_gradient * R_norm
        
    else:  # perlin
        # Perlin noise-based gradient
        from noise import snoise2
        
        if seed is None:
            seed = 0
        
        terrain = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # Use Perlin noise to create varying gradient
                noise_val = snoise2(
                    i / 25.0,
                    j / 25.0,
                    octaves=3,
                    persistence=0.3,
                    base=seed
                )
                # Map noise to gradient direction
                if direction == "x":
                    base_gradient = (X[i, j] + 1.0) / 2.0
                else:
                    base_gradient = (Y[i, j] + 1.0) / 2.0
                
                # Combine base gradient with noise variation
                terrain[i, j] = max_height_gradient * (base_gradient + noise_val * smoothness)
    
    # Normalize to [0, 1]
    terrain_min = terrain.min()
    terrain_max = terrain.max()
    if terrain_max > terrain_min:
        terrain = (terrain - terrain_min) / (terrain_max - terrain_min)
    else:
        terrain = np.zeros_like(terrain)
    
    return terrain.flatten()

