"""Hill/mound terrain generator for ballbot environment."""
import numpy as np
from typing import Optional, List, Tuple


def generate_hills_terrain(
    n: int,
    num_hills: int = 5,
    hill_height: float = 0.7,
    hill_radius: float = 0.15,
    flat_ratio: float = 0.4,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate terrain with multiple smooth hills/mounds.
    
    Creates smooth, rounded hills with flat areas between them, useful for
    testing navigation around obstacles and path planning.
    
    Args:
        n: Grid size (number of rows/columns, should be odd)
        num_hills: Number of hills to generate
        hill_height: Maximum hill height (normalized, 0-1)
        hill_radius: Hill radius as fraction of grid size (0-1)
        flat_ratio: Ratio of terrain that should be relatively flat
        seed: Random seed for hill placement
        
    Returns:
        1D array of height values, shape (n*n,), normalized to [0, 1]
        
    Raises:
        AssertionError: If n is not odd or parameters are invalid
    """
    assert n % 2 == 1, "n should be odd for heightfield symmetry"
    assert num_hills > 0, "num_hills must be positive"
    assert 0 <= hill_height <= 1.0, "hill_height should be between 0 and 1"
    assert 0 < hill_radius <= 0.5, "hill_radius should be between 0 and 0.5"
    
    # Initialize terrain
    terrain = np.zeros((n, n))
    
    # Set up random number generator
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(0)
    
    # Generate hill positions
    # Ensure hills are not too close to edges or each other
    min_distance = hill_radius * 2.0
    hill_positions: List[Tuple[float, float]] = []
    
    attempts = 0
    max_attempts = num_hills * 100
    
    while len(hill_positions) < num_hills and attempts < max_attempts:
        attempts += 1
        # Generate position in normalized coordinates [0, 1]
        x = rng.uniform(hill_radius, 1.0 - hill_radius)
        y = rng.uniform(hill_radius, 1.0 - hill_radius)
        
        # Check distance from existing hills
        too_close = False
        for existing_x, existing_y in hill_positions:
            dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
            if dist < min_distance:
                too_close = True
                break
        
        if not too_close:
            hill_positions.append((x, y))
    
    # Create coordinate grids normalized to [0, 1]
    x_coords = np.linspace(0, 1, n)
    y_coords = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Generate Gaussian hills
    for hill_x, hill_y in hill_positions:
        # Distance from hill center
        dx = X - hill_x
        dy = Y - hill_y
        r = np.sqrt(dx**2 + dy**2)
        
        # Gaussian hill with smooth falloff
        # Use exponential decay with smooth cutoff
        sigma = hill_radius / 3.0  # Standard deviation for Gaussian
        hill = hill_height * np.exp(-(r**2) / (2 * sigma**2))
        
        # Apply smooth cutoff at hill_radius
        cutoff_factor = np.clip(1.0 - (r / hill_radius), 0.0, 1.0)
        cutoff_factor = cutoff_factor * cutoff_factor * (3.0 - 2.0 * cutoff_factor)  # smoothstep
        hill = hill * cutoff_factor
        
        terrain += hill
    
    # Normalize to [0, 1] by clipping
    # We do NOT auto-scale min/max because hill_height should control physical height
    # relative to the global z-scale
    terrain = np.clip(terrain, 0.0, 1.0)
    
    return terrain.flatten()

