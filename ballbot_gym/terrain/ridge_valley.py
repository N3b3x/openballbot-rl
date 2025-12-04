"""Ridge and valley terrain generator for ballbot environment."""
import numpy as np
from typing import Optional


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Smooth interpolation function for smooth transitions."""
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def generate_ridge_valley_terrain(
    n: int,
    ridge_height: float = 0.6,
    valley_depth: float = 0.4,
    spacing: float = 0.2,
    orientation: str = "x",
    smoothness: float = 0.3,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate terrain with alternating ridges and valleys.
    
    Creates continuous ridges and valleys with smooth transitions, useful
    for testing navigation through terrain features and directional control.
    
    Args:
        n: Grid size (number of rows/columns, should be odd)
        ridge_height: Height of ridges (normalized, 0-1)
        valley_depth: Depth of valleys (normalized, 0-1)
        spacing: Spacing between features (cycles per grid unit)
        orientation: Feature orientation ("x", "y", or "diagonal")
        smoothness: Transition smoothness factor (0.0-1.0)
        seed: Random seed (unused, for API compatibility)
        
    Returns:
        1D array of height values, shape (n*n,), normalized to [0, 1]
        
    Raises:
        AssertionError: If n is not odd or parameters are invalid
    """
    assert n % 2 == 1, "n should be odd for heightfield symmetry"
    assert 0 <= ridge_height <= 1.0, "ridge_height should be between 0 and 1"
    assert 0 <= valley_depth <= 1.0, "valley_depth should be between 0 and 1"
    assert spacing > 0, "spacing must be positive"
    assert orientation in ["x", "y", "diagonal"], "orientation must be 'x', 'y', or 'diagonal'"
    
    # Create coordinate grids normalized to [0, 1]
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Generate ridge-valley pattern
    if orientation == "x":
        # Ridges and valleys along X axis
        pattern = np.cos(2 * np.pi * spacing * X)
    elif orientation == "y":
        # Ridges and valleys along Y axis
        pattern = np.cos(2 * np.pi * spacing * Y)
    else:  # diagonal
        # Diagonal pattern
        pattern = np.cos(2 * np.pi * spacing * (X + Y))
    
    # Map cosine pattern [-1, 1] to [valley_depth, ridge_height]
    # Cosine gives -1 at valleys, +1 at ridges
    terrain = valley_depth + (ridge_height - valley_depth) * (pattern + 1.0) / 2.0
    
    # Apply smoothing if requested
    if smoothness > 0:
        # Apply simple box filter for smoothing (no scipy dependency)
        kernel_size = int(smoothness * 5) + 1
        if kernel_size > 1:
            # Simple averaging filter
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
            # Pad terrain for convolution
            pad_size = kernel_size // 2
            padded = np.pad(terrain, pad_size, mode='edge')
            smoothed = np.zeros_like(terrain)
            for i in range(n):
                for j in range(n):
                    smoothed[i, j] = np.mean(padded[i:i+kernel_size, j:j+kernel_size])
            terrain = terrain * (1.0 - smoothness) + smoothed * smoothness
    
    # Clip to [0, 1]
    # We do NOT re-normalize because ridge_height/valley_depth control height
    terrain = np.clip(terrain, 0.0, 1.0)
    
    return terrain.flatten()

