"""Terraced terrain generator for ballbot environment."""
import numpy as np
from typing import Optional


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Smooth interpolation function for smooth transitions."""
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def generate_terraced_terrain(
    n: int,
    num_terraces: int = 5,
    terrace_height: float = 0.15,
    transition_width: float = 0.1,
    smoothness: float = 0.7,
    direction: str = "x",
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate smooth terraced terrain with gradual transitions.
    
    Creates multiple elevation levels with smooth transitions between them,
    improving upon stepped terrain for ballbot control.
    
    Args:
        n: Grid size (number of rows/columns, should be odd)
        num_terraces: Number of terrace levels
        terrace_height: Height difference between terraces (normalized, 0-1)
        transition_width: Width of transition zones as fraction of terrace width
        smoothness: Smoothness of transitions (0.0-1.0)
        direction: Terrace direction ("x" or "y")
        seed: Random seed (unused, for API compatibility)
        
    Returns:
        1D array of height values, shape (n*n,), normalized to [0, 1]
        
    Raises:
        AssertionError: If n is not odd or parameters are invalid
    """
    assert n % 2 == 1, "n should be odd for heightfield symmetry"
    assert num_terraces > 0, "num_terraces must be positive"
    assert 0 < terrace_height <= 1.0, "terrace_height should be between 0 and 1"
    assert 0 < transition_width < 1.0, "transition_width should be between 0 and 1"
    assert direction in ["x", "y"], "direction must be 'x' or 'y'"
    
    # Initialize terrain
    terrain = np.zeros((n, n))
    
    # Create coordinate grids normalized to [0, 1]
    x_coords = np.linspace(0, 1, n)
    y_coords = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Terrace width (excluding transitions)
    terrace_width = 1.0 / num_terraces
    transition_size = terrace_width * transition_width
    
    # Generate terraced pattern
    if direction == "x":
        coord = X
    else:  # y
        coord = Y
    
    for i in range(n):
        for j in range(n):
            c = coord[i, j]
            
            # Determine which terrace this point belongs to
            terrace_idx = int(c / terrace_width)
            terrace_idx = min(terrace_idx, num_terraces - 1)
            
            # Position within terrace [0, 1]
            pos_in_terrace = (c % terrace_width) / terrace_width
            
            # Base height for this terrace
            base_height = terrace_idx * terrace_height
            
            # Check if we're in a transition zone
            if pos_in_terrace < transition_size:
                # Transition from previous terrace
                if terrace_idx > 0:
                    prev_height = (terrace_idx - 1) * terrace_height
                    transition_pos = pos_in_terrace / transition_size
                    smooth_transition = smoothstep(0.0, 1.0, transition_pos)
                    terrain[i, j] = prev_height + (base_height - prev_height) * smooth_transition
                else:
                    terrain[i, j] = base_height
            elif pos_in_terrace > 1.0 - transition_size:
                # Transition to next terrace
                if terrace_idx < num_terraces - 1:
                    next_height = (terrace_idx + 1) * terrace_height
                    transition_pos = (pos_in_terrace - (1.0 - transition_size)) / transition_size
                    smooth_transition = smoothstep(0.0, 1.0, transition_pos)
                    terrain[i, j] = base_height + (next_height - base_height) * smooth_transition
                else:
                    terrain[i, j] = base_height
            else:
                # Flat terrace area
                terrain[i, j] = base_height
    
    # Clip to [0, 1]
    # We do NOT re-normalize because terrace_height controls height
    terrain = np.clip(terrain, 0.0, 1.0)
    
    return terrain.flatten()

