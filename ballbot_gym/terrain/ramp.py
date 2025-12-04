"""Ramp/inclined plane terrain generator for ballbot environment."""
import numpy as np
from typing import Optional


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """
    Smooth interpolation function for smooth transitions.
    
    Returns a smooth value between 0 and 1 for x in [edge0, edge1].
    Uses Hermite interpolation for C¹ continuity.
    """
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def generate_ramp_terrain(
    n: int,
    ramp_angle: float = 15.0,
    ramp_direction: str = "x",
    flat_ratio: float = 0.3,
    num_ramps: int = 1,
    transition_smoothness: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate ramp/inclined plane terrain with smooth transitions.
    
    Creates continuous slopes with flat areas at top/bottom. Suitable for
    testing ballbot slope compensation algorithms.
    
    Args:
        n: Grid size (number of rows/columns, should be odd)
        ramp_angle: Maximum slope angle in degrees (0-30 recommended)
        ramp_direction: Direction of ramp ("x", "y", or "radial")
        flat_ratio: Ratio of terrain that is flat (0.0-1.0)
        num_ramps: Number of ramps (for multiple ramps)
        transition_smoothness: Smoothness of transitions (0.0-1.0)
        seed: Random seed (unused, for API compatibility)
        
    Returns:
        1D array of height values, shape (n*n,), normalized to [0, 1]
        
    Raises:
        AssertionError: If n is not odd or parameters are invalid
    """
    assert n % 2 == 1, "n should be odd for heightfield symmetry"
    assert 0 <= ramp_angle <= 45, "ramp_angle should be between 0 and 45 degrees"
    assert 0 <= flat_ratio <= 1.0, "flat_ratio should be between 0 and 1"
    assert num_ramps > 0, "num_ramps must be positive"
    assert ramp_direction in ["x", "y", "radial"], "ramp_direction must be 'x', 'y', or 'radial'"
    
    # Initialize terrain
    terrain = np.zeros((n, n))
    
    # Convert angle to normalized height gradient
    # Assuming terrain spans from -1 to 1 in normalized coordinates
    max_height = np.tan(np.radians(ramp_angle)) * 2.0  # Scale appropriately
    
    # Create coordinate grids normalized to [-1, 1]
    center = n // 2
    x = (np.arange(n) - center) / center
    y = (np.arange(n) - center) / center
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    if ramp_direction == "x":
        # Ramp along X axis
        if num_ramps == 1:
            # Single ramp from left to right
            ramp_width = 1.0 - flat_ratio
            flat_width = flat_ratio / 2.0
            
            # Create smooth ramp
            for i in range(n):
                for j in range(n):
                    x_val = X[i, j]
                    if x_val < -flat_width:
                        # Flat area on left
                        terrain[i, j] = 0.0
                    elif x_val < flat_width:
                        # Ramp section
                        # Normalize x_val to [0, 1] for ramp section
                        ramp_x = (x_val + flat_width) / (flat_width * 2)
                        # Apply smoothstep for smooth transition
                        smooth_ramp_x = smoothstep(0.0, 1.0, ramp_x)
                        terrain[i, j] = smooth_ramp_x * max_height
                    else:
                        # Flat area on right
                        terrain[i, j] = max_height
        else:
            # Multiple ramps
            period = 2.0 / num_ramps
            for i in range(n):
                for j in range(n):
                    x_val = X[i, j]
                    # Map to periodic pattern
                    phase = (x_val + 1.0) % period
                    phase_norm = phase / period  # Normalize to [0, 1]
                    
                    if phase_norm < flat_ratio / 2:
                        terrain[i, j] = 0.0
                    elif phase_norm < 1.0 - flat_ratio / 2:
                        ramp_phase = (phase_norm - flat_ratio / 2) / (1.0 - flat_ratio)
                        smooth_phase = smoothstep(0.0, 1.0, ramp_phase)
                        terrain[i, j] = smooth_phase * max_height
                    else:
                        terrain[i, j] = max_height
                        
    elif ramp_direction == "y":
        # Ramp along Y axis (same as X but swap X and Y)
        if num_ramps == 1:
            ramp_width = 1.0 - flat_ratio
            flat_width = flat_ratio / 2.0
            
            for i in range(n):
                for j in range(n):
                    y_val = Y[i, j]
                    if y_val < -flat_width:
                        terrain[i, j] = 0.0
                    elif y_val < flat_width:
                        ramp_y = (y_val + flat_width) / (flat_width * 2)
                        smooth_ramp_y = smoothstep(0.0, 1.0, ramp_y)
                        terrain[i, j] = smooth_ramp_y * max_height
                    else:
                        terrain[i, j] = max_height
        else:
            period = 2.0 / num_ramps
            for i in range(n):
                for j in range(n):
                    y_val = Y[i, j]
                    phase = (y_val + 1.0) % period
                    phase_norm = phase / period
                    
                    if phase_norm < flat_ratio / 2:
                        terrain[i, j] = 0.0
                    elif phase_norm < 1.0 - flat_ratio / 2:
                        ramp_phase = (phase_norm - flat_ratio / 2) / (1.0 - flat_ratio)
                        smooth_phase = smoothstep(0.0, 1.0, ramp_phase)
                        terrain[i, j] = smooth_phase * max_height
                    else:
                        terrain[i, j] = max_height
                        
    else:  # radial
        # Radial ramp from center outward
        R = np.sqrt(X**2 + Y**2)
        max_radius = np.sqrt(2.0)  # Maximum distance from center
        
        # Normalize radius to [0, 1]
        R_norm = np.clip(R / max_radius, 0.0, 1.0)
        
        # Create radial ramp
        flat_radius = flat_ratio * max_radius / np.sqrt(2.0)
        for i in range(n):
            for j in range(n):
                r = R[i, j]
                if r < flat_radius:
                    terrain[i, j] = 0.0
                else:
                    ramp_r = (r - flat_radius) / (max_radius - flat_radius)
                    ramp_r = np.clip(ramp_r, 0.0, 1.0)
                    smooth_r = smoothstep(0.0, 1.0, ramp_r)
                    terrain[i, j] = smooth_r * max_height
    
    # Normalize to [0, 1]
    terrain_min = terrain.min()
    terrain_max = terrain.max()
    if terrain_max > terrain_min:
        terrain = (terrain - terrain_min) / (terrain_max - terrain_min)
    else:
        terrain = np.zeros_like(terrain)
    
    return terrain.flatten()

