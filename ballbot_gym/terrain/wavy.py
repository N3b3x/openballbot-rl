"""Wavy/undulating terrain generator for ballbot environment."""
import numpy as np
from typing import Optional, List


def generate_wavy_terrain(
    n: int,
    wave_amplitudes: Optional[List[float]] = None,
    wave_frequencies: Optional[List[float]] = None,
    wave_directions: Optional[List[float]] = None,
    phase_offsets: Optional[List[float]] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate wavy/undulating terrain with multiple wave frequencies.
    
    Creates natural-looking undulating terrain by combining multiple sine
    waves at different frequencies, amplitudes, and directions.
    
    Args:
        n: Grid size (number of rows/columns, should be odd)
        wave_amplitudes: List of wave amplitudes (normalized, 0-1)
        wave_frequencies: List of wave frequencies (cycles per grid unit)
        wave_directions: List of wave directions in degrees (0-360)
        phase_offsets: List of phase offsets in radians
        seed: Random seed (unused, for API compatibility)
        
    Returns:
        1D array of height values, shape (n*n,), normalized to [0, 1]
        
    Raises:
        AssertionError: If n is not odd or parameters are invalid
    """
    assert n % 2 == 1, "n should be odd for heightfield symmetry"
    
    # Default parameters for a natural-looking wave pattern
    if wave_amplitudes is None:
        wave_amplitudes = [0.3, 0.2, 0.1]
    if wave_frequencies is None:
        wave_frequencies = [0.05, 0.1, 0.2]
    if wave_directions is None:
        wave_directions = [0.0, 45.0, 90.0]
    if phase_offsets is None:
        phase_offsets = [0.0, 0.5, 1.0]
    
    # Validate parameters
    num_waves = len(wave_amplitudes)
    assert len(wave_frequencies) == num_waves, "wave_frequencies must match wave_amplitudes length"
    assert len(wave_directions) == num_waves, "wave_directions must match wave_amplitudes length"
    assert len(phase_offsets) == num_waves, "phase_offsets must match wave_amplitudes length"
    
    # Create coordinate grids
    x = np.linspace(0, 2 * np.pi, n)
    y = np.linspace(0, 2 * np.pi, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Initialize terrain
    terrain = np.zeros((n, n))
    
    # Combine multiple waves
    for amp, freq, direction_deg, phase in zip(
        wave_amplitudes, wave_frequencies, wave_directions, phase_offsets
    ):
        # Convert direction to radians
        direction_rad = np.radians(direction_deg)
        
        # Calculate wave direction vector
        dir_x = np.cos(direction_rad)
        dir_y = np.sin(direction_rad)
        
        # Project coordinates onto wave direction
        wave_coord = X * dir_x + Y * dir_y
        
        # Generate wave
        wave = amp * np.sin(freq * wave_coord + phase)
        terrain += wave
    
    # Center waves at 0.5 so they oscillate around mid-height
    terrain += 0.5
    
    # Clip to [0, 1]
    # We do NOT re-normalize because amplitudes should control physical height
    terrain = np.clip(terrain, 0.0, 1.0)
    
    return terrain.flatten()

