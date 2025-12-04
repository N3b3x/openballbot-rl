import pdb
import numpy as np
import quaternion
import matplotlib.pyplot as plt

from noise import snoise2  # more coherent than pnoise2

def generate_perlin_terrain(
        n: int,
        scale: float = 25.0,
        octaves: int = 4,
        persistence: float = 0.2,
        lacunarity: float = 2.0,
        amplitude: float = 1.0,  # Added amplitude control
        seed: int = 0
    ) -> np.ndarray:
    """
    Generate a Perlin noise-based terrain heightfield as a 1D array.

    This function creates a square Perlin noise height map. The output values
    are roughly in [0, amplitude], clipped to [0, 1].

    Args:
        n (int): 
            The grid size (number of rows and columns), should be odd for symmetry.
        scale (float, optional): 
            Controls the size of terrain features (higher => larger features). 
            Default is 25.0.
        octaves (int, optional): 
            Number of noise octaves (controls complexity/multiscale detail). 
            Default is 4.
        persistence (float, optional): 
            Amplitude scaling factor between octaves. Default is 0.2.
        lacunarity (float, optional): 
            Frequency scaling factor between octaves. Default is 2.0.
        amplitude (float, optional):
            Scaling factor for the noise height. Default is 1.0.
        seed (int, optional): 
            Noise seed for randomness/reproducibility. Default is 0.

    Returns:
        np.ndarray: 
            The flattened (1D) terrain array, shape (n * n,) with values in [0, 1].
    """
    assert n % 2 == 1, "n should be odd for heightfield symmetry"

    # Initialize empty terrain
    terrain = np.zeros((n, n))

    # Fill terrain with Perlin noise values
    for i in range(n):
        for j in range(n):
            x = i / scale
            y = j / scale
            # snoise2 returns values roughly in [-1, 1]
            noise_val = snoise2(
                x,
                y,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=seed
            )
            # Map roughly [-1, 1] to [0, 1]
            norm_val = (noise_val + 1.0) / 2.0
            terrain[i][j] = norm_val * amplitude
    
    # Clip to ensure [0, 1] range (snoise2 can slightly exceed bounds)
    terrain = np.clip(terrain, 0.0, 1.0)

    # Return as flat array (row-major) for MuJoCo usage
    return terrain.flatten()
