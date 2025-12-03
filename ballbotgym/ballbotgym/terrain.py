import pdb
import numpy as np
import quaternion
import matplotlib.pyplot as plt

import mujoco
import mujoco.viewer

from noise import snoise2  # more coherent than pnoise2

def generate_perlin_terrain(
        n: int,
        scale: float = 25.0,
        octaves: int = 4,
        persistence: float = 0.2,
        lacunarity: float = 2.0,
        seed: int = 0
    ) -> np.ndarray:
    """
    Generate a Perlin noise-based terrain heightfield as a 1D array.

    This function creates a square Perlin noise height map and normalizes the values 
    to [0, 1]. The heightfield can be directly used as MuJoCo's hfield data.

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
        seed (int, optional): 
            Noise seed for randomness/reproducibility. Default is 0.

    Returns:
        np.ndarray: 
            The flattened (1D) normalized terrain array, shape (n * n,) with values in [0, 1].

    Raises:
        AssertionError: If 'n' is not odd or terrain processing fails.

    Notes:
        - The output 1D array is row-major, which matches MuJoCo's heightfield.
        - The function uses snoise2 (simplex noise) for smooth, reproducible terrain.
        - Values are normalized after creation to fit the required range.

    Example:
        >>> hfield = generate_perlin_terrain(129, seed=42)
        >>> hfield.shape
        (16641,)
    """
    assert n % 2 == 1, "n should be odd for heightfield symmetry"

    # Initialize empty terrain
    terrain = np.zeros((n, n))

    # Fill terrain with Perlin noise values
    for i in range(n):
        for j in range(n):
            x = i / scale
            y = j / scale
            terrain[i][j] = snoise2(
                x,
                y,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=seed
            )
    
    # Normalize terrain to [0, 1]
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 1e-8)

    # Sanity checks
    assert (terrain >= 0).all(), "Terrain normalization failed: negative values found"
    assert (terrain.flatten().reshape(n, n) == terrain).all(), "Flattening did not preserve order"

    # Optionally visualize (uncomment for debugging)
    # plt.imshow(terrain)
    # plt.show()

    # Return as flat array (row-major) for MuJoCo usage
    return terrain.flatten()
