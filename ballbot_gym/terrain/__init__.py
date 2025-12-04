"""Terrain generation utilities for ballbot environment."""
import numpy as np
from ballbot_gym.core.registry import ComponentRegistry
from ballbot_gym.terrain.perlin import generate_perlin_terrain
from ballbot_gym.terrain.stepped import generate_stepped_terrain
from ballbot_gym.terrain.ramp import generate_ramp_terrain
from ballbot_gym.terrain.sinusoidal import generate_sinusoidal_terrain
from ballbot_gym.terrain.ridge_valley import generate_ridge_valley_terrain
from ballbot_gym.terrain.hills import generate_hills_terrain
from ballbot_gym.terrain.bowl import generate_bowl_terrain
from ballbot_gym.terrain.gradient import generate_gradient_terrain
from ballbot_gym.terrain.terraced import generate_terraced_terrain
from ballbot_gym.terrain.wavy import generate_wavy_terrain
from ballbot_gym.terrain.spiral import generate_spiral_terrain
from ballbot_gym.terrain.mixed import generate_mixed_terrain

# Auto-register components on import
ComponentRegistry.register_terrain("perlin", generate_perlin_terrain)
ComponentRegistry.register_terrain("stepped", generate_stepped_terrain)
ComponentRegistry.register_terrain("ramp", generate_ramp_terrain)
ComponentRegistry.register_terrain("sinusoidal", generate_sinusoidal_terrain)
ComponentRegistry.register_terrain("ridge_valley", generate_ridge_valley_terrain)
ComponentRegistry.register_terrain("hills", generate_hills_terrain)
ComponentRegistry.register_terrain("bowl", generate_bowl_terrain)
ComponentRegistry.register_terrain("gradient", generate_gradient_terrain)
ComponentRegistry.register_terrain("terraced", generate_terraced_terrain)
ComponentRegistry.register_terrain("wavy", generate_wavy_terrain)
ComponentRegistry.register_terrain("spiral", generate_spiral_terrain)
ComponentRegistry.register_terrain("mixed", generate_mixed_terrain)

# Register flat terrain as a simple function
def generate_flat_terrain(n: int, **kwargs) -> np.ndarray:
    """Generate flat terrain (all zeros)."""
    return np.zeros(n * n)

ComponentRegistry.register_terrain("flat", generate_flat_terrain)

__all__ = [
    "generate_perlin_terrain",
    "generate_stepped_terrain",
    "generate_flat_terrain",
    "generate_ramp_terrain",
    "generate_sinusoidal_terrain",
    "generate_ridge_valley_terrain",
    "generate_hills_terrain",
    "generate_bowl_terrain",
    "generate_gradient_terrain",
    "generate_terraced_terrain",
    "generate_wavy_terrain",
    "generate_spiral_terrain",
    "generate_mixed_terrain",
]

