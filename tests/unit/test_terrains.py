"""Unit tests for terrain generators."""
import pytest
import numpy as np
from ballbot_gym.core.factories import create_terrain
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


class TestTerrainGenerators:
    """Test terrain generator implementations."""
    
    def test_perlin_terrain(self):
        """Test Perlin terrain generation."""
        n = 129
        terrain = generate_perlin_terrain(n, seed=42)
        
        assert terrain.shape == (n * n,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
        assert np.allclose(terrain.min(), 0.0)  # Normalized to [0, 1]
    
    def test_perlin_terrain_reproducibility(self):
        """Test that Perlin terrain is reproducible with same seed."""
        n = 129
        terrain1 = generate_perlin_terrain(n, seed=42)
        terrain2 = generate_perlin_terrain(n, seed=42)
        assert np.allclose(terrain1, terrain2)
    
    def test_perlin_terrain_different_seeds(self):
        """Test that different seeds produce different terrain."""
        n = 129
        terrain1 = generate_perlin_terrain(n, seed=42)
        terrain2 = generate_perlin_terrain(n, seed=43)
        assert not np.allclose(terrain1, terrain2)
    
    def test_perlin_terrain_factory(self):
        """Test creating Perlin terrain via factory."""
        config = {
            "type": "perlin",
            "config": {
                "scale": 25.0,
                "octaves": 4,
                "seed": 42
            }
        }
        terrain_gen = create_terrain(config)
        terrain = terrain_gen(129)
        
        assert terrain.shape == (129 * 129,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_stepped_terrain(self):
        """Test stepped terrain generation."""
        n = 129
        terrain = generate_stepped_terrain(n, num_steps=5, step_height=0.1, seed=42)
        
        assert terrain.shape == (n * n,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_stepped_terrain_factory(self):
        """Test creating stepped terrain via factory."""
        config = {
            "type": "stepped",
            "config": {
                "num_steps": 8,
                "step_height": 0.15,
                "seed": 42
            }
        }
        terrain_gen = create_terrain(config)
        terrain = terrain_gen(129)
        
        assert terrain.shape == (129 * 129,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_flat_terrain(self):
        """Test flat terrain generation."""
        from ballbot_gym.terrain import generate_flat_terrain
        
        n = 129
        terrain = generate_flat_terrain(n)
        
        assert terrain.shape == (n * n,)
        assert np.allclose(terrain, 0.0)
    
    def test_flat_terrain_factory(self):
        """Test creating flat terrain via factory."""
        config = {
            "type": "flat",
            "config": {}
        }
        terrain_gen = create_terrain(config)
        terrain = terrain_gen(129)
        
        assert terrain.shape == (129 * 129,)
        assert np.allclose(terrain, 0.0)
    
    # ========================================================================
    # New terrain type tests
    # ========================================================================
    
    def test_ramp_terrain(self):
        """Test ramp terrain generation."""
        n = 129
        terrain = generate_ramp_terrain(n, ramp_angle=15.0, ramp_direction="x", seed=42)
        
        assert terrain.shape == (n * n,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_ramp_terrain_factory(self):
        """Test creating ramp terrain via factory."""
        config = {
            "type": "ramp",
            "config": {
                "ramp_angle": 15.0,
                "ramp_direction": "x",
                "flat_ratio": 0.3
            }
        }
        terrain_gen = create_terrain(config)
        terrain = terrain_gen(129)
        
        assert terrain.shape == (129 * 129,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_sinusoidal_terrain(self):
        """Test sinusoidal terrain generation."""
        n = 129
        terrain = generate_sinusoidal_terrain(n, amplitude=0.5, frequency=0.1, seed=42)
        
        assert terrain.shape == (n * n,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_sinusoidal_terrain_factory(self):
        """Test creating sinusoidal terrain via factory."""
        config = {
            "type": "sinusoidal",
            "config": {
                "amplitude": 0.5,
                "frequency": 0.1,
                "direction": "both"
            }
        }
        terrain_gen = create_terrain(config)
        terrain = terrain_gen(129)
        
        assert terrain.shape == (129 * 129,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_ridge_valley_terrain(self):
        """Test ridge-valley terrain generation."""
        n = 129
        terrain = generate_ridge_valley_terrain(n, spacing=0.2, seed=42)
        
        assert terrain.shape == (n * n,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_ridge_valley_terrain_factory(self):
        """Test creating ridge-valley terrain via factory."""
        config = {
            "type": "ridge_valley",
            "config": {
                "ridge_height": 0.6,
                "valley_depth": 0.4,
                "spacing": 0.2
            }
        }
        terrain_gen = create_terrain(config)
        terrain = terrain_gen(129)
        
        assert terrain.shape == (129 * 129,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_hills_terrain(self):
        """Test hills terrain generation."""
        n = 129
        terrain = generate_hills_terrain(n, num_hills=5, seed=42)
        
        assert terrain.shape == (n * n,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_hills_terrain_reproducibility(self):
        """Test that hills terrain is reproducible with same seed."""
        n = 129
        terrain1 = generate_hills_terrain(n, num_hills=5, seed=42)
        terrain2 = generate_hills_terrain(n, num_hills=5, seed=42)
        assert np.allclose(terrain1, terrain2)
    
    def test_hills_terrain_factory(self):
        """Test creating hills terrain via factory."""
        config = {
            "type": "hills",
            "config": {
                "num_hills": 5,
                "hill_height": 0.7,
                "hill_radius": 0.15,
                "seed": 42
            }
        }
        terrain_gen = create_terrain(config)
        terrain = terrain_gen(129)
        
        assert terrain.shape == (129 * 129,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_bowl_terrain(self):
        """Test bowl terrain generation."""
        n = 129
        terrain = generate_bowl_terrain(n, depth=0.6, radius=0.4, seed=42)
        
        assert terrain.shape == (n * n,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_bowl_terrain_factory(self):
        """Test creating bowl terrain via factory."""
        config = {
            "type": "bowl",
            "config": {
                "depth": 0.6,
                "radius": 0.4,
                "center_x": 0.5,
                "center_y": 0.5
            }
        }
        terrain_gen = create_terrain(config)
        terrain = terrain_gen(129)
        
        assert terrain.shape == (129 * 129,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_gradient_terrain(self):
        """Test gradient terrain generation."""
        n = 129
        terrain = generate_gradient_terrain(n, max_slope=20.0, gradient_type="linear", seed=42)
        
        assert terrain.shape == (n * n,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_gradient_terrain_factory(self):
        """Test creating gradient terrain via factory."""
        config = {
            "type": "gradient",
            "config": {
                "max_slope": 20.0,
                "gradient_type": "linear",
                "direction": "x"
            }
        }
        terrain_gen = create_terrain(config)
        terrain = terrain_gen(129)
        
        assert terrain.shape == (129 * 129,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_terraced_terrain(self):
        """Test terraced terrain generation."""
        n = 129
        terrain = generate_terraced_terrain(n, num_terraces=5, seed=42)
        
        assert terrain.shape == (n * n,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_terraced_terrain_factory(self):
        """Test creating terraced terrain via factory."""
        config = {
            "type": "terraced",
            "config": {
                "num_terraces": 5,
                "terrace_height": 0.15,
                "direction": "x"
            }
        }
        terrain_gen = create_terrain(config)
        terrain = terrain_gen(129)
        
        assert terrain.shape == (129 * 129,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_wavy_terrain(self):
        """Test wavy terrain generation."""
        n = 129
        terrain = generate_wavy_terrain(n, seed=42)
        
        assert terrain.shape == (n * n,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_wavy_terrain_factory(self):
        """Test creating wavy terrain via factory."""
        config = {
            "type": "wavy",
            "config": {
                "wave_amplitudes": [0.3, 0.2, 0.1],
                "wave_frequencies": [0.05, 0.1, 0.2]
            }
        }
        terrain_gen = create_terrain(config)
        terrain = terrain_gen(129)
        
        assert terrain.shape == (129 * 129,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_spiral_terrain(self):
        """Test spiral terrain generation."""
        n = 129
        terrain = generate_spiral_terrain(n, spiral_tightness=0.1, seed=42)
        
        assert terrain.shape == (n * n,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_spiral_terrain_factory(self):
        """Test creating spiral terrain via factory."""
        config = {
            "type": "spiral",
            "config": {
                "spiral_tightness": 0.1,
                "height_variation": 0.5,
                "direction": "cw"
            }
        }
        terrain_gen = create_terrain(config)
        terrain = terrain_gen(129)
        
        assert terrain.shape == (129 * 129,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_mixed_terrain(self):
        """Test mixed terrain generation."""
        n = 129
        components = [
            {
                "type": "hills",
                "weight": 0.5,
                "config": {"num_hills": 3, "seed": 42}
            },
            {
                "type": "ramp",
                "weight": 0.5,
                "config": {"ramp_angle": 10.0, "ramp_direction": "x"}
            }
        ]
        terrain = generate_mixed_terrain(n, components=components, blend_mode="additive", seed=42)
        
        assert terrain.shape == (n * n,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_mixed_terrain_factory(self):
        """Test creating mixed terrain via factory."""
        config = {
            "type": "mixed",
            "config": {
                "components": [
                    {
                        "type": "hills",
                        "weight": 0.5,
                        "config": {"num_hills": 3}
                    },
                    {
                        "type": "ramp",
                        "weight": 0.5,
                        "config": {"ramp_angle": 10.0, "ramp_direction": "x"}
                    }
                ],
                "blend_mode": "additive"
            }
        }
        terrain_gen = create_terrain(config)
        terrain = terrain_gen(129)
        
        assert terrain.shape == (129 * 129,)
        assert terrain.min() >= 0.0
        assert terrain.max() <= 1.0
    
    def test_all_terrains_normalized(self):
        """Test that all terrain types produce normalized output."""
        n = 129
        terrain_generators = [
            lambda: generate_ramp_terrain(n, seed=42),
            lambda: generate_sinusoidal_terrain(n, seed=42),
            lambda: generate_ridge_valley_terrain(n, seed=42),
            lambda: generate_hills_terrain(n, seed=42),
            lambda: generate_bowl_terrain(n, seed=42),
            lambda: generate_gradient_terrain(n, seed=42),
            lambda: generate_terraced_terrain(n, seed=42),
            lambda: generate_wavy_terrain(n, seed=42),
            lambda: generate_spiral_terrain(n, seed=42),
        ]
        
        for gen in terrain_generators:
            terrain = gen()
            assert terrain.min() >= 0.0, f"Terrain {gen} has negative values"
            assert terrain.max() <= 1.0, f"Terrain {gen} has values > 1.0"
            assert np.allclose(terrain.min(), 0.0) or np.allclose(terrain.max(), 1.0), \
                f"Terrain {gen} is not properly normalized"

