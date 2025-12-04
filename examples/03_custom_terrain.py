"""
Example 3: Adding a Custom Terrain Generator

This example shows how to:
1. Create a custom terrain generator function
2. Register it with the component registry
3. Use it in the environment via configuration

Run this example:
    python examples/03_custom_terrain.py
"""
import numpy as np
import gymnasium as gym
import ballbot_gym
from ballbot_gym.core.registry import ComponentRegistry
from ballbot_gym.core.factories import create_terrain


def generate_sine_wave_terrain(
    n: int,
    amplitude: float = 0.5,
    frequency: float = 0.1,
    seed: int = 0
) -> np.ndarray:
    """
    Generate terrain with sine wave pattern.
    
    This creates a simple sinusoidal terrain pattern, useful for testing
    the robot's ability to navigate periodic obstacles.
    
    Args:
        n: Grid size (number of rows/columns, should be odd)
        amplitude: Amplitude of sine wave (in normalized units)
        frequency: Frequency of sine wave (cycles per grid unit)
        seed: Random seed (unused, for API compatibility)
        
    Returns:
        1D array of height values, shape (n*n,), normalized to [0, 1]
    """
    assert n % 2 == 1, "n should be odd for heightfield symmetry"
    
    # Create coordinate grids
    x = np.linspace(0, 2 * np.pi * frequency * n, n)
    y = np.linspace(0, 2 * np.pi * frequency * n, n)
    X, Y = np.meshgrid(x, y)
    
    # Generate sine wave pattern
    terrain = amplitude * (np.sin(X) + np.sin(Y)) / 2.0
    
    # Normalize to [0, 1]
    terrain_min = terrain.min()
    terrain_max = terrain.max()
    if terrain_max > terrain_min:
        terrain = (terrain - terrain_min) / (terrain_max - terrain_min)
    else:
        terrain = np.zeros_like(terrain)
    
    return terrain.flatten()


def main():
    """Demonstrate custom terrain usage."""
    print("=" * 60)
    print("Example: Custom Terrain Generator")
    print("=" * 60)
    
    # Step 1: Register the custom terrain
    print("\n1. Registering custom terrain...")
    ComponentRegistry.register_terrain("sine_wave", generate_sine_wave_terrain)
    print(f"   ✓ Registered 'sine_wave' terrain")
    print(f"   Available terrains: {ComponentRegistry.list_terrains()}")
    
    # Step 2: Create terrain generator using factory
    print("\n2. Creating terrain generator from config...")
    terrain_config = {
        "type": "sine_wave",
        "config": {
            "amplitude": 0.5,
            "frequency": 0.1
        }
    }
    terrain_gen = create_terrain(terrain_config)
    print(f"   ✓ Created terrain generator")
    
    # Step 3: Generate terrain
    print("\n3. Generating terrain...")
    n = 129  # Standard MuJoCo heightfield size
    terrain = terrain_gen(n, seed=42)
    print(f"   Terrain shape: {terrain.shape}")
    print(f"   Terrain min: {terrain.min():.4f}, max: {terrain.max():.4f}")
    print(f"   Terrain mean: {terrain.mean():.4f}")
    
    # Step 4: Use in environment
    print("\n4. Using custom terrain in environment...")
    env = gym.make(
        "ballbot-v0.1",
        GUI=False,
        terrain_config=terrain_config,
        max_ep_steps=100
    )
    
    obs, _ = env.reset(seed=42)
    print(f"   ✓ Environment created with custom terrain")
    
    # Run a few steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    env.close()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nTo use this terrain in training, add to config.yaml:")
    print("""
problem:
  terrain:
    type: "sine_wave"
    config:
      amplitude: 0.5
      frequency: 0.1
      seed: null  # null = random each episode
""")


if __name__ == "__main__":
    main()

