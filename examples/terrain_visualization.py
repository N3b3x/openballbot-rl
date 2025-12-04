"""
Terrain Visualization Example

This script demonstrates all available terrain types and generates visualizations.
Run with: python examples/terrain_visualization.py
"""
import numpy as np
import matplotlib.pyplot as plt
from ballbot_gym.core.factories import create_terrain
import yaml
import os


def visualize_terrain(terrain_data, n, title, ax):
    """Visualize terrain as a 2D heatmap."""
    terrain_2d = terrain_data.reshape(n, n)
    im = ax.imshow(terrain_2d, cmap='terrain', origin='lower')
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)


def main():
    """Generate visualizations for all terrain types."""
    print("=" * 80)
    print("Terrain Visualization Example")
    print("=" * 80)
    
    n = 129  # Standard MuJoCo heightfield size
    
    # Define terrain configurations
    terrain_configs = {
        "Flat": {
            "type": "flat",
            "config": {}
        },
        "Perlin Noise": {
            "type": "perlin",
            "config": {
                "scale": 25.0,
                "octaves": 4,
                "persistence": 0.2,
                "seed": 42
            }
        },
        "Ramp (X)": {
            "type": "ramp",
            "config": {
                "ramp_angle": 15.0,
                "ramp_direction": "x",
                "flat_ratio": 0.3,
                "num_ramps": 1
            }
        },
        "Ramp (Radial)": {
            "type": "ramp",
            "config": {
                "ramp_angle": 12.0,
                "ramp_direction": "radial",
                "flat_ratio": 0.4
            }
        },
        "Sinusoidal": {
            "type": "sinusoidal",
            "config": {
                "amplitude": 0.5,
                "frequency": 0.1,
                "direction": "both"
            }
        },
        "Ridge-Valley": {
            "type": "ridge_valley",
            "config": {
                "ridge_height": 0.6,
                "valley_depth": 0.4,
                "spacing": 0.2,
                "orientation": "x"
            }
        },
        "Hills": {
            "type": "hills",
            "config": {
                "num_hills": 5,
                "hill_height": 0.7,
                "hill_radius": 0.15,
                "seed": 42
            }
        },
        "Bowl": {
            "type": "bowl",
            "config": {
                "depth": 0.6,
                "radius": 0.4,
                "center_x": 0.5,
                "center_y": 0.5
            }
        },
        "Gradient": {
            "type": "gradient",
            "config": {
                "max_slope": 20.0,
                "gradient_type": "linear",
                "direction": "x"
            }
        },
        "Terraced": {
            "type": "terraced",
            "config": {
                "num_terraces": 5,
                "terrace_height": 0.15,
                "direction": "x"
            }
        },
        "Wavy": {
            "type": "wavy",
            "config": {
                "wave_amplitudes": [0.3, 0.2, 0.1],
                "wave_frequencies": [0.05, 0.1, 0.2],
                "wave_directions": [0.0, 45.0, 90.0]
            }
        },
        "Spiral": {
            "type": "spiral",
            "config": {
                "spiral_tightness": 0.1,
                "height_variation": 0.5,
                "direction": "cw"
            }
        },
        "Mixed": {
            "type": "mixed",
            "config": {
                "components": [
                    {
                        "type": "hills",
                        "weight": 0.4,
                        "config": {"num_hills": 3, "hill_height": 0.5, "seed": 42}
                    },
                    {
                        "type": "ramp",
                        "weight": 0.3,
                        "config": {"ramp_angle": 10.0, "ramp_direction": "x"}
                    },
                    {
                        "type": "perlin",
                        "weight": 0.3,
                        "config": {"scale": 30.0, "octaves": 3, "seed": 42}
                    }
                ],
                "blend_mode": "additive"
            }
        }
    }
    
    # Create figure with subplots
    num_terrains = len(terrain_configs)
    cols = 4
    rows = (num_terrains + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten() if num_terrains > 1 else [axes]
    
    print(f"\nGenerating {num_terrains} terrain types...")
    
    for idx, (name, config) in enumerate(terrain_configs.items()):
        print(f"  [{idx+1}/{num_terrains}] Generating {name}...")
        
        try:
            # Create terrain generator
            terrain_gen = create_terrain(config)
            
            # Generate terrain
            terrain = terrain_gen(n, seed=42)
            
            # Visualize
            visualize_terrain(terrain, n, name, axes[idx])
            
            # Print stats
            print(f"    ✓ Shape: {terrain.shape}, Min: {terrain.min():.3f}, "
                  f"Max: {terrain.max():.3f}, Mean: {terrain.mean():.3f}")
            
        except Exception as e:
            print(f"    ✗ Error generating {name}: {e}")
            axes[idx].text(0.5, 0.5, f"Error:\n{str(e)}", 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(name, fontsize=10, color='red')
    
    # Hide unused subplots
    for idx in range(num_terrains, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = "terrain_visualizations.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualizations saved to: {output_path}")
    
    # Optionally display
    try:
        plt.show()
    except:
        print("  (Display not available, saved to file)")
    
    print("\n" + "=" * 80)
    print("Terrain visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

