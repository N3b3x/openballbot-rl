# Complete Guide to Terrain Types

*Comprehensive reference for all available terrain generators in openballbot-rl*

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Basic Terrains](#basic-terrains)
3. [Slope Terrains](#slope-terrains)
4. [Wave Terrains](#wave-terrains)
5. [Feature Terrains](#feature-terrains)
6. [Specialized Terrains](#specialized-terrains)
7. [Composite Terrains](#composite-terrains)
8. [Usage Examples](#usage-examples)
9. [Difficulty Progression](#difficulty-progression)

---

## 🎯 Overview

openballbot-rl provides **13 terrain types** designed for continuous, smooth terrain suitable for ballbot control. All terrains:

- Produce **continuous gradients** (C⁰ continuity, C¹ preferred)
- Output **normalized values** [0, 1]
- Support **randomization** via seed parameter
- Are **MuJoCo-compatible** (1D flattened arrays)

### Quick Reference

| Terrain Type | Use Case | Difficulty | Key Feature |
|--------------|----------|------------|-------------|
| `flat` | Baseline testing | Easy | Zero elevation |
| `perlin` | General purpose | Medium | Procedural noise |
| `ramp` | Slope compensation | Easy-Hard | Inclined planes |
| `sinusoidal` | Periodic patterns | Medium | Wave patterns |
| `ridge_valley` | Directional control | Medium | Alternating features |
| `hills` | Obstacle navigation | Medium | Smooth mounds |
| `bowl` | Recovery training | Medium | Radial depressions |
| `gradient` | Slope adaptation | Medium-Hard | Varying slopes |
| `terraced` | Elevation changes | Medium | Smooth steps |
| `wavy` | Natural undulation | Medium | Multi-frequency waves |
| `spiral` | Rotational control | Hard | Spiral patterns |
| `mixed` | Complex scenarios | Hard | Multiple types combined |

---

## 🏔️ Basic Terrains

### Flat Terrain (`flat`)

**Description**: Zero-height terrain for baseline testing.

**Configuration**:
```yaml
type: "flat"
config: {}
```

**Use Cases**:
- Initial training
- Baseline performance measurement
- Debugging control algorithms

**Example**:
```python
terrain_config = {
    "type": "flat",
    "config": {}
}
```

---

### Perlin Noise (`perlin`)

**Description**: Procedural noise-based terrain with natural appearance.

**Configuration**:
```yaml
type: "perlin"
config:
  scale: 25.0        # Feature size (higher = smoother)
  octaves: 4        # Detail levels
  persistence: 0.2  # Amplitude decay
  lacunarity: 2.0   # Frequency scaling
  seed: null        # Random seed
```

**Parameters**:
- `scale`: Controls feature size (10-50 recommended)
- `octaves`: Number of detail levels (1-8)
- `persistence`: Amplitude scaling (0.1-0.5)
- `lacunarity`: Frequency scaling (1.5-3.0)

**Example**:
```python
terrain_config = {
    "type": "perlin",
    "config": {
        "scale": 25.0,
        "octaves": 4,
        "persistence": 0.2,
        "seed": 42
    }
}
```

---

## ⛰️ Slope Terrains

### Ramp Terrain (`ramp`)

**Description**: Inclined planes with smooth transitions and flat areas.

**Configuration**:
```yaml
type: "ramp"
config:
  ramp_angle: 15.0          # Degrees (0-30 recommended)
  ramp_direction: "x"       # "x", "y", or "radial"
  flat_ratio: 0.3           # Ratio of flat area
  num_ramps: 1              # Number of ramps
  transition_smoothness: 0.5
  seed: null
```

**Parameters**:
- `ramp_angle`: Slope angle in degrees (0-45, 15-30 recommended for ballbots)
- `ramp_direction`: Direction of ramp ("x", "y", or "radial")
- `flat_ratio`: Fraction of terrain that is flat (0.0-1.0)
- `num_ramps`: Number of ramps (for periodic patterns)
- `transition_smoothness`: Smoothness of transitions (0.0-1.0)

**Use Cases**:
- Testing slope compensation algorithms
- Training on consistent inclines
- Simulating real-world ramps

**Examples**:

Single ramp:
```python
terrain_config = {
    "type": "ramp",
    "config": {
        "ramp_angle": 15.0,
        "ramp_direction": "x",
        "flat_ratio": 0.3
    }
}
```

Multiple ramps:
```python
terrain_config = {
    "type": "ramp",
    "config": {
        "ramp_angle": 12.0,
        "ramp_direction": "y",
        "num_ramps": 3,
        "flat_ratio": 0.2
    }
}
```

Radial ramp:
```python
terrain_config = {
    "type": "ramp",
    "config": {
        "ramp_angle": 10.0,
        "ramp_direction": "radial",
        "flat_ratio": 0.4
    }
}
```

---

### Gradient Terrain (`gradient`)

**Description**: Smooth gradient fields with varying slope angles.

**Configuration**:
```yaml
type: "gradient"
config:
  max_slope: 20.0        # Maximum slope angle (degrees)
  gradient_type: "linear" # "linear", "radial", or "perlin"
  smoothness: 0.5        # Smoothness factor
  direction: "x"         # "x" or "y" (for linear)
  seed: null
```

**Parameters**:
- `max_slope`: Maximum slope angle (0-45 degrees)
- `gradient_type`: Type of gradient pattern
- `smoothness`: Smoothness of transitions
- `direction`: Primary direction for linear gradients

**Use Cases**:
- Testing slope adaptation
- Training on varying inclines
- Simulating natural slopes

**Examples**:

Linear gradient:
```python
terrain_config = {
    "type": "gradient",
    "config": {
        "max_slope": 20.0,
        "gradient_type": "linear",
        "direction": "x"
    }
}
```

Perlin-based gradient:
```python
terrain_config = {
    "type": "gradient",
    "config": {
        "max_slope": 18.0,
        "gradient_type": "perlin",
        "smoothness": 0.4,
        "seed": 42
    }
}
```

---

## 🌊 Wave Terrains

### Sinusoidal Terrain (`sinusoidal`)

**Description**: Periodic wave patterns with smooth transitions.

**Configuration**:
```yaml
type: "sinusoidal"
config:
  amplitude: 0.5        # Wave amplitude (0-1)
  frequency: 0.1        # Cycles per grid unit
  direction: "both"     # "x", "y", or "both"
  phase: 0.0           # Phase offset (radians)
  seed: null
```

**Parameters**:
- `amplitude`: Wave amplitude (0.0-1.0)
- `frequency`: Wave frequency (cycles per grid unit)
- `direction`: Wave direction ("x", "y", or "both")
- `phase`: Phase offset in radians

**Use Cases**:
- Testing periodic obstacle navigation
- Training on repetitive patterns
- Simulating undulating terrain

**Example**:
```python
terrain_config = {
    "type": "sinusoidal",
    "config": {
        "amplitude": 0.5,
        "frequency": 0.1,
        "direction": "both"
    }
}
```

---

### Wavy Terrain (`wavy`)

**Description**: Multi-frequency undulating terrain with natural appearance.

**Configuration**:
```yaml
type: "wavy"
config:
  wave_amplitudes: [0.3, 0.2, 0.1]    # List of amplitudes
  wave_frequencies: [0.05, 0.1, 0.2]   # List of frequencies
  wave_directions: [0.0, 45.0, 90.0]  # Directions in degrees
  phase_offsets: [0.0, 0.5, 1.0]      # Phase offsets (radians)
  seed: null
```

**Parameters**:
- `wave_amplitudes`: List of wave amplitudes
- `wave_frequencies`: List of wave frequencies
- `wave_directions`: List of wave directions (degrees)
- `phase_offsets`: List of phase offsets (radians)

**Use Cases**:
- Testing balance on undulating surfaces
- Training on natural-looking terrain
- Simulating rolling hills

**Example**:
```python
terrain_config = {
    "type": "wavy",
    "config": {
        "wave_amplitudes": [0.3, 0.2, 0.1],
        "wave_frequencies": [0.05, 0.1, 0.2],
        "wave_directions": [0.0, 45.0, 90.0]
    }
}
```

---

## 🏞️ Feature Terrains

### Ridge-Valley Terrain (`ridge_valley`)

**Description**: Alternating ridges and valleys with smooth transitions.

**Configuration**:
```yaml
type: "ridge_valley"
config:
  ridge_height: 0.6      # Height of ridges (0-1)
  valley_depth: 0.4      # Depth of valleys (0-1)
  spacing: 0.2          # Spacing between features
  orientation: "x"      # "x", "y", or "diagonal"
  smoothness: 0.3        # Transition smoothness
  seed: null
```

**Use Cases**:
- Testing navigation through terrain features
- Training on directional challenges
- Simulating natural landscapes

**Example**:
```python
terrain_config = {
    "type": "ridge_valley",
    "config": {
        "ridge_height": 0.6,
        "valley_depth": 0.4,
        "spacing": 0.2,
        "orientation": "x"
    }
}
```

---

### Hills Terrain (`hills`)

**Description**: Multiple smooth hills/mounds with flat areas between.

**Configuration**:
```yaml
type: "hills"
config:
  num_hills: 5          # Number of hills
  hill_height: 0.7      # Maximum hill height (0-1)
  hill_radius: 0.15     # Hill radius (0-0.5)
  flat_ratio: 0.4       # Ratio of flat area
  seed: null
```

**Parameters**:
- `num_hills`: Number of hills to generate
- `hill_height`: Maximum hill height (0.0-1.0)
- `hill_radius`: Hill radius as fraction of grid (0.0-0.5)
- `flat_ratio`: Ratio of terrain that is relatively flat

**Use Cases**:
- Testing navigation around obstacles
- Training on varied elevation
- Simulating rolling hills

**Example**:
```python
terrain_config = {
    "type": "hills",
    "config": {
        "num_hills": 5,
        "hill_height": 0.7,
        "hill_radius": 0.15,
        "seed": 42
    }
}
```

---

### Bowl Terrain (`bowl`)

**Description**: Smooth bowl-shaped depression or elevation.

**Configuration**:
```yaml
type: "bowl"
config:
  depth: 0.6            # Depth/height (0-1)
  radius: 0.4          # Bowl radius (0-1)
  center_x: 0.5        # Center X position (0-1)
  center_y: 0.5        # Center Y position (0-1)
  smoothness: 0.5      # Edge smoothness
  seed: null
```

**Use Cases**:
- Testing recovery from depressions
- Training on concave/convex surfaces
- Simulating natural depressions

**Example**:
```python
terrain_config = {
    "type": "bowl",
    "config": {
        "depth": 0.6,
        "radius": 0.4,
        "center_x": 0.5,
        "center_y": 0.5
    }
}
```

---

## 🎨 Specialized Terrains

### Terraced Terrain (`terraced`)

**Description**: Smooth terraces with gradual transitions (improved stepped).

**Configuration**:
```yaml
type: "terraced"
config:
  num_terraces: 5          # Number of terrace levels
  terrace_height: 0.15      # Height difference (0-1)
  transition_width: 0.1    # Transition zone width
  smoothness: 0.7          # Transition smoothness
  direction: "x"           # "x" or "y"
  seed: null
```

**Use Cases**:
- Testing gradual elevation changes
- Training on stepped but smooth terrain
- Simulating terraced landscapes

**Example**:
```python
terrain_config = {
    "type": "terraced",
    "config": {
        "num_terraces": 5,
        "terrace_height": 0.15,
        "direction": "x"
    }
}
```

---

### Spiral Terrain (`spiral`)

**Description**: Spiral/radial patterns with smooth gradients.

**Configuration**:
```yaml
type: "spiral"
config:
  spiral_tightness: 0.1    # How tight the spiral
  height_variation: 0.5    # Height variation (0-1)
  direction: "cw"         # "cw" or "ccw"
  center_x: 0.5           # Spiral center X (0-1)
  center_y: 0.5           # Spiral center Y (0-1)
  seed: null
```

**Use Cases**:
- Testing rotational control
- Training on directional challenges
- Simulating spiral paths

**Example**:
```python
terrain_config = {
    "type": "spiral",
    "config": {
        "spiral_tightness": 0.1,
        "height_variation": 0.5,
        "direction": "cw"
    }
}
```

---

## 🔀 Composite Terrains

### Mixed Terrain (`mixed`)

**Description**: Combines multiple terrain types with configurable blending.

**Configuration**:
```yaml
type: "mixed"
config:
  components:
    - type: "hills"
      weight: 0.4
      config:
        num_hills: 3
        hill_height: 0.5
    - type: "ramp"
      weight: 0.3
      config:
        ramp_angle: 10.0
        ramp_direction: "x"
    - type: "perlin"
      weight: 0.3
      config:
        scale: 30.0
        octaves: 3
  blend_mode: "additive"  # "additive", "max", or "weighted"
  seed: null
```

**Blend Modes**:
- `additive`: Weighted sum of components
- `max`: Maximum value at each point
- `weighted`: Weighted average

**Use Cases**:
- Realistic terrain simulation
- Advanced training scenarios
- Testing on complex environments

**Example**:
```python
terrain_config = {
    "type": "mixed",
    "config": {
        "components": [
            {
                "type": "hills",
                "weight": 0.4,
                "config": {"num_hills": 3, "hill_height": 0.5}
            },
            {
                "type": "ramp",
                "weight": 0.3,
                "config": {"ramp_angle": 10.0, "ramp_direction": "x"}
            }
        ],
        "blend_mode": "additive"
    }
}
```

---

## 💻 Usage Examples

### Basic Usage

```python
from ballbot_gym.core.factories import create_terrain

# Create terrain generator
terrain_config = {
    "type": "ramp",
    "config": {
        "ramp_angle": 15.0,
        "ramp_direction": "x"
    }
}
terrain_gen = create_terrain(terrain_config)

# Generate terrain
n = 129  # Grid size (must be odd)
terrain = terrain_gen(n, seed=42)

# Use in environment
import gymnasium as gym
env = gym.make(
    "ballbot-v0.1",
    terrain_config=terrain_config
)
```

### Using YAML Configuration

```yaml
# config.yaml
terrain:
  type: "ramp"
  config:
    ramp_angle: 15.0
    ramp_direction: "x"
    flat_ratio: 0.3
```

### Randomization

```python
# Random seed each episode
terrain_config = {
    "type": "perlin",
    "config": {
        "scale": 25.0,
        "seed": None  # None = random each episode
    }
}
```

---

## 📈 Difficulty Progression

### Easy (Initial Training)

```python
# Gentle slopes
easy_ramp = {
    "type": "ramp",
    "config": {"ramp_angle": 5.0, "flat_ratio": 0.5}
}

# Few hills
easy_hills = {
    "type": "hills",
    "config": {"num_hills": 3, "hill_height": 0.3}
}
```

### Medium (Mid Training)

```python
# Moderate Perlin
medium_perlin = {
    "type": "perlin",
    "config": {"scale": 25.0, "octaves": 4}
}

# Moderate ramps
medium_ramp = {
    "type": "ramp",
    "config": {"ramp_angle": 12.0, "num_ramps": 2}
}
```

### Hard (Advanced Training)

```python
# Complex Perlin
hard_perlin = {
    "type": "perlin",
    "config": {"scale": 20.0, "octaves": 5, "persistence": 0.25}
}

# Steep ramps
hard_ramp = {
    "type": "ramp",
    "config": {"ramp_angle": 20.0, "num_ramps": 3}
}

# Mixed terrain
hard_mixed = {
    "type": "mixed",
    "config": {
        "components": [
            {"type": "perlin", "weight": 0.4, "config": {...}},
            {"type": "ramp", "weight": 0.3, "config": {...}},
            {"type": "hills", "weight": 0.3, "config": {...}}
        ]
    }
}
```

---

## 📚 See Also

- [Terrain Generation Tutorial](15_terrain_generation.md) - Perlin noise fundamentals
- [Terrain Types Research](../research/terrain_types_for_ballbot.md) - Detailed research
- [Example Configurations](../../examples/terrain_examples.yaml) - Complete examples
- [Visualization Script](../../examples/terrain_visualization.py) - Visualize all terrains

---

*Last Updated: 2025*

