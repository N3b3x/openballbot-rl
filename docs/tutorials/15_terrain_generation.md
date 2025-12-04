# 🌍 Terrain Generation for Robotics RL

*A comprehensive guide to procedural terrain generation using Perlin noise*

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [What is Procedural Terrain Generation?](#what-is-procedural-terrain-generation)
3. [Perlin Noise Fundamentals](#perlin-noise-fundamentals)
4. [Terrain Generation Parameters](#terrain-generation-parameters)
5. [Implementation: Ballbot Terrain](#implementation-ballbot-terrain)
6. [Parameter Tuning Guide](#parameter-tuning-guide)
7. [Domain Randomization with Terrain](#domain-randomization-with-terrain)
8. [Advanced Techniques](#advanced-techniques)
9. [Best Practices](#best-practices)
10. [Summary](#summary)

---

## 🎯 Introduction

Terrain generation is crucial for training robust robotics policies. By training on diverse, procedurally generated terrain, policies learn to generalize to unseen environments.

> "The terrain is the teacher. Diverse terrain teaches robust policies."  
> — *Common wisdom in robotics RL*

**Key Concepts:**
- **Procedural generation**: Create infinite terrain variations
- **Perlin noise**: Smooth, natural-looking terrain
- **Domain randomization**: Vary terrain parameters for robustness
- **Difficulty control**: Adjust terrain complexity

**Key Questions This Tutorial Answers:**
- How does Perlin noise work?
- What parameters control terrain appearance?
- How do we tune terrain for RL training?
- How do we use terrain for domain randomization?

---

## 🌍 What is Procedural Terrain Generation?

### The Problem

**Fixed Terrain:**
- Limited variation
- Policy overfits to specific terrain
- Poor generalization
- Requires manual design

**Procedural Terrain:**
- Infinite variations
- Policy learns robust behaviors
- Better generalization
- Automatic generation

### Why Perlin Noise?

**Perlin noise** is ideal for terrain because:
- **Smooth**: Natural-looking surfaces
- **Controllable**: Easy to adjust parameters
- **Reproducible**: Same seed = same terrain
- **Efficient**: Fast to generate

**Alternatives:**
- Random noise: Too jagged
- Fractal noise: More complex
- Real heightmaps: Limited, requires data

---

## 🎨 Perlin Noise Fundamentals

### What is Perlin Noise?

**Perlin noise** is a gradient noise function that produces smooth, natural-looking random values.

**Key Properties:**
- Smooth transitions
- Multi-scale detail
- Reproducible (with seed)
- Continuous derivatives

### How It Works

**Conceptual Process:**
1. **Grid of gradients**: Place random gradients on grid
2. **Interpolation**: Smoothly interpolate between gradients
3. **Multiple octaves**: Combine at different scales
4. **Final value**: Sum of all octaves

**Mathematical Form:**
\[
\text{Noise}(x, y) = \sum_{i=0}^{n-1} \text{persistence}^i \cdot \text{Noise}_i(\text{lacunarity}^i \cdot (x, y))
\]

Where:
- \(n\) = number of octaves
- **persistence** = amplitude scaling
- **lacunarity** = frequency scaling

### Visual Intuition

```
Single Octave:          Multiple Octaves:
┌─────────┐            ┌─────────┐
│  /\/\/  │            │  /\/\/  │  ← Base (low frequency)
│ /      \│    +       │/\/\/\/\/│  ← Detail (high frequency)
│/        \│           │         │
└─────────┘            └─────────┘
                        = Smooth + Detailed
```

---

## ⚙️ Terrain Generation Parameters

### Core Parameters

**1. Scale (Spatial Frequency)**

```python
scale = 25.0  # Default
```

**What it controls:**
- Size of terrain features
- **Higher scale** = larger features (smoother)
- **Lower scale** = smaller features (rougher)

**Effect:**
```python
scale = 50.0  # Very smooth, large hills
scale = 25.0  # Moderate (default)
scale = 10.0  # Rough, small bumps
```

**2. Octaves (Detail Levels)**

```python
octaves = 4  # Default
```

**What it controls:**
- Number of detail levels
- **More octaves** = more detail
- **Fewer octaves** = smoother

**Effect:**
```python
octaves = 1  # Very smooth, no detail
octaves = 4  # Moderate detail (default)
octaves = 8  # Very detailed, complex
```

**3. Persistence (Amplitude Decay)**

```python
persistence = 0.2  # Default
```

**What it controls:**
- How much each octave contributes
- **Higher persistence** = more variation
- **Lower persistence** = smoother

**Effect:**
```python
persistence = 0.1  # Very smooth
persistence = 0.2  # Moderate (default)
persistence = 0.5  # Very rough
```

**4. Lacunarity (Frequency Increase)**

```python
lacunarity = 2.0  # Default
```

**What it controls:**
- Frequency scaling between octaves
- **Higher lacunarity** = more variation
- **Lower lacunarity** = smoother

**Effect:**
```python
lacunarity = 1.5  # Smoother transitions
lacunarity = 2.0  # Standard (default)
lacunarity = 3.0  # More variation
```

**5. Seed (Randomness)**

```python
seed = 0  # Default
```

**What it controls:**
- Terrain pattern
- **Same seed** = same terrain
- **Different seed** = different terrain

---

## 💻 Implementation: Ballbot Terrain

### Basic Terrain Generation

**Function Signature:**
```python
def generate_perlin_terrain(
    n: int,              # Grid size (should be odd)
    scale: float = 25.0, # Spatial frequency
    octaves: int = 4,     # Detail levels
    persistence: float = 0.2,  # Amplitude decay
    lacunarity: float = 2.0,    # Frequency scaling
    seed: int = 0        # Random seed
) -> np.ndarray:
```

**Implementation:**
```python
import numpy as np
from noise import snoise2  # Simplex noise (improved Perlin)

def generate_perlin_terrain(n, scale=25.0, octaves=4, 
                           persistence=0.2, lacunarity=2.0, seed=0):
    """
    Generate Perlin noise terrain.
    """
    assert n % 2 == 1, "n should be odd for symmetry"
    
    terrain = np.zeros((n, n))
    
    # Generate noise at each point
    for i in range(n):
        for j in range(n):
            x = i / scale
            y = j / scale
            terrain[i][j] = snoise2(
                x, y,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=seed
            )
    
    # Normalize to [0, 1]
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 1e-8)
    
    # Return as flat array (MuJoCo format)
    return terrain.flatten()
```

### Using in MuJoCo

**Update Heightfield:**
```python
# Generate terrain
terrain_data = generate_perlin_terrain(
    n=129,           # 129x129 grid
    scale=25.0,
    octaves=4,
    seed=42
)

# Update MuJoCo heightfield
hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
model.hfield_data[hfield_id] = terrain_data

# Notify viewer
viewer.update_hfield(hfield_id)
```

---

## 🎛️ Parameter Tuning Guide

### Tuning for Difficulty

**Easy Terrain (Training Start):**
```python
easy_terrain = generate_perlin_terrain(
    n=129,
    scale=30.0,      # Large features (smooth)
    octaves=3,       # Fewer octaves (less detail)
    persistence=0.15, # Low persistence (smoother)
    lacunarity=2.0,
    seed=random_seed
)
```

**Medium Terrain (Mid Training):**
```python
medium_terrain = generate_perlin_terrain(
    n=129,
    scale=25.0,      # Moderate features
    octaves=4,       # Standard detail
    persistence=0.2, # Standard persistence
    lacunarity=2.0,
    seed=random_seed
)
```

**Hard Terrain (Advanced Training):**
```python
hard_terrain = generate_perlin_terrain(
    n=129,
    scale=20.0,      # Small features (rough)
    octaves=5,       # More octaves (more detail)
    persistence=0.25, # Higher persistence (rougher)
    lacunarity=2.5,  # Higher lacunarity
    seed=random_seed
)
```

### Parameter Effects Summary

| Parameter | Increase Effect | Decrease Effect |
|-----------|----------------|-----------------|
| **scale** | Smoother, larger features | Rougher, smaller features |
| **octaves** | More detail, complexity | Smoother, less detail |
| **persistence** | Rougher, more variation | Smoother, less variation |
| **lacunarity** | More variation | Smoother transitions |

### Visualizing Parameter Effects

**Create Comparison:**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Vary scale
terrain_scale_high = generate_perlin_terrain(129, scale=50.0, seed=42)
axes[0, 0].imshow(terrain_scale_high.reshape(129, 129))
axes[0, 0].set_title('Scale=50.0 (Smooth)')

terrain_scale_low = generate_perlin_terrain(129, scale=10.0, seed=42)
axes[0, 1].imshow(terrain_scale_low.reshape(129, 129))
axes[0, 1].set_title('Scale=10.0 (Rough)')

# Vary persistence
terrain_pers_low = generate_perlin_terrain(129, persistence=0.1, seed=42)
axes[1, 0].imshow(terrain_pers_low.reshape(129, 129))
axes[1, 0].set_title('Persistence=0.1 (Smooth)')

terrain_pers_high = generate_perlin_terrain(129, persistence=0.5, seed=42)
axes[1, 1].imshow(terrain_pers_high.reshape(129, 129))
axes[1, 1].set_title('Persistence=0.5 (Rough)')

plt.tight_layout()
plt.show()
```

---

## 🎲 Domain Randomization with Terrain

### Randomizing Terrain Parameters

**During Training:**
```python
class RandomizedTerrainGenerator:
    """
    Generate terrain with randomized parameters.
    """
    def __init__(self):
        # Parameter ranges
        self.scale_range = (20.0, 30.0)
        self.octaves_range = (3, 6)
        self.persistence_range = (0.15, 0.25)
        self.lacunarity_range = (1.8, 2.5)
    
    def generate_random_terrain(self, n, seed=None):
        """
        Generate terrain with random parameters.
        """
        if seed is None:
            seed = np.random.randint(0, 10000)
        
        # Sample random parameters
        scale = np.random.uniform(*self.scale_range)
        octaves = np.random.randint(*self.octaves_range)
        persistence = np.random.uniform(*self.persistence_range)
        lacunarity = np.random.uniform(*self.lacunarity_range)
        
        # Generate terrain
        terrain = generate_perlin_terrain(
            n=n,
            scale=scale,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            seed=seed
        )
        
        return terrain
```

### Progressive Randomization

**Increase Randomization Over Time:**
```python
class ProgressiveTerrainRandomization:
    """
    Gradually increase terrain difficulty.
    """
    def __init__(self, total_steps=10_000_000):
        self.total_steps = total_steps
        self.current_step = 0
    
    def get_parameter_ranges(self):
        """
        Get parameter ranges based on training progress.
        """
        progress = self.current_step / self.total_steps
        
        # Start easy, get harder
        scale_min = 30.0 - 10.0 * progress  # 30 → 20
        scale_max = 30.0 - 5.0 * progress   # 30 → 25
        
        persistence_min = 0.15 + 0.05 * progress  # 0.15 → 0.20
        persistence_max = 0.20 + 0.10 * progress   # 0.20 → 0.30
        
        return {
            'scale_range': (scale_min, scale_max),
            'persistence_range': (persistence_min, persistence_max)
        }
    
    def generate(self, n, seed=None):
        """
        Generate terrain with progressive difficulty.
        """
        ranges = self.get_parameter_ranges()
        
        scale = np.random.uniform(*ranges['scale_range'])
        persistence = np.random.uniform(*ranges['persistence_range'])
        
        return generate_perlin_terrain(
            n=n,
            scale=scale,
            persistence=persistence,
            seed=seed or np.random.randint(0, 10000)
        )
```

---

## 🔬 Advanced Techniques

### Multi-Scale Terrain

**Combine Multiple Scales:**
```python
def generate_multiscale_terrain(n, scales=[50.0, 25.0, 10.0], weights=[0.5, 0.3, 0.2]):
    """
    Generate terrain combining multiple scales.
    """
    terrain = np.zeros((n, n))
    
    for scale, weight in zip(scales, weights):
        terrain += weight * generate_perlin_terrain(n, scale=scale)
    
    # Normalize
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min() + 1e-8)
    
    return terrain.flatten()
```

### Terrain with Obstacles

**Add Obstacles:**
```python
def add_obstacles(terrain, obstacle_positions, obstacle_sizes):
    """
    Add obstacles to terrain.
    """
    terrain_2d = terrain.reshape(int(np.sqrt(len(terrain))), -1)
    
    for pos, size in zip(obstacle_positions, obstacle_sizes):
        # Add height bump at obstacle position
        i, j = int(pos[0]), int(pos[1])
        terrain_2d[i-size:i+size, j-size:j+size] += 0.3
    
    # Renormalize
    terrain_2d = np.clip(terrain_2d, 0, 1)
    
    return terrain_2d.flatten()
```

### Real-World Terrain Modeling

**Load Real Heightmaps:**
```python
def load_real_terrain(heightmap_path):
    """
    Load terrain from real heightmap.
    """
    import cv2
    
    # Load heightmap image
    heightmap = cv2.imread(heightmap_path, cv2.IMREAD_GRAYSCALE)
    
    # Normalize to [0, 1]
    heightmap = heightmap.astype(np.float32) / 255.0
    
    # Resize to match MuJoCo resolution
    heightmap = cv2.resize(heightmap, (129, 129))
    
    return heightmap.flatten()
```

---

## ✅ Best Practices

### 1. Start with Easy Terrain

- Begin training on flat or smooth terrain
- Gradually increase difficulty
- Use curriculum learning

### 2. Randomize Parameters

- Don't use fixed parameters
- Randomize scale, octaves, persistence
- Increase diversity

### 3. Match Real Conditions

- If deploying to real robot, match real terrain
- Use similar difficulty levels
- Consider real-world constraints

### 4. Validate Terrain

- Check terrain is valid (no NaN, in range)
- Verify robot can spawn correctly
- Test terrain generation speed

### 5. Document Parameters

- Record terrain parameters used
- Log terrain seeds for reproducibility
- Track terrain difficulty metrics

---

## 📊 Summary

### Key Takeaways

1. **Perlin noise creates natural terrain** - Smooth, controllable, efficient
2. **Parameters control appearance** - Scale, octaves, persistence, lacunarity
3. **Randomization improves robustness** - Vary parameters during training
4. **Difficulty can be controlled** - Adjust parameters for curriculum learning
5. **Terrain affects learning** - Choose parameters carefully

### Parameter Quick Reference

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| **scale** | 10-50 | Higher = smoother |
| **octaves** | 1-8 | More = more detail |
| **persistence** | 0.1-0.5 | Higher = rougher |
| **lacunarity** | 1.5-3.0 | Higher = more variation |

### Terrain Generation Checklist

- [ ] Parameters tuned for difficulty
- [ ] Randomization implemented
- [ ] Terrain validated (no errors)
- [ ] Generation is fast enough
- [ ] Seeds logged for reproducibility

---

## 📚 Further Reading

### Papers

- **Perlin (1985)** - "An Image Synthesizer" - Original Perlin noise
- **Perlin (2002)** - "Improving Noise" - Improved Perlin noise
- **Tobin et al. (2017)** - "Domain Randomization" - Terrain randomization

### Tutorials

- [Curriculum Learning](14_curriculum_learning.md) - Using terrain for curriculum
- [Sim-to-Real Transfer](12_sim_to_real_transfer.md) - Terrain for sim-to-real
- [Complete Training Guide](13_complete_training_guide.md) - Training workflow

### Code References

- `ballbot_gym/terrain.py` - Terrain generation implementation
- `ballbot_gym/bbot_env.py` - Terrain usage in environment

---

*Last Updated: 2025*

**Happy Terrain Generating! 🌍**

