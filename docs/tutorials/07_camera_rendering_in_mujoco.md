# 📷 Camera Rendering in MuJoCo

*A comprehensive guide to rendering RGB, Depth, and Segmentation images in MuJoCo for robotics RL*

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Rendering Architecture](#rendering-architecture)
3. [Defining Cameras in XML](#defining-cameras-in-xml)
4. [RGB Rendering](#rgb-rendering)
5. [Depth Rendering](#depth-rendering)
6. [Segmentation Rendering](#segmentation-rendering)
7. [Real-World Example: Ballbot RGB-D System](#real-world-example-ballbot-rgb-d-system)
8. [Multi-Camera Setups](#multi-camera-setups)
9. [Performance Optimization](#performance-optimization)
10. [Best Practices](#best-practices)
11. [Summary](#summary)

---

## 🎯 Introduction

Visual perception is crucial for many robotics tasks. MuJoCo provides powerful rendering capabilities that simulate realistic cameras, depth sensors, and segmentation masks—all essential for vision-based reinforcement learning.

> "Vision is the primary sense for navigation and manipulation. Simulated cameras must be realistic enough to enable sim-to-real transfer."  
> — *Sergey Levine, UC Berkeley*

**Key Concepts:**
- MuJoCo can render RGB, depth, and segmentation images
- Cameras are defined in the XML model
- Rendering is performed offscreen (no GUI required)
- Multiple cameras can be used simultaneously

---

## 🏗️ Rendering Architecture

### Core Components

```
┌─────────────────────────────────────┐
│      MuJoCo Model (XML)              │
│  - Geometry, materials, cameras       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      MuJoCo Data (State)             │
│  - Positions, orientations, etc.       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Renderer / Scene                │
│  - Renders from camera perspective   │
└──────────────┬──────────────────────┘
               │
       ┌───────┼───────┐
       ▼       ▼       ▼
    RGB    Depth   Segmentation
```

### High-Level vs Low-Level API

**High-Level (Recommended):**
```python
renderer = mujoco.Renderer(model, width=640, height=480)
renderer.update_scene(data, camera="cam_name")
image = renderer.render()
```

**Low-Level (Advanced):**
```python
scene = mujoco.MjvScene(model, maxgeom=10000)
mujoco.mjv_updateScene(model, data, camera, None, ...)
mujoco.mjr_render(viewport, scene, context)
```

---

## 📐 Defining Cameras in XML

### Basic Camera Definition

```xml
<camera name="front_cam" pos="0 0 1" lookat="0 0 0" />
```

**Parameters:**
- `name`: Camera identifier
- `pos`: Camera position [x, y, z]
- `lookat`: Point camera looks at [x, y, z]

### Body-Attached Camera

```xml
<camera name="head_cam" body="head" pos="0 0 0.1" euler="0 0 0" />
```

Camera moves with the body, useful for robot-mounted cameras.

### Fixed Camera

```xml
<camera name="overhead" mode="fixed" pos="0 0 3" />
```

Camera position is fixed in world frame.

### Example: Ballbot Cameras

```xml
<!-- From bbot.xml -->
<camera name="cam_0" pos="0.2 0 0.3" euler="0 -0.5 0" />
<camera name="cam_1" pos="-0.2 0 0.3" euler="0 0.5 0" />
```

Two cameras for stereo vision.

---

## 🎨 RGB Rendering

### Basic RGB Rendering

```python
import mujoco
import numpy as np

# Create renderer
renderer = mujoco.Renderer(model, width=640, height=480)

# Update scene from camera perspective
renderer.update_scene(data, camera="front_cam")

# Render RGB image
rgb = renderer.render()  # Shape: (H, W, 3), values [0, 255]

# Normalize to [0, 1]
rgb_normalized = rgb.astype(np.float32) / 255.0
```

### Features Included

- Lighting (directional, point, spot)
- Textures and materials
- Shadows
- Reflections
- Anti-aliasing

### Use Cases

- Vision-based RL
- Imitation learning
- Synthetic dataset generation
- Visualization

---

## 🌊 Depth Rendering

### Enabling Depth Rendering

```python
# Create renderer with depth enabled
renderer = mujoco.Renderer(model, width=640, height=480)
renderer.enable_depth_rendering()

# Render depth
renderer.update_scene(data, camera="front_cam")
depth = renderer.render()  # Shape: (H, W), values [0, 1]
```

### Depth Values

- **0.0**: Closest object (at near clipping plane)
- **1.0**: Farthest object (at far clipping plane)
- Values are normalized to [0, 1]

### Depth Clipping

```python
# Clip extreme values (sky, background)
depth[depth >= 1.0] = 1.0

# Add channel dimension
depth = depth[..., None]  # Shape: (H, W, 1)
```

### Use Cases

- Navigation
- 3D perception
- RGB-D policies
- Grasping

---

## 🎯 Segmentation Rendering

### Types of Segmentation

1. **Geom Segmentation**: Pixel → `geom_id`
2. **Body Segmentation**: Pixel → `body_id`
3. **Object Segmentation**: Pixel → `object_id`

### Implementation

```python
def get_segmentation_image(model, data, seg_mode="geom",
                           width=640, height=480):
    """
    Render segmentation mask.
    
    Args:
        seg_mode: "geom", "body", or "object"
    
    Returns:
        Segmentation mask (H, W) with integer IDs
    """
    # Setup scene
    scene = mujoco.MjvScene(model, maxgeom=10000)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    viewport = mujoco.MjrRect(0, 0, width, height)
    
    # Update scene
    mujoco.mjv_updateScene(model, data, cam, None,
                          mujoco.mjtCatBit.mjCAT_ALL, scene)
    
    # Set segmentation type
    if seg_mode == "geom":
        ctx.segmentation = mujoco.mjtSeg.mjSEGCATEGORY_GEOM
    elif seg_mode == "body":
        ctx.segmentation = mujoco.mjtSeg.mjSEGCATEGORY_BODY
    elif seg_mode == "object":
        ctx.segmentation = mujoco.mjtSeg.mjSEGCATEGORY_OBJECT
    
    # Render
    mujoco.mjr_render(viewport, scene, ctx)
    
    # Read segmentation buffer
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    seg = np.zeros((height, width), dtype=np.int32)
    mujoco.mjr_readPixels(rgb, seg, viewport, ctx)
    
    return seg.T  # Transpose to (H, W)
```

### Use Cases

- Semantic segmentation datasets
- Object-aware policies
- Attention mechanisms
- Reward shaping

---

## 🤖 Real-World Example: Ballbot RGB-D System

### RGBDInputs Class

The Ballbot uses a custom `RGBDInputs` class to manage rendering:

```python
# From bbot_env.py
class RGBDInputs:
    def __init__(self, mjc_model, height, width, cams, disable_rgb):
        self.width = width
        self.height = height
        
        # RGB renderer (optional)
        self._renderer_rgb = mujoco.Renderer(
            mjc_model, width=width, height=height
        ) if not disable_rgb else None
        
        # Depth renderer (always enabled)
        self._renderer_d = mujoco.Renderer(
            mjc_model, width=width, height=height
        )
        self._renderer_d.enable_depth_rendering()
        
        self.cams = cams
```

### Rendering Method

```python
def __call__(self, data, cam_name):
    # Render RGB if enabled
    if self._renderer_rgb is not None:
        self._renderer_rgb.update_scene(data, camera=cam_name)
        rgb = self._renderer_rgb.render().astype(np.float32) / 255.0
    
    # Render depth
    self._renderer_d.update_scene(data, camera=cam_name)
    depth = np.expand_dims(self._renderer_d.render(), axis=-1)
    
    # Clip depth values
    depth[depth >= 1.0] = 1.0
    
    # Combine
    if self._renderer_rgb is not None:
        arr = np.concatenate([rgb, depth], -1)  # (H, W, 4)
    else:
        arr = depth  # (H, W, 1)
    
    return arr
```

### Usage in Environment

```python
# In _get_obs()
if delta_time >= 1.0 / self.camera_frame_rate:
    # Update cameras
    rgbd_0 = self.rgbd_inputs(self.data, "cam_0")
    rgbd_1 = self.rgbd_inputs(self.data, "cam_1")
    
    # Store for reuse
    self.prev_im_pair.im_0 = rgbd_0.copy()
    self.prev_im_pair.im_1 = rgbd_1.copy()
    self.prev_im_pair.ts = self.data.time
else:
    # Reuse previous frames
    rgbd_0 = self.prev_im_pair.im_0.copy()
    rgbd_1 = self.prev_im_pair.im_1.copy()
```

### Camera Update Frequency

Cameras update at lower frequency than physics:

```python
self.camera_frame_rate = 90  # Hz (vs 500 Hz physics)

# Only update if enough time has passed
if delta_time >= 1.0 / self.camera_frame_rate:
    # Render new images
```

**Why?**
- Rendering is expensive
- Matches realistic camera frame rates
- Reduces computational cost

---

## 📹 Multi-Camera Setups

### Stereo Vision

```python
# Two cameras for depth estimation
rgbd_left = renderer.render(camera="left_cam")
rgbd_right = renderer.render(camera="right_cam")
```

### Multiple Viewpoints

```python
# Different perspectives
front_view = renderer.render(camera="front")
side_view = renderer.render(camera="side")
overhead_view = renderer.render(camera="overhead")
```

### Observation Construction

```python
obs = {
    "rgbd_0": rgbd_0.transpose(2, 0, 1),  # Channels-first
    "rgbd_1": rgbd_1.transpose(2, 0, 1),
    # ... other observations
}
```

---

## ⚡ Performance Optimization

### 1. Lower Resolution

```python
# Smaller images = faster rendering
renderer = mujoco.Renderer(model, width=64, height=64)
```

### 2. Depth-Only Mode

```python
# Skip RGB if not needed
disable_rgb = True  # Only render depth
```

### 3. Lower Frame Rate

```python
# Update cameras less frequently
camera_frame_rate = 30  # Instead of 90 Hz
```

### 4. Reuse Renderers

```python
# Create once, reuse many times
renderer = mujoco.Renderer(model, width, height)
# ... use in loop ...
```

---

## ✅ Best Practices

### 1. Normalize Images

```python
# RGB: [0, 255] → [0, 1]
rgb = rgb.astype(np.float32) / 255.0

# Depth: Already [0, 1], but clip extremes
depth = np.clip(depth, 0.0, 1.0)
```

### 2. Use Channels-First Format

```python
# For PyTorch/Stable-Baselines3
image = image.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
```

### 3. Handle Camera Updates

```python
# Update only when needed
if time_since_last_update >= desired_interval:
    render_new_image()
else:
    reuse_previous_image()
```

### 4. Clip Depth Values

```python
# Prevent infinite depth (sky, background)
depth[depth >= 1.0] = 1.0
```

---

## 🎨 Advanced Rendering Techniques ⭐⭐

### Neural Rendering

**Concept:** Use neural networks to render or enhance images instead of traditional graphics.

**Why Neural Rendering?**
- Can learn complex lighting/material effects
- Faster than ray tracing
- Can fill in missing information
- Better sim-to-real transfer

**Neural Radiance Fields (NeRF) for Simulation:**
```python
class NeRFRenderer:
    """
    Neural Radiance Field renderer for enhanced simulation images.
    """
    def __init__(self):
        self.nerf_model = NeRFModel()  # Pretrained NeRF
    
    def render(self, camera_pose, camera_intrinsics):
        """
        Render image using NeRF instead of traditional graphics.
        """
        # Query NeRF for pixel colors
        rays = generate_rays(camera_pose, camera_intrinsics)
        colors = self.nerf_model(rays)
        
        return colors.reshape(image_height, image_width, 3)
```

**Benefits:**
- More realistic rendering
- Can learn from real images
- Better sim-to-real alignment

### Domain Randomization for Rendering

**Concept:** Randomize rendering parameters to improve sim-to-real transfer.

**Implementation:**
```python
class DomainRandomizedRenderer:
    """
    Renderer with domain randomization.
    """
    def __init__(self, model, camera_name):
        self.renderer = mujoco.Renderer(model)
        self.camera_name = camera_name
    
    def render(self, data, randomize=True):
        """
        Render with randomized parameters.
        """
        if randomize:
            # Randomize lighting
            self.renderer.scene.option.ambient = np.random.uniform(0.1, 0.5)
            
            # Randomize camera noise
            noise = np.random.normal(0, 0.01, (height, width, 3))
            
            # Randomize texture
            texture_scale = np.random.uniform(0.8, 1.2)
        else:
            noise = 0
            texture_scale = 1.0
        
        # Render
        rgb = self.renderer.render()
        
        # Apply noise
        rgb = rgb + noise
        rgb = np.clip(rgb, 0, 255)
        
        return rgb
```

**Benefits:**
- Better sim-to-real transfer
- More robust policies
- Reduces overfitting to simulation

### Learned Image Compression

**Concept:** Use learned encoders to compress images for efficient RL training.

**Implementation:**
```python
class LearnedImageEncoder:
    """
    Learned encoder for compressing camera images.
    """
    def __init__(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128)
        )
    
    def encode(self, image):
        """
        Compress image to compact representation.
        """
        # image: (H, W, 3) → (3, H, W)
        image = image.transpose(2, 0, 1)
        features = self.encoder(image)
        return features  # (128,)
```

**Benefits:**
- Reduces observation dimensionality
- Faster training
- Can be pretrained

### Multi-View Rendering

**Concept:** Render from multiple viewpoints and fuse information.

**Implementation:**
```python
class MultiViewRenderer:
    """
    Render from multiple camera viewpoints.
    """
    def __init__(self, model, camera_names):
        self.renderers = {
            name: mujoco.Renderer(model, camera_name=name)
            for name in camera_names
        }
    
    def render_all(self, data):
        """
        Render from all cameras.
        """
        images = {}
        for name, renderer in self.renderers.items():
            images[name] = renderer.render()
        return images
    
    def fuse_views(self, images):
        """
        Fuse multiple views into single representation.
        """
        # Option 1: Concatenate
        fused = np.concatenate(list(images.values()), axis=1)
        
        # Option 2: Attention-based fusion
        # (see Multi-Modal Fusion tutorial)
        
        return fused
```

---

## 📊 Summary

### Key Takeaways

1. **MuJoCo provides powerful rendering** - RGB, depth, segmentation

2. **Cameras defined in XML** - Flexible positioning and attachment

3. **High-level API recommended** - `mujoco.Renderer` is easiest

4. **Performance matters** - Lower resolution, depth-only, lower frame rate

5. **Real-world example** - Ballbot shows complete RGB-D system

6. **Consider neural rendering** - For better realism and sim-to-real transfer ⭐

7. **Use domain randomization** - For robust sim-to-real policies ⭐

### Rendering Checklist

- [ ] Cameras defined in XML
- [ ] Renderers created and configured
- [ ] Images normalized appropriately
- [ ] Depth values clipped
- [ ] Channels-first format for PyTorch
- [ ] Camera update frequency optimized
- [ ] Considered domain randomization for sim-to-real ⭐
- [ ] Considered learned compression for efficiency ⭐

---

## 📚 Further Reading

### Papers

**Classic Rendering:**
- **Todorov et al. (2012)** - "MuJoCo: A physics engine for model-based control" - MuJoCo rendering

**Modern Rendering:**
- **Mildenhall et al. (2020)** - "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" - Neural rendering
- **Tobin et al. (2017)** - "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" - Domain randomization
- **Peng et al. (2018)** - "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization" - Dynamics randomization
- **Schwarzer et al. (2021)** - "Data-Efficient Reinforcement Learning with Self-Predictive Representations" - Learned representations

### Documentation
- [MuJoCo Rendering](https://mujoco.readthedocs.io/en/latest/rendering.html)
- [MuJoCo Camera API](https://mujoco.readthedocs.io/en/latest/APIreference.html#camera)

### Papers
- **Tassa et al. (2018)** - "Control Suite" - Vision-based control

### Code Examples
- Ballbot RGB-D system: `ballbot_gym/bbot_env.py` - `RGBDInputs` class

---

## 🎓 Exercises

1. **Add New Camera**: Add a third camera to the Ballbot environment.

2. **Segmentation**: Implement segmentation rendering for the Ballbot.

3. **Performance Test**: Measure rendering time at different resolutions.

---

*Next Tutorial: [Terrain Generation for Locomotion](08_terrain_generation_for_locomotion.md)*

---

**Happy Learning! 🚀**

