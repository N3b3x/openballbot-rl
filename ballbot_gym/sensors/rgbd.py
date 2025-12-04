"""RGB-D camera rendering for MuJoCo environments."""
import numpy as np
import mujoco
from typing import List

# Default data type for all arrays (float32 for efficiency and compatibility)
_default_dtype = np.float32


class RGBDInputs:
    """
    Manages RGB-D (color and depth) camera rendering for Mujoco environments.

    Provides quick rendering of RGB and/or Depth images from multiple virtual cameras
    (for example, for training a robot with pixel-based RL).

    Args:
        mjc_model: The Mujoco model, used for scene rendering.
        height (int): The image output height in pixels.
        width (int): The image output width in pixels.
        cams (List[str]): List of camera names available in the Mujoco environment.
        disable_rgb (bool): If True, disables RGB channel (outputs only the depth channel); otherwise, both RGB and depth.

    Attributes:
        width (int): Image width.
        height (int): Image height.
        cams (List[str]): List of available camera names.
        _renderer_rgb: Mujoco Renderer for RGB images (None if disabled).
        _renderer_d: Mujoco Renderer for depth images (always enabled).
    """
    def __init__(self, mjc_model, height, width, cams: List[str], disable_rgb: bool):
        self.width = width
        self.height = height

        # Initialize RGB renderer unless disabled
        self._renderer_rgb = mujoco.Renderer(
            mjc_model, width=width, height=height) if not disable_rgb else None
        # Always initialize the depth renderer
        self._renderer_d = mujoco.Renderer(mjc_model,
                                           width=width,
                                           height=height)
        self._renderer_d.enable_depth_rendering()

        self.cams = cams

    def __call__(self, data, cam_name: str):
        """
        Render either RGB-D or Depth-only image arrays from specified camera.

        Args:
            data: Mujoco simulation data (mjData).
            cam_name (str): Name of the camera to use for rendering.

        Returns:
            np.ndarray: Combined (H, W, 4) RGB-D array with values in [0,1], or (H, W, 1) Depth array if RGB is disabled.

        Raises:
            AssertionError: If the provided camera name is not in the allowed list.
        """
        assert cam_name in self.cams, f"wrong cam name (got {cam_name}, available ones are {self.cams})"

        # Render RGB image if renderer is enabled
        if self._renderer_rgb is not None:
            self._renderer_rgb.update_scene(data, camera=cam_name)
            rgb = self._renderer_rgb.render().astype(_default_dtype) / 255  # Normalize to [0, 1]

        # Render Depth image
        self._renderer_d.update_scene(data, camera=cam_name)
        depth = np.expand_dims(self._renderer_d.render(), axis=-1)

        # Clip extreme depth values (sky, background), keeping max at 1.0
        # This is important: robot orientations near failures may let camera rays see skybox,
        # resulting in artificially huge depths. Downstream perception may fail with wild depths.
        depth[depth >= 1.0] = 1.0

        # Combine channels or report only depth
        if self._renderer_rgb is not None:
            arr = np.concatenate([rgb, depth], -1)  # Shape: (H, W, 4)
        else:
            arr = depth                             # Shape: (H, W, 1)

        return arr

    def reset(self, mjc_model):
        """
        Reset renderer state, for when the environment changes (e.g. model reloaded).

        Args:
            mjc_model: New or reset Mujoco model.
        """
        self._renderer_d.close()
        if self._renderer_rgb is not None:
            self._renderer_rgb.close()
            self._renderer_rgb = mujoco.Renderer(mjc_model,
                                                 width=self.width,
                                                 height=self.height)
        self._renderer_d = mujoco.Renderer(mjc_model,
                                           width=self.width,
                                           height=self.height)
        self._renderer_d.enable_depth_rendering()

    def close(self):
        """
        Properly close and cleanup all renderers.
        """
        if self._renderer_rgb is not None:
            self._renderer_rgb.close()
            self._renderer_rgb = None
        self._renderer_d.close()
        self._renderer_d = None

