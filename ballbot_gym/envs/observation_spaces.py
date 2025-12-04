"""Observation space definitions for ballbot environment."""
import gymnasium as gym
import numpy as np

# Default data type for all arrays (float32 for efficiency and compatibility)
_default_dtype = np.float32


def create_observation_space(
    im_shape: dict,
    num_channels: int,
    disable_cameras: bool
) -> gym.spaces.Dict:
    """
    Create observation space with or without cameras.
    
    Args:
        im_shape (dict): Image dimensions with keys "h" (height) and "w" (width).
        num_channels (int): Number of image channels (1 for depth-only, 4 for RGB-D).
        disable_cameras (bool): If True, creates proprioceptive-only observation space.
    
    Returns:
        gym.spaces.Dict: Observation space dictionary with appropriate keys.
    """
    # Define observation space with cameras enabled
    # Note: All values are normalized and clipped to reasonable ranges
    obs_space_with_cameras = gym.spaces.Dict({
        "orientation": gym.spaces.Box(
            low=-np.pi, high=np.pi, shape=(3, ), dtype=_default_dtype
        ),  # Rotation vector representation (3D)
        "angular_vel": gym.spaces.Box(
            low=-2, high=2, shape=(3, ), dtype=_default_dtype
        ),  # Angular velocity in rad/s (clipped)
        "vel": gym.spaces.Box(
            low=-2, high=2, shape=(3, ), dtype=_default_dtype
        ),  # Linear velocity in m/s (clipped)
        "motor_state": gym.spaces.Box(
            -2.0, 2.0, shape=(3, ), dtype=_default_dtype
        ),  # Normalized wheel velocities
        "actions": gym.spaces.Box(
            -1.0, 1.0, shape=(3, ), dtype=_default_dtype
        ),  # Previous action (for temporal context)
        "rgbd_0": gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(num_channels, im_shape["h"], im_shape["w"]),
            dtype=_default_dtype
        ),  # RGB-D image from camera 0 (channels-first format)
        "rgbd_1": gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(num_channels, im_shape["h"], im_shape["w"]),
            dtype=_default_dtype
        ),  # RGB-D image from camera 1 (channels-first format)
        "relative_image_timestamp": gym.spaces.Box(
            low=0.0, high=0.1, shape=(1, ), dtype=_default_dtype
        ),  # Time lag between proprioceptive and visual data (cameras update slower)
    })
    
    # Define observation space without cameras (proprioceptive only)
    obs_space_no_cameras = gym.spaces.Dict({
        "orientation": gym.spaces.Box(
            low=-np.pi, high=np.pi, shape=(3, ), dtype=_default_dtype
        ),
        "angular_vel": gym.spaces.Box(
            low=-2, high=2, shape=(3, ), dtype=_default_dtype
        ),
        "vel": gym.spaces.Box(
            low=-2, high=2, shape=(3, ), dtype=_default_dtype
        ),
        "motor_state": gym.spaces.Box(
            -2.0, 2.0, shape=(3, ), dtype=_default_dtype
        ),
        "actions": gym.spaces.Box(
            -1.0, 1.0, shape=(3, ), dtype=_default_dtype
        ),
        "relative_image_timestamp": gym.spaces.Box(
            low=0.0, high=0.1, shape=(1, ), dtype=_default_dtype
        ),
    })
    
    # Select appropriate observation space based on camera configuration
    return obs_space_with_cameras if not disable_cameras else obs_space_no_cameras

