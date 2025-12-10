"""
Ballbot Reinforcement Learning Environment

This module implements a Gymnasium-compatible environment for training a ballbot robot
using reinforcement learning. The ballbot is a dynamically balanced robot that moves
on a ball, requiring careful control to maintain balance while navigating terrain.

Key Features:
    - MuJoCo physics simulation for realistic robot dynamics
    - RGB-D camera support for visual observations
    - Procedural terrain generation (Perlin noise or flat)
    - Directional reward system for goal-directed navigation
    - Comprehensive logging and visualization support

The environment follows the standard Gymnasium API:
    - reset(): Initialize or reset the environment
    - step(action): Execute one simulation step
    - close(): Clean up resources

Example Usage:
    ```python
    import gymnasium as gym
    env = gym.make("ballbot-v0.1")
    obs, info = env.reset()
    
    for _ in range(1000):
        action = agent.select_action(obs)  # Shape: (3,)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    ```
"""

import gymnasium as gym
import numpy as np
import quaternion
import os
import string
import shutil
from typing import Optional

from termcolor import colored

import mujoco
import mujoco.viewer

from ballbot_gym.core.factories import create_reward, create_terrain
from ballbot_gym.core.config import get_component_config
from ballbot_gym.sensors import RGBDInputs, StampedImPair
from ballbot_gym.utils.helpers import warnings_stdout_off, sample_direction_uniform
from ballbot_gym.utils.logging import save_episode_logs
from ballbot_gym.envs.observation_spaces import create_observation_space

# Default data type for all arrays (float32 for efficiency and compatibility)
_default_dtype = np.float32


class BBotSimulation(gym.Env):
    """
    Ballbot Simulation Environment for Reinforcement Learning.
    
    Metadata:
        render_modes: List of supported render modes. Supports "rgb_array" for video recording.
    """
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 30,  # FPS for video recording
    }
    """
    Ballbot Simulation Environment for Reinforcement Learning
    
    This class implements a Gymnasium environment for a ballbot robot. The ballbot
    is a dynamically balanced mobile robot that moves by controlling three omniwheels
    that contact a ball. The robot must maintain balance while navigating terrain.
    
    The environment provides:
        - Proprioceptive observations: orientation, velocities, motor states
        - Visual observations: RGB-D images from two cameras (optional)
        - Directional rewards: encourages movement toward a target direction
        - Terrain generation: Perlin noise or flat terrain
    
    Action Space:
        Box(-1.0, 1.0, shape=(3,), dtype=float32)
        - Normalized omniwheel commands for three wheels
        - Values are scaled to [-10, 10] rad/s internally
    
    Observation Space (with cameras):
        Dict({
            "orientation": Box(-π, π, shape=(3,)) - Rotation vector (Euler-like)
            "angular_vel": Box(-2, 2, shape=(3,)) - Angular velocity (rad/s)
            "vel": Box(-2, 2, shape=(3,)) - Linear velocity (m/s)
            "motor_state": Box(-2, 2, shape=(3,)) - Normalized wheel velocities
            "actions": Box(-1, 1, shape=(3,)) - Previous action
            "rgbd_0": Box(0, 1, shape=(C, H, W)) - RGB-D from camera 0
            "rgbd_1": Box(0, 1, shape=(C, H, W)) - RGB-D from camera 1
            "relative_image_timestamp": Box(0, 0.1, shape=(1,)) - Camera lag
        })
        where C=1 if depth_only else 4, H=im_shape["h"], W=im_shape["w"]
    
    Observation Space (without cameras):
        Same as above but without "rgbd_0", "rgbd_1", and "relative_image_timestamp"
    
    Reward Structure:
        1. Directional reward: velocity component in target direction (scaled by 1/100)
        2. Action regularization: -0.0001 * ||action||² (penalizes large actions)
        3. Survival bonus: +0.02 per step (if robot hasn't failed)
    
    Termination Conditions:
        - Episode exceeds max_ep_steps (default: 4000)
        - Robot tilt exceeds 20 degrees (failure)
    
    Args:
        xml_path (str): Path to MuJoCo XML model file defining the ballbot
        GUI (bool, optional): If True, launches passive MuJoCo viewer. 
            Note: On macOS, requires mjpython instead of python. Defaults to False.
        im_shape (dict, optional): Camera image dimensions. Keys: "h" (height), "w" (width).
            Defaults to {"h": 64, "w": 64}.
        disable_cameras (bool, optional): If True, disables camera rendering entirely.
            Reduces computational cost. Defaults to False.
        depth_only (bool, optional): If True, only renders depth (no RGB). 
            Reduces memory and computation. Defaults to True.
        log_options (dict, optional): Logging configuration. Keys:
            - "cams": If True, saves camera frames to disk
            - "reward_terms": If True, saves reward component history
            Defaults to {"cams": False, "reward_terms": False}.
        max_ep_steps (int, optional): Maximum steps per episode. 
            Defaults to 4000 if None.
        terrain_type (str, optional): Terrain generation type. Options:
            - "perlin": Procedural Perlin noise terrain (randomized each episode)
            - "flat": Flat terrain (no variation)
            Defaults to "perlin".
        eval_env (list, optional): [is_eval, seed]. If is_eval is True, uses fixed
            seed for reproducibility. Defaults to [False, None].
    
    Attributes:
        model (mujoco.MjModel): MuJoCo physics model
        data (mujoco.MjData): MuJoCo simulation data
        action_space (gym.spaces.Box): Action space definition
        observation_space (gym.spaces.Dict): Observation space definition
        reward_obj (Rewards.DirectionalReward): Reward function object
        goal_2d (list): Target direction [x, y] for directional reward
    
    Example:
        ```python
        env = BBotSimulation(
            xml_path="ballbot.xml",
            GUI=False,
            im_shape={"h": 64, "w": 64},
            terrain_type="perlin"
        )
        obs, info = env.reset()
        ```
    """
    
    def __init__(
            self,
            xml_path,
            GUI=False,  #full mujoco gui
            im_shape={
                "h": 64,
                "w": 64
            },
            disable_cameras=False,
            depth_only=True,
            log_options={
                "cams": False,
                "reward_terms": False
            },
            max_ep_steps=None,
            terrain_type: str = "perlin",
            eval_env=[False, None],
            reward_config=None,
            terrain_config=None,
            env_config=None,
            render_mode: Optional[str] = None,
            viewer_title: Optional[str] = None):
        """
        Initialize ballbot environment.
        
        Args:
            render_mode: Render mode for video recording. Options: "rgb_array" or None.
                        Defaults to None. Set to "rgb_array" for video recording compatibility.
        """
        super().__init__()
        
        # Validate render_mode
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render_mode: {render_mode}. "
                f"Supported modes: {self.metadata['render_modes']}"
            )

        # Store configuration parameters
        # Backward compatibility: if terrain_type is provided as string, convert to config
        if terrain_config is None:
            terrain_config = {"type": terrain_type, "config": {}}
        self.terrain_config = terrain_config
        self.terrain_type = terrain_config.get("type", terrain_type)  # Keep for backward compat
        
        # Backward compatibility: if reward_config is None, use default directional reward
        if reward_config is None:
            reward_config = {"type": "directional", "config": {"target_direction": [0.0, 1.0]}}
        self.reward_config = reward_config
        
        # Extract env config values (with defaults)
        if env_config is None:
            env_config = {}
        env_settings = env_config.get("env", {})
        camera_settings = env_config.get("camera", {})
        logging_settings = env_config.get("logging", {})
        
        # Override log_options with config if provided
        if logging_settings:
            self.log_options = {**log_options, **logging_settings}
        else:
            self.log_options = log_options
        
        self.xml_path = xml_path
        self.max_ep_steps = env_settings.get("max_ep_steps", max_ep_steps if max_ep_steps is not None else 4000)
        self.max_allowed_tilt = env_settings.get("max_allowed_tilt", 20.0)  # degrees
        self.max_wheel_velocity = env_settings.get("max_wheel_velocity", 10.0)  # rad/s
        self.camera_frame_rate = camera_settings.get("frame_rate", 90)  # Hz
        self.depth_only = depth_only
        
        # Extract reward config values (for scaling coefficients)
        reward_config_dict = reward_config.get("config", {})
        self.reward_scale = reward_config_dict.get("scale", 0.01)  # Default: /100
        self.action_reg_coef = reward_config_dict.get("action_reg_coef", -0.0001)
        self.survival_bonus = reward_config_dict.get("survival_bonus", 0.02)

        # Define action space: 3D continuous actions for three omniwheels
        # Actions are normalized to [-1, 1] and scaled internally to [-10, 10] rad/s
        self.action_space = gym.spaces.Box(-1.0,
                                           1.0,
                                           shape=(3, ),
                                           dtype=_default_dtype)

        # Determine number of channels: 1 for depth-only, 4 for RGB-D (RGB + depth)
        # Use camera config if available, otherwise use depth_only parameter
        actual_depth_only = camera_settings.get("disable_rgb", self.depth_only) if camera_settings else self.depth_only
        num_channels = 1 if actual_depth_only else 4
        
        # Use camera config dimensions if available
        obs_im_shape = {
            "h": camera_settings.get("height", im_shape["h"]) if camera_settings else im_shape["h"],
            "w": camera_settings.get("width", im_shape["w"]) if camera_settings else im_shape["w"]
        }
        
        # Create observation space using helper function
        self.observation_space = create_observation_space(
            im_shape=obs_im_shape,
            num_channels=num_channels,
            disable_cameras=disable_cameras
        )

        # Load MuJoCo model and create simulation data structure
        # The model contains all physical properties (masses, inertias, geometries, etc.)
        # The data structure holds the current simulation state (positions, velocities, etc.)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Initialize RGB-D camera system if enabled
        # Cameras are updated at a lower frequency than physics (see camera_frame_rate)
        # Use config values if provided, otherwise use im_shape parameter
        camera_height = camera_settings.get("height", im_shape["h"])
        camera_width = camera_settings.get("width", im_shape["w"])
        camera_disable_rgb = camera_settings.get("disable_rgb", self.depth_only)
        
        self.rgbd_inputs = None
        if not disable_cameras:
            self.rgbd_inputs = RGBDInputs(
                self.model,
                height=camera_height,
                width=camera_width,
                cams=["cam_0", "cam_1"],  # Two cameras for stereo vision
                disable_rgb=camera_disable_rgb
            )
            # Store previous image pair to reuse when cameras haven't updated
            self.prev_im_pair = StampedImPair(im_0=None, im_1=None, ts=0)
            # Initialize history lists for logging if enabled
            if self.log_options["cams"]:
                self.rgbd_hist_0 = []
                self.rgbd_hist_1 = []
        else:
            # Initialize empty lists even when cameras are disabled (for logging function)
            self.rgbd_hist_0 = []
            self.rgbd_hist_1 = []
        self.disable_cameras = disable_cameras
        
        # Set render_mode for video recording compatibility
        # VecVideoRecorder requires this attribute to be set
        # Use provided render_mode or default to "rgb_array" for video recording
        self.render_mode = render_mode if render_mode is not None else "rgb_array"
        
        # Create a dedicated renderer for video recording (lazy initialization)
        # This uses a world-view camera (third-person) to see the robot in action
        # Separate from observation cameras (cam_0, cam_1) which are robot-mounted
        self._video_renderer = None
        self._video_renderer_size = (640, 480)  # Standard video resolution (can be higher quality than obs cameras)

        # Store viewer title for later use
        self._viewer_title = viewer_title
        
        # Initialize passive viewer if GUI is requested
        # On macOS, this requires running with mjpython instead of python
        if GUI:
            try:
                self.passive_viewer = mujoco.viewer.launch_passive(
                    self.model, self.data)
                
                # Try to set window title if viewer_title is provided
                if viewer_title and self.passive_viewer is not None:
                    try:
                        import glfw
                        window = None
                        
                        # Try multiple ways to access the GLFW window
                        # Method 1: Direct access through Handle.sim.window
                        if hasattr(self.passive_viewer, 'sim'):
                            sim = self.passive_viewer.sim
                            if hasattr(sim, 'window'):
                                window = sim.window
                            elif hasattr(sim, '_window'):
                                window = sim._window
                        
                        # Method 2: Try accessing through Handle attributes
                        if window is None and hasattr(self.passive_viewer, 'window'):
                            window = self.passive_viewer.window
                        
                        # Method 3: Try accessing through Handle's internal _simulate
                        if window is None and hasattr(self.passive_viewer, '_simulate'):
                            sim = self.passive_viewer._simulate
                            if hasattr(sim, 'window'):
                                window = sim.window
                            elif hasattr(sim, '_window'):
                                window = sim._window
                        
                        # Set the title if we found the window
                        if window is not None:
                            glfw.set_window_title(window, viewer_title)
                    except (AttributeError, ImportError, Exception) as e:
                        # Silently fail if we can't set the title
                        # This is expected if the viewer structure changes or GLFW isn't available
                        pass
            except RuntimeError as e:
                if "mjpython" in str(e).lower():
                    print(colored(
                        "Warning: GUI requested but launch_passive requires mjpython on macOS. "
                        "Running without viewer. To enable GUI, run with: mjpython your_script.py",
                        "yellow",
                        attrs=["bold"]
                    ))
                    self.passive_viewer = None
                else:
                    raise
        else:
            self.passive_viewer = None

        # Logging directory (created in reset() method)
        self.log_dir = None

        # Track maximum absolute values of observations (for normalization/debugging)
        self.max_abs_obs = {
            x: -1 for x in ["orientation", "angular_vel", "vel", "pos"]
        }

        # History of reward components for logging and analysis
        self.reward_term_1_hist = []  # Directional reward
        self.reward_term_2_hist = []  # Action regularization
        self.reward_term_3_hist = []  # Reserved for future use

        # Episode counter (incremented in reset())
        self.num_episodes = -1
        
        # Evaluation mode: if True, uses fixed random seed for reproducibility
        self.eval_env = eval_env[0]
        if self.eval_env:
            # Initialize random number generator with fixed seed for evaluation
            self._np_random, _ = gym.utils.seeding.np_random(eval_env[1])
        else:
            # Will be initialized in reset() with gymnasium's seeding
            self._np_random = None

        # Verbose mode: print detailed information during execution
        self.verbose = False

    def effective_camera_frame_rate(self):
        """
        Calculate the effective camera frame rate given physics timestep constraints.
        
        Cameras update at a lower frequency than physics simulation. This method
        computes the actual frame rate, which may be slightly lower than the desired
        rate due to discrete physics timesteps.
        
        Returns:
            float: Effective camera frame rate in Hz
        
        Note:
            The effective rate is computed by finding the smallest number of physics
            steps that exceeds the desired camera interval, ensuring regular timestamps.
        """
        dt_mj = self.opt_timestep  # Physics timestep (e.g., 0.002s for 500Hz)
        desired_cam_dt = 1 / self.camera_frame_rate  # Desired camera interval (e.g., 1/90s)

        # Find number of physics steps needed to exceed desired camera interval
        N = np.ceil(desired_cam_dt / dt_mj)

        # Effective frame rate is inverse of actual interval (N * dt_mj)
        effective_framre_rate = 1.0 / (N * dt_mj)

        return effective_framre_rate

    @property
    def opt_timestep(self):
        """
        Get the physics simulation timestep from MuJoCo model.
        
        Returns:
            float: Physics timestep in seconds (typically 0.002 for 500Hz)
        """
        return self.model.opt.timestep

    def _reset_goal_and_reward_objs(self):
        """
        Initialize or reset the goal direction and reward function.
        
        Uses the factory pattern to create reward function from configuration.
        Supports backward compatibility with goal_2d attribute.
        """
        # Create reward using factory
        self.reward_obj = create_reward(self.reward_config)
        
        # Extract goal_2d for backward compatibility (if reward has target_direction)
        if hasattr(self.reward_obj, 'target_direction'):
            self.goal_2d = self.reward_obj.target_direction.tolist() if hasattr(self.reward_obj.target_direction, 'tolist') else list(self.reward_obj.target_direction)
        else:
            # Default goal for backward compatibility
            self.goal_2d = [0.0, 1.0]

    def _reset_terrain(self):
        """
        Generate or reset the terrain heightfield and compute initial robot height.
        
        This method:
        1. Generates terrain data (Perlin noise or flat)
        2. Updates the MuJoCo heightfield
        3. Computes the initial robot height to place it on the terrain surface
        
        The robot height is computed by:
        - Finding the ball's axis-aligned bounding box (AABB)
        - Sampling terrain heights under the ball
        - Setting robot height to maximum terrain height + small epsilon
        
        This ensures the robot starts on the terrain surface, not embedded in it.
        
        Returns:
            float: Initial height offset to apply to robot joints (in meters)
        
        Raises:
            AssertionError: If terrain dimensions are not square
            AssertionError: If ball is not centered at origin (required assumption)
            Exception: If terrain_type is not "perlin" or "flat"
        """
        # Get terrain dimensions from MuJoCo model
        nrows = self.model.hfield_nrow.item()  # Number of heightfield rows
        ncols = self.model.hfield_ncol.item()  # Number of heightfield columns

        # Validate terrain is square (simplifies indexing)
        assert nrows == ncols, (
            f"Terrain must be square (got {nrows}x{ncols} in XML file)"
        )
        # Validate terrain size is square
        assert self.model.hfield_size[0, 0] == self.model.hfield_size[0, 1], (
            f"Terrain must have equal length and width "
            f"(got {self.model.hfield_size[0,:2]} in XML file)"
        )

        # Extract terrain parameters
        sz = self.model.hfield_size[0, 0]  # Terrain size (length/width in meters)
        
        # Dynamic vertical scaling for angle-based terrains
        # This ensures physical slopes match requested angles regardless of terrain width
        terrain_config_dict = self.terrain_config.get("config", {})
        if self.terrain_type == "ramp":
            angle = terrain_config_dict.get("ramp_angle", 15.0)
            # Rise = Run * tan(theta), where Run is full width (2 * sz)
            required_height = (2 * sz) * np.tan(np.radians(angle))
            self.model.hfield_size[0, 2] = required_height
            
        elif self.terrain_type == "gradient":
            angle = terrain_config_dict.get("max_slope", 20.0)
            required_height = (2 * sz) * np.tan(np.radians(angle))
            self.model.hfield_size[0, 2] = required_height
            
        # Update height coef variable after potential modification
        hfield_height_coef = self.model.hfield_size[0, 2]  # Height scaling factor

        # Generate terrain data using factory pattern
        terrain_gen = create_terrain(self.terrain_config)
        
        # Get seed from config or generate random one
        terrain_config_dict = self.terrain_config.get("config", {})
        if terrain_config_dict.get("seed") is None:
            r_seed = self._np_random.integers(0, 10000)
            self.last_r_seed = r_seed  # Store for logging/reproducibility
        else:
            r_seed = terrain_config_dict["seed"]
            self.last_r_seed = r_seed
        
        # Generate terrain with seed
        self.model.hfield_data = terrain_gen(nrows, seed=r_seed)

        # Update viewer if active (so terrain is visible)
        if self.passive_viewer is not None:
            terrain_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain"
            )
            self.passive_viewer.update_hfield(terrain_id)

        # Reset simulation data and forward kinematics
        # This is needed to get accurate positions after terrain generation
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)  # Compute positions, velocities, etc.

        # Get ball geometry information
        ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "the_ball")
        
        # Verify ball is centered at origin (required for terrain generation logic)
        assert (self.data.geom_xpos[ball_id][0] == 0 and 
                self.data.geom_xpos[ball_id][1] == 0), (
            "Terrain generation assumes ball is centered at (0,0). "
            "Check XML file."
        )
        
        ball_pos = self.data.geom_xpos[ball_id]  # Ball center position [x, y, z]
        ball_size = self.model.geom_size[ball_id]  # Ball radius (scalar)
        
        # Compute axis-aligned bounding box (AABB) of the ball
        # Note: ball_size is a scalar (same radius in all directions)
        aabb_min = ball_pos - ball_size[0]  # Minimum corner [x, y, z]
        aabb_max = ball_pos + ball_size[0]  # Maximum corner [x, y, z]

        # Convert terrain to 2D matrix and extract region under the ball
        cell_size = sz / nrows  # Size of each terrain cell in meters
        center_idx = nrows // 2  # Index of center cell (where ball is)

        terrain_mat = self.model.hfield_data.reshape(nrows, ncols)
        
        # Extract terrain region that the ball covers
        # Convert AABB bounds to cell indices
        x_min_idx = center_idx - abs(int(aabb_min[0] // cell_size))
        x_max_idx = center_idx + int(aabb_max[0] // cell_size) + 1
        y_min_idx = center_idx - abs(int(aabb_min[1] // cell_size))
        y_max_idx = center_idx + int(aabb_max[1] // cell_size) + 1
        
        sub_terr = terrain_mat[x_min_idx:x_max_idx, y_min_idx:y_max_idx]

        # Compute initial robot height: maximum terrain height under ball + epsilon
        # This ensures robot sits on terrain surface, not inside it
        eps = 0.01  # Small safety margin (1cm)
        init_robot_height_offset = sub_terr.max() * hfield_height_coef + eps

        return init_robot_height_offset

    def reset(self, seed=None, goal: str = "random", **kwargs):
        """
        Reset the environment to an initial state for a new episode.
        
        This method:
        1. Seeds the random number generator
        2. Resets goal and reward function
        3. Generates new terrain
        4. Positions robot on terrain surface
        5. Resets all state tracking variables
        6. Creates logging directory (if needed)
        
        Args:
            seed (int, optional): Random seed for reproducibility. If None, uses
                random seed. Defaults to None.
            goal (str, optional): Goal specification (currently unused, goal is fixed).
                Defaults to "random".
            **kwargs: Additional keyword arguments (for Gymnasium compatibility).
        
        Returns:
            tuple: (observation, info)
                - observation (dict): Initial observation dictionary
                - info (dict): Information dictionary with episode metadata
        
        Note:
            The robot is positioned at the terrain center (0, 0) with height computed
            to place it on the terrain surface. All velocities are zero initially.
        """
        # Initialize random number generator (Gymnasium standard)
        super().reset(seed=seed)
        if self._np_random is None:
            # Initialize if not already done (e.g., in eval mode)
            self._np_random = self.np_random

        # Reset goal direction and reward function
        self._reset_goal_and_reward_objs()

        # Reset episode counters
        self.step_counter = 0
        self.prev_data_time = 0

        # Generate new terrain and get initial height offset
        init_robot_height_offset = self._reset_terrain()

        # Reset MuJoCo simulation data
        mujoco.mj_resetData(self.model, self.data)
        
        # Apply height offset to base and ball joints
        # This positions the robot on the terrain surface
        self.data.joint("base_free_joint").qpos[2] += init_robot_height_offset
        self.data.joint("ball_free_joint").qpos[2] += init_robot_height_offset
        
        # Forward kinematics: compute positions, velocities, etc.
        mujoco.mj_forward(self.model, self.data)

        # Reset camera renderers (needed after terrain change)
        if not self.disable_cameras:
            self.rgbd_inputs.reset(self.model)
            self.prev_im_pair = StampedImPair(im_0=None, im_1=None, ts=0)

        # Reset state tracking variables
        # Note: prev_pos and prev_orientation are no longer needed since we use data.cvel
        # Keeping prev_motor_state for potential future use
        self.prev_motor_state = None
        self.prev_time = 0

        # Get initial observation (with zero action)
        obs = self._get_obs(np.zeros(3).astype(_default_dtype))
        info = self._get_info()

        # Reset camera logging history if enabled
        if self.log_options.get("cams", False):
            if not self.disable_cameras:
                self.rgbd_hist_0 = []
                self.rgbd_hist_1 = []

        # Reset episode return accumulator
        self.G_tau = 0.0
        self.num_episodes += 1

        # Print effective camera frame rate on first episode (if verbose)
        if self.num_episodes == 0 and self.verbose:
            print(colored(
                f"effective_frame_rate=={self.effective_camera_frame_rate()}",
                "cyan",
                attrs=["bold"]
            ))

        # Create logging directory (only for training, not evaluation)
        if self.log_dir is None and not self.eval_env:
            # Generate random directory name
            rand_str = ''.join(
                self._np_random.permutation(
                    list(string.ascii_letters + string.digits)
                )[:12]
            )
            self.log_dir = "/tmp/log_" + rand_str
            if os.path.exists(self.log_dir):
                print(f"log_dir {self.log_dir} already exists. Overwriting!")
                shutil.rmtree(self.log_dir)
            else:
                print(f"Creating log_dir {self.log_dir}")
            os.mkdir(self.log_dir)

        return obs, info

    def _save_logs(self):
        """
        Save episode logs to disk (cameras, rewards, terrain seeds).
        
        This method is called at the end of each episode (when terminated=True).
        It saves:
        - Camera frames (RGB and/or depth) if logging is enabled
        - Reward component histories for analysis
        - Terrain seeds for reproducibility
        
        Note:
            Logs are only saved during training (not evaluation).
            Files are saved to self.log_dir, which is created in reset().
        """
        save_episode_logs(
            log_dir=self.log_dir,
            rgbd_hist_0=getattr(self, 'rgbd_hist_0', []),
            rgbd_hist_1=getattr(self, 'rgbd_hist_1', []),
            reward_term_1_hist=self.reward_term_1_hist,
            reward_term_2_hist=self.reward_term_2_hist,
            terrain_seed=getattr(self, 'last_r_seed', None) if self.terrain_type == "perlin" else None,
            terrain_type=self.terrain_type,
            depth_only=self.depth_only,
            log_options=self.log_options,
            num_episodes=self.num_episodes,
            eval_env=self.eval_env
        )

    def _get_obs(self, last_ctrl):
        """
        Construct the observation dictionary from current simulation state.
        
        This method collects:
        - Visual observations: RGB-D images from cameras (updated at lower frequency)
        - Proprioceptive observations: orientation, velocities, motor states
        - Previous action: for temporal context
        
        Camera Update Logic:
            Cameras update at a lower frequency (camera_frame_rate) than physics
            simulation. This is done to:
            1. Reduce computational cost (rendering is expensive)
            2. Match realistic camera frame rates
            3. Provide temporal context (previous frames are reused)
            
            The time-based update ensures regular timestamps, which is important
            for learning temporal patterns. The effective frame rate may be slightly
            lower than desired due to discrete physics timesteps.
        
        Velocity Computation:
            - Linear velocity: Extracted from data.cvel (MuJoCo's computed body velocity)
            - Angular velocity: Extracted from data.cvel (MuJoCo's computed body velocity)
                MuJoCo automatically computes 6D body velocities (3D linear + 3D angular)
                in the world frame at the body's center of mass. This is more accurate
                than finite difference and doesn't require storing previous state.
        
        Args:
            last_ctrl (np.ndarray): Previous action (shape: (3,))
        
        Returns:
            dict: Observation dictionary with keys:
                - "orientation": Rotation vector (3,)
                - "angular_vel": Angular velocity (3,)
                - "vel": Linear velocity (3,)
                - "motor_state": Normalized wheel velocities (3,)
                - "actions": Previous action (3,)
                - "rgbd_0": RGB-D from camera 0 (C, H, W) [if cameras enabled]
                - "rgbd_1": RGB-D from camera 1 (C, H, W) [if cameras enabled]
                - "relative_image_timestamp": Camera lag (1,) [if cameras enabled]
        """
        # ========== CAMERA OBSERVATIONS ==========
        if not self.disable_cameras:
            # Compute time since last camera update
            delta_time = self.data.time - self.prev_im_pair.ts

            # Update cameras only if enough time has passed
            # This ensures regular timestamps (may be slightly lower than desired rate)
            # See docstring for detailed explanation of frame rate logic
            if self.prev_im_pair.im_0 is None or delta_time >= 1.0 / self.camera_frame_rate:
                # Render new images from both cameras
                rgbd_0 = self.rgbd_inputs(self.data, "cam_0").astype(_default_dtype)
                rgbd_1 = self.rgbd_inputs(self.data, "cam_1").astype(_default_dtype)
                
                # Log images if enabled
                if self.log_options["cams"]:
                    self.rgbd_hist_0.append(rgbd_0)
                    self.rgbd_hist_1.append(rgbd_1)

                # Store current images and timestamp
                self.prev_im_pair.im_0 = rgbd_0.copy()
                self.prev_im_pair.im_1 = rgbd_1.copy()
                self.prev_im_pair.ts = self.data.time
            else:
                # Reuse previous images (cameras haven't updated yet)
                rgbd_0 = self.prev_im_pair.im_0.copy()
                rgbd_1 = self.prev_im_pair.im_1.copy()

        # ========== PROPRIOCEPTIVE OBSERVATIONS ==========
        
        # Get robot base body ID and extract position/orientation
        body_id = self.model.body("base").id
        position = self.data.xpos[body_id].copy().astype(_default_dtype)  # [x, y, z]
        
        # Convert quaternion to rotation vector (Euler-like representation)
        # Quaternions are more stable than Euler angles but rotation vectors are
        # easier for neural networks to learn from
        orientation_quat = quaternion.quaternion(*self.data.xquat[body_id].copy())
        rot_vec = quaternion.as_rotation_vector(orientation_quat).astype(_default_dtype)

        # Get motor (wheel) velocities
        # These are the actual angular velocities of the three omniwheels
        motor_state = np.array([
            self.data.qvel[self.model.joint(f"wheel_joint_{motor_idx}").id]
            for motor_idx in range(3)
        ]).astype(_default_dtype)
        motor_state /= self.max_wheel_velocity  # Normalize by max wheel velocity from config
        motor_state = np.clip(motor_state, a_min=-2.0, a_max=2.0)
        self.prev_motor_state = motor_state.copy()

        # Get body velocities from MuJoCo (computed automatically)
        # data.cvel provides 6D velocity: [linear_vel (3D), angular_vel (3D)]
        # This is more accurate than finite difference and doesn't require state storage
        cvel = self.data.cvel[body_id]
        linear_vel = cvel[:3].copy().astype(_default_dtype)  # Linear velocity in world frame [m/s]
        angular_vel = cvel[3:].copy().astype(_default_dtype)  # Angular velocity in world frame [rad/s]
        
        # Clip velocities to reasonable ranges for observations
        vel = np.clip(linear_vel, a_min=-2.0, a_max=2.0)
        angular_vel = np.clip(angular_vel, a_min=-2.0, a_max=2.0)

        # ========== CONSTRUCT OBSERVATION DICTIONARY ==========
        if self.disable_cameras:
            # Proprioceptive-only observation
            obs = {
                "orientation": rot_vec,
                "angular_vel": angular_vel,
                "vel": vel,
                "motor_state": motor_state,
                "actions": last_ctrl
            }
        else:
            # Full observation with cameras
            obs = {
                "orientation": rot_vec,
                "angular_vel": angular_vel,
                "vel": vel,
                "motor_state": motor_state,
                "actions": last_ctrl,
                # Convert images from (H, W, C) to (C, H, W) format (channels-first)
                "rgbd_0": rgbd_0.transpose(2, 0, 1),
                "rgbd_1": rgbd_1.transpose(2, 0, 1),
                # Time lag between proprioceptive and visual data
                "relative_image_timestamp": np.array([
                    self.data.time - self.prev_im_pair.ts
                ]).astype(_default_dtype),
            }

        return obs

    def _get_info(self):
        """
        Get information dictionary for the current state.
        
        This method provides metadata about the current episode state,
        which is useful for debugging, logging, and analysis.
        
        Returns:
            dict: Information dictionary with keys:
                - "success": Whether episode ended successfully (always False currently)
                - "failure": Whether episode ended due to failure (set in step())
                - "step_counter": Current step number in episode
                - "pos2d": 2D position [x, y] of robot base (z is height)
        """
        position = self.data.xpos[self.model.body("base").id].copy().astype(_default_dtype)
        info = {
            "success": False,  # Currently unused (could indicate goal reached)
            "failure": False,   # Set to True in step() if robot falls
            "step_counter": self.step_counter,  # Current step in episode
            "pos2d": position[:-1]  # 2D position [x, y] (exclude height z)
        }
        return info

    def step(self, omniwheel_commands):
        """
        Execute one simulation step.
        
        This is the core method that:
        1. Applies actions to the robot (omniwheel commands)
        2. Steps the physics simulation forward
        3. Computes observations
        4. Calculates rewards
        5. Checks termination conditions
        
        Action Processing:
            Actions are normalized to [-1, 1] and scaled to [-10, 10] rad/s.
            The negative sign accounts for coordinate system conventions.
        
        Reward Structure:
            1. Directional reward: Encourages velocity in target direction
            2. Action regularization: Penalizes large actions (energy efficiency)
            3. Survival bonus: Small positive reward for staying upright
        
        Termination Conditions:
            - Episode length exceeds max_ep_steps
            - Robot tilt exceeds 20 degrees (failure)
        
        Args:
            omniwheel_commands (np.ndarray): Normalized wheel commands, shape (3,).
                Values should be in [-1, 1]. Each value controls one omniwheel.
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation (dict): Current observation dictionary
                - reward (float): Reward for this step
                - terminated (bool): True if episode ended (failure or max steps)
                - truncated (bool): Always False (reserved for time limits)
                - info (dict): Information dictionary with episode metadata
        
        Note:
            The reward is computed based on velocity alignment with goal direction,
            not position. This encourages continuous movement rather than just
            reaching a point.
        """

        # Handle GUI reset (if user manually resets in viewer)
        if self.data.time == 0.0 and self.step_counter > 0.0:
            print("RESET DUE TO RESET FROM GUI")
            self.reset()

        # Scale actions from [-1, 1] to [-max_wheel_velocity, max_wheel_velocity] rad/s
        # This is the actual wheel velocity command
        ctrl = omniwheel_commands * self.max_wheel_velocity
        ctrl = np.clip(ctrl, a_min=-self.max_wheel_velocity, a_max=self.max_wheel_velocity)  # Safety: prevent extreme commands

        # Apply control commands (negative for correct coordinate system)
        self.data.ctrl[:] = -ctrl
        
        # Step physics simulation forward
        # Suppress warnings about objective convexity (benign MuJoCo warnings)
        with warnings_stdout_off():
            mujoco.mj_step(self.model, self.data)

        # Reset applied forces (MuJoCo applies forces until explicitly reset)
        # This prevents forces from accumulating across steps
        self.data.xfrc_applied[self.model.body("base").id, :3] = np.zeros(3)

        # Get current observation and info
        obs = self._get_obs(omniwheel_commands.astype(_default_dtype))
        info = self._get_info()
        terminated = False
        truncated = False  # Not used (reserved for time limits)

        # ========== REWARD COMPUTATION ==========
        
        # 1. Directional reward: velocity component in target direction
        # This encourages movement toward the goal direction
        # Scaled by reward_scale (default: 0.01 = /100) to keep rewards in reasonable range
        reward = self.reward_obj(obs) * self.reward_scale
        self.reward_term_1_hist.append(reward)

        # 2. Action regularization: penalize large actions
        # This encourages energy-efficient control and smooth motions
        # The L2 norm squared gives quadratic penalty
        action_regularization = self.action_reg_coef * (np.linalg.norm(omniwheel_commands)**2)
        self.reward_term_2_hist.append(action_regularization)
        reward += action_regularization

        # Update passive viewer if enabled (for visualization)
        if self.passive_viewer:
            with self.passive_viewer.lock():
                # Optional: visualize goal direction (currently disabled)
                # This code would draw a sphere and line showing the target direction
                if 0:  # Set to 1 to enable goal visualization
                    self.passive_viewer.user_scn.ngeom = 1  # Reset geometry count
                    factor = 20  # Scaling factor for display
                    hh = 0.5  # Height offset
                    mujoco.mjv_initGeom(
                        self.passive_viewer.user_scn.geoms[
                            self.passive_viewer.user_scn.ngeom - 1
                        ],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.1] * 3,
                        pos=[
                            self.goal_2d[0] * factor,
                            self.goal_2d[1] * factor, hh
                        ],
                        mat=np.eye(3).flatten(),
                        rgba=[1, 0, 1, 1]  # Magenta color
                    )
                    mujoco.mjv_connector(
                        self.passive_viewer.user_scn.geoms[
                            self.passive_viewer.user_scn.ngeom - 1
                        ],
                        type=mujoco.mjtGeom.mjGEOM_LINE,
                        width=200,
                        from_=[0, 0, hh],
                        to=[
                            self.goal_2d[0] * factor,
                            self.goal_2d[1] * factor, hh
                        ]
                    )
            self.passive_viewer.sync()

        # Track simulation time to detect GUI resets
        self.prev_data_time = self.data.time
        self.step_counter += 1

        # ========== TERMINATION CONDITIONS ==========
        
        # Check if episode exceeded maximum steps
        if self.step_counter >= self.max_ep_steps:
            if self.verbose:
                print(f"terminated. Cause: {self.step_counter} >= {self.max_ep_steps}")
            terminated = True

        # Check if robot has fallen (tilted too much)
        # Compute angle between robot's up-axis and gravity vector
        gravity = np.array([0, 0, -1.0]).astype(_default_dtype).reshape(3, 1)  # Global gravity
        
        # Transform gravity to robot's local frame
        # This tells us which direction "down" is from the robot's perspective
        R_local = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(obs["orientation"][-3:])
        )
        gravity_local = (R_local.T @ gravity).reshape(3)
        
        # Robot's local up-axis (should align with -gravity_local when upright)
        up_axis_local = np.array([0, 0, 1]).astype(_default_dtype)
        
        # Compute tilt angle: angle between up-axis and negative gravity
        # When upright: up_axis_local ≈ -gravity_local, angle ≈ 0°
        # When fallen: up_axis_local ⊥ -gravity_local, angle ≈ 90°
        angle_in_degrees = np.arccos(
            up_axis_local.dot(-gravity_local)
        ).item() * 180 / np.pi

        if angle_in_degrees > self.max_allowed_tilt:
            # Robot has fallen: terminate episode
            if self.verbose:
                print(
                    f"failure after {self.step_counter} steps. "
                    f"Reason: tilt_angle ({angle_in_degrees:.1f}°) > {self.max_allowed_tilt}°"
                )
            info["success"] = False
            info["failure"] = True
            terminated = True
        else:
            # Robot is still upright: add survival bonus
            reward += self.survival_bonus

        # Accumulate episode return (for logging/analysis)
        gamma = 1.0  # Discount factor (not used algorithmically, but tracked)
        self.G_tau += (gamma**self.step_counter) * reward
        
        # Save logs if episode ended
        if terminated:
            if self.verbose:
                print(colored(
                    f"Episode ended. Return: {self.G_tau:.2f}, Steps: {self.step_counter}",
                    "magenta",
                    attrs=["bold"]
                ))
            self._save_logs()

        return obs, reward, terminated, truncated, info

    def close(self):
        """
        Clean up resources and close the environment.
        
        This method should be called when done with the environment to:
        - Close camera renderers
        - Close visualization window
        - Free MuJoCo model and data structures
        
        Note:
            After calling close(), the environment should not be used further.
            Create a new instance if needed.
        """
        # Close camera renderers if enabled
        if not self.disable_cameras:
            self.rgbd_inputs.close()

        # Close passive viewer if active
        if self.passive_viewer:
            self.passive_viewer.close()

        # Note: GLFW termination is commented out as it can cause freezing
        # The viewer.close() above should handle cleanup properly
        # if glfw.get_current_context() is not None:
        #     glfw.terminate()

        # Clean up video renderer if it exists
        if hasattr(self, '_video_renderer') and self._video_renderer is not None:
            try:
                self._video_renderer.close()
            except Exception:
                # Ignore errors during cleanup (renderer may already be closed)
                pass
            self._video_renderer = None

        # Explicitly delete MuJoCo structures to free memory
        del self.model
        del self.data

    def render(self):
        """
        Render the environment and return an RGB array for video recording.
        
        This method is required by VecVideoRecorder for video generation.
        Uses a world-view (third-person) camera that tracks the robot.
        This is separate from observation cameras (cam_0, cam_1) which are robot-mounted.
        
        Returns:
            np.ndarray: RGB image array of shape (H, W, 3) with values in [0, 255] (uint8).
                       Always returns a valid RGB array using a world-view camera.
        
        Note:
            - Uses a tracking camera that follows the robot's position
            - Camera position: behind and above robot, looking down at an angle
            - This provides a good view of the robot navigating terrain
            - Observation cameras (cam_0, cam_1) remain for robot perception
        """
        # Initialize video renderer if needed (lazy initialization)
        if self._video_renderer is None:
            video_height, video_width = self._video_renderer_size
            self._video_renderer = mujoco.Renderer(self.model, width=video_width, height=video_height)
        
        try:
            # Get robot position for camera tracking
            body_id = self.model.body("base").id
            robot_pos = self.data.xpos[body_id].copy()  # [x, y, z]
            
            # Create a tracking camera that follows the robot
            # Position: behind robot (-Y direction), elevated, looking down
            camera_offset = np.array([0, -2.0, 1.5])  # Behind and above robot
            camera_pos = robot_pos + camera_offset
            lookat_pos = robot_pos + np.array([0, 0, 0.2])  # Look at robot center
            
            # Use MuJoCo's scene camera API to create a tracking camera
            # We'll update the scene with a custom camera view
            scene = self._video_renderer._scene
            if scene is not None:
                # Set up tracking camera
                scene.camera.lookat[:] = lookat_pos
                scene.camera.distance = np.linalg.norm(camera_offset)
                scene.camera.elevation = -20  # Look down angle (degrees)
                scene.camera.azimuth = 90  # Behind robot (90 degrees = -Y direction)
            
            # Update scene and render
            self._video_renderer.update_scene(self.data, camera=None)
            rgb = self._video_renderer.render().astype(_default_dtype) / 255.0
            rgb_uint8 = (rgb * 255).astype(np.uint8)
            return rgb_uint8
        except Exception:
            # Fallback: use cam_0 (robot-mounted camera) if tracking fails
            try:
                self._video_renderer.update_scene(self.data, camera="cam_0")
                rgb = self._video_renderer.render().astype(_default_dtype) / 255.0
                rgb_uint8 = (rgb * 255).astype(np.uint8)
                return rgb_uint8
            except Exception:
                # Last resort: return a black image (should never happen)
                return np.zeros(self._video_renderer_size + (3,), dtype=np.uint8)


