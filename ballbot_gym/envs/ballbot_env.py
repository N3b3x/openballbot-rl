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
from ballbot_gym.utils.contact_frame import ContactFrameStepper
from ballbot_gym.sensors.tof import ToFArray
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
        if camera_settings.get("enabled", True) is False:
            disable_cameras = True  # proprioception-only ablation
        logging_settings = env_config.get("logging", {})
        
        # Override log_options with config if provided
        if logging_settings:
            self.log_options = {**log_options, **logging_settings}
        else:
            self.log_options = log_options
        
        self.xml_path = xml_path
        self.max_ep_steps = env_settings.get("max_ep_steps", max_ep_steps if max_ep_steps is not None else 4000)
        self.max_allowed_tilt = env_settings.get("max_allowed_tilt", 20.0)  # degrees
        # robot must stay on the ball: angle of base-above-ball-centre line from vertical (None = no check)
        self.max_ball_offset_deg = env_settings.get("max_ball_offset_deg", None)
        self.lean_coef = env_settings.get("lean_coef", 0.0)
        # domain randomisation + disturbances (None = nominal). Keys (all optional):
        #   mass: +-frac on base and ball mass   friction: +-frac on wheel-ball pair friction
        #   damping: [lo, hi] wheel damping       torque: +-frac motor torque scale
        #   latency_p: prob. an action is applied one step late   obs_noise: std scale (vel 1x, gyro 2x, orient 0.25x)
        #   push_force: [lo, hi] N horizontal push on the base   push_every: [lo, hi] s   push_dur: s
        self.dr = env_settings.get("dr", None)
        self.init_yaw_random = env_settings.get("init_yaw_random", False)
        self.obs_vint = env_settings.get("obs_vint", False)   # add integrated velocity error to observations
        # command-direction cone and minimum speed (e.g. step-climbing training toward +y)
        self.cmd_ang_range = tuple(env_settings.get("cmd_ang_range", [0.0, 2 * np.pi]))
        self.cmd_vmin = env_settings.get("cmd_vmin", 0.0)
        self._vint = np.zeros(2)
        # residual control: action = LQR(model-based) action + residual_scale * policy action
        self.residual_lqr = env_settings.get("residual_lqr", False)
        self._env_settings_lqr_json = env_settings.get("lqr_model_json")   # per-morphology LQR model (design studies)
        # hybrid controller: the policy outputs 3 wheel actions + 1 gate logit; applied = (1-g) a_LQR + g a_pi
        self.hybrid_gate = env_settings.get("hybrid_gate", False)
        self.gate_init = float(env_settings.get("gate_init", 0.1))    # (not used by env; documentation of the zero-init bias)
        self._morph_pool = env_settings.get("morph_pool") or []        # list of design dicts (xml/json/tau/damp/armature/ctx)
        self._morph_ctx = np.array(env_settings.get("morph_ctx", [0, 0, 0, 0]), dtype=np.float32)
        self._morph_idx = None
        self._obs_morph = bool(self._morph_pool) or ("morph_ctx" in env_settings)
        self._lqr_act = np.zeros(3, dtype=np.float32)
        self._env_settings_all = env_settings
        self._motor = env_settings.get("motor")     # {omega_nl, tau_stall, fc, lag}: realistic actuator model (off by default)
        self._tau_act = np.zeros(3)
        self.residual_scale = env_settings.get("residual_scale", 0.3)
        self._lqr = None
        self._last_obs = None
        self.ball_offset_coef = env_settings.get("ball_offset_coef", 0.0)   # soft penalty on (offset/limit)^2
        self.action_rate_coef = env_settings.get("action_rate_coef", 0.0)  # <= 0, penalty on |a_t - a_t-1|^2
        self.max_wheel_velocity = env_settings.get("max_wheel_velocity", 10.0)  # rad/s
        # Physics steps per policy decision (1 = 500 Hz control as in the paper)
        self.action_repeat = int(env_settings.get("action_repeat", 1))
        # Goal distance along the target direction [m]. Reaching it ends the episode as a
        # success (time-limit style truncation so PPO bootstraps the value). None = disabled.
        self.goal_progress = env_settings.get("goal_progress", None)
        # Waypoint task (H3). goal_mode: None | "relative" (obs = goal - position, closed loop)
        # | "direction" (obs = unit direction to the goal at reset, open loop).
        self.goal_mode = env_settings.get("goal_mode", None)
        self.goal_radius = float(env_settings.get("goal_radius", 0.3))
        self.goal_dist_range = tuple(env_settings.get("goal_dist_range", (2.0, 3.0)))
        self.goal_bearing_deg = float(env_settings.get("goal_bearing_deg", 60.0))
        self.goal_xy = np.array([0.0, 3.0])
        self.goal_dir0 = np.array([0.0, 1.0])
        # Velocity-command mode: obs["cmd"] = [vx, vy, wz] (world frame, m/s and rad/s), resampled every
        # cmd_resample_steps decisions; reward = exp tracking of planar velocity and yaw rate.
        self.command_mode = bool(env_settings.get("command_mode", False))
        self.cmd_vmax = float(env_settings.get("cmd_vmax", 1.0))
        self.cmd_wmax = float(env_settings.get("cmd_wmax", 2.0))
        self.cmd_resample_steps = int(env_settings.get("cmd_resample_steps", 200))
        self.cmd_zero_prob = float(env_settings.get("cmd_zero_prob", 0.1))
        self.cmd_sigma_v = float(env_settings.get("cmd_sigma_v", 0.25))
        self.cmd_sigma_w = float(env_settings.get("cmd_sigma_w", 0.5))
        self.cmd_reward = env_settings.get("cmd_reward", "exp")  # "exp" | "dot"
        self.cmd_fixed = env_settings.get("cmd_fixed")  # optional fixed [vx, vy, wz] for evaluation
        self.wheel_damping = env_settings.get("wheel_damping")  # override joint damping [N m s/rad]
        self.obs_vel_clip = float(env_settings.get("obs_vel_clip", 2.0))
        self.cmd = np.zeros(3)
        self.cmd_err_hist = []
        # Re-align omniwheel contact frames in Python (needed for stock, unpatched MuJoCo)
        self.fix_wheel_contacts = bool(env_settings.get("fix_wheel_contacts", True))
        # Optional depth corruption for the perception-robustness study (H2)
        self.depth_noise_std = float(camera_settings.get("depth_noise_std", 0.0))
        self.depth_dropout = float(camera_settings.get("depth_dropout", 0.0))
        self.depth_occlusion = float(camera_settings.get("depth_occlusion", 0.0))
        self.camera_frame_rate = camera_settings.get("frame_rate", 90)  # Hz
        self.depth_only = depth_only
        
        # Extract reward config values (for scaling coefficients)
        reward_config_dict = reward_config.get("config", {})
        self.reward_scale = reward_config_dict.get("scale", 0.01)  # Default: /100
        self.action_reg_coef = reward_config_dict.get("action_reg_coef", -0.0001)
        self.survival_bonus = reward_config_dict.get("survival_bonus", 0.02)
        self.fall_penalty = float(reward_config_dict.get("fall_penalty", 0.0))
        self.reward_cap = reward_config_dict.get("cap", None)  # max raw directional speed [m/s]

        # Define action space: 3D continuous actions for three omniwheels
        # Actions are normalized to [-1, 1] and scaled internally to [-10, 10] rad/s
        self.action_space = gym.spaces.Box(-1.0,
                                           1.0,
                                           shape=(4 if env_settings.get("hybrid_gate", False) else 3, ),
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

        if self.goal_mode:
            spaces = dict(self.observation_space.spaces)
            spaces["goal"] = gym.spaces.Box(-1.5, 1.5, shape=(2, ), dtype=_default_dtype)
            self.observation_space = gym.spaces.Dict(spaces)
        if self.command_mode:
            spaces = dict(self.observation_space.spaces)
            spaces["cmd"] = gym.spaces.Box(-5.0, 5.0, shape=(3, ), dtype=_default_dtype)
            spaces["heading"] = gym.spaces.Box(-1.0, 1.0, shape=(2, ), dtype=_default_dtype)
            if self.obs_vint:
                spaces["vint"] = gym.spaces.Box(-1.0, 1.0, shape=(2, ), dtype=_default_dtype)
            if self.hybrid_gate:
                spaces["lqr_act"] = gym.spaces.Box(-1.0, 1.0, shape=(3, ), dtype=_default_dtype)
            if self._obs_morph:
                spaces["morph"] = gym.spaces.Box(-5.0, 5.0, shape=(4, ), dtype=_default_dtype)
            self.observation_space = gym.spaces.Dict(spaces)

        # Optional multizone ToF sensors (env_config["tof"] = list of sensor dicts) and the privileged
        # height scan for teacher policies (env_config["height_scan"] = {"x": [lo, hi, n], "y": [lo, hi, n]}).
        self._tof_cfg = env_config.get("tof") or []
        self._scan_cfg = env_config.get("height_scan")
        self._tof_extra_noise = float(env_settings.get("tof_extra_noise", 0.0))
        self._tof_extra_dropout = float(env_settings.get("tof_extra_dropout", 0.0))

        # Load MuJoCo model and create simulation data structure
        # The model contains all physical properties (masses, inertias, geometries, etc.)
        # The data structure holds the current simulation state (positions, velocities, etc.)
        if env_settings.get("xml_path"):                      # morphology variants (design studies)
            self.xml_path = str(env_settings["xml_path"])
        self._integrator = env_settings.get("integrator", "rk4")   # "rk4" (model default) or "implicitfast" (~3x faster)
        self._wheel_armature = env_settings.get("wheel_armature")
        self._load_model()
        self._tq_scale, self._prev_act, self._push_t, self._push_left, self._push_f = 1.0, None, 0, 0, np.zeros(3)
        self._tof = ToFArray(self.model, self._tof_cfg) if self._tof_cfg else None
        spaces = dict(self.observation_space.spaces)
        if self._tof is not None:
            spaces["tof"] = gym.spaces.Box(0.0, 1.0, shape=(self._tof.size,), dtype=_default_dtype)
        if self._scan_cfg:
            n_scan = int(self._scan_cfg["x"][2]) * int(self._scan_cfg["y"][2])
            spaces["height_scan"] = gym.spaces.Box(-1.0, 1.0, shape=(n_scan,), dtype=_default_dtype)
        self.observation_space = gym.spaces.Dict(spaces)

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
        
        # Optional per-episode difficulty: amplitude ~ U(lo, hi) (Perlin only)
        extra = {}
        amp_range = terrain_config_dict.get("amplitude_range")
        if amp_range:
            self.last_amplitude = float(self._np_random.uniform(*amp_range))
            extra["amplitude"] = self.last_amplitude

        # Generic per-episode parameter ranges, e.g. {"step_height_m": [0.0, 0.04]}
        for name, (lo, hi) in (terrain_config_dict.get("ranges") or {}).items():
            extra[name] = float(self._np_random.uniform(lo, hi))
        self.last_terrain_params = dict(extra)

        # Generate terrain with seed
        self.model.hfield_data = terrain_gen(nrows, seed=r_seed, **extra)

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

    def _load_model(self):
        """(Re)build model, data and stepper from self.xml_path and the current actuator/damping/armature settings."""
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        if self._integrator == "implicitfast":
            self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self._stepper = ContactFrameStepper(self.model, fix_frames=self.fix_wheel_contacts)
        # The actuators are torque motors: "max_wheel_velocity" scales the commanded torque [N m]
        # (the XML hard-codes +-10), so widen the control range when asking for more torque.
        self.model.actuator_ctrlrange[:] = np.array([-1.0, 1.0]) * max(self.max_wheel_velocity, 1e-6)
        if self._wheel_armature is not None:   # rotor inertia override (drive design studies)
            for k in range(3):
                self.model.dof_armature[self.model.joint(f"wheel_joint_{k}").dofadr[0]] = float(self._wheel_armature)
        if self.wheel_damping is not None:
            for k in range(3):
                self.model.dof_damping[self.model.joint(f"wheel_joint_{k}").dofadr[0]] = float(self.wheel_damping)
        self._nom = dict(mass=self.model.body_mass.copy(), damping=self.model.dof_damping.copy(),
                         pair_friction=self.model.pair_friction.copy())
        self._lqr = None
        self._wheel_dofs = np.array([self.model.joint(f"wheel_joint_{k}").dofadr[0] for k in range(3)])

    def _pick_morphology(self):
        """Morphology randomisation: draw a design from the pool at reset and rebuild the model when it changes."""
        i = int(self._np_random.integers(len(self._morph_pool)))
        if i == self._morph_idx:
            return
        sp = self._morph_pool[i]
        self._morph_idx = i
        self.xml_path = str(sp["xml"])
        self.max_wheel_velocity = float(sp["tau"])
        self.wheel_damping = float(sp["damp"])
        self._wheel_armature = float(sp["armature"])
        self._env_settings_lqr_json = sp.get("json")
        self._morph_ctx = np.array(sp["ctx"], dtype=np.float32)
        self._load_model()

    def _randomize(self):
        """Sample physical parameters for this episode around the nominal model (domain randomisation)."""
        r, dr, nom, m = self._np_random, self.dr, self._nom, self.model
        m.body_mass[:] = nom["mass"]
        if dr.get("mass"):
            for b in ("base", "ball"):
                i = m.body(b).id
                m.body_mass[i] = nom["mass"][i] * (1 + r.uniform(-dr["mass"], dr["mass"]))
        m.pair_friction[:] = nom["pair_friction"]
        if dr.get("friction"):
            m.pair_friction[:, 1] = nom["pair_friction"][:, 1] * (1 + r.uniform(-dr["friction"], dr["friction"]))
        m.dof_damping[:] = nom["damping"]
        if dr.get("damping"):
            for k in range(3):
                # the range is specified for the nominal robot (centre 0.1 N m s/rad); scale it with this design's nominal damping
                a = m.joint(f"wheel_joint_{k}").dofadr[0]
                m.dof_damping[a] = r.uniform(*dr["damping"]) * nom["damping"][a] / 0.1
        self._tq_scale = 1 + r.uniform(-dr.get("torque", 0.0), dr.get("torque", 0.0))
        self._prev_act = None
        self._push_left = 0
        self._push_t = int(r.uniform(*dr.get("push_every", [2.0, 4.0])) * 100)

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

        if self._morph_pool:
            self._pick_morphology()
        # Reset goal direction and reward function
        self._reset_goal_and_reward_objs()
        if self.command_mode:
            self._sample_cmd()
        if self.goal_mode:
            phi = np.radians(self._np_random.uniform(-self.goal_bearing_deg, self.goal_bearing_deg))
            dist = self._np_random.uniform(*self.goal_dist_range)
            self.goal_dir0 = np.array([np.sin(phi), np.cos(phi)])
            self.goal_xy = dist * self.goal_dir0

        if self.dr:
            self._randomize()

        # Reset episode counters
        self.step_counter = 0
        self._prev_cmd = None
        self._vint = np.zeros(2)
        self._tau_act = np.zeros(3)
        self.prev_data_time = 0

        # Generate new terrain and get initial height offset
        init_robot_height_offset = self._reset_terrain()

        # Reset MuJoCo simulation data
        mujoco.mj_resetData(self.model, self.data)
        
        # Apply height offset to base and ball joints
        # This positions the robot on the terrain surface
        self.data.joint("base_free_joint").qpos[2] += init_robot_height_offset
        self.data.joint("ball_free_joint").qpos[2] += init_robot_height_offset
        if self.init_yaw_random:   # uniform starting heading so world-frame commands are trained at every yaw
            yaw0 = self._np_random.uniform(-np.pi, np.pi)
            self.data.joint("base_free_joint").qpos[3:7] = [np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2)]
        
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

        self._last_obs = obs
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

    def _corrupt_depth(self, img):
        """Apply optional noise / dropout / occlusion to the depth channel (last channel)."""
        if not (self.depth_noise_std or self.depth_dropout or self.depth_occlusion):
            return img
        rng = self._np_random
        depth = img[..., -1]
        if self.depth_noise_std:
            depth = depth + rng.normal(0.0, self.depth_noise_std, depth.shape)
        if self.depth_dropout:
            depth = np.where(rng.random(depth.shape) < self.depth_dropout, 1.0, depth)
        if self.depth_occlusion:
            h, w = depth.shape
            frac = np.sqrt(self.depth_occlusion)
            bh, bw = max(1, int(h * frac)), max(1, int(w * frac))
            y0 = int(rng.integers(0, h - bh + 1))
            x0 = int(rng.integers(0, w - bw + 1))
            depth[y0:y0 + bh, x0:x0 + bw] = 0.0  # near-range occluder
        img[..., -1] = np.clip(depth, 0.0, 1.0)
        return img

    def _yaw(self):
        w, x, y, z = self.data.xquat[self.model.body("base").id]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def _heading_sincos(self):
        yaw = self._yaw()
        return np.array([np.sin(yaw), np.cos(yaw)])

    def _sample_cmd(self):
        if self.cmd_fixed is not None:
            self.cmd = np.asarray(self.cmd_fixed, dtype=float)
            return
        rng = self._np_random
        if rng.random() < self.cmd_zero_prob:
            self.cmd = np.zeros(3)
            return
        sp = rng.uniform(self.cmd_vmin, self.cmd_vmax)
        ang = rng.uniform(*self.cmd_ang_range)          # world-frame direction; pi/2 = +y
        self.cmd = np.array([sp * np.cos(ang), sp * np.sin(ang), rng.uniform(-self.cmd_wmax, self.cmd_wmax)])

    def _height_scan(self, position):
        """Terrain heights [m] relative to the terrain under the robot, on a grid around/ahead of it
        (privileged information for teacher policies), clipped to +-10 cm and scaled to [-1, 1]."""
        n = self.model.hfield_nrow.item()
        half = self.model.hfield_size[0, 0]
        z = self.model.hfield_data.reshape(n, n) * self.model.hfield_size[0, 2]

        def idx(v):
            return np.clip(np.round((v + half) / (2 * half) * (n - 1)).astype(int), 0, n - 1)

        xs = np.linspace(*self._scan_cfg["x"][:2], int(self._scan_cfg["x"][2])) + position[0]
        ys = np.linspace(*self._scan_cfg["y"][:2], int(self._scan_cfg["y"][2])) + position[1]
        ref = z[idx(np.array(position[1])), idx(np.array(position[0]))]
        h = z[np.ix_(idx(ys), idx(xs))].T  # hfield rows = y, cols = x; scan is [x, y]
        return np.clip((h - ref).ravel() / 0.1, -1.0, 1.0).astype(_default_dtype)

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
                rgbd_0 = self._corrupt_depth(
                    self.rgbd_inputs(self.data, "cam_0").astype(_default_dtype))
                rgbd_1 = self._corrupt_depth(
                    self.rgbd_inputs(self.data, "cam_1").astype(_default_dtype))
                
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
        # MuJoCo's cvel is (rotational:3, translational:3). The original code had the two
        # halves swapped, so the "directional" reward was paying for angular velocity.
        angular_vel = cvel[:3].copy().astype(_default_dtype)  # Angular velocity in world frame [rad/s]
        linear_vel = cvel[3:].copy().astype(_default_dtype)  # Linear velocity in world frame [m/s]
        
        # Clip velocities to reasonable ranges for observations
        vel = np.clip(linear_vel, a_min=-self.obs_vel_clip, a_max=self.obs_vel_clip)
        angular_vel = np.clip(angular_vel, a_min=-self.obs_vel_clip, a_max=self.obs_vel_clip)

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

        if self._tof is not None:
            obs["tof"] = self._tof(self.data, self._np_random, self._tof_extra_noise, self._tof_extra_dropout)
        if self._scan_cfg:
            obs["height_scan"] = self._height_scan(position)

        if self.command_mode:
            obs["cmd"] = self.cmd.astype(_default_dtype)
            obs["heading"] = self._heading_sincos().astype(_default_dtype)
            if self.obs_vint:
                # integral of the planar velocity error (integral-action state for the policy), clipped
                self._vint = np.clip(self._vint + (self.cmd[:2] - obs["vel"][:2]) * 0.01, -1.0, 1.0)
                obs["vint"] = self._vint.astype(_default_dtype)
            if self.hybrid_gate:
                obs["lqr_act"] = self._lqr_act.astype(_default_dtype)
            if self._obs_morph:
                obs["morph"] = self._morph_ctx.astype(_default_dtype)

        if self.goal_mode:
            if self.goal_mode == "relative":
                g = np.clip((self.goal_xy - position[:2]) / 3.0, -1.5, 1.5)
            else:
                g = self.goal_dir0
            obs["goal"] = np.asarray(g, dtype=_default_dtype)

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
            "pos2d": position[:-1],  # 2D position [x, y] (exclude height z)
            "progress": float(np.dot(position[:-1], getattr(self, "goal_2d", [0.0, 1.0]))),
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
        if self.residual_lqr or self.hybrid_gate:
            if self._lqr is None:
                from ballbot_gym.controllers.lqr import LQRPolicy
                _mj = self._env_settings_lqr_json
                self._lqr = LQRPolicy(self, model_json=_mj) if _mj else LQRPolicy(self)
            o = dict(self._last_obs)
            o["cmd"] = np.asarray(self.cmd, dtype=float)     # current command (may be set externally)
            base_act, _ = self._lqr.predict(o)
            base_act = np.asarray(base_act, dtype=float)
            self._lqr_act = np.clip(base_act, -1, 1).astype(np.float32)
            if self.hybrid_gate:
                oc = np.asarray(omniwheel_commands, dtype=float)
                g = 0.5 * (1.0 + np.clip(oc[3], -1.0, 1.0))
                self._gate = float(g)
                omniwheel_commands = np.clip((1.0 - g) * base_act + g * oc[:3], -1.0, 1.0)
            else:
                omniwheel_commands = np.clip(base_act + self.residual_scale * np.asarray(omniwheel_commands), -1.0, 1.0)
        applied = omniwheel_commands
        if self.dr:
            if self._prev_act is not None and self._np_random.random() < self.dr.get("latency_p", 0.0):
                applied = self._prev_act          # this action arrives one control step late
            self._prev_act = np.array(omniwheel_commands, copy=True)
        ctrl = applied * self.max_wheel_velocity * self._tq_scale
        ctrl = np.clip(ctrl, a_min=-self.max_wheel_velocity, a_max=self.max_wheel_velocity)  # Safety: prevent extreme commands

        # Apply control commands (negative for correct coordinate system)
        self.data.ctrl[:] = -ctrl
        
        # Step physics simulation forward
        # Suppress warnings about objective convexity (benign MuJoCo warnings)
        if self.dr and self.dr.get("push_force"):
            self._push_t -= 1
            if self._push_t <= 0 and self._push_left <= 0:
                ang = self._np_random.uniform(0, 2 * np.pi)
                self._push_f = self._np_random.uniform(*self.dr["push_force"]) * np.array([np.cos(ang), np.sin(ang), 0.0])
                self._push_left = max(1, int(round(self.dr.get("push_dur", 0.05) * 100)))
                self._push_t = int(self._np_random.uniform(*self.dr.get("push_every", [2.0, 4.0])) * 100)
            if self._push_left > 0:
                self.data.xfrc_applied[self.model.body("base").id, :3] = self._push_f
                self._push_left -= 1
        mm = self._motor
        for _ in range(self.action_repeat):
            if mm:
                # realistic actuator: first-order current lag, torque-speed curve (back-EMF), Coulomb friction
                w = self.data.qvel[self._wheel_dofs]
                cmd = -ctrl
                self._tau_act += (cmd - self._tau_act) * (self.model.opt.timestep / (mm.get("lag", 0.002) + self.model.opt.timestep))
                tlim = mm.get("tau_stall", self.max_wheel_velocity) * np.clip(1.0 - np.abs(w) / mm.get("omega_nl", 1e9), 0.0, 1.0)
                tau = np.where(self._tau_act * w > 0, np.clip(self._tau_act, -tlim, tlim), self._tau_act)
                tau = tau - mm.get("fc", 0.0) * np.tanh(w / 0.1)
                self.data.ctrl[:] = np.clip(tau, -self.max_wheel_velocity, self.max_wheel_velocity)
            self._stepper.step(self.data)

        # Reset applied forces (MuJoCo applies forces until explicitly reset)
        # This prevents forces from accumulating across steps
        self.data.xfrc_applied[self.model.body("base").id, :3] = np.zeros(3)

        # Get current observation and info
        obs = self._get_obs(omniwheel_commands.astype(_default_dtype))
        if self.dr and self.dr.get("obs_noise"):
            sd = float(self.dr["obs_noise"])
            for k, f in (("vel", 1.0), ("angular_vel", 2.0), ("orientation", 0.25)):
                obs[k] = (obs[k] + self._np_random.normal(0, sd * f, size=obs[k].shape)).astype(obs[k].dtype)
        info = self._get_info()
        terminated = False
        truncated = False  # Not used (reserved for time limits)

        # ========== REWARD COMPUTATION ==========
        
        # 1. Directional reward: velocity component in target direction
        # This encourages movement toward the goal direction
        # Scaled by reward_scale (default: 0.01 = /100) to keep rewards in reasonable range
        if self.command_mode:
            if self.step_counter > 0 and self.step_counter % self.cmd_resample_steps == 0:
                self._sample_cmd()
                obs["cmd"] = self.cmd.astype(_default_dtype)
            ev = float(np.linalg.norm(obs["vel"][:2] - self.cmd[:2]))
            ew = float(obs["angular_vel"][2] - self.cmd[2])
            # dense linear terms give a gradient far from the command (standing still no longer plateaus),
            # exponential terms reward precision near it
            if self.cmd_reward == "err":
                # zero baseline at standing (cn - |v - c|), positive slope all the way to the command, plus a
                # precision bonus near it; overshoot is penalised symmetrically
                v = obs["vel"][:2].astype(float)
                c = self.cmd[:2]
                cn = float(np.linalg.norm(c))
                e = float(np.linalg.norm(v - c))
                u = c / cn if cn > 0.05 else np.zeros(2)
                p = float(v @ u) if cn > 0.05 else 0.0
                wz, cw = float(obs["angular_vel"][2]), float(self.cmd[2])
                ew_ = abs(wz - cw)
                raw_reward = (cn - e) + 0.5 * np.exp(-e ** 2 / 0.25 ** 2) + 0.5 * (abs(cw) - ew_) + 0.25 * np.exp(-ew_ ** 2 / 0.5 ** 2)
                if self.lean_coef and cn > 0.05:
                    _g = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "the_ball")
                    _rel = self.data.xpos[self.model.body("base").id] - self.data.geom_xpos[_g]
                    lean = float(_rel[:2] @ u) / max(float(_rel[2]), 1e-3)
                    raw_reward += self.lean_coef * (1.0 - min(max(p / cn, 0.0), 1.0)) * float(np.clip(lean / 0.3, -1.0, 1.0))
            elif self.cmd_reward == "dot":
                # zero-baseline projection reward (standing still earns 0), as in the directional task
                v = obs["vel"][:2].astype(float)
                c = self.cmd[:2]
                cn = float(np.linalg.norm(c))
                if cn > 0.05:
                    u = c / cn
                    p = float(v @ u)
                    raw_t = min(p, cn) - 0.5 * float(np.linalg.norm(v - p * u))
                else:
                    raw_t = -float(np.linalg.norm(v))
                wz, cw = float(obs["angular_vel"][2]), float(self.cmd[2])
                raw_w = min(np.sign(cw) * wz, abs(cw)) if abs(cw) > 0.05 else -0.5 * abs(wz)
                raw_reward = raw_t + 0.5 * raw_w
                if self.lean_coef and cn > 0.05:
                    # teach acceleration: reward leaning the base ahead of the ball along the command while below
                    # the commanded speed (fades out as the speed is reached, where steady lean is zero)
                    _g = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "the_ball")
                    _rel = self.data.xpos[self.model.body("base").id] - self.data.geom_xpos[_g]
                    lean = float(_rel[:2] @ u) / max(float(_rel[2]), 1e-3)
                    raw_reward += self.lean_coef * (1.0 - min(max(p / cn, 0.0), 1.0)) * float(np.clip(lean / 0.3, -1.0, 1.0))
            else:
                raw_reward = float((1.0 - ev / self.cmd_vmax) + (1.0 - abs(ew) / max(self.cmd_wmax, 1.0))
                                   + 0.5 * np.exp(-ev ** 2 / self.cmd_sigma_v ** 2)
                                   + 0.5 * np.exp(-ew ** 2 / self.cmd_sigma_w ** 2))
            info["vel_err"], info["yaw_err"] = ev, ew
            info["speed"] = float(np.linalg.norm(obs["vel"][:2]))
            info["yaw_rate"] = float(obs["angular_vel"][2])
            info["cmd"] = self.cmd.copy()
        elif self.goal_mode:
            if self.goal_mode == "relative":
                vec = self.goal_xy - info["pos2d"]
                heading = vec / (np.linalg.norm(vec) + 1e-6)
            else:
                heading = self.goal_dir0
            raw_reward = float(np.dot(obs["vel"][:2], heading))
        else:
            raw_reward = self.reward_obj(obs)
        if self.reward_cap is not None:
            raw_reward = min(raw_reward, self.reward_cap)  # cap speed reward: no incentive to lunge
        reward = raw_reward * self.reward_scale * self.action_repeat
        self.reward_term_1_hist.append(reward)

        # 2. Action regularization: penalize large actions
        # This encourages energy-efficient control and smooth motions
        # The L2 norm squared gives quadratic penalty
        action_regularization = (self.action_reg_coef * self.action_repeat *
                                 (np.linalg.norm(omniwheel_commands)**2))
        self.reward_term_2_hist.append(action_regularization)
        reward += action_regularization

        # 2b. Smoothness: penalize change between consecutive commands (squared L2)
        cmd_now = np.asarray(omniwheel_commands, dtype=float).ravel()
        prev = getattr(self, "_prev_cmd", None)
        act_rate = 0.0 if prev is None else float(np.sum((cmd_now - prev) ** 2))
        self._prev_cmd = cmd_now
        info["act_rate"] = act_rate
        reward += self.action_rate_coef * self.action_repeat * act_rate

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
                print(f"truncated. Cause: {self.step_counter} >= {self.max_ep_steps}")
            truncated = True  # time limit (bootstrapped by SB3 via TimeLimit.truncated)

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

        ball_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "the_ball")
        rel = self.data.xpos[self.model.body("base").id] - self.data.geom_xpos[ball_gid]
        off_ball_deg = float(np.degrees(np.arctan2(np.hypot(rel[0], rel[1]), max(rel[2], 1e-6))))
        info["off_ball_deg"] = off_ball_deg
        off_ball = self.max_ball_offset_deg is not None and off_ball_deg > self.max_ball_offset_deg
        if self.max_ball_offset_deg and self.ball_offset_coef:
            reward -= self.ball_offset_coef * self.action_repeat * (off_ball_deg / self.max_ball_offset_deg) ** 2

        if angle_in_degrees > self.max_allowed_tilt or off_ball:
            # Robot has fallen (or left the ball): terminate episode
            if self.verbose:
                print(
                    f"failure after {self.step_counter} steps. "
                    f"Reason: tilt_angle ({angle_in_degrees:.1f}°) > {self.max_allowed_tilt}°"
                )
            info["success"] = False
            info["failure"] = True
            terminated = True
            reward -= self.fall_penalty
        else:
            # Robot is still upright: add survival bonus
            reward += self.survival_bonus * self.action_repeat

        # Accumulate episode return (for logging/analysis)
        gamma = 1.0  # Discount factor (not used algorithmically, but tracked)
        self.G_tau += (gamma**self.step_counter) * reward
        
        # Save logs if episode ended
        if self.goal_mode:
            info["goal_dist"] = float(np.linalg.norm(self.goal_xy - info["pos2d"]))
            if not info["failure"] and info["goal_dist"] <= self.goal_radius:
                info["success"] = True
                truncated = True
        if (self.goal_progress is not None and not info["failure"]
                and info["progress"] >= self.goal_progress):
            info["success"] = True
            truncated = True
        truncated = truncated and not terminated

        if terminated or truncated:
            if self.verbose:
                print(colored(
                    f"Episode ended. Return: {self.G_tau:.2f}, Steps: {self.step_counter}",
                    "magenta",
                    attrs=["bold"]
                ))
            self._save_logs()

        self._last_obs = obs
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


