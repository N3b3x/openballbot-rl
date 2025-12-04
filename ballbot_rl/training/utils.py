import matplotlib.pyplot as plt
from typing import List, Optional
import numpy as np

from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

import ballbot_gym


def make_ballbot_env(terrain_type=None,
                     reward_config=None,
                     terrain_config=None,
                     env_config=None,
                     gui=False,
                     disable_cams=False,
                     seed=0,
                     log_options={
                         "cams": False,
                         "reward_terms": False
                     },
                     eval_env=False):
    """
    Factory function to create ballbot environment.
    
    Args:
        terrain_type: (deprecated) Terrain type as string ("perlin" or "flat")
        reward_config: Reward configuration dictionary with "type" and "config" keys
        terrain_config: Terrain configuration dictionary with "type" and "config" keys
        env_config: Environment configuration dictionary (camera, env, logging settings)
        gui: Whether to show GUI
        disable_cams: Whether to disable cameras
        seed: Random seed
        log_options: Logging options dictionary
        eval_env: Whether this is an evaluation environment
        
    Returns:
        Environment factory function
        
    Note:
        eval_env is just to ensure repeatability with stable_baselines3. During training,
        stable_baselines3 seeds the env at init (with base_seed+i where i is in [0,num_envs]),
        but it doesn't do so during eval (seed is always None, and so no _np_random is created).
        To ensure that we can reproduce the terrains used during eval, we pass a seed to the env
        which will then manually create _np_random.
        
        Therefore, it's not required to be set eval_env=True if you're testing a policy from
        outside stablebaseline3's framework.
    """
    from ballbot_gym.core.config import get_component_config
    
    # Backward compatibility: convert terrain_type string to config if needed
    if terrain_config is None and terrain_type is not None:
        terrain_config = {"type": terrain_type, "config": {}}
    elif terrain_config is None:
        terrain_config = {"type": "perlin", "config": {}}
    
    # Default reward config if not provided
    if reward_config is None:
        reward_config = {"type": "directional", "config": {"target_direction": [0.0, 1.0]}}

    def _init():
        env = gym.make(
            "ballbot-v0.1",
            GUI=gui,  #should be disabled in parallel training
            log_options=log_options,
            terrain_type=terrain_config.get("type", "perlin"),  # Backward compat
            reward_config=reward_config,
            terrain_config=terrain_config,
            env_config=env_config,  # Pass env config for camera/env settings
            eval_env=[eval_env, seed]
        )  #because stablebaselines's EvalCallback, in contrast with training, doesn't seed at the first iteration

        return Monitor(
            env
        )  #using a Monitor wrapper to enable logging rollout avg rewards

    return _init


def deg2rad(d):

    return d * np.pi / 180


def rad2deg(r):

    return r * 180 / np.pi
