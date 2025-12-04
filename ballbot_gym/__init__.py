"""
ballbot_gym
===========

Module for registering the Ballbot custom reinforcement learning environment
with Gymnasium. This enables usage via gymnasium.make("ballbot-v0.1").

The Ballbot environment simulates a dynamically balanced robot
with omnidirectional wheels and supports both standard continuous control 
and rich (optionally RGB-D) visual observations.

Environment Registration Details:
---------------------------------
- id: "ballbot-v0.1"
- entry_point: "ballbot_gym.envs.ballbot_env:BBotSimulation"
- assets: Uses embedded MuJoCo model XML ("ballbot.xml") from the models package
- kwargs: Passes the default path to the MuJoCo XML file as "xml_path" argument

Usage Example:
--------------
    import gymnasium as gym
    import ballbot_gym  # noqa: F401 (registers the Ballbot env)

    env = gym.make("ballbot-v0.1")
    obs, info = env.reset()
    # ... interact using the Gymnasium API

References:
-----------
- For environment implementation: ballbot_gym/envs/ballbot_env.py
- For MuJoCo model XML: ballbot_gym/models/ballbot.xml
"""

from gymnasium.envs.registration import register
import importlib.resources

# Import terrain and reward modules to trigger component registration
# This ensures all terrains and rewards are registered before the environment is used
import ballbot_gym.terrain  # noqa: F401 - Registers all terrain generators
import ballbot_gym.rewards  # noqa: F401 - Registers all reward functions

# Obtain the path to the embedded MuJoCo XML model for the Ballbot
with importlib.resources.path("ballbot_gym.models", "ballbot.xml") as path:
    _xml_path = str(path)

# Register the Ballbot environment with Gymnasium under a unique ID
register(
    id="ballbot-v0.1",
    entry_point="ballbot_gym.envs.ballbot_env:BBotSimulation",
    kwargs={
        "xml_path": _xml_path
    }
)

