"""
ballbotgym
==========

Module for registering the Ballbot custom reinforcement learning environment
with Gymnasium. This enables usage via gymnasium.make("ballbot-v0.1").

The Ballbot environment simulates a dynamically balanced robot
with omnidirectional wheels and supports both standard continuous control 
and rich (optionally RGB-D) visual observations.

Environment Registration Details:
---------------------------------
- id: "ballbot-v0.1"
- entry_point: "ballbotgym.bbot_env:BBotSimulation"
- assets: Uses embedded MuJoCo model XML ("bbot.xml") from the assets package
- kwargs: Passes the default path to the MuJoCo XML file as "xml_path" argument

Usage Example:
--------------
    import gymnasium as gym
    import ballbotgym  # noqa: F401 (registers the Ballbot env)

    env = gym.make("ballbot-v0.1")
    obs, info = env.reset()
    # ... interact using the Gymnasium API

References:
-----------
- For environment implementation: ballbotgym/ballbotgym/bbot_env.py
- For MuJoCo model XML: ballbotgym/assets/bbot.xml
"""

from gymnasium.envs.registration import register
import importlib.resources

# Obtain the path to the embedded MuJoCo XML model for the Ballbot
with importlib.resources.path("ballbotgym.assets", "bbot.xml") as path:
    _xml_path = str(path)

# Register the Ballbot environment with Gymnasium under a unique ID
register(
    id="ballbot-v0.1",
    entry_point="ballbotgym.bbot_env:BBotSimulation",
    kwargs={
        "xml_path": _xml_path
    }
)
