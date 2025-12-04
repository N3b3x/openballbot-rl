"""
Example 2: Adding a Custom Reward Function

This example shows how to:
1. Create a custom reward function inheriting from BaseReward
2. Register it with the component registry
3. Use it in the environment via configuration

Run this example:
    python examples/02_custom_reward.py
"""
import numpy as np
import gymnasium as gym
import ballbot_gym
from ballbot_gym.rewards.base import BaseReward
from ballbot_gym.core.registry import ComponentRegistry
from ballbot_gym.core.factories import create_reward


class VelocityMagnitudeReward(BaseReward):
    """
    Custom reward that encourages high velocity magnitude.
    
    This is a simple example reward that rewards the agent for moving fast,
    regardless of direction. Useful for testing or as a component in a
    composite reward function.
    """
    
    def __init__(self, scale: float = 0.1):
        """
        Initialize VelocityMagnitudeReward.
        
        Args:
            scale: Scaling factor for velocity reward
        """
        self.scale = scale
    
    def __call__(self, state: dict) -> float:
        """
        Compute reward based on velocity magnitude.
        
        Args:
            state: Observation dictionary containing "vel" key
            
        Returns:
            Reward proportional to velocity magnitude
        """
        velocity = state["vel"][:2]  # 2D velocity (x, y)
        velocity_magnitude = np.linalg.norm(velocity)
        return self.scale * velocity_magnitude


def main():
    """Demonstrate custom reward usage."""
    print("=" * 60)
    print("Example: Custom Reward Function")
    print("=" * 60)
    
    # Step 1: Register the custom reward
    print("\n1. Registering custom reward...")
    ComponentRegistry.register_reward("velocity_magnitude", VelocityMagnitudeReward)
    print(f"   ✓ Registered 'velocity_magnitude' reward")
    print(f"   Available rewards: {ComponentRegistry.list_rewards()}")
    
    # Step 2: Create reward using factory
    print("\n2. Creating reward from config...")
    reward_config = {
        "type": "velocity_magnitude",
        "config": {"scale": 0.1}
    }
    reward = create_reward(reward_config)
    print(f"   ✓ Created reward: {type(reward).__name__}")
    
    # Step 3: Test reward function
    print("\n3. Testing reward function...")
    test_state = {
        "vel": np.array([0.5, 0.3, 0.0])  # 2D velocity [x, y, z]
    }
    reward_value = reward(test_state)
    print(f"   Test state velocity: {test_state['vel'][:2]}")
    print(f"   Reward value: {reward_value:.4f}")
    
    # Step 4: Use in environment (via config)
    print("\n4. Using custom reward in environment...")
    env = gym.make(
        "ballbot-v0.1",
        GUI=False,
        terrain_type="flat",
        reward_config=reward_config,
        max_ep_steps=100
    )
    
    obs, _ = env.reset(seed=42)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"   ✓ Environment created with custom reward")
    print(f"   Step reward: {reward:.4f}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nTo use this reward in training, add to config.yaml:")
    print("""
problem:
  reward:
    type: "velocity_magnitude"
    config:
      scale: 0.1
""")


if __name__ == "__main__":
    main()

