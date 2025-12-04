"""
Example 1: Basic Environment Usage

This example demonstrates the minimal code needed to:
1. Create a ballbot environment
2. Reset and step through episodes
3. Use the environment with default settings

Run this example:
    python examples/01_basic_usage.py
"""
import gymnasium as gym
import ballbot_gym


def main():
    """Run basic environment example."""
    print("Creating ballbot environment...")
    
    # Create environment using Gymnasium
    env = gym.make(
        "ballbot-v0.1",
        GUI=False,  # Set to True to visualize (requires mjpython on macOS)
        terrain_type="flat",  # Start with flat terrain for simplicity
        max_ep_steps=1000
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset environment
    print("\nResetting environment...")
    obs, info = env.reset(seed=42)
    print(f"Initial observation keys: {obs.keys()}")
    print(f"Initial position: {info['pos2d']}")
    
    # Run a short episode
    print("\nRunning episode...")
    total_reward = 0.0
    for step in range(100):
        # Sample random action (in practice, use your policy)
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            print(f"Reason: {'terminated' if terminated else 'truncated'}")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Final position: {info['pos2d']}")
    
    # Clean up
    env.close()
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()

