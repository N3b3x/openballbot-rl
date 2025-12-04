"""
Example 4: Adding a Custom Policy Architecture

This example shows how to:
1. Create a custom policy feature extractor inheriting from BaseFeaturesExtractor
2. Register it with the component registry
3. Use it in training via configuration

Run this example:
    python examples/04_custom_policy.py
"""
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from ballbot_gym.core.registry import ComponentRegistry


class CustomPolicyExtractor(BaseFeaturesExtractor):
    """
    Custom policy feature extractor with different architecture.
    
    This is a simple example policy that uses a different neural network
    architecture than the default MLP policy. Useful for experimentation
    or when you need a specific architecture for your task.
    
    Architecture:
    - Proprioceptive features: 2-layer MLP
    - Visual features: 2-layer MLP (if depth encoder provided)
    - Fusion: Concatenation + 1-layer MLP
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 64,
        hidden_size: int = 128,
        **kwargs
    ):
        """
        Initialize CustomPolicyExtractor.
        
        Args:
            observation_space: Gymnasium Dict observation space
            features_dim: Dimension of output features
            hidden_size: Hidden layer size
            **kwargs: Additional arguments (e.g., frozen_encoder_path)
        """
        super().__init__(observation_space, features_dim)
        
        self.hidden_size = hidden_size
        
        # Extract observation dimensions
        proprioceptive_dim = observation_space["proprioceptive"].shape[0]
        
        # Check if depth encoder is provided
        self.use_depth = "depth" in observation_space.spaces
        if self.use_depth:
            # If frozen encoder path provided, load it
            # Otherwise, assume depth features are already extracted
            depth_dim = observation_space["depth"].shape[0] if "depth" in observation_space.spaces else 0
        else:
            depth_dim = 0
        
        # Proprioceptive feature extractor
        self.proprioceptive_net = nn.Sequential(
            nn.Linear(proprioceptive_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU()
        )
        
        # Visual feature extractor (if depth available)
        if self.use_depth and depth_dim > 0:
            self.visual_net = nn.Sequential(
                nn.Linear(depth_dim, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LeakyReLU()
            )
            fusion_input_dim = (hidden_size // 2) * 2
        else:
            self.visual_net = None
            fusion_input_dim = hidden_size // 2
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.LeakyReLU()
        )
    
    def forward(self, observations: dict) -> torch.Tensor:
        """
        Forward pass through the policy network.
        
        Args:
            observations: Dictionary containing observation tensors
                - "proprioceptive": Proprioceptive features [batch, dim]
                - "depth": Depth features [batch, dim] (optional)
                
        Returns:
            Combined features [batch, features_dim]
        """
        # Extract proprioceptive features
        proprio_features = self.proprioceptive_net(observations["proprioceptive"])
        
        # Extract visual features (if available)
        if self.use_depth and self.visual_net is not None and "depth" in observations:
            visual_features = self.visual_net(observations["depth"])
            # Concatenate features
            combined = torch.cat([proprio_features, visual_features], dim=-1)
        else:
            combined = proprio_features
        
        # Fusion
        features = self.fusion_net(combined)
        
        return features


def main():
    """Demonstrate custom policy usage."""
    print("=" * 60)
    print("Example: Custom Policy Architecture")
    print("=" * 60)
    
    # Step 1: Register the custom policy
    print("\n1. Registering custom policy...")
    ComponentRegistry.register_policy("custom", CustomPolicyExtractor)
    print(f"   ✓ Registered 'custom' policy")
    print(f"   Available policies: {ComponentRegistry.list_policies()}")
    
    # Step 2: Create policy using factory
    print("\n2. Creating policy from config...")
    policy_config = {
        "type": "custom",
        "config": {
            "features_dim": 64,
            "hidden_size": 128
        }
    }
    
    # Note: In practice, policies are created by Stable-Baselines3
    # This is just for demonstration
    print(f"   ✓ Policy config created")
    print(f"   Config: {policy_config}")
    
    # Step 3: Test policy architecture
    print("\n3. Testing policy architecture...")
    
    # Create mock observation space
    observation_space = gym.spaces.Dict({
        "proprioceptive": gym.spaces.Box(low=-1, high=1, shape=(12,)),
        "depth": gym.spaces.Box(low=0, high=1, shape=(20,))  # Encoded depth features
    })
    
    # Create policy extractor
    policy_extractor = CustomPolicyExtractor(
        observation_space=observation_space,
        features_dim=64,
        hidden_size=128
    )
    
    # Test forward pass
    batch_size = 4
    test_observations = {
        "proprioceptive": torch.randn(batch_size, 12),
        "depth": torch.randn(batch_size, 20)
    }
    
    with torch.no_grad():
        features = policy_extractor(test_observations)
    
    print(f"   Input proprioceptive shape: {test_observations['proprioceptive'].shape}")
    print(f"   Input depth shape: {test_observations['depth'].shape}")
    print(f"   Output features shape: {features.shape}")
    print(f"   ✓ Policy forward pass successful")
    
    # Step 4: Show how to use in training
    print("\n4. Using custom policy in training...")
    print("   To use this policy in training, add to config.yaml:")
    print("""
problem:
  policy:
    type: "custom"
    config:
      features_dim: 64
      hidden_size: 128

algo:
  name: ppo
  policy_kwargs:
    features_extractor_class: CustomPolicyExtractor
    features_extractor_kwargs:
      features_dim: 64
      hidden_size: 128
""")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nNote: Policies are typically used through Stable-Baselines3")
    print("during training. See examples/05_training_workflow.py for")
    print("a complete training example.")


if __name__ == "__main__":
    main()

