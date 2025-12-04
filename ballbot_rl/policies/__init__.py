"""Policy network architectures for ballbot RL."""
from ballbot_gym.core.registry import ComponentRegistry
from ballbot_rl.policies.mlp_policy import Extractor

# Auto-register components on import
# Note: Extractor is a feature extractor, not a full policy, but we register it
# for consistency with the component system
ComponentRegistry.register_policy("mlp", Extractor)

__all__ = ["Extractor"]

