"""Encoder models for visual observations."""
from ballbot_rl.encoders.models import TinyAutoencoder
from ballbot_rl.encoders.training import train_autoencoder

__all__ = ["TinyAutoencoder", "train_autoencoder"]
