"""Training module for model definition and training."""

from .model import HousingMLP
from .train import train_model

__all__ = ["HousingMLP", "train_model"]
