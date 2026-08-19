"""Loss functions for deepfake classification, auxiliary tasks, and hyperspherical manifold regularization."""
from .hyperspherical import HypersphericalCosineMarginLoss
from .multi_task_loss import MultiTaskForensicLoss

__all__ = [
    "HypersphericalCosineMarginLoss",
    "MultiTaskForensicLoss",
]
