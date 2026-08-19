"""Model architectures, forensic branches, auxiliary heads, and fusion modules."""
from .backbones import FeatureEncoder, HypersphericalProjector
from .detector import AdaptiveFusionDetector
from .losses.hyperspherical import HypersphericalCosineMarginLoss
from .losses.multi_task_loss import MultiTaskForensicLoss

__all__ = [
    "FeatureEncoder",
    "HypersphericalProjector",
    "AdaptiveFusionDetector",
    "HypersphericalCosineMarginLoss",
    "MultiTaskForensicLoss",
]
