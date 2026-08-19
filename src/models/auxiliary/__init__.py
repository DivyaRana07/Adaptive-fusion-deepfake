"""Auxiliary self-supervised shift heads and domain uncertainty estimators."""
from .compression_head import CompressionLevelHead
from .blending_head import BlendingRatioHead
from .motion_head import MotionStabilityHead
from .shift_estimator import DomainShiftEstimator

__all__ = [
    "CompressionLevelHead",
    "BlendingRatioHead",
    "MotionStabilityHead",
    "DomainShiftEstimator",
]
