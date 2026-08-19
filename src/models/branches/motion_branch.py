"""
Motion and Spatiotemporal Forensic Branch.
Operates on dense optical flow fields (u, v, magnitude) or temporal frame differences.
Captures temporal discontinuities, unnatural blinking dynamics, warping jitter, and lip-sync anomalies.
"""

import torch
import torch.nn as nn
from ..backbones import FeatureEncoder


class MotionBranch(nn.Module):
    """Branch 4: Spatiotemporal Dense Optical Flow Forensic Stream."""

    def __init__(
        self,
        in_channels: int = 3, # [u, v, magnitude] or HSV flow map
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = False,
        feature_dim: int = 256,
        use_hyperspherical: bool = True
    ):
        super().__init__()
        self.encoder = FeatureEncoder(
            in_channels=in_channels,
            backbone_name=backbone_name,
            pretrained=pretrained,
            feature_dim=feature_dim,
            use_hyperspherical=use_hyperspherical
        )
        self.feature_dim = feature_dim

    def forward(self, motion_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_tensor: (B, 3, H, W)
        Returns:
            (B, feature_dim) normalized hyperspherical feature embedding
        """
        return self.encoder(motion_tensor)
