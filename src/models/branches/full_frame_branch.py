"""
Full-Frame Forensic Branch.
Operates on the entire uncropped scene (RGB) to capture global lighting inconsistencies,
head-to-body scale/orientation mismatches, and background compositing flaws that face-only detectors miss.
"""

import torch
import torch.nn as nn
from ..backbones import FeatureEncoder


class FullFrameBranch(nn.Module):
    """Branch 2: Global Full-Frame Forensic Stream."""

    def __init__(
        self,
        in_channels: int = 3,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
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

    def forward(self, full_frame_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            full_frame_tensor: (B, 3, H, W)
        Returns:
            (B, feature_dim) normalized hyperspherical feature embedding
        """
        return self.encoder(full_frame_tensor)
