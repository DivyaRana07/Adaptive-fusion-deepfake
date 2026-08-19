"""
Face-Crop Forensic Branch.
Operates on tightly cropped, aligned facial regions (RGB) to capture local texture artifacts,
facial boundary seams, eye/mouth blending edges, and fine skin-pore inconsistencies.
"""

import torch
import torch.nn as nn
from ..backbones import FeatureEncoder


class FaceCropBranch(nn.Module):
    """Branch 1: Local Face-Crop Forensic Stream."""

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

    def forward(self, face_crop_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            face_crop_tensor: (B, 3, H, W)
        Returns:
            (B, feature_dim) normalized hyperspherical feature embedding
        """
        return self.encoder(face_crop_tensor)
