"""
Frequency / Noise Domain Forensic Branch.
Operates on 2D Discrete Wavelet Transform (DWT subbands: LL, LH, HL, HH) or FFT magnitude spectra.
Captures spectral fingerprints, checkerboard upsampling artifacts, and high-frequency GAN/diffusion traces.
"""

import torch
import torch.nn as nn
from ..backbones import FeatureEncoder


class FrequencyBranch(nn.Module):
    """Branch 3: Wavelet and Noise Residual Forensic Stream."""

    def __init__(
        self,
        in_channels: int = 4, # 4-channel DWT [LL, LH, HL, HH]
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = False, # Specialized spectral filters
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

    def forward(self, freq_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            freq_tensor: (B, 4, H, W)
        Returns:
            (B, feature_dim) normalized hyperspherical feature embedding
        """
        return self.encoder(freq_tensor)
