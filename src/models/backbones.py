"""
Backbone encoders and hyperspherical manifold projection heads.
Implements GenD-style hyperspherical normalization (L2 feature normalization) to constrain
representation spaces and prevent source-specific overfitting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class HypersphericalProjector(nn.Module):
    """Projects high-dimensional feature embeddings onto a unit hypersphere S^(d-1)."""

    def __init__(self, in_features: int, out_features: int = 256, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features)
        )

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Projects and optionally L2-normalizes embeddings.
        Args:
            x: (B, in_features)
            normalize: if True, projects onto S^(d-1)
        Returns:
            (B, out_features) unit-norm vectors
        """
        proj = self.projection(x)
        if normalize:
            proj = F.normalize(proj, p=2, dim=-1, eps=1e-8)
        return proj


class FeatureEncoder(nn.Module):
    """
    Modular per-branch backbone encoder.
    Supports timm pretrained models or custom lightweight convolutional backbones.
    """

    def __init__(
        self,
        in_channels: int = 3,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        feature_dim: int = 256,
        use_hyperspherical: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.use_hyperspherical = use_hyperspherical

        # Build backbone
        self.encoder, raw_dim = self._build_backbone(backbone_name, pretrained, in_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Hyperspherical projector
        self.projector = HypersphericalProjector(raw_dim, feature_dim)

    def _build_backbone(self, name: str, pretrained: bool, in_channels: int) -> Tuple[nn.Module, int]:
        try:
            import timm
            # Try loading timm backbone
            model = timm.create_model(
                name,
                pretrained=pretrained,
                in_chans=in_channels,
                num_classes=0 # removes classification head
            )
            raw_dim = model.num_features
            return model, raw_dim
        except Exception:
            # Robust fallback CNN when timm or internet weights are unavailable
            return self._build_custom_cnn(in_channels)

    def _build_custom_cnn(self, in_channels: int) -> Tuple[nn.Module, int]:
        """Lightweight ConvNet backbone for rapid CPU execution and self-contained tests."""
        cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU()
        )
        return cnn, 512

    def forward(self, x: torch.Tensor, normalize: Optional[bool] = None) -> torch.Tensor:
        """
        Forward pass extracting L2-normalized hyperspherical feature embedding.
        Args:
            x: (B, C, H, W)
        Returns:
            (B, feature_dim)
        """
        if normalize is None:
            normalize = self.use_hyperspherical

        feat_map = self.encoder(x)
        if feat_map.ndim == 4:
            pooled = self.pool(feat_map).flatten(1)
        else:
            pooled = feat_map.flatten(1)
            
        proj = self.projector(pooled, normalize=normalize)
        return proj
