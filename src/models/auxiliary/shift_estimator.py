"""
Domain Shift and Uncertainty Aggregator.
Combines predictive uncertainty metrics from the three auxiliary self-supervised heads
(compression, blending, motion stability) into a label-free domain shift vector u.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .compression_head import CompressionLevelHead
from .blending_head import BlendingRatioHead
from .motion_head import MotionStabilityHead


class DomainShiftEstimator(nn.Module):
    """Aggregates multi-cue auxiliary heads to generate label-free shift signals."""

    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 5,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.compression_head = CompressionLevelHead(feature_dim, num_classes, hidden_dim)
        self.blending_head = BlendingRatioHead(feature_dim, num_classes, hidden_dim)
        self.motion_head = MotionStabilityHead(feature_dim, num_classes, hidden_dim)

    def forward(
        self,
        feat_freq: torch.Tensor,
        feat_face: torch.Tensor,
        feat_motion: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            feat_freq: (B, feature_dim) frequency/noise features (used for compression estimation)
            feat_face: (B, feature_dim) face crop features (used for blending estimation)
            feat_motion: (B, feature_dim) motion features (used for motion stability estimation)
        Returns:
            shift_vector: (B, 3) domain shift vector u = [u_comp, u_blend, u_motion]
            aux_outputs: Dictionary containing logits and individual uncertainty scores
        """
        comp_logits, comp_unc = self.compression_head(feat_freq)
        blend_logits, blend_unc = self.blending_head(feat_face)
        motion_logits, motion_unc = self.motion_head(feat_motion)

        # Concatenate normalized uncertainties into domain shift descriptor u
        shift_vector = torch.cat([comp_unc, blend_unc, motion_unc], dim=-1) # (B, 3)

        aux_outputs = {
            "compression_logits": comp_logits,
            "blending_logits": blend_logits,
            "motion_logits": motion_logits,
            "compression_uncertainty": comp_unc,
            "blending_uncertainty": blend_unc,
            "motion_uncertainty": motion_unc,
            "shift_vector": shift_vector
        }

        return shift_vector, aux_outputs
