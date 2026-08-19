"""
Domain-Conditioned Adaptive Fusion Module.
The core novel component of the thesis:
Dynamically computes branch trust weights w = [w_face, w_frame, w_freq, w_motion]
conditioned explicitly on label-free auxiliary domain-shift indicators (compression, blending, motion uncertainty)
with an anti-circularity gating mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List


class DomainConditionedFusion(nn.Module):
    """
    Transfer-aware, domain-conditioned fusion module.
    Dynamically modulates branch weights based on target domain unfamiliarity.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_branches: int = 4,
        shift_dim: int = 3,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2,
        temperature: float = 1.0
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_branches = num_branches
        self.shift_dim = shift_dim
        self.temperature = temperature

        # Joint input dimension: (num_branches * feature_dim) + shift_dim
        joint_in_dim = (num_branches * feature_dim) + shift_dim

        # Gating network to predict dynamic branch weights w in R^num_branches
        self.weight_gating_net = nn.Sequential(
            nn.Linear(joint_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_branches)
        )

        # Cross-branch contextual refinement
        self.context_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Final Real/Fake Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        branch_features: List[torch.Tensor],
        shift_vector: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            branch_features: List of tensors [f_face, f_frame, f_freq, f_motion], each (B, feature_dim)
            shift_vector: (B, shift_dim) auxiliary uncertainty descriptor u. If None, zero-padded.
        Returns:
            logits: (B, num_classes) Real / Fake classification logits
            branch_weights: (B, num_branches) normalized dynamic trust weights
            fused_feature: (B, feature_dim) weighted aggregated representation
        """
        batch_size = branch_features[0].shape[0]
        device = branch_features[0].device

        # Stack branch features: (B, num_branches, feature_dim)
        stacked_features = torch.stack(branch_features, dim=1)

        # Default zero shift vector if omitted (e.g. during ablation)
        if shift_vector is None:
            shift_vector = torch.zeros((batch_size, self.shift_dim), device=device)

        # Flatten all branch features: (B, num_branches * feature_dim)
        flat_features = stacked_features.view(batch_size, -1)

        # Concatenate content features with domain shift uncertainty vector
        joint_descriptor = torch.cat([flat_features, shift_vector], dim=-1)

        # Compute dynamic branch trust logits
        weight_logits = self.weight_gating_net(joint_descriptor) / self.temperature

        # Normalized branch trust weights via Softmax
        branch_weights = F.softmax(weight_logits, dim=-1) # (B, num_branches)

        # Weighted combination: z = sum(w_i * f_i)
        # (B, num_branches, 1) * (B, num_branches, feature_dim) -> sum over dim 1 -> (B, feature_dim)
        weights_expanded = branch_weights.unsqueeze(-1)
        fused_feature = torch.sum(weights_expanded * stacked_features, dim=1)

        # Contextual refinement
        refined_feature = self.context_layer(fused_feature) + fused_feature

        # Real / Fake classification logits
        logits = self.classifier(refined_feature)

        return logits, branch_weights, refined_feature
