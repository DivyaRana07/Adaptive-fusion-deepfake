"""
Compression Level Auxiliary Head.
Predicts JPEG/video compression quality tiers and computes predictive entropy/uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CompressionLevelHead(nn.Module):
    """Auxiliary Head 1: Self-Supervised Compression Estimator."""

    def __init__(self, in_features: int = 256, num_classes: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature: (B, in_features)
        Returns:
            logits: (B, num_classes)
            uncertainty: (B, 1) normalized Shannon entropy in [0, 1]
        """
        logits = self.mlp(feature)
        probs = F.softmax(logits, dim=-1)
        
        # Calculate Shannon entropy: - sum(p * log(p + eps))
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
        
        # Normalize entropy by max possible entropy log(num_classes)
        max_entropy = torch.log(torch.tensor(probs.shape[-1], dtype=torch.float32, device=feature.device))
        norm_entropy = entropy / (max_entropy + 1e-8)
        
        return logits, norm_entropy
