"""
Baseline fusion architectures for comparative evaluation and ablation studies:
1. Fixed Uniform Average Fusion (w_i = 0.25)
2. Direct Feature Concatenation MLP
3. Multi-Head Self-Attention Fusion (unconditioned)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class FixedAverageFusion(nn.Module):
    """Baseline 1: Equal Uniform Weighting across all branches."""

    def __init__(self, feature_dim: int = 256, num_branches: int = 4, num_classes: int = 2):
        super().__init__()
        self.num_branches = num_branches
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(
        self, branch_features: List[torch.Tensor], shift_vector: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = branch_features[0].shape[0]
        device = branch_features[0].device

        stacked = torch.stack(branch_features, dim=1) # (B, num_branches, D)
        fused = torch.mean(stacked, dim=1) # (B, D)

        # Equal weights (1 / num_branches)
        weights = torch.full((batch_size, self.num_branches), 1.0 / self.num_branches, device=device)
        logits = self.classifier(fused)
        return logits, weights, fused


class ConcatMLPFusion(nn.Module):
    """Baseline 2: Concatenation of all branch embeddings followed by MLP."""

    def __init__(self, feature_dim: int = 256, num_branches: int = 4, num_classes: int = 2):
        super().__init__()
        self.num_branches = num_branches
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * num_branches, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(
        self, branch_features: List[torch.Tensor], shift_vector: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = branch_features[0].shape[0]
        device = branch_features[0].device

        concatenated = torch.cat(branch_features, dim=-1) # (B, num_branches * D)
        logits = self.mlp(concatenated)
        weights = torch.full((batch_size, self.num_branches), 1.0 / self.num_branches, device=device)
        return logits, weights, concatenated


class SelfAttentionFusion(nn.Module):
    """Baseline 3: Multi-Head Self-Attention over branch tokens without domain shift conditioning."""

    def __init__(self, feature_dim: int = 256, num_branches: int = 4, num_heads: int = 4, num_classes: int = 2):
        super().__init__()
        self.num_branches = num_branches
        self.mha = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(
        self, branch_features: List[torch.Tensor], shift_vector: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = branch_features[0].shape[0]
        device = branch_features[0].device

        # Stack branch features as sequence tokens: (B, num_branches, D)
        tokens = torch.stack(branch_features, dim=1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_tokens, tokens], dim=1) # (B, num_branches + 1, D)

        attn_out, attn_weights = self.mha(seq, seq, seq)
        cls_out = attn_out[:, 0, :] # Output at CLS token

        logits = self.classifier(cls_out)
        # Extract attention weights assigned by CLS token to the branch tokens
        weights = attn_weights[:, 0, 1:] # (B, num_branches)
        weights = F.softmax(weights, dim=-1)

        return logits, weights, cls_out
