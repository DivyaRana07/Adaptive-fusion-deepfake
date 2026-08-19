"""
Hyperspherical Manifold Regularization Loss (GenD-style).
Constrains feature embeddings to unit hypersphere S^(d-1) and enforces angular margin separation.
Reduces source-specific high-frequency overfitting before fusion is applied.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HypersphericalCosineMarginLoss(nn.Module):
    """
    GenD-style Hyperspherical Angular / Cosine Margin Loss.
    Ensures authentic features cluster closely on S^(d-1) while manipulated features are separated by margin m.
    """

    def __init__(self, margin: float = 0.35, scale: float = 30.0):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) L2-normalized feature embeddings
            labels: (B,) binary labels (0=real, 1=fake)
        Returns:
            Scalar loss value
        """
        if features.shape[0] < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # Ensure unit normalization
        norm_feat = F.normalize(features, p=2, dim=-1)

        # Pairwise cosine similarity matrix (B, B)
        cos_sim = torch.matmul(norm_feat, norm_feat.T)

        # Label equality mask: 1 if same class (positive pair), 0 if different (negative pair)
        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float()
        mask_neg = 1.0 - mask_pos

        # Remove diagonal self-similarity from positive mask
        diag_mask = torch.eye(features.shape[0], device=features.device)
        mask_pos = mask_pos * (1.0 - diag_mask)

        # Positive loss: encourage high cosine similarity (1 - cos_sim)
        num_pos = mask_pos.sum()
        if num_pos > 0:
            loss_pos = (mask_pos * (1.0 - cos_sim)).sum() / num_pos
        else:
            loss_pos = torch.tensor(0.0, device=features.device)

        # Negative loss: penalize cosine similarity exceeding margin (max(0, cos_sim - (1 - m)))
        threshold = 1.0 - self.margin
        neg_penalty = F.relu(cos_sim - threshold)
        num_neg = mask_neg.sum()
        if num_neg > 0:
            loss_neg = (mask_neg * neg_penalty).sum() / num_neg
        else:
            loss_neg = torch.tensor(0.0, device=features.device)

        total_loss = loss_pos + loss_neg
        return total_loss
