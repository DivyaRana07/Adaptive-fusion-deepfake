"""
Multi-Task Forensic Loss Function.
Combines primary Real/Fake binary classification loss with auxiliary self-supervised
heads and hyperspherical metric regularization.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .hyperspherical import HypersphericalCosineMarginLoss


class MultiTaskForensicLoss(nn.Module):
    """Computes joint multi-task loss for the complete detector."""

    def __init__(
        self,
        cls_weight: float = 1.0,
        aux_compression_weight: float = 0.2,
        aux_blending_weight: float = 0.2,
        aux_motion_weight: float = 0.2,
        hyperspherical_weight: float = 0.1,
        margin: float = 0.35
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.aux_comp_weight = aux_compression_weight
        self.aux_blend_weight = aux_blending_weight
        self.aux_motion_weight = aux_motion_weight
        self.hyper_weight = hyperspherical_weight

        # Loss components
        self.ce_loss = nn.CrossEntropyLoss()
        self.hyper_loss_fn = HypersphericalCosineMarginLoss(margin=margin)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        branch_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Dictionary output from AdaptiveFusionDetector
            targets: Dictionary containing ground truth 'label', 'compression_target', 'blending_target', 'motion_target'
            branch_features: Optional fused or concatenated features for hyperspherical regularization
        Returns:
            Dictionary containing 'total_loss' and individual loss terms.
        """
        loss_dict = {}

        # 1. Primary Real/Fake Classification Loss
        cls_logits = predictions["logits"]
        cls_targets = targets["label"].long()
        cls_loss = self.ce_loss(cls_logits, cls_targets)
        loss_dict["cls_loss"] = cls_loss

        # 2. Auxiliary Compression Loss
        if "compression_logits" in predictions and "compression_target" in targets:
            comp_loss = self.ce_loss(predictions["compression_logits"], targets["compression_target"].long())
        else:
            comp_loss = torch.tensor(0.0, device=cls_logits.device)
        loss_dict["aux_comp_loss"] = comp_loss

        # 3. Auxiliary Blending Loss
        if "blending_logits" in predictions and "blending_target" in targets:
            blend_loss = self.ce_loss(predictions["blending_logits"], targets["blending_target"].long())
        else:
            blend_loss = torch.tensor(0.0, device=cls_logits.device)
        loss_dict["aux_blend_loss"] = blend_loss

        # 4. Auxiliary Motion Stability Loss
        if "motion_logits" in predictions and "motion_target" in targets:
            motion_loss = self.ce_loss(predictions["motion_logits"], targets["motion_target"].long())
        else:
            motion_loss = torch.tensor(0.0, device=cls_logits.device)
        loss_dict["aux_motion_loss"] = motion_loss

        # 5. Hyperspherical Manifold Regularization Loss
        if self.hyper_weight > 0 and branch_features is not None:
            hyper_loss = self.hyper_loss_fn(branch_features, cls_targets)
        else:
            hyper_loss = torch.tensor(0.0, device=cls_logits.device)
        loss_dict["hyper_loss"] = hyper_loss

        # Total Weighted Loss
        total_loss = (
            self.cls_weight * cls_loss +
            self.aux_comp_weight * comp_loss +
            self.aux_blend_weight * blend_loss +
            self.aux_motion_weight * motion_loss +
            self.hyper_weight * hyper_loss
        )
        loss_dict["total_loss"] = total_loss

        return loss_dict
