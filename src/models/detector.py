"""
Complete End-to-End Multi-Branch Deepfake Detector.
Integrates the 4 forensic streams (face-crop, full-frame, frequency, motion),
auxiliary self-supervised shift heads, hyperspherical feature regularization,
and domain-conditioned adaptive fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple

from .branches.face_crop_branch import FaceCropBranch
from .branches.full_frame_branch import FullFrameBranch
from .branches.frequency_branch import FrequencyBranch
from .branches.motion_branch import MotionBranch
from .auxiliary.shift_estimator import DomainShiftEstimator
from .fusion.domain_conditioned import DomainConditionedFusion
from .fusion.baseline_fusion import (
    FixedAverageFusion,
    ConcatMLPFusion,
    SelfAttentionFusion,
)


class AdaptiveFusionDetector(nn.Module):
    """
    End-to-End Detector with Domain-Conditioned Adaptive Fusion.
    Fuses spatial, frequency, and motion cues weighted dynamically by self-supervised shift signals.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        feature_dim: int = 256,
        fusion_type: str = "domain_conditioned",
        use_hyperspherical: bool = True,
        use_auxiliary: bool = True,
        active_branches: Optional[List[str]] = None
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type
        self.use_hyperspherical = use_hyperspherical
        self.use_auxiliary = use_auxiliary
        
        if active_branches is None:
            self.active_branches = ["face_crop", "full_frame", "frequency", "motion"]
        else:
            self.active_branches = active_branches

        # 1. Initialize Forensic Branches
        self.face_branch = FaceCropBranch(
            in_channels=3, backbone_name=backbone_name, pretrained=pretrained,
            feature_dim=feature_dim, use_hyperspherical=use_hyperspherical
        )
        self.frame_branch = FullFrameBranch(
            in_channels=3, backbone_name=backbone_name, pretrained=pretrained,
            feature_dim=feature_dim, use_hyperspherical=use_hyperspherical
        )
        self.freq_branch = FrequencyBranch(
            in_channels=4, backbone_name=backbone_name, pretrained=False,
            feature_dim=feature_dim, use_hyperspherical=use_hyperspherical
        )
        self.motion_branch = MotionBranch(
            in_channels=3, backbone_name=backbone_name, pretrained=False,
            feature_dim=feature_dim, use_hyperspherical=use_hyperspherical
        )

        num_active = len(self.active_branches)

        # 2. Initialize Auxiliary Self-Supervised Shift Estimator
        if self.use_auxiliary:
            self.shift_estimator = DomainShiftEstimator(
                feature_dim=feature_dim, num_classes=5, hidden_dim=128
            )
        else:
            self.shift_estimator = None

        # 3. Initialize Fusion Module
        if fusion_type == "domain_conditioned":
            self.fusion_module = DomainConditionedFusion(
                feature_dim=feature_dim,
                num_branches=num_active,
                shift_dim=3 if self.use_auxiliary else 0,
                hidden_dim=128,
                num_classes=2
            )
        elif fusion_type == "concat":
            self.fusion_module = ConcatMLPFusion(
                feature_dim=feature_dim, num_branches=num_active, num_classes=2
            )
        elif fusion_type == "fixed_average":
            self.fusion_module = FixedAverageFusion(
                feature_dim=feature_dim, num_branches=num_active, num_classes=2
            )
        elif fusion_type == "self_attention":
            self.fusion_module = SelfAttentionFusion(
                feature_dim=feature_dim, num_branches=num_active, num_classes=2
            )
        elif fusion_type == "none":
            # Single-branch mode
            self.fusion_module = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 2)
            )
        else:
            raise ValueError(f"Unknown fusion type '{fusion_type}'")

    def forward(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Forward pass for multi-branch inputs.
        Args:
            batch_dict: Dictionary containing:
                - 'face_crop': (B, 3, H, W)
                - 'full_frame': (B, 3, H, W)
                - 'frequency': (B, 4, H, W)
                - 'motion': (B, 3, H, W)
        Returns:
            Dictionary containing logits, probs, weights, uncertainty vectors, and embeddings.
        """
        branch_feats = {}
        ordered_feats = []

        # Extract features for active branches
        if "face_crop" in self.active_branches:
            f_face = self.face_branch(batch_dict["face_crop"])
            branch_feats["face_crop"] = f_face
            ordered_feats.append(f_face)
            
        if "full_frame" in self.active_branches:
            f_frame = self.frame_branch(batch_dict["full_frame"])
            branch_feats["full_frame"] = f_frame
            ordered_feats.append(f_frame)
            
        if "frequency" in self.active_branches:
            f_freq = self.freq_branch(batch_dict["frequency"])
            branch_feats["frequency"] = f_freq
            ordered_feats.append(f_freq)
            
        if "motion" in self.active_branches:
            f_motion = self.motion_branch(batch_dict["motion"])
            branch_feats["motion"] = f_motion
            ordered_feats.append(f_motion)

        # Compute auxiliary shift indicators
        shift_vector = None
        aux_outputs = {}
        if self.use_auxiliary and self.shift_estimator is not None:
            # Need features for auxiliary prediction
            feat_freq = branch_feats.get("frequency", ordered_feats[0])
            feat_face = branch_feats.get("face_crop", ordered_feats[0])
            feat_motion = branch_feats.get("motion", ordered_feats[0])

            shift_vector, aux_outputs = self.shift_estimator(feat_freq, feat_face, feat_motion)

        # Single branch baseline routing
        if self.fusion_type == "none":
            single_feat = ordered_feats[0]
            logits = self.fusion_module(single_feat)
            batch_size = single_feat.shape[0]
            weights = torch.ones((batch_size, 1), device=single_feat.device)
            fused_feat = single_feat
        else:
            # Multi-branch fusion
            logits, weights, fused_feat = self.fusion_module(ordered_feats, shift_vector=shift_vector)

        # Calculate softmax probabilities & calibrated confidence
        probs = F.softmax(logits, dim=-1)
        is_fake_prob = probs[:, 1]
        confidence = torch.max(probs, dim=-1).values

        # Formulate output dictionary
        output = {
            "logits": logits,
            "probs": probs,
            "is_fake_prob": is_fake_prob,
            "confidence": confidence,
            "branch_weights": weights,
            "fused_feature": fused_feat,
            "branch_features": branch_feats,
            "shift_vector": shift_vector,
        }

        # Merge auxiliary head outputs
        output.update(aux_outputs)

        return output
