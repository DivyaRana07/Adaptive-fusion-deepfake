"""
Test domain-conditioned adaptive fusion and baseline fusion mechanisms.
"""

import pytest
import torch
from src.models.fusion.domain_conditioned import DomainConditionedFusion
from src.models.fusion.baseline_fusion import FixedAverageFusion, ConcatMLPFusion, SelfAttentionFusion
from src.models.detector import AdaptiveFusionDetector


def test_domain_conditioned_fusion():
    fusion = DomainConditionedFusion(feature_dim=128, num_branches=4, shift_dim=3)
    b_feats = [torch.randn(4, 128) for _ in range(4)]
    shift_vec = torch.rand(4, 3)

    logits, weights, fused = fusion(b_feats, shift_vector=shift_vec)
    assert logits.shape == (4, 2)
    assert weights.shape == (4, 4)
    assert fused.shape == (4, 128)

    # Verify weights sum to 1.0 per sample
    weight_sums = torch.sum(weights, dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


def test_baseline_fusions():
    b_feats = [torch.randn(4, 128) for _ in range(4)]
    
    # 1. Fixed Average
    f_avg = FixedAverageFusion(feature_dim=128, num_branches=4)
    logits, weights, _ = f_avg(b_feats)
    assert logits.shape == (4, 2)
    assert torch.allclose(weights, torch.full_like(weights, 0.25))

    # 2. Concat MLP
    f_cat = ConcatMLPFusion(feature_dim=128, num_branches=4)
    logits, _, _ = f_cat(b_feats)
    assert logits.shape == (4, 2)

    # 3. Self-Attention
    f_attn = SelfAttentionFusion(feature_dim=128, num_branches=4, num_heads=4)
    logits, _, _ = f_attn(b_feats)
    assert logits.shape == (4, 2)


def test_complete_detector_forward():
    detector = AdaptiveFusionDetector(
        backbone_name="efficientnet_b0",
        pretrained=False,
        feature_dim=128,
        fusion_type="domain_conditioned",
        use_hyperspherical=True,
        use_auxiliary=True
    )

    batch = {
        "face_crop": torch.randn(2, 3, 224, 224),
        "full_frame": torch.randn(2, 3, 224, 224),
        "frequency": torch.randn(2, 4, 224, 224),
        "motion": torch.randn(2, 3, 224, 224),
    }

    out = detector(batch)
    assert out["logits"].shape == (2, 2)
    assert out["is_fake_prob"].shape == (2,)
    assert out["branch_weights"].shape == (2, 4)
    assert out["shift_vector"].shape == (2, 3)
