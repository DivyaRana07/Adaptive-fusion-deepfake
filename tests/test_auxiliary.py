"""
Test auxiliary self-supervised shift heads and uncertainty estimation.
"""

import pytest
import torch
from src.models.auxiliary.compression_head import CompressionLevelHead
from src.models.auxiliary.blending_head import BlendingRatioHead
from src.models.auxiliary.motion_head import MotionStabilityHead
from src.models.auxiliary.shift_estimator import DomainShiftEstimator


def test_compression_head():
    head = CompressionLevelHead(in_features=128, num_classes=5)
    feat = torch.randn(4, 128)
    logits, unc = head(feat)
    assert logits.shape == (4, 5)
    assert unc.shape == (4, 1)
    assert torch.all(unc >= 0.0) and torch.all(unc <= 1.0)


def test_blending_head():
    head = BlendingRatioHead(in_features=128, num_classes=5)
    feat = torch.randn(4, 128)
    logits, unc = head(feat)
    assert logits.shape == (4, 5)
    assert unc.shape == (4, 1)


def test_motion_head():
    head = MotionStabilityHead(in_features=128, num_classes=5)
    feat = torch.randn(4, 128)
    logits, unc = head(feat)
    assert logits.shape == (4, 5)
    assert unc.shape == (4, 1)


def test_shift_estimator():
    estimator = DomainShiftEstimator(feature_dim=128, num_classes=5)
    f_freq = torch.randn(4, 128)
    f_face = torch.randn(4, 128)
    f_motion = torch.randn(4, 128)

    shift_vec, aux_out = estimator(f_freq, f_face, f_motion)
    assert shift_vec.shape == (4, 3) # [u_comp, u_blend, u_motion]
    assert "compression_logits" in aux_out
    assert "blending_logits" in aux_out
    assert "motion_logits" in aux_out
