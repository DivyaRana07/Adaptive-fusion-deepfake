"""
Test all four forensic streams and feature extractors.
"""

import pytest
import torch
import numpy as np
from src.models.branches.face_crop_branch import FaceCropBranch
from src.models.branches.full_frame_branch import FullFrameBranch
from src.models.branches.frequency_branch import FrequencyBranch
from src.models.branches.motion_branch import MotionBranch
from src.data.frequency_extractor import FrequencyExtractor
from src.data.motion_extractor import MotionExtractor


def test_face_crop_branch():
    branch = FaceCropBranch(in_channels=3, feature_dim=128, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = branch(x)
    assert out.shape == (2, 128)
    # Check L2 hyperspherical normalization
    norms = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_full_frame_branch():
    branch = FullFrameBranch(in_channels=3, feature_dim=128, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = branch(x)
    assert out.shape == (2, 128)


def test_frequency_branch_and_extractor():
    ext = FrequencyExtractor(target_size=224)
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dwt = ext.extract_dwt(dummy_img)
    assert dwt.shape == (224, 224, 4)

    branch = FrequencyBranch(in_channels=4, feature_dim=128, pretrained=False)
    x = torch.from_numpy(dwt).permute(2, 0, 1).unsqueeze(0).float()
    out = branch(x)
    assert out.shape == (1, 128)


def test_motion_branch_and_extractor():
    ext = MotionExtractor(target_size=224)
    f1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    f2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    flow = ext.compute_dense_flow(f1, f2)
    assert flow.shape == (224, 224, 3)

    branch = MotionBranch(in_channels=3, feature_dim=128, pretrained=False)
    x = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float()
    out = branch(x)
    assert out.shape == (1, 128)
