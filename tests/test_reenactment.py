"""
Test LivePortrait motion extraction and paired reenactment OOD generation.
"""

import pytest
import os
import numpy as np
from src.ood_reenactment.expression_extractor import MotionExpressionExtractor
from src.ood_reenactment.liveportrait_pipeline import LivePortraitReenactor
from src.ood_reenactment.ood_dataset_builder import ReenactmentOODBuilder


def test_motion_extractor_and_reenactor():
    extractor = MotionExpressionExtractor()
    reenactor = LivePortraitReenactor(target_size=112)

    # Synthetic driving frames
    f1 = np.ones((112, 112, 3), dtype=np.uint8) * 128
    f2 = np.ones((112, 112, 3), dtype=np.uint8) * 130
    motion_seq = extractor.extract_motion_from_frames([f1, f2])

    assert len(motion_seq) == 2
    assert "dx" in motion_seq[0]

    # Source portrait photo
    source_photo = np.ones((112, 112, 3), dtype=np.uint8) * 180
    animated_frames = reenactor.animate_photo(source_photo, motion_seq)

    assert len(animated_frames) == 2
    assert animated_frames[0].shape == (112, 112, 3)


def test_ood_benchmark_builder(tmp_path):
    builder = ReenactmentOODBuilder(output_dir=str(tmp_path / "reenactment_test"), target_size=112)
    manifest_path = builder.build_synthetic_benchmark(num_pairs=2, frames_per_pair=3)

    assert os.path.exists(manifest_path)
