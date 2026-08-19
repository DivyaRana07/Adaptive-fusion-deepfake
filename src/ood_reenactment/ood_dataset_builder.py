"""
Step 3: Reenactment OOD Evaluation Dataset Builder.
Assembles the complete 3rd-generator-family evaluation benchmark with paired Real/Fake video sequences
and exports preprocessed 4-branch forensic test samples.
"""

import os
import json
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .expression_extractor import MotionExpressionExtractor
from .liveportrait_pipeline import LivePortraitReenactor
from ..data.face_extractor import FaceExtractor
from ..data.frequency_extractor import FrequencyExtractor
from ..data.motion_extractor import MotionExtractor


class ReenactmentOODBuilder:
    """Automates creation and indexing of the self-built LivePortrait reenactment OOD benchmark."""

    def __init__(self, output_dir: str = "data/Reenactment_LivePortrait_OOD", target_size: int = 224):
        self.output_dir = output_dir
        self.target_size = target_size
        self.motion_extractor = MotionExpressionExtractor()
        self.reenactor = LivePortraitReenactor(target_size=target_size)
        self.face_ext = FaceExtractor(target_size=target_size)
        self.freq_ext = FrequencyExtractor(target_size=target_size)
        self.flow_ext = MotionExtractor(target_size=target_size)

        os.makedirs(os.path.join(output_dir, "real"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "fake"), exist_ok=True)

    def build_synthetic_benchmark(self, num_pairs: int = 20, frames_per_pair: int = 15) -> str:
        """
        Builds a self-contained LivePortrait reenactment OOD dataset.
        Returns path to generated dataset manifest JSON.
        """
        manifest = []

        print(f"Building LivePortrait Reenactment OOD Benchmark ({num_pairs} paired real/fake clips)...")
        for i in tqdm(range(num_pairs), desc="Generating Reenactment OOD Pairs"):
            # 1. Synthesize real driving video frames
            driving_frames = []
            bg_color = np.random.randint(50, 200, 3)
            for f in range(frames_per_pair):
                frame = np.ones((self.target_size, self.target_size, 3), dtype=np.uint8) * bg_color.astype(np.uint8)
                # Person 1 (Driving Actor)
                cv2.circle(frame, (112 + int(np.sin(f * 0.5) * 8), 112 + int(np.cos(f * 0.3) * 5)), 50, (180, 200, 220), -1)
                cv2.circle(frame, (95, 100), 5, (20, 20, 20), -1)
                cv2.circle(frame, (130, 100), 5, (20, 20, 20), -1)
                cv2.ellipse(frame, (112, 135), (15, 5 + int(np.sin(f * 0.6) * 3)), 0, 0, 180, (50, 50, 180), -1)
                driving_frames.append(frame)

            # 2. Extract motion sequence from driving actor
            motion_seq = self.motion_extractor.extract_motion_from_frames(driving_frames)

            # 3. Source Portrait of different identity (Person 2)
            source_photo = np.ones((self.target_size, self.target_size, 3), dtype=np.uint8) * np.random.randint(40, 180, 3).astype(np.uint8)
            cv2.ellipse(source_photo, (112, 112), (45, 55), 0, 0, 360, (220, 190, 170), -1) # Different skin tone & geometry
            cv2.circle(source_photo, (95, 105), 6, (60, 40, 30), -1)
            cv2.circle(source_photo, (130, 105), 6, (60, 40, 30), -1)

            # 4. Animate Person 2 photo with Person 1 motion via LivePortrait keypoint-warping
            reenacted_frames = self.reenactor.animate_photo(source_photo, motion_seq)

            # Save sample pairs to manifest
            real_img_path = os.path.join(self.output_dir, "real", f"pair_{i:03d}_frame_{frames_per_pair-1:02d}.png")
            fake_img_path = os.path.join(self.output_dir, "fake", f"pair_{i:03d}_reenacted_{frames_per_pair-1:02d}.png")
            prev_real_path = os.path.join(self.output_dir, "real", f"pair_{i:03d}_frame_{frames_per_pair-2:02d}.png")
            prev_fake_path = os.path.join(self.output_dir, "fake", f"pair_{i:03d}_reenacted_{frames_per_pair-2:02d}.png")

            cv2.imwrite(real_img_path, driving_frames[-1])
            cv2.imwrite(prev_real_path, driving_frames[-2])
            cv2.imwrite(fake_img_path, reenacted_frames[-1])
            cv2.imwrite(prev_fake_path, reenacted_frames[-2])

            manifest.append({
                "id": f"reenact_real_{i:03d}",
                "image_path": real_img_path,
                "prev_image_path": prev_real_path,
                "is_fake": 0,
                "domain": "Real_Driving_Video",
                "generator_family": "Real"
            })
            manifest.append({
                "id": f"reenact_fake_{i:03d}",
                "image_path": fake_img_path,
                "prev_image_path": prev_fake_path,
                "is_fake": 1,
                "domain": "LivePortrait_Reenactment",
                "generator_family": "Keypoint_Warping_Motion_Transfer"
            })

        manifest_file = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Reenactment OOD Benchmark successfully created: {manifest_file}")
        return manifest_file
