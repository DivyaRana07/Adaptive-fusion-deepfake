"""
Fusion Weight Analyzer & Domain-Shift Diagnostic Tool.
Tracks and visualizes how dynamic branch weights w = [w_face, w_frame, w_freq, w_motion]
shift when exposed to different perturbation types, compression levels, and generator families.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

from ..models.detector import AdaptiveFusionDetector
from ..data.dataset import MultiBranchDeepfakeDataset
from ..data.augmentation import ForensicAugmentationPipeline


class FusionWeightAnalyzer:
    """Diagnoses transfer-adaptiveness of learned fusion weights."""

    def __init__(self, model: AdaptiveFusionDetector, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.aug = ForensicAugmentationPipeline()

    @torch.no_grad()
    def analyze_compression_shift(
        self,
        dataset: MultiBranchDeepfakeDataset,
        quality_levels: List[int] = [95, 80, 60, 40, 20],
        num_samples: int = 20
    ) -> pd.DataFrame:
        """
        Measures branch weight trajectory as JPEG compression severity increases.
        """
        self.model.eval()
        branch_names = self.model.active_branches
        records = []

        for q in quality_levels:
            weights_per_q = []
            for i in range(min(num_samples, len(dataset))):
                sample = dataset[i]
                # Apply specific compression level
                full_frame = sample["full_frame"].permute(1, 2, 0).numpy() * 255.0
                face_crop = sample["face_crop"].permute(1, 2, 0).numpy() * 255.0
                
                c_face, _, _ = self.aug.apply_jpeg_compression(face_crop.astype(np.uint8), quality=q)
                c_frame, _, _ = self.aug.apply_jpeg_compression(full_frame.astype(np.uint8), quality=q)

                # Convert to tensors
                def to_t(arr):
                    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0

                inputs = {
                    "face_crop": to_t(c_face).to(self.device),
                    "full_frame": to_t(c_frame).to(self.device),
                    "frequency": sample["frequency"].unsqueeze(0).to(self.device),
                    "motion": sample["motion"].unsqueeze(0).to(self.device),
                }

                out = self.model(inputs)
                if "branch_weights" in out:
                    weights_per_q.append(out["branch_weights"].cpu().numpy().squeeze(0))

            if len(weights_per_q) > 0:
                mean_w = np.mean(weights_per_q, axis=0)
                row = {"JPEG_Quality": q}
                for idx, b_name in enumerate(branch_names):
                    if idx < len(mean_w):
                        row[f"Weight_{b_name}"] = round(float(mean_w[idx]), 3)
                records.append(row)

        df = pd.DataFrame(records)
        return df

    def plot_compression_shift(self, df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
        """Plots branch trust trajectory as a function of JPEG quality."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        weight_cols = [c for c in df.columns if c.startswith("Weight_")]
        for col in weight_cols:
            branch_label = col.replace("Weight_", "").replace("_", " ").title()
            ax.plot(df["JPEG_Quality"], df[col], marker="o", linewidth=2.5, label=branch_label)

        ax.set_xlabel("JPEG Quality Factor (Lower = More Compressed)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Assigned Branch Trust Weight", fontsize=11, fontweight="bold")
        ax.set_title("Adaptive Weight Shift under Domain Compression", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 0.6)
        ax.invert_xaxis() # High quality to low quality
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig
