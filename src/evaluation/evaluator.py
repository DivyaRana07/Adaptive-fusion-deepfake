"""
Cross-Dataset Zero-Shot Evaluator.
Benchmarks trained detectors across FaceForensics++, Celeb-DF, DFDC, Diffusion OOD,
and the LivePortrait Reenactment OOD dataset.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..models.detector import AdaptiveFusionDetector
from ..data.dataset import MultiBranchDeepfakeDataset
from .metrics import compute_forensic_metrics


class CrossDatasetEvaluator:
    """Evaluates zero-shot generalization across distinct dataset and generator distributions."""

    def __init__(self, model: AdaptiveFusionDetector, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataset: MultiBranchDeepfakeDataset,
        dataset_name: str = "Target_Dataset",
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """
        Runs full zero-shot evaluation on a single target dataset.
        Returns metrics dictionary and collected branch trust weights.
        """
        self.model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_labels = []
        all_probs = []
        all_weights = []

        for batch in tqdm(loader, desc=f"Zero-Shot Eval: {dataset_name}", leave=False):
            inputs = {
                "face_crop": batch["face_crop"].to(self.device),
                "full_frame": batch["full_frame"].to(self.device),
                "frequency": batch["frequency"].to(self.device),
                "motion": batch["motion"].to(self.device),
            }

            outputs = self.model(inputs)
            
            all_labels.extend(batch["label"].cpu().numpy().tolist())
            all_probs.extend(outputs["is_fake_prob"].cpu().numpy().tolist())
            
            if "branch_weights" in outputs:
                all_weights.append(outputs["branch_weights"].cpu().numpy())

        metrics = compute_forensic_metrics(all_labels, all_probs)
        
        # Calculate mean branch trust weights
        if len(all_weights) > 0:
            stacked_w = np.concatenate(all_weights, axis=0) # (N, num_branches)
            mean_weights = np.mean(stacked_w, axis=0).tolist()
            metrics["mean_branch_weights"] = mean_weights

        return metrics

    def run_benchmark_suite(
        self,
        datasets: Dict[str, MultiBranchDeepfakeDataset],
        batch_size: int = 16
    ) -> pd.DataFrame:
        """
        Runs evaluation across all benchmark datasets and formats into a comparative DataFrame.
        """
        results = []
        for name, dset in datasets.items():
            res = self.evaluate_dataset(dset, dataset_name=name, batch_size=batch_size)
            row = {
                "Dataset": name,
                "AUC (%)": res["auc"],
                "Accuracy (%)": res["accuracy"],
                "EER (%)": res["eer"],
                "F1-Score (%)": res["f1_score"],
                "Brier Score": res["brier_score"]
            }
            if "mean_branch_weights" in res:
                for idx, b_name in enumerate(self.model.active_branches):
                    if idx < len(res["mean_branch_weights"]):
                        row[f"Weight_{b_name}"] = round(res["mean_branch_weights"][idx], 3)
            results.append(row)

        df = pd.DataFrame(results)
        return df
