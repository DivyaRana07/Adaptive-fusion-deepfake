"""
Automated 5-Stage Ablation Matrix Runner.
Isolates individual contributions of:
1. Single branches (face-crop, full-frame, frequency, motion)
2. Fusion architectures (fixed average, concat MLP, self-attention, domain-conditioned)
3. Episodic meta-learning (MLDG on vs off)
4. Auxiliary self-supervised shift heads (on vs off)
5. Hyperspherical manifold regularization (on vs off)
"""

import os
import torch
import pandas as pd
from typing import Dict, Any, List

from ..models.detector import AdaptiveFusionDetector
from ..models.losses.multi_task_loss import MultiTaskForensicLoss
from ..data.dataset import MultiBranchDeepfakeDataset
from .evaluator import CrossDatasetEvaluator


class AblationMatrixRunner:
    """Executes systematic ablation suite and formats comparative empirical tables."""

    def __init__(self, device: str = "cpu", quick_mode: bool = True):
        self.device = device
        self.quick_mode = quick_mode
        self.epochs = 2 if quick_mode else 10

    def run_ablation_experiment(
        self,
        config: Dict[str, Any],
        train_dset: MultiBranchDeepfakeDataset,
        val_dset: MultiBranchDeepfakeDataset,
        eval_datasets: Dict[str, MultiBranchDeepfakeDataset]
    ) -> Dict[str, Any]:
        """Trains and evaluates a specific model variant."""
        name = config.get("name", "experiment")
        active_branches = config.get("active_branches", ["face_crop", "full_frame", "frequency", "motion"])
        fusion_type = config.get("fusion_type", "domain_conditioned")
        use_meta = config.get("meta_learning", False)
        use_aux = config.get("auxiliary", True)
        use_hyper = config.get("hyperspherical", True)

        print(f"\n{'='*70}\nRunning Ablation: {name}\n{'='*70}")

        # Instantiate model variant
        model = AdaptiveFusionDetector(
            backbone_name="efficientnet_b0",
            pretrained=False if self.quick_mode else True,
            feature_dim=128 if self.quick_mode else 256,
            fusion_type=fusion_type,
            use_hyperspherical=use_hyper,
            use_auxiliary=use_aux,
            active_branches=active_branches
        ).to(self.device)

        loss_fn = MultiTaskForensicLoss(
            cls_weight=1.0,
            aux_compression_weight=0.2 if use_aux else 0.0,
            aux_blending_weight=0.2 if use_aux else 0.0,
            aux_motion_weight=0.2 if use_aux else 0.0,
            hyperspherical_weight=0.1 if use_hyper else 0.0
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        # Train with Meta-Learning or Standard Multi-Task
        if use_meta:
            from ..training.meta_trainer import MetaLearningTrainer
            trainer = MetaLearningTrainer(model, loss_fn, optimizer, device=self.device)
            trainer.train_meta(train_dset, val_dset, episodes_per_epoch=10 if self.quick_mode else 50, epochs=self.epochs)
        else:
            from ..training.standard_trainer import StandardTrainer
            train_loader = torch.utils.data.DataLoader(train_dset, batch_size=8, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dset, batch_size=8, shuffle=False)
            trainer = StandardTrainer(model, loss_fn, optimizer, device=self.device)
            trainer.train(train_loader, val_loader, epochs=self.epochs)

        # Cross-Dataset Zero-Shot Evaluation
        evaluator = CrossDatasetEvaluator(model, device=self.device)
        df_results = evaluator.run_benchmark_suite(eval_datasets, batch_size=8)
        
        return {
            "variant_name": name,
            "config": config,
            "results_df": df_results
        }

    def run_all_ablations(
        self,
        train_dset: MultiBranchDeepfakeDataset,
        val_dset: MultiBranchDeepfakeDataset,
        eval_datasets: Dict[str, MultiBranchDeepfakeDataset]
    ) -> pd.DataFrame:
        """Runs the complete suite of ablations and builds a unified summary table."""
        variants = [
            # 1. Single Branches
            {"name": "1. Face-Crop Only", "active_branches": ["face_crop"], "fusion_type": "none", "meta_learning": False, "auxiliary": False, "hyperspherical": False},
            {"name": "2. Full-Frame Only", "active_branches": ["full_frame"], "fusion_type": "none", "meta_learning": False, "auxiliary": False, "hyperspherical": False},
            {"name": "3. Frequency Only", "active_branches": ["frequency"], "fusion_type": "none", "meta_learning": False, "auxiliary": False, "hyperspherical": False},
            {"name": "4. Motion Only", "active_branches": ["motion"], "fusion_type": "none", "meta_learning": False, "auxiliary": False, "hyperspherical": False},
            
            # 2. Baseline Fusions
            {"name": "5. Fixed Average Fusion", "active_branches": ["face_crop", "full_frame", "frequency", "motion"], "fusion_type": "fixed_average", "meta_learning": False, "auxiliary": False, "hyperspherical": False},
            {"name": "6. Concat MLP Fusion", "active_branches": ["face_crop", "full_frame", "frequency", "motion"], "fusion_type": "concat", "meta_learning": False, "auxiliary": False, "hyperspherical": False},
            {"name": "7. Self-Attention Fusion", "active_branches": ["face_crop", "full_frame", "frequency", "motion"], "fusion_type": "self_attention", "meta_learning": False, "auxiliary": False, "hyperspherical": False},
            
            # 3. Component Ablations
            {"name": "8. Ours (No Meta, No Aux)", "active_branches": ["face_crop", "full_frame", "frequency", "motion"], "fusion_type": "domain_conditioned", "meta_learning": False, "auxiliary": False, "hyperspherical": True},
            {"name": "9. Ours (Meta Only)", "active_branches": ["face_crop", "full_frame", "frequency", "motion"], "fusion_type": "domain_conditioned", "meta_learning": True, "auxiliary": False, "hyperspherical": True},
            {"name": "10. Ours (Aux Only)", "active_branches": ["face_crop", "full_frame", "frequency", "motion"], "fusion_type": "domain_conditioned", "meta_learning": False, "auxiliary": True, "hyperspherical": True},
            {"name": "11. Ours (Full Proposed)", "active_branches": ["face_crop", "full_frame", "frequency", "motion"], "fusion_type": "domain_conditioned", "meta_learning": True, "auxiliary": True, "hyperspherical": True},
        ]

        summary_rows = []
        for var in variants:
            res = self.run_ablation_experiment(var, train_dset, val_dset, eval_datasets)
            df = res["results_df"]
            
            row = {"Method / Variant": var["name"]}
            for _, r in df.iterrows():
                dname = r["Dataset"]
                row[f"{dname} AUC"] = f"{r['AUC (%)']:.1f}%"
                row[f"{dname} EER"] = f"{r['EER (%)']:.1f}%"
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        return summary_df
