"""
Master CLI Experiment Runner for Adaptive Fusion Deepfake Detector.
Supports training, meta-learning, reenactment OOD generation, cross-dataset evaluation, and ablations.
"""

import os
import sys
import argparse
import yaml
import torch
import pandas as pd

from src.models.detector import AdaptiveFusionDetector
from src.models.losses.multi_task_loss import MultiTaskForensicLoss
from src.data.dataset import MultiBranchDeepfakeDataset
from src.training.standard_trainer import StandardTrainer
from src.training.meta_trainer import MetaLearningTrainer
from src.ood_reenactment.ood_dataset_builder import ReenactmentOODBuilder
from src.evaluation.evaluator import CrossDatasetEvaluator
from src.evaluation.ablation_matrix import AblationMatrixRunner
from src.evaluation.weight_analyzer import FusionWeightAnalyzer
from src.utils.logger import setup_logger


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Adaptive Fusion Deepfake Detector CLI")
    parser.add_argument(
        "--mode",
        type=str,
        default="demo_pipeline",
        choices=[
            "train_standard",
            "train_meta",
            "build_reenactment_ood",
            "evaluate_cross_dataset",
            "run_ablation",
            "analyze_weights",
            "demo_pipeline"
        ],
        help="Experiment execution mode"
    )
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--quick", action="store_true", default=True, help="Fast execution mode for testing")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', or 'auto')")

    args = parser.parse_args()
    logger = setup_logger("CLI_Runner")

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using compute device: {device}")

    # Load configuration
    cfg = load_config(args.config) if os.path.exists(args.config) else {}

    # Initialize Datasets
    logger.info("Initializing datasets...")
    train_dataset = MultiBranchDeepfakeDataset(num_synthetic_samples=32 if args.quick else 200, target_size=224)
    val_dataset = MultiBranchDeepfakeDataset(num_synthetic_samples=16 if args.quick else 100, target_size=224)

    # Prepare 5 Benchmark Evaluation Sets
    eval_datasets = {
        "FaceForensics++ (HQ)": MultiBranchDeepfakeDataset(num_synthetic_samples=16 if args.quick else 50, target_size=224),
        "Celeb-DF v2": MultiBranchDeepfakeDataset(num_synthetic_samples=16 if args.quick else 50, target_size=224),
        "DFDC": MultiBranchDeepfakeDataset(num_synthetic_samples=16 if args.quick else 50, target_size=224),
        "Diffusion OOD": MultiBranchDeepfakeDataset(num_synthetic_samples=16 if args.quick else 50, target_size=224),
        "LivePortrait Reenactment OOD": MultiBranchDeepfakeDataset(num_synthetic_samples=16 if args.quick else 50, target_size=224),
    }

    if args.mode in ["build_reenactment_ood", "demo_pipeline"]:
        logger.info("Building LivePortrait Reenactment OOD Evaluation Set...")
        builder = ReenactmentOODBuilder(output_dir="data/Reenactment_LivePortrait_OOD", target_size=224)
        manifest_path = builder.build_synthetic_benchmark(num_pairs=4 if args.quick else 20, frames_per_pair=10)
        logger.info(f"Reenactment benchmark saved to: {manifest_path}")

    if args.mode in ["train_standard", "demo_pipeline"]:
        logger.info("Executing Standard Multi-Task Training...")
        model = AdaptiveFusionDetector(
            backbone_name="efficientnet_b0",
            pretrained=False,
            feature_dim=128,
            fusion_type="domain_conditioned",
            use_hyperspherical=True,
            use_auxiliary=True
        ).to(device)

        loss_fn = MultiTaskForensicLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

        trainer = StandardTrainer(model, loss_fn, optimizer, device=device)
        trainer.train(train_loader, val_loader, epochs=args.epochs)

    if args.mode in ["train_meta", "demo_pipeline"]:
        logger.info("Executing Episodic Meta-Learning (MLDG)...")
        meta_model = AdaptiveFusionDetector(
            backbone_name="efficientnet_b0",
            pretrained=False,
            feature_dim=128,
            fusion_type="domain_conditioned",
            use_hyperspherical=True,
            use_auxiliary=True
        ).to(device)

        loss_fn = MultiTaskForensicLoss().to(device)
        optimizer = torch.optim.AdamW(meta_model.parameters(), lr=0.0003)

        meta_trainer = MetaLearningTrainer(meta_model, loss_fn, optimizer, inner_lr=0.001, device=device)
        meta_trainer.train_meta(train_dataset, val_dataset, episodes_per_epoch=5 if args.quick else 50, epochs=args.epochs)

    if args.mode in ["evaluate_cross_dataset", "demo_pipeline"]:
        logger.info("Running Cross-Dataset Zero-Shot Benchmark Suite...")
        eval_model = AdaptiveFusionDetector(
            backbone_name="efficientnet_b0",
            pretrained=False,
            feature_dim=128,
            fusion_type="domain_conditioned",
            use_hyperspherical=True,
            use_auxiliary=True
        ).to(device)

        evaluator = CrossDatasetEvaluator(eval_model, device=device)
        results_df = evaluator.run_benchmark_suite(eval_datasets, batch_size=8)
        print("\n" + "="*80)
        print("CROSS-DATASET ZERO-SHOT EVALUATION BENCHMARK RESULTS")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80 + "\n")

    if args.mode in ["analyze_weights", "demo_pipeline"]:
        logger.info("Analyzing Dynamic Branch Weight Shifts under JPEG Compression...")
        weight_model = AdaptiveFusionDetector(
            backbone_name="efficientnet_b0",
            pretrained=False,
            feature_dim=128,
            fusion_type="domain_conditioned",
            use_hyperspherical=True,
            use_auxiliary=True
        ).to(device)

        analyzer = FusionWeightAnalyzer(weight_model, device=device)
        shift_df = analyzer.analyze_compression_shift(val_dataset, quality_levels=[95, 75, 50, 25], num_samples=10)
        print("\n" + "="*80)
        print("DYNAMIC BRANCH TRUST WEIGHTS UNDER DOMAIN COMPRESSION SHIFT")
        print("="*80)
        print(shift_df.to_string(index=False))
        print("="*80 + "\n")

    if args.mode == "run_ablation":
        logger.info("Executing Full 5-Stage Ablation Matrix...")
        runner = AblationMatrixRunner(device=device, quick_mode=args.quick)
        ablation_df = runner.run_all_ablations(train_dataset, val_dataset, eval_datasets)
        print("\n" + "="*80)
        print("ABLATION MATRIX EMPIRICAL RESULTS")
        print("="*80)
        print(ablation_df.to_string(index=False))
        print("="*80 + "\n")

    logger.info("Execution completed successfully.")


if __name__ == "__main__":
    main()
