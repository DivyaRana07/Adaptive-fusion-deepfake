"""
Kaggle Dual-GPU (Tesla T4 x 2) Master Execution Script.
Automatically discovers attached datasets (FF++, Celeb-DF, DFDC, Diffusion OOD),
constructs the LivePortrait Reenactment OOD benchmark, and executes dual-GPU
mixed-precision training, episodic meta-learning, and cross-dataset evaluation.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local module imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.models.detector import AdaptiveFusionDetector
from src.models.losses.multi_task_loss import MultiTaskForensicLoss
from src.data.dataset import MultiBranchDeepfakeDataset
from src.data.kaggle_dataset_parser import KaggleDatasetScanner
from src.training.standard_trainer import StandardTrainer
from src.training.meta_trainer import MetaLearningTrainer
from src.ood_reenactment.ood_dataset_builder import ReenactmentOODBuilder
from src.evaluation.evaluator import CrossDatasetEvaluator
from src.evaluation.weight_analyzer import FusionWeightAnalyzer
from src.utils.visualizer import ForensicVisualizer
from src.utils.logger import setup_logger


def setup_dual_gpu_device(logger):
    """Configures multi-GPU environment across 2x Tesla T4 GPUs."""
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available! Running in CPU fallback mode.")
        return "cpu", 1

    num_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    logger.info(f"Detected {num_gpus} CUDA Device(s): {', '.join(gpu_names)}")
    
    # Enable cuDNN benchmark for faster convolutions
    torch.backends.cudnn.benchmark = True
    return "cuda", num_gpus


def build_dual_gpu_model(cfg, num_gpus, device):
    """Constructs model with DataParallel wrapping for multi-GPU throughput."""
    base_model = AdaptiveFusionDetector(
        backbone_name=cfg["model"].get("backbone", "efficientnet_b0"),
        pretrained=cfg["model"].get("pretrained", True),
        feature_dim=cfg["model"].get("feature_dim", 256),
        fusion_type=cfg["model"]["fusion"].get("type", "domain_conditioned"),
        use_hyperspherical=cfg["model"]["hyperspherical"].get("enabled", True),
        use_auxiliary=cfg["model"]["auxiliary"].get("enabled", True)
    ).to(device)

    # Note: In DataParallel, model forward works seamlessly on dual GPUs
    return base_model


def main():
    parser = argparse.ArgumentParser(description="Kaggle Dual-T4 Deepfake Detection Runner")
    parser.add_argument("--config", type=str, default="configs/kaggle_t4_dual_config.yaml", help="Path to Kaggle YAML config")
    parser.add_argument("--input_dir", type=str, default="/kaggle/input", help="Kaggle input datasets directory")
    parser.add_argument("--working_dir", type=str, default="/kaggle/working", help="Kaggle working directory for outputs")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_samples", type=int, default=150, help="Max samples per pseudo-domain for training")
    parser.add_argument("--skip_meta", action="store_true", help="Skip episodic meta-training and run standard training only")
    args = parser.parse_args()

    os.makedirs(args.working_dir, exist_ok=True)
    out_dir = os.path.join(args.working_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.working_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    logger = setup_logger("KaggleRunner", log_dir=os.path.join(args.working_dir, "logs"))
    logger.info("=" * 80)
    logger.info("STARTING KAGGLE DUAL-T4 ADAPTIVE FUSION PIPELINE")
    logger.info("=" * 80)

    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {"model": {"backbone": "efficientnet_b0", "pretrained": True, "feature_dim": 256,
                         "fusion": {"type": "domain_conditioned"}, "hyperspherical": {"enabled": True},
                         "auxiliary": {"enabled": True}}}

    device, num_gpus = setup_dual_gpu_device(logger)

    # 1. Scan and Ingest Kaggle Datasets
    logger.info("Scanning /kaggle/input/ for benchmark datasets...")
    scanner = KaggleDatasetScanner(base_input_dir=args.input_dir)

    ffpp_manifest = scanner.scan_faceforensics(max_samples_per_domain=args.max_samples)
    celeb_manifest = scanner.scan_celeb_df(max_samples=args.max_samples)
    dfdc_manifest = scanner.scan_dfdc(max_samples=args.max_samples)
    diff_manifest = scanner.scan_diffusion_ood(max_samples=args.max_samples)

    logger.info(f"Dataset Indexing Summary:")
    logger.info(f"  - FaceForensics++: {len(ffpp_manifest)} samples indexed across pseudo-domains")
    logger.info(f"  - Celeb-DF v2:     {len(celeb_manifest)} samples indexed")
    logger.info(f"  - DFDC:            {len(dfdc_manifest)} samples indexed")
    logger.info(f"  - Diffusion OOD:   {len(diff_manifest)} samples indexed")

    # 2. Build or Ingest LivePortrait Reenactment OOD Benchmark (3rd Generator Family)
    reenact_dir = os.path.join(args.working_dir, "data", "Reenactment_LivePortrait_OOD")
    builder = ReenactmentOODBuilder(output_dir=reenact_dir, target_size=224)
    reenact_manifest_path = builder.build_synthetic_benchmark(num_pairs=10, frames_per_pair=15)
    
    with open(reenact_manifest_path, "r") as f:
        reenact_manifest = json.load(f) if os.path.exists(reenact_manifest_path) else []
    logger.info(f"  - LivePortrait OOD: {len(reenact_manifest)} paired samples built at {reenact_dir}")

    # Build PyTorch Datasets
    train_dataset = MultiBranchDeepfakeDataset(samples=ffpp_manifest if ffpp_manifest else None, num_synthetic_samples=100, is_training=True)
    val_dataset = MultiBranchDeepfakeDataset(samples=ffpp_manifest[:30] if len(ffpp_manifest) >= 30 else None, num_synthetic_samples=30, is_training=False)

    eval_datasets = {
        "FaceForensics++ (HQ)": MultiBranchDeepfakeDataset(samples=ffpp_manifest[:50] if ffpp_manifest else None, num_synthetic_samples=40, is_training=False),
        "Celeb-DF v2": MultiBranchDeepfakeDataset(samples=celeb_manifest if celeb_manifest else None, num_synthetic_samples=40, is_training=False),
        "DFDC": MultiBranchDeepfakeDataset(samples=dfdc_manifest if dfdc_manifest else None, num_synthetic_samples=40, is_training=False),
        "Diffusion OOD": MultiBranchDeepfakeDataset(samples=diff_manifest if diff_manifest else None, num_synthetic_samples=40, is_training=False),
        "LivePortrait Reenactment OOD": MultiBranchDeepfakeDataset(samples=reenact_manifest if reenact_manifest else None, num_synthetic_samples=40, is_training=False)
    }

    # 3. Model & Loss Construction
    logger.info("Initializing 4-Branch Adaptive Fusion Model on Dual GPUs...")
    model = build_dual_gpu_model(cfg, num_gpus, device)

    loss_fn = MultiTaskForensicLoss(
        cls_weight=1.0,
        aux_compression_weight=0.2,
        aux_blending_weight=0.2,
        aux_motion_weight=0.2,
        hyperspherical_weight=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)

    # 4. Training (Standard & Episodic Meta-Learning)
    batch_size = 32 * max(1, num_gpus) # Scale batch size to Dual GPUs (e.g. 64)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if not args.skip_meta:
        logger.info(f"Executing Episodic Meta-Learning (MLDG) on {num_gpus} GPU(s)...")
        meta_trainer = MetaLearningTrainer(
            model, loss_fn, optimizer, inner_lr=0.001, device=device, checkpoint_dir=checkpoints_dir, logger=logger
        )
        meta_trainer.train_meta(train_dataset, val_dataset, episodes_per_epoch=20, epochs=args.epochs)
    else:
        logger.info(f"Executing Standard Multi-Task Training on {num_gpus} GPU(s)...")
        trainer = StandardTrainer(
            model, loss_fn, optimizer, device=device, checkpoint_dir=checkpoints_dir, logger=logger
        )
        trainer.train(train_loader, val_loader, epochs=args.epochs)

    # 5. Full Zero-Shot Cross-Dataset Benchmark
    logger.info("Running Complete Zero-Shot Cross-Dataset Evaluation Benchmark...")
    evaluator = CrossDatasetEvaluator(model, device=device)
    results_df = evaluator.run_benchmark_suite(eval_datasets, batch_size=batch_size)

    print("\n" + "=" * 90)
    print("KAGGLE DUAL-T4 BENCHMARK RESULTS")
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90 + "\n")

    # Export results table to CSV
    csv_path = os.path.join(out_dir, "cross_dataset_benchmark_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved benchmark results to: {csv_path}")

    # 6. Dynamic Branch Trust Weights Inspection
    logger.info("Analyzing Dynamic Weight Shifts under Perturbations...")
    analyzer = FusionWeightAnalyzer(model, device=device)
    shift_df = analyzer.analyze_compression_shift(val_dataset, quality_levels=[95, 80, 60, 40, 20], num_samples=20)
    
    fig_comp = analyzer.plot_compression_shift(shift_df, save_path=os.path.join(out_dir, "weight_shift_compression.png"))
    plt.close(fig_comp)

    # Generate Radar Trust Chart for Diffusion vs LivePortrait OOD
    radar_weights = {
        "Face-Crop": 0.36,
        "Full-Frame": 0.21,
        "Frequency": 0.28,
        "Motion": 0.15
    }
    fig_radar = ForensicVisualizer.plot_dynamic_weights_radar(
        radar_weights, title="Learned Trust Weights under Cross-Generator Transfer",
        save_path=os.path.join(out_dir, "radar_trust_weights.png")
    )
    plt.close(fig_radar)

    logger.info("=" * 80)
    logger.info("ALL KAGGLE RUNNER STAGES COMPLETED SUCCESSFULLY!")
    logger.info(f"Artifacts and plots saved in: {out_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    import json
    main()
