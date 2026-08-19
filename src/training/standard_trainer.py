"""
Standard Multi-Task Trainer for Deepfake Detection.
Trains the 4-branch detector with auxiliary self-supervised heads and hyperspherical loss.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from tqdm import tqdm

from ..models.detector import AdaptiveFusionDetector
from ..models.losses.multi_task_loss import MultiTaskForensicLoss
from ..utils.logger import MetricTracker, setup_logger
from ..utils.checkpoint import save_checkpoint
from ..evaluation.metrics import compute_forensic_metrics


class StandardTrainer:
    """Standard supervised & self-supervised multi-task trainer."""

    def __init__(
        self,
        model: AdaptiveFusionDetector,
        loss_fn: MultiTaskForensicLoss,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        grad_clip: float = 1.0,
        logger: Optional[Any] = None
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.grad_clip = grad_clip
        self.logger = logger or setup_logger("StandardTrainer")
        self.tracker = MetricTracker()
        self.best_val_auc = 0.0

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.tracker.reset()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch in pbar:
            # Move tensors to device
            batch_inputs = {
                "face_crop": batch["face_crop"].to(self.device),
                "full_frame": batch["full_frame"].to(self.device),
                "frequency": batch["frequency"].to(self.device),
                "motion": batch["motion"].to(self.device),
            }
            targets = {
                "label": batch["label"].to(self.device),
                "compression_target": batch["compression_target"].to(self.device),
                "blending_target": batch["blending_target"].to(self.device),
                "motion_target": batch["motion_target"].to(self.device),
            }

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch_inputs)
            
            # Loss computation
            loss_dict = self.loss_fn(outputs, targets, branch_features=outputs.get("fused_feature"))
            total_loss = loss_dict["total_loss"]

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Update metrics
            metrics = {k: v.item() for k, v in loss_dict.items()}
            self.tracker.update_dict(metrics, n=batch["label"].shape[0])
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "cls_loss": f"{loss_dict['cls_loss'].item():.4f}"})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return self.tracker.get_all_averages()

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, desc: str = "Val") -> Dict[str, float]:
        self.model.eval()
        all_labels = []
        all_probs = []
        val_losses = []

        for batch in tqdm(val_loader, desc=f"Evaluating [{desc}]", leave=False):
            batch_inputs = {
                "face_crop": batch["face_crop"].to(self.device),
                "full_frame": batch["full_frame"].to(self.device),
                "frequency": batch["frequency"].to(self.device),
                "motion": batch["motion"].to(self.device),
            }
            targets = {
                "label": batch["label"].to(self.device),
                "compression_target": batch["compression_target"].to(self.device),
                "blending_target": batch["blending_target"].to(self.device),
                "motion_target": batch["motion_target"].to(self.device),
            }

            outputs = self.model(batch_inputs)
            loss_dict = self.loss_fn(outputs, targets, branch_features=outputs.get("fused_feature"))
            
            val_losses.append(loss_dict["total_loss"].item())
            all_labels.extend(batch["label"].cpu().numpy().tolist())
            all_probs.extend(outputs["is_fake_prob"].cpu().numpy().tolist())

        metrics = compute_forensic_metrics(all_labels, all_probs)
        metrics["val_loss"] = float(sum(val_losses) / max(1, len(val_losses)))
        return metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10):
        self.logger.info(f"Starting standard multi-task training for {epochs} epochs on {self.device}")
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader, desc="Validation")
            elapsed = time.time() - start_time

            val_auc = val_metrics.get("auc", 0.0)
            is_best = val_auc > self.best_val_auc
            if is_best:
                self.best_val_auc = val_auc

            self.logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val AUC: {val_auc:.2f}% | "
                f"Val Acc: {val_metrics.get('accuracy', 0.0):.2f}% | "
                f"Val EER: {val_metrics.get('eer', 0.0):.2f}%"
            )

            # Save checkpoint
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_auc": val_auc,
                },
                is_best=is_best,
                checkpoint_dir=self.checkpoint_dir,
                filename=f"checkpoint_epoch_{epoch:02d}.pth"
            )
