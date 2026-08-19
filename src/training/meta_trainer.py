"""
Episodic Meta-Learning Trainer (MLDG: Meta-Learning for Domain Generalization).
Optimizes the domain-conditioned fusion module and feature projectors over pseudo-domains
carved from FaceForensics++, explicitly practicing domain transfer at every training step.
"""

import time
import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm

from ..models.detector import AdaptiveFusionDetector
from ..models.losses.multi_task_loss import MultiTaskForensicLoss
from ..data.dataset import MultiBranchDeepfakeDataset, EpisodicPseudoDomainSampler
from ..utils.logger import MetricTracker, setup_logger
from ..utils.checkpoint import save_checkpoint
from ..evaluation.metrics import compute_forensic_metrics


class MetaLearningTrainer:
    """
    MLDG-style Episodic Meta-Trainer.
    Simulates domain-shift at train time by meta-training on subset of pseudo-domains
    and meta-testing on a held-out pseudo-domain.
    """

    def __init__(
        self,
        model: AdaptiveFusionDetector,
        loss_fn: MultiTaskForensicLoss,
        optimizer: torch.optim.Optimizer,
        inner_lr: float = 0.001,
        meta_test_weight: float = 1.0,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints_meta",
        grad_clip: float = 1.0,
        logger: Optional[Any] = None
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.inner_lr = inner_lr
        self.beta = meta_test_weight
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.grad_clip = grad_clip
        self.logger = logger or setup_logger("MetaTrainer")
        self.tracker = MetricTracker()
        self.best_val_auc = 0.0

    def _prepare_batch(self, dataset: MultiBranchDeepfakeDataset, indices: List[int]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Collates indexed samples into device tensors."""
        face_crops, full_frames, freqs, motions = [], [], [], []
        labels, comps, blends, mots = [], [], [], []

        for idx in indices:
            sample = dataset[idx]
            face_crops.append(sample["face_crop"])
            full_frames.append(sample["full_frame"])
            freqs.append(sample["frequency"])
            motions.append(sample["motion"])
            labels.append(sample["label"])
            comps.append(sample["compression_target"])
            blends.append(sample["blending_target"])
            mots.append(sample["motion_target"])

        inputs = {
            "face_crop": torch.stack(face_crops).to(self.device),
            "full_frame": torch.stack(full_frames).to(self.device),
            "frequency": torch.stack(freqs).to(self.device),
            "motion": torch.stack(motions).to(self.device),
        }
        targets = {
            "label": torch.stack(labels).to(self.device),
            "compression_target": torch.stack(comps).to(self.device),
            "blending_target": torch.stack(blends).to(self.device),
            "motion_target": torch.stack(mots).to(self.device),
        }
        return inputs, targets

    def train_epoch(self, dataset: MultiBranchDeepfakeDataset, sampler: EpisodicPseudoDomainSampler, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.tracker.reset()

        pbar = tqdm(sampler, desc=f"Meta-Epoch {epoch}", leave=False)
        for episode in pbar:
            train_indices = episode["meta_train_indices"]
            test_indices = episode["meta_test_indices"]

            train_inputs, train_targets = self._prepare_batch(dataset, train_indices)
            test_inputs, test_targets = self._prepare_batch(dataset, test_indices)

            self.optimizer.zero_grad()

            # 1. Forward pass on Meta-Train pseudo-domains
            train_outputs = self.model(train_inputs)
            train_loss_dict = self.loss_fn(train_outputs, train_targets, branch_features=train_outputs.get("fused_feature"))
            l_train = train_loss_dict["total_loss"]

            # 2. Inner virtual step using functional parameter mapping (non-destructive)
            train_params = dict(self.model.named_parameters())
            grads = torch.autograd.grad(l_train, train_params.values(), create_graph=False, retain_graph=True, allow_unused=True)

            virtual_params = {}
            for (name, p), g in zip(train_params.items(), grads):
                if g is not None:
                    virtual_params[name] = p - self.inner_lr * g
                else:
                    virtual_params[name] = p

            # 3. Forward pass on Held-Out Meta-Test pseudo-domain using virtual parameters
            try:
                from torch.func import functional_call
                test_outputs = functional_call(self.model, virtual_params, (test_inputs,))
            except Exception:
                test_outputs = self.model(test_inputs)

            test_loss_dict = self.loss_fn(test_outputs, test_targets, branch_features=test_outputs.get("fused_feature"))
            l_test = test_loss_dict["total_loss"]

            # 4. Meta-Optimization Objective: L_meta = L_train + beta * L_test
            l_meta = l_train + self.beta * l_test
            l_meta.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Track metrics
            self.tracker.update("meta_loss", l_meta.item())
            self.tracker.update("meta_train_loss", l_train.item())
            self.tracker.update("meta_test_loss", l_test.item())
            pbar.set_postfix({"l_meta": f"{l_meta.item():.4f}", "l_test": f"{l_test.item():.4f}"})

        return self.tracker.get_all_averages()

    def train_meta(
        self,
        dataset: MultiBranchDeepfakeDataset,
        val_dataset: MultiBranchDeepfakeDataset,
        episodes_per_epoch: int = 50,
        epochs: int = 10
    ):
        self.logger.info(f"Starting Episodic Meta-Learning (MLDG) on {self.device}")
        
        sampler = EpisodicPseudoDomainSampler(
            dataset,
            episodes_per_epoch=episodes_per_epoch,
            meta_train_domains_per_ep=3,
            meta_test_domains_per_ep=1,
            batch_size_per_domain=8
        )

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

        for epoch in range(1, epochs + 1):
            start = time.time()
            train_res = self.train_epoch(dataset, sampler, epoch)
            
            # Validation
            val_res = self.evaluate(val_loader)
            elapsed = time.time() - start

            val_auc = val_res.get("auc", 0.0)
            is_best = val_auc > self.best_val_auc
            if is_best:
                self.best_val_auc = val_auc

            self.logger.info(
                f"Meta-Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
                f"Meta-Loss: {train_res['meta_loss']:.4f} | "
                f"Val AUC: {val_auc:.2f}% | "
                f"Val Acc: {val_res.get('accuracy', 0.0):.2f}% | "
                f"Val EER: {val_res.get('eer', 0.0):.2f}%"
            )

            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_auc": val_auc
                },
                is_best=is_best,
                checkpoint_dir=self.checkpoint_dir,
                filename=f"meta_checkpoint_{epoch:02d}.pth"
            )

    @torch.no_grad()
    def evaluate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.eval()
        labels, probs = [], []
        for batch in val_loader:
            inputs = {
                "face_crop": batch["face_crop"].to(self.device),
                "full_frame": batch["full_frame"].to(self.device),
                "frequency": batch["frequency"].to(self.device),
                "motion": batch["motion"].to(self.device),
            }
            outputs = self.model(inputs)
            labels.extend(batch["label"].cpu().numpy().tolist())
            probs.extend(outputs["is_fake_prob"].cpu().numpy().tolist())

        return compute_forensic_metrics(labels, probs)
