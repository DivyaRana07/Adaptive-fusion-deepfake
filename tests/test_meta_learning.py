"""
Test episodic meta-learning optimization loops.
"""

import pytest
import torch
from src.models.detector import AdaptiveFusionDetector
from src.models.losses.multi_task_loss import MultiTaskForensicLoss
from src.data.dataset import MultiBranchDeepfakeDataset
from src.training.meta_trainer import MetaLearningTrainer


def test_meta_learning_train_step():
    dataset = MultiBranchDeepfakeDataset(num_synthetic_samples=16, target_size=112)
    val_dataset = MultiBranchDeepfakeDataset(num_synthetic_samples=8, target_size=112)

    detector = AdaptiveFusionDetector(
        backbone_name="efficientnet_b0",
        pretrained=False,
        feature_dim=64,
        fusion_type="domain_conditioned",
        use_hyperspherical=True,
        use_auxiliary=True
    )

    loss_fn = MultiTaskForensicLoss()
    optimizer = torch.optim.AdamW(detector.parameters(), lr=0.001)

    trainer = MetaLearningTrainer(
        detector, loss_fn, optimizer, inner_lr=0.001, device="cpu"
    )

    # Run 1 meta-epoch
    metrics = trainer.train_epoch(
        dataset,
        dataset=dataset,
        sampler=torch.utils.data.DataLoader(dataset, batch_size=4),
        epoch=1
    ) if False else None # Direct call below

    from src.data.dataset import EpisodicPseudoDomainSampler
    sampler = EpisodicPseudoDomainSampler(dataset, episodes_per_epoch=2, batch_size_per_domain=2)
    epoch_res = trainer.train_epoch(dataset, sampler, epoch=1)

    assert "meta_loss" in epoch_res
    assert epoch_res["meta_loss"] > 0.0
