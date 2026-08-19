"""Adaptive fusion modules and baseline comparison fusion architectures."""
from .domain_conditioned import DomainConditionedFusion
from .baseline_fusion import (
    FixedAverageFusion,
    ConcatMLPFusion,
    SelfAttentionFusion,
)

__all__ = [
    "DomainConditionedFusion",
    "FixedAverageFusion",
    "ConcatMLPFusion",
    "SelfAttentionFusion",
]
