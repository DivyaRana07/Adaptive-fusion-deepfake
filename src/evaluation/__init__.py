"""Evaluation metrics, cross-dataset zero-shot benchmarkers, and ablation suites."""
from .metrics import compute_forensic_metrics, calculate_eer
from .evaluator import CrossDatasetEvaluator
from .ablation_matrix import AblationMatrixRunner
from .weight_analyzer import FusionWeightAnalyzer

__all__ = [
    "compute_forensic_metrics",
    "calculate_eer",
    "CrossDatasetEvaluator",
    "AblationMatrixRunner",
    "FusionWeightAnalyzer",
]
