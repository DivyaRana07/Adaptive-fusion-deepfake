"""
Forensic Evaluation Metrics.
Computes Area Under ROC Curve (AUC), Equal Error Rate (EER), Classification Accuracy,
F1-Score, and Calibration Reliability.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, brier_score_loss
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from typing import Dict, List, Union


def calculate_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Computes the Equal Error Rate (EER) where False Positive Rate == False Negative Rate.
    Returns: EER percentage (0.0 to 100.0)
    """
    if len(np.unique(y_true)) < 2:
        return 50.0

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr

    # Find the intersection point where fpr == fnr
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except Exception:
        # Fallback to closest point
        diff = np.abs(fpr - fnr)
        idx = np.argmin(diff)
        eer = (fpr[idx] + fnr[idx]) / 2.0

    return float(eer * 100.0)


def compute_forensic_metrics(
    labels: Union[List[int], np.ndarray],
    probabilities: Union[List[float], np.ndarray],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes all standard deepfake evaluation metrics.
    Args:
        labels: Ground truth binary labels (0=real, 1=fake)
        probabilities: Predicted probabilities of being fake in [0, 1]
        threshold: Decision threshold for discrete classification
    Returns:
        Dictionary with AUC, EER, Accuracy, F1-Score, Brier Score
    """
    y_true = np.array(labels)
    y_prob = np.array(probabilities)

    # Check for single-class degenerate case
    if len(np.unique(y_true)) < 2:
        return {
            "auc": 50.0,
            "eer": 50.0,
            "accuracy": float(accuracy_score(y_true, (y_prob >= threshold).astype(int)) * 100.0),
            "f1_score": 0.0,
            "brier_score": 0.25
        }

    # 1. AUC-ROC
    auc = float(roc_auc_score(y_true, y_prob) * 100.0)

    # 2. Equal Error Rate (EER)
    eer = calculate_eer(y_true, y_prob)

    # 3. Accuracy
    preds = (y_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_true, preds) * 100.0)

    # 4. F1-Score
    f1 = float(f1_score(y_true, preds, zero_division=0) * 100.0)

    # 5. Brier Score / Calibration Error
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "auc": round(auc, 2),
        "eer": round(eer, 2),
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 2),
        "brier_score": round(brier, 4)
    }
