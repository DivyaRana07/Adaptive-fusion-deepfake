"""
Forensic visualizer for 4-branch inspection, auxiliary uncertainty, and radar weight distribution charts.
"""

import numpy as np
import matplotlib.pyplot as plt
import io
from typing import Dict, List, Optional, Tuple


class ForensicVisualizer:
    """Visualizes multi-branch forensic cues and adaptive fusion decisions."""

    @staticmethod
    def plot_4branch_inputs(
        face_crop: np.ndarray,
        full_frame: np.ndarray,
        freq_map: np.ndarray,
        motion_flow: np.ndarray,
        title: str = "4-Branch Forensic Stream Inputs",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plots the four parallel forensic stream inputs side-by-side."""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # 1. Face Crop
        axes[0].imshow(np.clip(face_crop, 0, 1) if face_crop.max() <= 1.0 else face_crop.astype(np.uint8))
        axes[0].set_title("1. Face-Crop (Local Seams)")
        axes[0].axis("off")
        
        # 2. Full Frame
        axes[1].imshow(np.clip(full_frame, 0, 1) if full_frame.max() <= 1.0 else full_frame.astype(np.uint8))
        axes[1].set_title("2. Full-Frame (Scene Context)")
        axes[1].axis("off")
        
        # 3. Frequency / Wavelet Map
        if freq_map.ndim == 3 and freq_map.shape[-1] >= 3:
            axes[2].imshow(np.clip(freq_map[..., :3], 0, 1) if freq_map.max() <= 1.0 else freq_map[..., :3].astype(np.uint8))
        else:
            axes[2].imshow(freq_map.squeeze(), cmap="magma")
        axes[2].set_title("3. Frequency (DWT / Spectral)")
        axes[2].axis("off")
        
        # 4. Motion / Optical Flow
        if motion_flow.ndim == 3 and motion_flow.shape[-1] == 3:
            axes[3].imshow(np.clip(motion_flow, 0, 1) if motion_flow.max() <= 1.0 else motion_flow.astype(np.uint8))
        else:
            axes[3].imshow(np.linalg.norm(motion_flow, axis=-1) if motion_flow.ndim == 3 else motion_flow, cmap="viridis")
        axes[3].set_title("4. Motion (Optical Flow / RAFT)")
        axes[3].axis("off")
        
        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    @staticmethod
    def plot_dynamic_weights_radar(
        branch_weights: Dict[str, float],
        title: str = "Adaptive Fusion Branch Trust Weights",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Draws a radar chart of branch trust weights allocated by the domain-conditioned fusion module."""
        categories = list(branch_weights.keys())
        values = list(branch_weights.values())
        
        # Number of variables
        num_vars = len(categories)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values_closed = values + values[:1]
        angles_closed = angles + angles[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        
        # Plot and fill
        ax.plot(angles_closed, values_closed, color="#1f77b4", linewidth=2.5, linestyle="solid")
        ax.fill(angles_closed, values_closed, color="#1f77b4", alpha=0.35)
        
        # Set category labels
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(0.6, max(values) + 0.1))
        
        # Styling
        ax.grid(color="gray", linestyle="--", alpha=0.5)
        ax.set_title(title, size=13, weight="bold", y=1.1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    @staticmethod
    def plot_cross_dataset_comparison(
        results_dict: Dict[str, Dict[str, float]],
        metric_name: str = "AUC",
        title: str = "Cross-Dataset Zero-Shot Performance Benchmark",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plots bar chart comparison across benchmark datasets (FF++, Celeb-DF, DFDC, Diffusion, Reenactment)."""
        datasets = list(results_dict.keys())
        methods = list(next(iter(results_dict.values())).keys())
        
        x = np.arange(len(datasets))
        width = 0.8 / len(methods)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, method in enumerate(methods):
            scores = [results_dict[d].get(method, 0.0) for d in datasets]
            rects = ax.bar(x + i * width, scores, width, label=method)
            # Add text labels on bars
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f"{height:.1f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)
                            
        ax.set_ylabel(f"{metric_name} (%)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(datasets, fontsize=11, fontweight="bold")
        ax.legend(loc="lower right")
        ax.set_ylim(40, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig
