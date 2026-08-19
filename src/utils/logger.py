"""
Structured logger and metric tracking utilities for experiments and meta-learning.
"""

import os
import sys
import logging
import json
import time
from typing import Dict, Any, Optional
from collections import defaultdict


def setup_logger(name: str = "deepfake_detector", log_dir: Optional[str] = "logs", level: int = logging.INFO) -> logging.Logger:
    """Configures console and file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{name}_{int(time.time())}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger


class MetricTracker:
    """Tracks running averages, losses, and accuracy metrics across batches and epochs."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.metrics = defaultdict(float)
        self.counts = defaultdict(int)
        self.history = defaultdict(list)
        
    def update(self, key: str, value: float, n: int = 1):
        """Accumulates metric values."""
        self.metrics[key] += float(value) * n
        self.counts[key] += n
        
    def update_dict(self, metrics_dict: Dict[str, float], n: int = 1):
        for k, v in metrics_dict.items():
            self.update(k, v, n)
            
    def get_average(self, key: str) -> float:
        if self.counts[key] == 0:
            return 0.0
        return self.metrics[key] / self.counts[key]
        
    def get_all_averages(self) -> Dict[str, float]:
        return {k: self.get_average(k) for k in self.metrics}
        
    def step_epoch(self) -> Dict[str, float]:
        """Saves current averages to history and resets current accumulator."""
        avg = self.get_all_averages()
        for k, v in avg.items():
            self.history[k].append(v)
        self.metrics = defaultdict(float)
        self.counts = defaultdict(int)
        return avg
        
    def save_json(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(dict(self.history), f, indent=2)
