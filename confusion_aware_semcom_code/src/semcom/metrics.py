from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    min_recall: float
    worst_directed_confusion: float
    per_class_recall: list[float]
    per_class_f1: list[float]
    confusion_matrix: list[list[int]]

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "min_recall": self.min_recall,
            "worst_directed_confusion": self.worst_directed_confusion,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
            "confusion_matrix": self.confusion_matrix,
        }


def confusion_from_predictions(targets: torch.Tensor, predictions: torch.Tensor, num_classes: int) -> torch.Tensor:
    targets = targets.detach().to(torch.int64).cpu()
    predictions = predictions.detach().to(torch.int64).cpu()
    valid = (targets >= 0) & (targets < num_classes)
    indices = num_classes * targets[valid] + predictions[valid]
    matrix = torch.bincount(indices, minlength=num_classes * num_classes)
    return matrix.reshape(num_classes, num_classes)


def metrics_from_confusion(confusion: torch.Tensor) -> ClassificationMetrics:
    cm = confusion.to(torch.float64)
    total = cm.sum().clamp_min(1.0)
    tp = torch.diag(cm)
    support = cm.sum(dim=1)
    predicted = cm.sum(dim=0)
    recall = tp / support.clamp_min(1.0)
    precision = tp / predicted.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)

    row_rates = cm / support[:, None].clamp_min(1.0)
    off_diag = row_rates.clone()
    off_diag.fill_diagonal_(0.0)
    worst = off_diag.max().item() if off_diag.numel() else 0.0

    return ClassificationMetrics(
        accuracy=(tp.sum() / total).item(),
        macro_f1=f1.mean().item(),
        min_recall=recall.min().item(),
        worst_directed_confusion=worst,
        per_class_recall=recall.tolist(),
        per_class_f1=f1.tolist(),
        confusion_matrix=confusion.to(torch.int64).tolist(),
    )


def symmetric_confusion_weights(confusion: torch.Tensor) -> torch.Tensor:
    cm = confusion.to(torch.float64)
    row = cm / cm.sum(dim=1, keepdim=True).clamp_min(1.0)
    weights = 0.5 * (row + row.T)
    weights.fill_diagonal_(0.0)
    return weights.to(torch.float32)


def mean_confidence_interval(values: list[float], z: float = 1.96) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    mean = float(array.mean())
    if array.size == 1:
        return mean, 0.0
    half = float(z * array.std(ddof=1) / np.sqrt(array.size))
    return mean, half
