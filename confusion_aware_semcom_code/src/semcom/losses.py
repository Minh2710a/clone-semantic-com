from __future__ import annotations

import torch
import torch.nn.functional as F


def channel_weighted_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    snr_db: torch.Tensor,
    weights: torch.Tensor,
    snr_grid: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    batch, num_classes = logits.shape
    if snr_db.numel() == 1:
        snr_db = snr_db.expand(batch)
    nearest = torch.argmin(torch.abs(snr_db[:, None] - snr_grid[None, :].to(snr_db.device)), dim=1)
    selected_weights = weights.to(logits.device)[nearest, targets]  # [B,C]
    true_logits = logits.gather(1, targets[:, None])
    hinge = torch.relu(float(margin) - (true_logits - logits))
    mask = 1.0 - F.one_hot(targets, num_classes=num_classes).to(logits.dtype)
    numerator = (selected_weights * hinge * mask).sum()
    denominator = (selected_weights * mask).sum().clamp_min(1e-8)
    return numerator / denominator


def classwise_loss_variance(per_sample_ce: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    class_means: list[torch.Tensor] = []
    for class_index in range(num_classes):
        mask = targets == class_index
        if mask.any():
            class_means.append(per_sample_ce[mask].mean())
    if len(class_means) <= 1:
        return per_sample_ce.new_zeros(())
    stacked = torch.stack(class_means)
    return torch.mean((stacked - stacked.mean()) ** 2)
