from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .channel import TransmissionPlan


@dataclass
class PolicyArtifacts:
    pair_rankings: torch.Tensor  # [C,C,G]
    class_rankings: torch.Tensor  # [C,G]
    global_ranking: torch.Tensor  # [G]
    confusion_weights: torch.Tensor  # [S,C,C]
    snr_grid: torch.Tensor  # [S]

    @classmethod
    def load(cls, codebook_path: str, graph_path: str, map_location: str | torch.device = "cpu"):
        codebook = torch.load(codebook_path, map_location=map_location, weights_only=False)
        graph = torch.load(graph_path, map_location=map_location, weights_only=False)
        return cls(
            pair_rankings=codebook["pair_rankings"].long(),
            class_rankings=codebook["class_rankings"].long(),
            global_ranking=codebook["global_ranking"].long(),
            confusion_weights=graph["weights"].float(),
            snr_grid=graph["snr_grid"].float(),
        )


class SemanticTransmissionPolicy:
    """Deterministic fixed-budget selector and repetition allocator."""

    SUPPORTED = {
        "full",
        "random",
        "global",
        "class_specific",
        "snr_only",
        "pairwise",
        "proposed",
    }

    def __init__(
        self,
        *,
        scheme: str,
        artifacts: PolicyArtifacts,
        num_groups: int,
        budget_slots: int,
        policy_cfg: dict[str, Any],
    ) -> None:
        if scheme not in self.SUPPORTED:
            raise ValueError(f"Unsupported scheme {scheme}; choose from {sorted(self.SUPPORTED)}")
        if budget_slots <= 0:
            raise ValueError("budget_slots must be positive")
        self.scheme = scheme
        self.artifacts = artifacts
        self.num_groups = int(num_groups)
        self.budget_slots = int(budget_slots)
        self.cfg = policy_cfg
        self.min_groups = int(policy_cfg.get("min_groups", 1))
        self.keep_table = sorted(
            [(float(item["max_snr"]), float(item["keep_ratio"])) for item in policy_cfg["snr_keep_table"]],
            key=lambda item: item[0],
        )
        self.severity_low = float(policy_cfg["severity_thresholds"]["low"])
        self.severity_high = float(policy_cfg["severity_thresholds"]["high"])
        self.severity_multipliers = {
            "low": float(policy_cfg["severity_multipliers"]["low"]),
            "medium": float(policy_cfg["severity_multipliers"]["medium"]),
            "high": float(policy_cfg["severity_multipliers"]["high"]),
        }

    def make_plans(self, local_logits: torch.Tensor, snr_db: torch.Tensor) -> list[TransmissionPlan]:
        if local_logits.ndim != 2:
            raise ValueError("local_logits must have shape [batch, classes]")
        batch = local_logits.shape[0]
        if snr_db.numel() == 1:
            snr_db = snr_db.expand(batch)
        top2 = torch.topk(local_logits.detach(), k=2, dim=1).indices.cpu()
        snr_cpu = snr_db.detach().float().cpu().reshape(-1)
        plans: list[TransmissionPlan] = []
        for b in range(batch):
            first, second = int(top2[b, 0]), int(top2[b, 1])
            current_snr = float(snr_cpu[b])
            severity = self._severity(first, second, current_snr)
            selected_count = self._selected_count(current_snr, severity)
            ranking = self._ranking(first, second)
            selected = tuple(int(value) for value in ranking[:selected_count].tolist())
            copies = self._allocate_copies(selected_count)
            plans.append(
                TransmissionPlan(
                    selected_groups=selected,
                    copies=tuple(copies),
                    pair=(first, second),
                    severity=severity,
                )
            )
        return plans

    def _ranking(self, first: int, second: int) -> torch.Tensor:
        if self.scheme == "random":
            return torch.randperm(self.num_groups)
        if self.scheme in {"global", "snr_only"}:
            return self.artifacts.global_ranking
        if self.scheme == "class_specific":
            return self.artifacts.class_rankings[first]
        if self.scheme in {"pairwise", "proposed"}:
            return self.artifacts.pair_rankings[first, second]
        if self.scheme == "full":
            return torch.arange(self.num_groups)
        raise AssertionError(self.scheme)

    def _selected_count(self, snr: float, severity: float) -> int:
        max_distinct = min(self.budget_slots, self.num_groups)
        if self.scheme in {"full", "random", "global", "class_specific"}:
            if self.scheme == "full" and self.budget_slots < self.num_groups:
                raise ValueError(
                    f"full scheme requires budget_slots >= num_groups ({self.num_groups}), got {self.budget_slots}"
                )
            return max_distinct
        ratio = self._base_keep_ratio(snr)
        if self.scheme == "proposed":
            if severity >= self.severity_high:
                ratio *= self.severity_multipliers["high"]
            elif severity >= self.severity_low:
                ratio *= self.severity_multipliers["medium"]
            else:
                ratio *= self.severity_multipliers["low"]
        selected = int(round(max_distinct * ratio))
        return max(self.min_groups, min(selected, max_distinct))

    def _base_keep_ratio(self, snr: float) -> float:
        for max_snr, ratio in self.keep_table:
            if snr <= max_snr:
                return ratio
        return self.keep_table[-1][1]

    def _severity(self, first: int, second: int, snr: float) -> float:
        index = int(torch.argmin(torch.abs(self.artifacts.snr_grid - snr)).item())
        return float(self.artifacts.confusion_weights[index, first, second].item())

    def _allocate_copies(self, selected_count: int) -> list[int]:
        if selected_count > self.budget_slots:
            raise ValueError("selected_count cannot exceed budget_slots")
        copies = [1] * selected_count
        remaining = self.budget_slots - selected_count
        # Ranking order is importance order, so cycle extra copies from most important downward.
        position = 0
        while remaining > 0:
            copies[position % selected_count] += 1
            position += 1
            remaining -= 1
        if sum(copies) != self.budget_slots:
            raise AssertionError("Copy allocation failed to fill budget")
        return copies
