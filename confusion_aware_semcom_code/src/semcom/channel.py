from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class TransmissionPlan:
    selected_groups: tuple[int, ...]
    copies: tuple[int, ...]
    pair: tuple[int, int]
    severity: float

    def __post_init__(self) -> None:
        if len(self.selected_groups) != len(self.copies):
            raise ValueError("selected_groups and copies must have equal length")
        if any(value <= 0 for value in self.copies):
            raise ValueError("Every selected group requires at least one copy")

    @property
    def used_slots(self) -> int:
        return int(sum(self.copies))


@dataclass
class LinkStatistics:
    complex_symbols_per_sample: int
    selected_groups_mean: float
    replicas_mean: float
    tx_power_mean: float
    channel_type: str


class AnalogFeatureLink(torch.nn.Module):
    """Differentiable real-feature to complex-baseband link with repetition UEP.

    A feature group of D real values occupies D/2 complex symbols. The slot budget
    is the number of feature-group slots, so the total channel uses per sample are
    budget_slots * D/2. Extra slots repeat high-priority groups. The receiver
    equalizes and averages copies before placing groups back in their original
    positions.
    """

    def __init__(
        self,
        average_power: float = 1.0,
        fill_mode: str = "zero",
        normalization_scales: torch.Tensor | float = 1.0,
    ) -> None:
        super().__init__()

        self.average_power = float(average_power)
        self.fill_mode = fill_mode

        if not math.isfinite(self.average_power) \
                or self.average_power <= 0:
            raise ValueError(
                "average_power must be finite and positive"
            )

        scales = torch.as_tensor(
            normalization_scales,
            dtype=torch.float32,
        )

        if scales.ndim == 0:
            scales = scales.reshape(1)

        if scales.ndim != 1:
            raise ValueError(
                "normalization_scales must be scalar "
                "or 1-D tensor"
            )

        if not torch.isfinite(scales).all():
            raise ValueError(
                "normalization_scales contains NaN/Inf"
            )

        if (scales <= 0).any():
            raise ValueError(
                "Every normalization scale must be positive"
            )

        self.register_buffer(
            "normalization_scales",
            scales.clone(),
        )
    def forward(
        self,
        groups: torch.Tensor,
        plans: Sequence[TransmissionPlan],
        snr_db: torch.Tensor,
        channel_type: str,
        mask_token: torch.Tensor | None = None,
        csi_error_std: float = 0.0,
    ) -> tuple[torch.Tensor, LinkStatistics]:
        if groups.ndim != 3:
            raise ValueError("groups must have shape [batch, num_groups, group_dim]")
        batch, num_groups, group_dim = groups.shape
        if self.normalization_scales.numel() == 1:
            group_scales = (
                self.normalization_scales.expand(num_groups)
            )

        elif self.normalization_scales.numel() == num_groups:
            group_scales = self.normalization_scales

        else:
            raise ValueError(
                "Number of normalization scales does not "
                "match num_groups: "
                f"{self.normalization_scales.numel()} "
                f"vs {num_groups}"
    )
        if len(plans) != batch:
            raise ValueError("One transmission plan is required per sample")
        if group_dim % 2 != 0:
            raise ValueError("group_dim must be even")
        used_slots = {plan.used_slots for plan in plans}
        if len(used_slots) != 1:
            raise ValueError("All samples in a batch must use the same fixed slot budget")
        budget_slots = used_slots.pop()
        complex_per_group = group_dim // 2
        device = groups.device
        dtype = groups.dtype

        tx = torch.zeros(batch, budget_slots, complex_per_group, dtype=torch.complex64, device=device)
        slot_group_ids = torch.full((batch, budget_slots), -1, dtype=torch.long, device=device)
        cursor_by_sample: list[int] = []
        for b, plan in enumerate(plans):
            cursor = 0
            for group_id, copies in zip(plan.selected_groups, plan.copies):
                if group_id < 0 or group_id >= num_groups:
                    raise IndexError(f"Group id {group_id} outside [0,{num_groups})")
                group_complex = torch.view_as_complex(
                    groups[b, group_id]
                    .float()
                    .reshape(complex_per_group, 2)
                    .contiguous()
                )

                group_scale = (
                    group_scales[group_id]
                    .to(device=device)
                    .clamp_min(1e-8)
                )

                group_gain = (
                    self.average_power ** 0.5
                ) / group_scale

                normalized_group = (
                    group_complex * group_gain
                )

                for _ in range(copies):
                    tx[b, cursor] = normalized_group
                    slot_group_ids[b, cursor] = group_id
                    cursor += 1
                cursor_by_sample.append(cursor)
        if any(cursor != budget_slots for cursor in cursor_by_sample):
            raise RuntimeError("Transmission plan does not fill the fixed slot budget")

        # A fixed feature scale is calibrated on the training split and shared by
        # transmitter and receiver. This enforces the average power constraint in
        # expectation without requiring a per-sample normalization side channel.

        snr_db = snr_db.to(device=device, dtype=torch.float32).reshape(batch, 1, 1)
        noise_variance = self.average_power / torch.pow(10.0, snr_db / 10.0)
        noise = torch.sqrt(noise_variance / 2.0) * (
            torch.randn_like(tx.real) + 1j * torch.randn_like(tx.real)
        )
        channel_type = channel_type.lower()
        if channel_type == "awgn":
            h = torch.ones(batch, 1, 1, dtype=torch.complex64, device=device)
        elif channel_type == "rayleigh":
            h = (torch.randn(batch, 1, 1, device=device) + 1j * torch.randn(batch, 1, 1, device=device)) / (2.0**0.5)
        else:
            raise ValueError(f"Unsupported channel type: {channel_type}")
        received = h * tx + noise

        if csi_error_std > 0:
            error = csi_error_std * (
                torch.randn_like(h.real) + 1j * torch.randn_like(h.real)
            ) / (2.0**0.5)
            h_hat = h + error
        else:
            h_hat = h
        equalized = received / h_hat.clamp_min(1e-5) if not torch.is_complex(h_hat) else received / _safe_complex(h_hat)

        if self.fill_mode == "zero":
            recovered = torch.zeros_like(groups)
        elif self.fill_mode == "mask":
            if mask_token is None:
                raise ValueError("mask_token is required when fill_mode='mask'")
            recovered = mask_token.to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1).clone()
        else:
            raise ValueError(f"Unsupported fill mode: {self.fill_mode}")

        for b, plan in enumerate(plans):
            for group_id in plan.selected_groups:
                positions = torch.where(slot_group_ids[b] == group_id)[0]
                estimate = (
                    equalized[b, positions]
                    .mean(dim=0)
                )

                group_scale = (
                    group_scales[group_id]
                    .to(device=device)
                    .clamp_min(1e-8)
                )

                group_gain = (
                    self.average_power ** 0.5
                ) / group_scale

                estimate = estimate / group_gain
                recovered[b, group_id] = torch.view_as_real(estimate).reshape(group_dim).to(dtype)

        stats = LinkStatistics(
            complex_symbols_per_sample=budget_slots * complex_per_group,
            selected_groups_mean=float(sum(len(plan.selected_groups) for plan in plans) / batch),
            replicas_mean=float(budget_slots / max(sum(len(plan.selected_groups) for plan in plans) / batch, 1e-8)),
            tx_power_mean=float(tx.abs().square().mean().detach().cpu()),
            channel_type=channel_type,
        )
        return recovered, stats


def _safe_complex(value: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    magnitude = value.abs().clamp_min(epsilon)
    phase = value / value.abs().clamp_min(epsilon)
    return magnitude * phase
