from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torchvision import models


class SmallCNNBackbone(nn.Module):
    """Small fallback backbone used by the smoke test and CPU-only debugging."""

    def __init__(self, out_channels: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    name = name.lower()
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        network = models.mobilenet_v3_small(weights=weights)
        return network.features, 576
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        network = models.resnet18(weights=weights)
        backbone = nn.Sequential(*list(network.children())[:-2])
        return backbone, 512
    if name == "small_cnn":
        backbone = SmallCNNBackbone(out_channels=128)
        return backbone, backbone.out_channels
    raise ValueError(f"Unsupported backbone: {name}")


@dataclass
class SemanticOutput:
    groups: torch.Tensor
    vector: torch.Tensor
    local_logits: torch.Tensor
    receiver_logits: torch.Tensor | None = None


class FeatureSemanticModel(nn.Module):
    """Transmitter feature extractor, local pair head, and receiver classifier."""

    def __init__(
        self,
        *,
        num_classes: int,
        backbone_name: str,
        pretrained: bool,
        feature_dim: int,
        num_groups: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dim % num_groups != 0:
            raise ValueError("feature_dim must be divisible by num_groups")
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.num_groups = int(num_groups)
        self.group_dim = feature_dim // num_groups
        if self.group_dim % 2 != 0:
            raise ValueError("group_dim must be even for real-to-complex mapping")

        self.backbone, backbone_channels = build_backbone(backbone_name, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_channels, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.local_head = nn.Linear(feature_dim, num_classes)
        self.receiver_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.mask_token = nn.Parameter(torch.zeros(num_groups, self.group_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.backbone(images)
        vector = self.projector(self.pool(feature_map))
        groups = vector.reshape(vector.shape[0], self.num_groups, self.group_dim)
        return groups, vector

    def local_logits_from_groups(self, groups: torch.Tensor) -> torch.Tensor:
        return self.local_head(groups.flatten(1))

    def receiver_logits_from_groups(self, groups: torch.Tensor) -> torch.Tensor:
        return self.receiver_head(groups.flatten(1))

    def forward_clean(self, images: torch.Tensor) -> SemanticOutput:
        groups, vector = self.encode(images)
        local_logits = self.local_head(vector)
        receiver_logits = self.receiver_head(vector)
        return SemanticOutput(groups, vector, local_logits, receiver_logits)

    def initialize_receiver_from_local(self) -> None:
        """Initialize the final receiver layer from the clean local classifier."""
        final = self.receiver_head[-1]
        if isinstance(final, nn.Linear) and final.weight.shape == self.local_head.weight.shape:
            with torch.no_grad():
                final.weight.copy_(self.local_head.weight)
                final.bias.copy_(self.local_head.bias)

    def set_trainable_for_semcom(self, mode: str) -> None:
        mode = mode.lower()
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.receiver_head.parameters():
            parameter.requires_grad = True
        self.mask_token.requires_grad = True
        if mode == "head":
            return
        if mode == "projector":
            for parameter in self.projector.parameters():
                parameter.requires_grad = True
            return
        if mode == "partial":
            for parameter in self.projector.parameters():
                parameter.requires_grad = True
            # Unfreeze the last two backbone blocks when the backbone is sequential.
            children = list(self.backbone.children())
            for child in children[-2:]:
                for parameter in child.parameters():
                    parameter.requires_grad = True
            return
        if mode == "all":
            for parameter in self.parameters():
                parameter.requires_grad = True
            # Local head remains fixed to avoid moving the pair estimator during policy training.
            for parameter in self.local_head.parameters():
                parameter.requires_grad = False
            return
        raise ValueError(f"Unknown semcom trainable mode: {mode}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
