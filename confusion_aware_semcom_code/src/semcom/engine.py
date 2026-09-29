from __future__ import annotations

import math
import signal
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .channel import AnalogFeatureLink, TransmissionPlan
from .checkpoint import CheckpointManager
from .config import config_hash
from .io import append_csv_row, atomic_json_dump, atomic_torch_save
from .losses import channel_weighted_margin_loss, classwise_loss_variance
from .metrics import confusion_from_predictions, metrics_from_confusion, symmetric_confusion_weights
from .models import FeatureSemanticModel, count_parameters, count_trainable_parameters
from .policy import PolicyArtifacts, SemanticTransmissionPolicy


class StopRequested:
    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None

    def install(self) -> None:
        def _handler(signum, _frame):
            self.requested = True
            self.signal_number = signum

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)


def get_device(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg["project"].get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def build_model(cfg: dict[str, Any], num_classes: int) -> FeatureSemanticModel:
    model_cfg = cfg["model"]
    return FeatureSemanticModel(
        num_classes=num_classes,
        backbone_name=model_cfg["backbone"],
        pretrained=bool(model_cfg.get("pretrained", False)),
        feature_dim=int(model_cfg["feature_dim"]),
        num_groups=int(model_cfg["num_groups"]),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def train_clean_classifier(
    cfg: dict[str, Any],
    loaders: dict[str, torch.utils.data.DataLoader],
    num_classes: int,
    run_dir: str | Path,
    resume: str = "auto",
) -> Path:
    device = get_device(cfg)
    model = build_model(cfg, num_classes).to(device)
    train_cfg = cfg["clean_training"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(int(train_cfg["epochs"]), 1))
    use_amp = bool(train_cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    manager = CheckpointManager(run_dir, "clean")
    cfg_hash = config_hash(cfg)
    start_epoch = 0
    resume_batch = 0
    resume_loader_state = None
    resume_running = None
    global_step = 0
    best_metric = -math.inf
    checkpoint = manager.load(
        model,
        optimizer,
        scheduler,
        scaler,
        mode=resume,
        expected_hash=cfg_hash,
        map_location=device,
    )
    if checkpoint:
        extra = checkpoint.get("extra", {})
        status = extra.get("status", "epoch_complete")
        if status == "epoch_complete":
            start_epoch = int(checkpoint["epoch"]) + 1
        else:
            start_epoch = int(checkpoint["epoch"])
            resume_batch = int(extra.get("batch_in_epoch", -1)) + 1
            resume_loader_state = extra.get("epoch_loader_rng_state")
            resume_running = extra.get("running")
        global_step = int(checkpoint["global_step"])
        best_metric = float(checkpoint["best_metric"])
        print(
            f"[resume] clean stage from {checkpoint['_loaded_from']} at epoch {start_epoch}, "
            f"batch {resume_batch}"
        )

    stop = StopRequested()
    stop.install()
    history_path = Path(run_dir) / "logs" / "clean_history.csv"
    save_every_steps = int(train_cfg.get("save_every_steps", 500))
    model.train()

    try:
        for epoch in range(start_epoch, int(train_cfg["epochs"])):
            model.train()
            train_loader = loaders["train"]
            if epoch == start_epoch and resume_batch > 0 and resume_loader_state is not None:
                train_loader.generator.set_state(resume_loader_state)
            epoch_loader_rng_state = train_loader.generator.get_state()
            if epoch == start_epoch and resume_running is not None:
                running_loss = float(resume_running.get("running_loss", 0.0))
                seen = int(resume_running.get("seen", 0))
            else:
                running_loss = 0.0
                seen = 0
            progress = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"clean epoch {epoch + 1}",
                leave=False,
            )
            for batch_index, (images, labels, _paths) in progress:
                if epoch == start_epoch and batch_index < resume_batch:
                    continue
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp) if device.type == "cuda" else nullcontext()
                with autocast_ctx:
                    output = model.forward_clean(images)
                    local_loss = F.cross_entropy(output.local_logits, labels)
                    receiver_loss = F.cross_entropy(output.receiver_logits, labels)
                    loss = local_loss + float(train_cfg.get("receiver_loss_weight", 1.0)) * receiver_loss
                scaler.scale(loss).backward()
                if float(train_cfg.get("grad_clip", 0.0)) > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()

                batch_size = labels.numel()
                running_loss += float(loss.detach()) * batch_size
                seen += batch_size
                global_step += 1
                progress.set_postfix(loss=running_loss / max(seen, 1))

                if save_every_steps > 0 and global_step % save_every_steps == 0:
                    manager.save(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        best_metric=best_metric,
                        config_hash=cfg_hash,
                        extra={
                            "status": "mid_epoch",
                            "batch_in_epoch": batch_index,
                            "epoch_loader_rng_state": epoch_loader_rng_state,
                            "running": {"running_loss": running_loss, "seen": seen},
                            "train_loss": running_loss / max(seen, 1),
                        },
                        tag=f"step_{global_step:08d}",
                    )
                if stop.requested:
                    raise InterruptedError(f"Stop requested by signal {stop.signal_number}")

            resume_batch = 0
            resume_loader_state = None
            resume_running = None
            scheduler.step()
            val_result = evaluate_clean(model, loaders["val"], device, num_classes)
            metric = val_result["macro_f1"]
            is_best = metric > best_metric
            if is_best:
                best_metric = metric
            append_csv_row(
                history_path,
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": running_loss / max(seen, 1),
                    "val_accuracy": val_result["accuracy"],
                    "val_macro_f1": val_result["macro_f1"],
                    "val_min_recall": val_result["min_recall"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
            )
            manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                config_hash=cfg_hash,
                extra={"status": "epoch_complete", "validation": val_result},
                tag=f"epoch_{epoch:04d}",
                is_best=is_best,
            )
    except (KeyboardInterrupt, InterruptedError) as exc:
        print(f"[checkpoint] interrupted clean training: {exc}")
        manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=max(start_epoch, locals().get("epoch", start_epoch)),
            global_step=global_step,
            best_metric=best_metric,
            config_hash=cfg_hash,
            extra={
                "status": "interrupted",
                "reason": str(exc),
                "batch_in_epoch": locals().get("batch_index", -1),
                "epoch_loader_rng_state": locals().get("epoch_loader_rng_state"),
                "running": {
                    "running_loss": locals().get("running_loss", 0.0),
                    "seen": locals().get("seen", 0),
                },
            },
            tag="emergency",
        )
        raise

    if not manager.best_path.exists():
        raise RuntimeError("Clean training completed without a best checkpoint")
    return manager.best_path


@torch.inference_mode()
def evaluate_clean(
    model: FeatureSemanticModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for images, labels, _paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model.forward_clean(images)
        predictions = output.local_logits.argmax(dim=1)
        confusion += confusion_from_predictions(labels, predictions, num_classes)
    return metrics_from_confusion(confusion).as_dict()


@torch.inference_mode()
def calibrate_feature_scale(
    model: FeatureSemanticModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_path: str | Path,
) -> torch.Tensor:
    model.eval()

    num_groups = model.num_groups

    # Tổng power riêng cho từng feature group
    group_power_sum = torch.zeros(
        num_groups,
        dtype=torch.float64,
        device=device,
    )

    complex_samples_per_group = 0

    for images, _labels, _paths in tqdm(
        loader,
        desc="calibrate feature scales",
        leave=False,
    ):
        images = images.to(device, non_blocking=True)

        groups, _vector = model.encode(images)

        # groups shape:
        # [batch, num_groups, group_dim]
        #
        # complex_groups shape:
        # [batch, num_groups, group_dim / 2]
        complex_groups = torch.view_as_complex(
            groups.float().reshape(
                groups.shape[0],
                groups.shape[1],
                groups.shape[2] // 2,
                2,
            ).contiguous()
        )

        group_power_sum += (
            complex_groups
            .abs()
            .square()
            .sum(dim=(0, 2))
            .to(torch.float64)
        )

        complex_samples_per_group += (
            complex_groups.shape[0]
            * complex_groups.shape[2]
        )

    group_mean_power = (
        group_power_sum
        / max(complex_samples_per_group, 1)
    )

    normalization_scales = (
        torch.sqrt(group_mean_power)
        .clamp_min(1e-8)
        .to(torch.float32)
    )

    payload = {
        "artifact_version": 2,
        "normalization_scales": normalization_scales.cpu(),
        "complex_samples_per_group": complex_samples_per_group,
    }

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_torch_save(payload, output_path)

    return normalization_scales.cpu()

def load_feature_scales(
    path: str | Path,
) -> torch.Tensor:

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if "normalization_scales" not in payload:
        raise RuntimeError(
            "Old feature-scale artifact detected. "
            "Rebuild feature_scale.pt."
        )

    scales = torch.as_tensor(
        payload["normalization_scales"],
        dtype=torch.float32,
    )

    if scales.ndim != 1:
        raise ValueError(
            "normalization_scales must be 1-D"
        )

    if not torch.isfinite(scales).all():
        raise ValueError(
            "normalization_scales contains NaN/Inf"
        )

    if (scales <= 0).any():
        raise ValueError(
            "Every normalization scale must be positive"
        )

    return scales

@torch.inference_mode()
def build_confusion_graph(
    cfg: dict[str, Any],
    model: FeatureSemanticModel,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    normalization_scales: torch.Tensor,
    output_path: str | Path,
) -> Path:
    device = next(model.parameters()).device
    model.eval()
    snr_grid = [float(value) for value in cfg["channel"]["snr_grid"]]
    graph_channel = str(cfg["artifacts"].get("graph_channel", "awgn"))
    link = AnalogFeatureLink(
        average_power=float(cfg["channel"].get("average_power", 1.0)),
        fill_mode="zero",
        normalization_scales=normalization_scales,
    ).to(device)
    weights = []
    confusions = []
    all_groups = tuple(range(model.num_groups))
    copies = tuple([1] * model.num_groups)
    for snr in snr_grid:
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        for images, labels, _paths in tqdm(loader, desc=f"graph {graph_channel} {snr:g} dB", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            groups, _ = model.encode(images)
            plans = [TransmissionPlan(all_groups, copies, (0, 1), 0.0) for _ in range(images.shape[0])]
            snr_tensor = torch.full((images.shape[0],), snr, device=device)
            recovered, _stats = link(groups, plans, snr_tensor, graph_channel)
            logits = model.local_logits_from_groups(recovered)
            predictions = logits.argmax(dim=1)
            confusion += confusion_from_predictions(labels, predictions, num_classes)
        confusions.append(confusion)
        weights.append(symmetric_confusion_weights(confusion))
    payload = {
        "snr_grid": torch.tensor(snr_grid, dtype=torch.float32),
        "confusions": torch.stack(confusions),
        "weights": torch.stack(weights),
        "channel": graph_channel,
        "normalization_scales": normalization_scales.detach().cpu(),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, output_path)
    return output_path


@torch.inference_mode()
def build_pairwise_codebook(
    model: FeatureSemanticModel,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    output_path: str | Path,
) -> Path:
    """Build pair-specific group rankings for the linear local head.

    For a linear local head, the gradient of logit_i-logit_j with respect to z is
    exactly w_i-w_j. Therefore the mean gradient-times-activation score over samples
    in classes i or j is E[|z|] * |w_i-w_j|, aggregated by feature group.
    """
    device = next(model.parameters()).device
    model.eval()
    feature_dim = model.feature_dim
    class_sum_abs = torch.zeros(num_classes, feature_dim, dtype=torch.float64, device=device)
    class_count = torch.zeros(num_classes, dtype=torch.float64, device=device)
    for images, labels, _paths in tqdm(loader, desc="pairwise codebook", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        _groups, vector = model.encode(images)
        for class_index in labels.unique().tolist():
            mask = labels == int(class_index)
            class_sum_abs[int(class_index)] += vector[mask].abs().sum(dim=0).to(torch.float64)
            class_count[int(class_index)] += mask.sum()
    class_mean_abs = class_sum_abs / class_count[:, None].clamp_min(1.0)
    weights = model.local_head.weight.detach().to(torch.float64)
    pair_scores = torch.zeros(num_classes, num_classes, model.num_groups, dtype=torch.float64, device=device)
    for first in range(num_classes):
        for second in range(num_classes):
            if first == second:
                continue
            mean_abs = 0.5 * (class_mean_abs[first] + class_mean_abs[second])
            score_dim = mean_abs * (weights[first] - weights[second]).abs()
            pair_scores[first, second] = score_dim.reshape(model.num_groups, model.group_dim).sum(dim=1)
    pair_rankings = torch.argsort(pair_scores, dim=-1, descending=True)
    class_scores = torch.zeros(num_classes, model.num_groups, dtype=torch.float64, device=device)
    for class_index in range(num_classes):
        mask = torch.ones(num_classes, dtype=torch.bool, device=device)
        mask[class_index] = False
        class_scores[class_index] = pair_scores[class_index, mask].mean(dim=0)
    global_scores = pair_scores[~torch.eye(num_classes, dtype=torch.bool, device=device)].reshape(-1, model.num_groups).mean(dim=0)
    payload = {
        "pair_scores": pair_scores.cpu().float(),
        "pair_rankings": pair_rankings.cpu().long(),
        "class_scores": class_scores.cpu().float(),
        "class_rankings": torch.argsort(class_scores, dim=-1, descending=True).cpu().long(),
        "global_scores": global_scores.cpu().float(),
        "global_ranking": torch.argsort(global_scores, descending=True).cpu().long(),
        "scoring": "mean_abs_activation_times_linear_pairwise_gradient",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, output_path)
    return output_path


def train_semcom_scheme(
    cfg: dict[str, Any],
    loaders: dict[str, torch.utils.data.DataLoader],
    num_classes: int,
    run_dir: str | Path,
    clean_checkpoint: str | Path,
    graph_path: str | Path,
    codebook_path: str | Path,
    scale_path: str | Path,
    scheme: str,
    budget_slots: int,
    seed: int,
    resume: str = "auto",
) -> Path:
    device = get_device(cfg)
    model = build_model(cfg, num_classes).to(device)
    clean_payload = torch.load(clean_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(clean_payload["model"], strict=True)
    model.set_trainable_for_semcom(str(cfg["semcom_training"].get("trainable", "head")))

    artifacts = PolicyArtifacts.load(str(codebook_path), str(graph_path), map_location="cpu")
    policy = SemanticTransmissionPolicy(
        scheme=scheme,
        artifacts=artifacts,
        num_groups=model.num_groups,
        budget_slots=budget_slots,
        policy_cfg=cfg["policy"],
    )
    normalization_scales = load_feature_scales(
        scale_path
    )
    link = AnalogFeatureLink(
        average_power=float(
            cfg["channel"].get("average_power", 1.0)
        ),
        fill_mode=str(
            cfg["semcom_training"].get(
                "fill_mode",
                "mask"
            )
        ),
        normalization_scales=normalization_scales,
    ).to(device)

    train_cfg = cfg["semcom_training"]
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters for semantic communication stage")
    optimizer = AdamW(
        trainable_parameters,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(int(train_cfg["epochs"]), 1))
    use_amp = bool(train_cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    manager = CheckpointManager(run_dir, f"semcom_{scheme}_slots{budget_slots}")
    cfg_hash = config_hash(cfg)
    start_epoch = 0
    resume_batch = 0
    resume_loader_state = None
    resume_running = None
    global_step = 0
    best_metric = -math.inf
    checkpoint = manager.load(
        model,
        optimizer,
        scheduler,
        scaler,
        mode=resume,
        expected_hash=cfg_hash,
        map_location=device,
    )
    if checkpoint:
        extra = checkpoint.get("extra", {})
        status = extra.get("status", "epoch_complete")
        if status == "epoch_complete":
            start_epoch = int(checkpoint["epoch"]) + 1
        else:
            start_epoch = int(checkpoint["epoch"])
            resume_batch = int(extra.get("batch_in_epoch", -1)) + 1
            resume_loader_state = extra.get("epoch_loader_rng_state")
            resume_running = extra.get("running")
        global_step = int(checkpoint["global_step"])
        best_metric = float(checkpoint["best_metric"])
        print(
            f"[resume] {scheme}/slots{budget_slots} from {checkpoint['_loaded_from']} "
            f"at epoch {start_epoch}, batch {resume_batch}"
        )

    stop = StopRequested()
    stop.install()
    history_path = Path(run_dir) / "logs" / f"semcom_{scheme}_slots{budget_slots}.csv"
    snr_grid = torch.tensor(cfg["channel"]["train_snr_grid"], dtype=torch.float32, device=device)
    channel_types = list(cfg["channel"].get("train_channel_types", ["awgn"]))
    lambda_margin = float(train_cfg.get("lambda_margin", 0.0))
    lambda_fair = float(train_cfg.get("lambda_fair", 0.0))
    margin_value = float(train_cfg.get("margin", 1.0))
    save_every_steps = int(train_cfg.get("save_every_steps", 500))

    try:
        for epoch in range(start_epoch, int(train_cfg["epochs"])):
            model.train()
            # The local head is a frozen transmitter-side pair estimator.
            model.local_head.eval()
            train_loader = loaders["train"]
            if epoch == start_epoch and resume_batch > 0 and resume_loader_state is not None:
                train_loader.generator.set_state(resume_loader_state)
            epoch_loader_rng_state = train_loader.generator.get_state()
            if epoch == start_epoch and resume_running is not None:
                running = {
                    "loss": float(resume_running.get("loss", 0.0)),
                    "ce": float(resume_running.get("ce", 0.0)),
                    "margin": float(resume_running.get("margin", 0.0)),
                    "fair": float(resume_running.get("fair", 0.0)),
                    "samples": int(resume_running.get("samples", 0)),
                }
            else:
                running = {"loss": 0.0, "ce": 0.0, "margin": 0.0, "fair": 0.0, "samples": 0}
            progress = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"{scheme} slots{budget_slots} epoch {epoch + 1}",
                leave=False,
            )
            for batch_index, (images, labels, _paths) in progress:
                if epoch == start_epoch and batch_index < resume_batch:
                    continue
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                batch = labels.numel()
                sampled_indices = torch.randint(0, len(snr_grid), (batch,), device=device)
                snr_db = snr_grid[sampled_indices]
                channel_type = channel_types[torch.randint(0, len(channel_types), (1,)).item()]

                optimizer.zero_grad(set_to_none=True)
                autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp) if device.type == "cuda" else nullcontext()
                with autocast_ctx:
                    groups, _vector = model.encode(images)
                    local_logits = model.local_logits_from_groups(groups)
                    plans = policy.make_plans(local_logits, snr_db)
                    recovered, _stats = link(
                        groups,
                        plans,
                        snr_db,
                        channel_type,
                        mask_token=model.mask_token,
                        csi_error_std=float(cfg["channel"].get("train_csi_error_std", 0.0)),
                    )
                    logits = model.receiver_logits_from_groups(recovered)
                    per_sample_ce = F.cross_entropy(logits, labels, reduction="none")
                    ce_loss = per_sample_ce.mean()
                    margin_loss = channel_weighted_margin_loss(
                        logits,
                        labels,
                        snr_db,
                        artifacts.confusion_weights,
                        artifacts.snr_grid,
                        margin_value,
                    )
                    fair_loss = classwise_loss_variance(per_sample_ce, labels, num_classes)
                    loss = ce_loss + lambda_margin * margin_loss + lambda_fair * fair_loss
                scaler.scale(loss).backward()
                if float(train_cfg.get("grad_clip", 0.0)) > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable_parameters, float(train_cfg["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()

                running["loss"] += float(loss.detach()) * batch
                running["ce"] += float(ce_loss.detach()) * batch
                running["margin"] += float(margin_loss.detach()) * batch
                running["fair"] += float(fair_loss.detach()) * batch
                running["samples"] += batch
                global_step += 1
                progress.set_postfix(loss=running["loss"] / running["samples"])

                if save_every_steps > 0 and global_step % save_every_steps == 0:
                    manager.save(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        best_metric=best_metric,
                        config_hash=cfg_hash,
                        extra={
                            "status": "mid_epoch",
                            "scheme": scheme,
                            "budget_slots": budget_slots,
                            "batch_in_epoch": batch_index,
                            "epoch_loader_rng_state": epoch_loader_rng_state,
                            "running": dict(running),
                        },
                        tag=f"step_{global_step:08d}",
                    )
                if stop.requested:
                    raise InterruptedError(f"Stop requested by signal {stop.signal_number}")

            resume_batch = 0
            resume_loader_state = None
            resume_running = None
            scheduler.step()
            validation_snrs = [float(value) for value in train_cfg.get("validation_snrs", [0.0, 5.0])]
            validation_values = []
            for validation_snr in validation_snrs:
                result = evaluate_semcom(
                    model=model,
                    loader=loaders["val"],
                    device=device,
                    num_classes=num_classes,
                    policy=policy,
                    link=link,
                    snr_db=validation_snr,
                    channel_type=str(train_cfg.get("validation_channel", "awgn")),
                    csi_error_std=0.0,
                )
                validation_values.append(result["macro_f1"])
            metric = float(sum(validation_values) / len(validation_values))
            is_best = metric > best_metric
            if is_best:
                best_metric = metric
            append_csv_row(
                history_path,
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": running["loss"] / max(running["samples"], 1),
                    "train_ce": running["ce"] / max(running["samples"], 1),
                    "train_margin": running["margin"] / max(running["samples"], 1),
                    "train_fair": running["fair"] / max(running["samples"], 1),
                    "val_macro_f1_mean": metric,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "trainable_parameters": count_trainable_parameters(model),
                },
            )
            manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                config_hash=cfg_hash,
                extra={
                    "status": "epoch_complete",
                    "scheme": scheme,
                    "budget_slots": budget_slots,
                    "validation_macro_f1": validation_values,
                },
                tag=f"epoch_{epoch:04d}",
                is_best=is_best,
            )
    except (KeyboardInterrupt, InterruptedError) as exc:
        print(f"[checkpoint] interrupted semcom training: {exc}")
        manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=max(start_epoch, locals().get("epoch", start_epoch)),
            global_step=global_step,
            best_metric=best_metric,
            config_hash=cfg_hash,
            extra={
                "status": "interrupted",
                "reason": str(exc),
                "scheme": scheme,
                "budget_slots": budget_slots,
                "batch_in_epoch": locals().get("batch_index", -1),
                "epoch_loader_rng_state": locals().get("epoch_loader_rng_state"),
                "running": dict(locals().get("running", {})),
            },
            tag="emergency",
        )
        raise

    if not manager.best_path.exists():
        raise RuntimeError(f"No best checkpoint for {scheme}/slots{budget_slots}")
    return manager.best_path


@torch.inference_mode()
def evaluate_semcom(
    *,
    model: FeatureSemanticModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    policy: SemanticTransmissionPolicy,
    link: AnalogFeatureLink,
    snr_db: float,
    channel_type: str,
    csi_error_std: float,
) -> dict[str, Any]:
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    total_time = 0.0
    total_samples = 0
    symbols_sum = 0.0
    selected_sum = 0.0
    replicas_sum = 0.0
    tx_power_sum = 0.0
    pair_correct = 0
    pair_total = 0
    for images, labels, _paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch = labels.numel()
        snr_tensor = torch.full((batch,), float(snr_db), device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        groups, _ = model.encode(images)
        local_logits = model.local_logits_from_groups(groups)
        plans = policy.make_plans(local_logits, snr_tensor)
        recovered, stats = link(
            groups,
            plans,
            snr_tensor,
            channel_type,
            mask_token=model.mask_token,
            csi_error_std=csi_error_std,
        )
        tx_power_sum += (
            stats.tx_power_mean * batch
        )
        logits = model.receiver_logits_from_groups(recovered)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start
        predictions = logits.argmax(dim=1)
        confusion += confusion_from_predictions(labels, predictions, num_classes)
        total_samples += batch
        symbols_sum += stats.complex_symbols_per_sample * batch
        selected_sum += stats.selected_groups_mean * batch
        replicas_sum += stats.replicas_mean * batch
        top2 = torch.topk(local_logits, k=2, dim=1).indices
        pair_correct += int((top2 == labels[:, None]).any(dim=1).sum().item())
        pair_total += batch

    result = metrics_from_confusion(confusion).as_dict()
    result.update(
        {
            "snr_db": float(snr_db),
            "channel_type": channel_type,
            "complex_symbols_per_sample": symbols_sum / max(total_samples, 1),
            "selected_groups_mean": selected_sum / max(total_samples, 1),
            "replicas_per_selected_group": replicas_sum / max(total_samples, 1),
            "runtime_ms_per_sample": 1000.0 * total_time / max(total_samples, 1),
            "top2_pair_contains_label": pair_correct / max(pair_total, 1),
            "task_efficiency": result["macro_f1"] / max(symbols_sum / max(total_samples, 1), 1e-12),
            "model_parameters": count_parameters(model),
            "tx_power_mean": tx_power_sum / max(total_samples, 1),
        }
    )
    return result


def save_run_metadata(cfg: dict[str, Any], run_dir: str | Path, metadata: dict[str, Any]) -> None:
    payload = {
        "config_hash": config_hash(cfg),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **metadata,
    }
    atomic_json_dump(payload, Path(run_dir) / "run_metadata.json")
