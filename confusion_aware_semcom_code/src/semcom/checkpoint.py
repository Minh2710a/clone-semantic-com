from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from .seed import capture_rng_state, restore_rng_state


class CheckpointManager:
    """Atomic, resumable checkpoint manager.

    Each save first writes a temporary file, fsyncs it, and atomically replaces the
    destination. A power loss therefore leaves either the previous valid checkpoint
    or the new valid checkpoint, not a partially written file.
    """

    def __init__(self, run_dir: str | Path, stage: str) -> None:
        self.dir = Path(run_dir) / "checkpoints" / stage
        self.dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.dir / "latest.pt"
        self.best_path = self.dir / "best.pt"

    def save(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        scaler: Any | None,
        epoch: int,
        global_step: int,
        best_metric: float,
        config_hash: str,
        extra: dict[str, Any] | None = None,
        is_best: bool = False,
        tag: str | None = None,
    ) -> Path:
        payload = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_metric": float(best_metric),
            "config_hash": config_hash,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng_state": capture_rng_state(),
            "extra": extra or {},
        }
        target = self.dir / (f"{tag}.pt" if tag else "latest.pt")
        self._atomic_torch_save(payload, target)
        if tag:
            self._atomic_copy(target, self.latest_path)
        if is_best:
            self._atomic_copy(target, self.best_path)
        return target

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        mode: str = "auto",
        expected_hash: str | None = None,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> dict[str, Any] | None:
        path = self._resolve_path(mode)
        if path is None:
            return None
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        stored_hash = checkpoint.get("config_hash")
        if expected_hash and stored_hash and expected_hash != stored_hash:
            raise RuntimeError(
                f"Checkpoint/config mismatch: checkpoint={stored_hash}, config={expected_hash}. "
                "Use the exact resolved_config.yaml from the run directory or start a new run."
            )
        model.load_state_dict(checkpoint["model"], strict=strict)
        if optimizer is not None and checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        restore_rng_state(checkpoint.get("rng_state"))
        checkpoint["_loaded_from"] = str(path)
        return checkpoint

    def _resolve_path(self, mode: str) -> Path | None:
        if mode in {"none", "off", "false"}:
            return None
        if mode == "auto":
            return self.latest_path if self.latest_path.exists() else None
        if mode == "best":
            return self.best_path if self.best_path.exists() else None
        path = Path(mode)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
        fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        os.close(fd)
        try:
            torch.save(payload, temp_name)
            with open(temp_name, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        fd, temp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
        os.close(fd)
        try:
            shutil.copy2(source, temp_name)
            with open(temp_name, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temp_name, destination)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
