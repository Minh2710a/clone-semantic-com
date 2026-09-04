from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    cfg = copy.deepcopy(cfg)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply key=value overrides using dotted keys, e.g. training.epochs=5."""
    if not overrides:
        return cfg
    result = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor: dict[str, Any] = result
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return result


def config_hash(cfg: dict[str, Any], length: int = 10) -> str:
    serializable = {k: v for k, v in cfg.items() if not k.startswith("_")}
    payload = json.dumps(serializable, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def resolve_run_dir(cfg: dict[str, Any], scheme: str | None = None, seed: int | None = None) -> Path:
    root = Path(cfg["project"]["output_root"])
    name = str(cfg["project"]["name"])
    run_id = config_hash(cfg)
    parts = [f"{name}_{run_id}"]
    if scheme:
        parts.append(scheme)
    if seed is not None:
        parts.append(f"seed{seed}")
    run_dir = root.joinpath(*parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_resolved_config(cfg: dict[str, Any], run_dir: str | Path) -> Path:
    path = Path(run_dir) / "resolved_config.yaml"
    clean_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(clean_cfg, handle, sort_keys=False, allow_unicode=True)
    return path


def experiment_root(cfg: dict[str, Any], seed: int) -> Path:
    root = Path(cfg["project"]["output_root"])
    directory = root / f"{cfg['project']['name']}_{config_hash(cfg)}" / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory
