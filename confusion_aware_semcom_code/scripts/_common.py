from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semcom.config import apply_overrides, load_config  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--override", action="append", default=[], help="Dotted key=value override; repeatable")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def load_cfg_from_args(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    cfg = apply_overrides(load_config(args.config), args.override)
    seed = int(args.seed if args.seed is not None else cfg["project"]["seeds"][0])
    return cfg, seed
