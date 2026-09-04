from __future__ import annotations

from pathlib import Path

import torch

from _common import base_parser, load_cfg_from_args

from semcom.config import experiment_root


def main() -> None:
    parser = base_parser("Verify that all checkpoint files are readable after interruption or file transfer")
    args = parser.parse_args()
    cfg, seed = load_cfg_from_args(args)
    root = experiment_root(cfg, seed)
    checkpoints = sorted(root.rglob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pt files under {root}")
    failures = []
    for path in checkpoints:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if path.name in {"latest.pt", "best.pt", "emergency.pt"} and "model" not in payload:
                raise ValueError("missing model state")
            print(f"[ok] {path}")
        except Exception as exc:
            failures.append((path, exc))
            print(f"[fail] {path}: {exc}")
    if failures:
        raise RuntimeError(f"{len(failures)} checkpoint/artifact files are unreadable")
    print(f"Verified {len(checkpoints)} .pt files")


if __name__ == "__main__":
    main()
