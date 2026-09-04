from __future__ import annotations

from pathlib import Path

import torch

from _common import base_parser, load_cfg_from_args

from semcom.config import experiment_root
from semcom.data import make_loaders
from semcom.engine import (
    build_confusion_graph,
    build_model,
    build_pairwise_codebook,
    calibrate_feature_scale,
    get_device,
)
from semcom.seed import seed_everything


def main() -> None:
    parser = base_parser("Build fixed feature scale, SNR-conditioned confusion graph, and pairwise codebook")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg, seed = load_cfg_from_args(args)
    seed_everything(seed, deterministic=bool(cfg["project"].get("deterministic", True)))
    run_dir = experiment_root(cfg, seed)
    loaders, metadata = make_loaders(cfg, seed)
    num_classes = len(metadata["class_to_idx"])
    device = get_device(cfg)
    model = build_model(cfg, num_classes).to(device)
    clean_path = run_dir / "checkpoints" / "clean" / "best.pt"
    if not clean_path.exists():
        raise FileNotFoundError(f"Train the clean classifier first: {clean_path}")
    model.load_state_dict(torch.load(clean_path, map_location=device, weights_only=False)["model"])

    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scale_path = artifact_dir / "feature_scale.pt"
    graph_path = artifact_dir / "confusion_graph.pt"
    codebook_path = artifact_dir / "pairwise_codebook.pt"

    if args.force or not scale_path.exists():
        scale = calibrate_feature_scale(model, loaders["train"], device, scale_path)
        print(f"Feature scale: {scale:.6f}")
    else:
        scale = float(torch.load(scale_path, map_location="cpu", weights_only=False)["normalization_scale"])
        print(f"Reuse feature scale: {scale:.6f}")
    if args.force or not graph_path.exists():
        build_confusion_graph(cfg, model, loaders["val"], num_classes, scale, graph_path)
        print(f"Built confusion graph: {graph_path}")
    else:
        print(f"Reuse confusion graph: {graph_path}")
    if args.force or not codebook_path.exists():
        build_pairwise_codebook(model, loaders["val"], num_classes, codebook_path)
        print(f"Built pairwise codebook: {codebook_path}")
    else:
        print(f"Reuse pairwise codebook: {codebook_path}")


if __name__ == "__main__":
    main()
