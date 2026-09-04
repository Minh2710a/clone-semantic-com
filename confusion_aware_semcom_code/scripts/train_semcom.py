from __future__ import annotations

from _common import base_parser, load_cfg_from_args

from semcom.config import experiment_root
from semcom.data import make_loaders
from semcom.engine import train_semcom_scheme
from semcom.seed import seed_everything


def main() -> None:
    parser = base_parser("Train one semantic transmission scheme with automatic resume")
    parser.add_argument("--scheme", required=True)
    parser.add_argument("--budget-slots", type=int, required=True)
    parser.add_argument("--resume", default="auto")
    args = parser.parse_args()
    cfg, seed = load_cfg_from_args(args)
    seed_everything(seed, deterministic=bool(cfg["project"].get("deterministic", True)))
    root = experiment_root(cfg, seed)
    loaders, metadata = make_loaders(cfg, seed)
    artifact_dir = root / "artifacts"
    run_dir = root / "schemes" / args.scheme / f"slots{args.budget_slots}"
    best = train_semcom_scheme(
        cfg=cfg,
        loaders=loaders,
        num_classes=len(metadata["class_to_idx"]),
        run_dir=run_dir,
        clean_checkpoint=root / "checkpoints" / "clean" / "best.pt",
        graph_path=artifact_dir / "confusion_graph.pt",
        codebook_path=artifact_dir / "pairwise_codebook.pt",
        scale_path=artifact_dir / "feature_scale.pt",
        scheme=args.scheme,
        budget_slots=args.budget_slots,
        seed=seed,
        resume=args.resume,
    )
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
