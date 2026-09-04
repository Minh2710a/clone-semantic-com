from __future__ import annotations

from _common import base_parser, load_cfg_from_args

from semcom.config import experiment_root, save_resolved_config
from semcom.data import make_loaders
from semcom.engine import save_run_metadata, train_clean_classifier
from semcom.seed import seed_everything


def main() -> None:
    parser = base_parser("Train the channel-free transmitter classifier with automatic resume")
    parser.add_argument("--resume", default="auto", help="auto, none, best, or checkpoint path")
    args = parser.parse_args()
    cfg, seed = load_cfg_from_args(args)
    seed_everything(seed, deterministic=bool(cfg["project"].get("deterministic", True)))
    run_dir = experiment_root(cfg, seed)
    save_resolved_config(cfg, run_dir)
    loaders, metadata = make_loaders(cfg, seed)
    save_run_metadata(cfg, run_dir, {"seed": seed, "dataset": metadata})
    best = train_clean_classifier(cfg, loaders, len(metadata["class_to_idx"]), run_dir, resume=args.resume)
    print(f"Best clean checkpoint: {best}")


if __name__ == "__main__":
    main()
