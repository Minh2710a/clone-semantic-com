from __future__ import annotations

import traceback
from pathlib import Path

import torch

from _common import base_parser, load_cfg_from_args

from semcom.config import experiment_root, save_resolved_config
from semcom.data import create_manifests, make_loaders
from semcom.engine import (
    build_confusion_graph,
    build_model,
    build_pairwise_codebook,
    calibrate_feature_scale,
    get_device,
    save_run_metadata,
    train_clean_classifier,
    train_semcom_scheme,
)
from semcom.io import atomic_json_dump, read_json
from semcom.seed import seed_everything

from evaluate import evaluate_grid


STAGES = ["data", "clean", "artifacts", "semcom", "evaluation"]


def mark_state(state_path: Path, state: dict, stage: str, status: str, detail=None) -> None:
    state.setdefault("stages", {})[stage] = {"status": status, "detail": detail}
    atomic_json_dump(state, state_path)


def main() -> None:
    parser = base_parser("Run the complete recoverable simulation pipeline")
    parser.add_argument("--from-stage", choices=STAGES, default="data")
    parser.add_argument("--to-stage", choices=STAGES, default="evaluation")
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg, seed = load_cfg_from_args(args)
    seed_everything(seed, deterministic=bool(cfg["project"].get("deterministic", True)))
    root = experiment_root(cfg, seed)
    save_resolved_config(cfg, root)
    state_path = root / "pipeline_state.json"
    state = read_json(state_path, default={"seed": seed, "stages": {}, "completed_runs": []})
    start = STAGES.index(args.from_stage)
    end = STAGES.index(args.to_stage)
    if start > end:
        raise ValueError("from-stage must not be after to-stage")

    try:
        if start <= 0 <= end:
            mark_state(state_path, state, "data", "running")
            create_manifests(cfg, force=args.force)
            mark_state(state_path, state, "data", "complete")

        loaders, metadata = make_loaders(cfg, seed)
        num_classes = len(metadata["class_to_idx"])
        save_run_metadata(cfg, root, {"seed": seed, "dataset": metadata})

        if start <= 1 <= end:
            mark_state(state_path, state, "clean", "running")
            clean_best = root / "checkpoints" / "clean" / "best.pt"
            if args.force or not clean_best.exists():
                clean_best = train_clean_classifier(cfg, loaders, num_classes, root, resume=args.resume)
            mark_state(state_path, state, "clean", "complete", str(clean_best))

        clean_best = root / "checkpoints" / "clean" / "best.pt"
        if end >= 2 and not clean_best.exists():
            raise FileNotFoundError(f"Missing clean checkpoint: {clean_best}")

        artifact_dir = root / "artifacts"
        scale_path = artifact_dir / "feature_scale.pt"
        graph_path = artifact_dir / "confusion_graph.pt"
        codebook_path = artifact_dir / "pairwise_codebook.pt"
        if start <= 2 <= end:
            mark_state(state_path, state, "artifacts", "running")
            device = get_device(cfg)
            model = build_model(cfg, num_classes).to(device)
            model.load_state_dict(torch.load(clean_best, map_location=device, weights_only=False)["model"])
            if args.force or not scale_path.exists():
                scale = calibrate_feature_scale(model, loaders["train"], device, scale_path)
            else:
                scale = float(torch.load(scale_path, map_location="cpu", weights_only=False)["normalization_scale"])
            if args.force or not graph_path.exists():
                build_confusion_graph(cfg, model, loaders["val"], num_classes, scale, graph_path)
            if args.force or not codebook_path.exists():
                build_pairwise_codebook(model, loaders["val"], num_classes, codebook_path)
            mark_state(
                state_path,
                state,
                "artifacts",
                "complete",
                {"scale": str(scale_path), "graph": str(graph_path), "codebook": str(codebook_path)},
            )

        if end >= 3:
            for required in (scale_path, graph_path, codebook_path):
                if not required.exists():
                    raise FileNotFoundError(f"Missing artifact: {required}")

        if start <= 3 <= end:
            mark_state(state_path, state, "semcom", "running")
            schemes = list(cfg["evaluation"]["schemes"])
            budgets = [int(value) for value in cfg["evaluation"]["budget_slots"]]
            for scheme in schemes:
                for budget in budgets:
                    if scheme == "full" and budget < int(cfg["model"]["num_groups"]):
                        continue
                    run_key = f"{scheme}:slots{budget}"
                    scheme_dir = root / "schemes" / scheme / f"slots{budget}"
                    best_path = scheme_dir / "checkpoints" / f"semcom_{scheme}_slots{budget}" / "best.pt"
                    if best_path.exists() and not args.force:
                        if run_key not in state["completed_runs"]:
                            state["completed_runs"].append(run_key)
                            atomic_json_dump(state, state_path)
                        continue
                    train_semcom_scheme(
                        cfg=cfg,
                        loaders=loaders,
                        num_classes=num_classes,
                        run_dir=scheme_dir,
                        clean_checkpoint=clean_best,
                        graph_path=graph_path,
                        codebook_path=codebook_path,
                        scale_path=scale_path,
                        scheme=scheme,
                        budget_slots=budget,
                        seed=seed,
                        resume=args.resume,
                    )
                    if run_key not in state["completed_runs"]:
                        state["completed_runs"].append(run_key)
                    atomic_json_dump(state, state_path)
            mark_state(state_path, state, "semcom", "complete", state["completed_runs"])

        if start <= 4 <= end:
            mark_state(state_path, state, "evaluation", "running")
            summary = evaluate_grid(cfg, seed, force=args.force)
            mark_state(state_path, state, "evaluation", "complete", str(summary))

    except BaseException as exc:
        failed_stage = next(
            (stage for stage in reversed(STAGES) if state.get("stages", {}).get(stage, {}).get("status") == "running"),
            "unknown",
        )
        mark_state(state_path, state, failed_stage, "interrupted", {"error": str(exc), "traceback": traceback.format_exc()})
        print(f"Pipeline interrupted at stage '{failed_stage}'. Re-run the same command; checkpoints are resumed automatically.")
        raise

    print(f"Pipeline complete. Run directory: {root}")


if __name__ == "__main__":
    main()
