from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from _common import base_parser, load_cfg_from_args

from semcom.channel import AnalogFeatureLink
from semcom.config import experiment_root
from semcom.data import make_loaders
from semcom.engine import build_model, evaluate_semcom, get_device, load_feature_scales
from semcom.io import atomic_json_dump
from semcom.policy import PolicyArtifacts, SemanticTransmissionPolicy
from semcom.seed import seed_everything


def evaluate_grid(
    cfg: dict,
    seed: int,
    schemes: list[str] | None = None,
    budgets: list[int] | None = None,
    force: bool = False,
) -> Path:
    root = experiment_root(cfg, seed)
    loaders, metadata = make_loaders(cfg, seed)
    num_classes = len(metadata["class_to_idx"])
    device = get_device(cfg)
    artifact_dir = root / "artifacts"
    graph_path = artifact_dir / "confusion_graph.pt"
    codebook_path = artifact_dir / "pairwise_codebook.pt"
    scale_path = artifact_dir / "feature_scale.pt"
    artifacts = PolicyArtifacts.load(str(codebook_path), str(graph_path), map_location="cpu")
    normalization_scales = (
        load_feature_scales(scale_path)
    )

    schemes = schemes or list(cfg["evaluation"]["schemes"])
    budgets = budgets or [int(value) for value in cfg["evaluation"]["budget_slots"]]
    channels = list(cfg["evaluation"]["channel_types"])
    snrs = [float(value) for value in cfg["channel"]["snr_grid"]]
    monte_carlo_seeds = [int(value) for value in cfg["evaluation"]["monte_carlo_seeds"]]
    result_root = root / "results"
    result_root.mkdir(parents=True, exist_ok=True)

    for scheme in schemes:
        for budget in budgets:
            if scheme == "full" and budget < int(cfg["model"]["num_groups"]):
                continue
            run_dir = root / "schemes" / scheme / f"slots{budget}"
            checkpoint_path = run_dir / "checkpoints" / f"semcom_{scheme}_slots{budget}" / "best.pt"
            missing_checkpoints = []
            if not checkpoint_path.exists():
                missing_checkpoints.append(
                    checkpoint_path
                )
                continue
            model = build_model(cfg, num_classes).to(device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model"])
            policy = SemanticTransmissionPolicy(
                scheme=scheme,
                artifacts=artifacts,
                num_groups=model.num_groups,
                budget_slots=budget,
                policy_cfg=cfg["policy"],
            )
            link = AnalogFeatureLink(
                average_power=float(cfg["channel"].get("average_power", 1.0)),
                fill_mode=str(cfg["semcom_training"].get("fill_mode", "mask")),
                normalization_scales=normalization_scales,
            ).to(device)
            for channel in channels:
                for snr in snrs:
                    for mc_seed in monte_carlo_seeds:
                        cell_dir = result_root / scheme / f"slots{budget}" / channel
                        cell_path = cell_dir / f"snr{snr:+05.1f}_mc{mc_seed}.json"
                        if cell_path.exists() and not force:
                            continue
                        seed_everything(seed * 100000 + mc_seed * 100 + int((snr + 50) * 10), deterministic=True)
                        result = evaluate_semcom(
                            model=model,
                            loader=loaders["test"],
                            device=device,
                            num_classes=num_classes,
                            policy=policy,
                            link=link,
                            snr_db=snr,
                            channel_type=channel,
                            csi_error_std=float(cfg["evaluation"].get("csi_error_std", 0.0)),
                        )
                        result.update(
                            {
                                "scheme": scheme,
                                "budget_slots": budget,
                                "seed": seed,
                                "monte_carlo_seed": mc_seed,
                                "class_names": metadata["selected_classes"],
                            }
                        )
                        atomic_json_dump(result, cell_path)
                        print(
                            f"[done] {scheme} slots={budget} {channel} {snr:g} dB mc={mc_seed}: "
                            f"F1={result['macro_f1']:.4f}, minR={result['min_recall']:.4f}"
                        )
            if missing_checkpoints:
                missing = "\n".join(
                    str(path)
                    for path in missing_checkpoints
                )

                raise FileNotFoundError(
                    "Evaluation incomplete. "
                    "Missing checkpoints:\n"
                    + missing
                )
    return summarize_results(result_root)


def summarize_results(result_root: Path) -> Path:
    rows = []
    for path in result_root.rglob("*.json"):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append(
            {
                "scheme": payload["scheme"],
                "budget_slots": payload["budget_slots"],
                "channel_type": payload["channel_type"],
                "snr_db": payload["snr_db"],
                "monte_carlo_seed": payload["monte_carlo_seed"],
                "accuracy": payload["accuracy"],
                "macro_f1": payload["macro_f1"],
                "min_recall": payload["min_recall"],
                "worst_directed_confusion": payload["worst_directed_confusion"],
                "complex_symbols_per_sample": payload["complex_symbols_per_sample"],
                "selected_groups_mean": payload["selected_groups_mean"],
                "replicas_per_selected_group": payload["replicas_per_selected_group"],
                "task_efficiency": payload["task_efficiency"],
                "runtime_ms_per_sample": payload["runtime_ms_per_sample"],
                "top2_pair_contains_label": payload["top2_pair_contains_label"],
                "source_file": str(path),
                "tx_power_mean": payload["tx_power_mean"],
            }
        )
    output = result_root / "summary_raw.csv"
    if rows:
        frame = pd.DataFrame(rows).sort_values(
            ["scheme", "budget_slots", "channel_type", "snr_db", "monte_carlo_seed"]
        )
        frame.to_csv(output, index=False)
        grouped = (
            frame.groupby(["scheme", "budget_slots", "channel_type", "snr_db"], as_index=False)
            .agg(
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                macro_f1_mean=("macro_f1", "mean"),
                macro_f1_std=("macro_f1", "std"),
                min_recall_mean=("min_recall", "mean"),
                min_recall_std=("min_recall", "std"),
                worst_confusion_mean=("worst_directed_confusion", "mean"),
                task_efficiency_mean=("task_efficiency", "mean"),
                runtime_ms_mean=("runtime_ms_per_sample", "mean"),
                tx_power_mean=("tx_power_mean", "mean",),
                tx_power_std=("tx_power_mean", "std",),
            )
        )
        grouped.to_csv(result_root / "summary_aggregated.csv", index=False)
    else:
        output.write_text("", encoding="utf-8")
    print(f"Summary: {output}")
    return output


def main() -> None:
    parser = base_parser("Evaluate trained schemes; completed cells are skipped automatically")
    parser.add_argument("--scheme", action="append", default=[])
    parser.add_argument("--budget-slots", type=int, action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg, seed = load_cfg_from_args(args)
    evaluate_grid(
        cfg,
        seed,
        schemes=args.scheme or None,
        budgets=args.budget_slots or None,
        force=args.force,
    )


if __name__ == "__main__":
    main()
