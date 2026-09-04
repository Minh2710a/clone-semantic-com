from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = {
    "macro_f1": ("Macro-F1", "macro_f1_vs_snr", "Macro-F1"),
    "min_recall": (
        "Minimum-class recall",
        "min_recall_vs_snr",
        "Minimum-class recall",
    ),
    "worst_directed_confusion": (
        "Worst directed confusion",
        "worst_confusion_vs_snr",
        "Nhầm lẫn cặp lớp xấu nhất",
    ),
    "selected_groups_mean": (
        "Selected groups / sample",
        "selected_groups_vs_snr",
        "Số nhóm đặc trưng được chọn",
    ),
    "replicas_per_selected_group": (
        "Replicas / selected group",
        "replicas_vs_snr",
        "Số bản lặp trên mỗi nhóm được chọn",
    ),
}

# Two-sided 95% Student-t critical values, keyed by sample count n.
T95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
    11: 2.228,
    12: 2.201,
    13: 2.179,
    14: 2.160,
    15: 2.145,
    16: 2.131,
    17: 2.120,
    18: 2.110,
    19: 2.101,
    20: 2.093,
    21: 2.086,
    22: 2.080,
    23: 2.074,
    24: 2.069,
    25: 2.064,
    26: 2.060,
    27: 2.056,
    28: 2.052,
    29: 2.048,
    30: 2.045,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate independent training seeds and plot mean curves"
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="Directory containing seed11, seed22, ...",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--channel", default="awgn")
    parser.add_argument("--budget-slots", type=int, nargs="+", default=None)
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_seed_results(experiment_root: Path, seeds: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[Path] = []
    for seed in seeds:
        path = experiment_root / f"seed{seed}" / "results" / "summary_raw.csv"
        if not path.exists():
            missing.append(path)
            continue
        frame = pd.read_csv(path)
        frame["training_seed"] = seed
        frames.append(frame)
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing result summaries:\n{joined}")
    if not frames:
        raise RuntimeError("No seed results found")
    return pd.concat(frames, ignore_index=True)


def aggregate_training_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["training_seed", "scheme", "budget_slots", "channel_type", "snr_db"]
    available_metrics = [metric for metric in METRICS if metric in frame.columns]

    # Average Monte-Carlo channel draws within each trained model first. This keeps
    # independent training seeds as the statistical sample for the outer CI.
    per_seed = frame.groupby(keys, as_index=False)[available_metrics].mean()
    outer_keys = ["scheme", "budget_slots", "channel_type", "snr_db"]
    rows: list[dict[str, float | int | str]] = []
    for key, group in per_seed.groupby(outer_keys, sort=True):
        row: dict[str, float | int | str] = dict(zip(outer_keys, key, strict=True))
        row["num_training_seeds"] = int(group["training_seed"].nunique())
        for metric in available_metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            count = len(values)
            mean = float(values.mean()) if count else float("nan")
            std = float(values.std(ddof=1)) if count > 1 else 0.0
            critical = T95.get(count, 1.96) if count > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = critical * std / math.sqrt(count) if count else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric(
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_stem: Path,
    formats: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for scheme, group in frame.groupby("scheme", sort=True):
        group = group.sort_values("snr_db")
        x = group["snr_db"].to_numpy(dtype=float)
        y = group[f"{metric}_mean"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", label=scheme)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    for extension in formats:
        fig.savefig(output_stem.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    formats = [value.lower().lstrip(".") for value in args.formats]
    unsupported = sorted(set(formats) - {"pdf", "png", "svg"})
    if unsupported:
        raise ValueError(f"Unsupported output format(s): {unsupported}")

    frame = load_seed_results(args.experiment_root, args.seeds)
    frame = frame[frame["channel_type"] == args.channel]
    if args.budget_slots:
        frame = frame[frame["budget_slots"].isin(args.budget_slots)]
    if frame.empty:
        raise RuntimeError("No rows match the requested channel and budget")

    summary = aggregate_training_seeds(frame)
    output_dir = args.output_dir or args.experiment_root / "multiseed" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir.parent / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary_path = table_dir / "summary_multiseed.csv"
    summary.to_csv(summary_path, index=False)

    for budget in sorted(summary["budget_slots"].unique()):
        subset = summary[summary["budget_slots"] == budget]
        for metric, (ylabel, filename, title_prefix) in METRICS.items():
            if f"{metric}_mean" not in subset.columns:
                continue
            stem = output_dir / f"{filename}_{args.channel}_slots{int(budget)}"
            title = (
                f"{title_prefix} theo SNR, {args.channel.upper()}, "
                f"budget={int(budget)} slots"
            )
            plot_metric(subset, metric, ylabel, title, stem, formats)

    print(f"Multi-seed table: {summary_path}")
    print(f"Figures: {output_dir}")


if __name__ == "__main__":
    main()
