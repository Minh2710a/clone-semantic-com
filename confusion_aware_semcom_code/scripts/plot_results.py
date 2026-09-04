from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _common import base_parser, load_cfg_from_args

from semcom.config import experiment_root


def main() -> None:
    parser = base_parser("Generate paper-oriented plots from resumable JSON result cells")
    parser.add_argument("--channel", default="awgn")
    parser.add_argument("--budget-slots", type=int, default=None)
    args = parser.parse_args()
    cfg, seed = load_cfg_from_args(args)
    root = experiment_root(cfg, seed)
    summary = root / "results" / "summary_aggregated.csv"
    if not summary.exists():
        raise FileNotFoundError(f"Run evaluation first: {summary}")
    frame = pd.read_csv(summary)
    frame = frame[frame["channel_type"] == args.channel]
    budgets = [args.budget_slots] if args.budget_slots is not None else sorted(frame["budget_slots"].unique())
    output_dir = root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    for budget in budgets:
        subset = frame[frame["budget_slots"] == budget]
        if subset.empty:
            continue
        _line_plot(
            subset,
            value="macro_f1_mean",
            ylabel="Macro-F1",
            title=f"Macro-F1 theo SNR, {args.channel.upper()}, budget={budget} slots",
            output=output_dir / f"macro_f1_vs_snr_{args.channel}_slots{budget}.pdf",
        )
        _line_plot(
            subset,
            value="min_recall_mean",
            ylabel="Minimum-class recall",
            title=f"Minimum-class recall theo SNR, {args.channel.upper()}, budget={budget} slots",
            output=output_dir / f"min_recall_vs_snr_{args.channel}_slots{budget}.pdf",
        )
        _line_plot(
            subset,
            value="worst_confusion_mean",
            ylabel="Worst directed confusion",
            title=f"Nhầm lẫn cặp lớp xấu nhất theo SNR, {args.channel.upper()}, budget={budget} slots",
            output=output_dir / f"worst_confusion_vs_snr_{args.channel}_slots{budget}.pdf",
        )
    print(f"Figures written to: {output_dir}")


def _line_plot(frame: pd.DataFrame, value: str, ylabel: str, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for scheme, group in frame.groupby("scheme"):
        group = group.sort_values("snr_db")
        ax.plot(group["snr_db"], group[value], marker="o", label=scheme)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
