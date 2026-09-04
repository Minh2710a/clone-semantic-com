from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from _common import base_parser, load_cfg_from_args

from semcom.config import experiment_root


def main() -> None:
    parser = base_parser("Export confusion-graph and codebook diagnostics")
    parser.add_argument("--top-pairs", type=int, default=10)
    args = parser.parse_args()
    cfg, seed = load_cfg_from_args(args)
    root = experiment_root(cfg, seed)
    artifact_dir = root / "artifacts"
    graph = torch.load(artifact_dir / "confusion_graph.pt", map_location="cpu", weights_only=False)
    codebook = torch.load(artifact_dir / "pairwise_codebook.pt", map_location="cpu", weights_only=False)
    metadata_path = Path(cfg["data"]["manifest_dir"]) / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    class_names = metadata["selected_classes"]

    rows = []
    for snr_index, snr in enumerate(graph["snr_grid"].tolist()):
        weights = graph["weights"][snr_index]
        for first in range(len(class_names)):
            for second in range(first + 1, len(class_names)):
                rows.append(
                    {
                        "snr_db": snr,
                        "class_i": class_names[first],
                        "class_j": class_names[second],
                        "weight": float(weights[first, second]),
                    }
                )
    frame = pd.DataFrame(rows).sort_values(["snr_db", "weight"], ascending=[True, False])
    output_dir = root / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "confusion_edges_all.csv", index=False)
    frame.groupby("snr_db", group_keys=False).head(args.top_pairs).to_csv(
        output_dir / "confusion_edges_top.csv", index=False
    )

    rank_rows = []
    pair_scores = codebook["pair_scores"]
    pair_rankings = codebook["pair_rankings"]
    for first in range(len(class_names)):
        for second in range(first + 1, len(class_names)):
            top_groups = pair_rankings[first, second, : min(10, pair_rankings.shape[-1])].tolist()
            rank_rows.append(
                {
                    "class_i": class_names[first],
                    "class_j": class_names[second],
                    "top_groups": " ".join(map(str, top_groups)),
                    "top_score": float(pair_scores[first, second, top_groups[0]]),
                }
            )
    pd.DataFrame(rank_rows).to_csv(output_dir / "pairwise_top_groups.csv", index=False)
    print(f"Diagnostics written to: {output_dir}")


if __name__ == "__main__":
    main()
