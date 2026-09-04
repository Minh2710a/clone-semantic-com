from __future__ import annotations

from _common import base_parser, load_cfg_from_args

from semcom.data import create_manifests
from semcom.io import read_json


def main() -> None:
    parser = base_parser("Create deterministic PlantVillage train/validation/test manifests")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg, _seed = load_cfg_from_args(args)
    paths = create_manifests(cfg, force=args.force)
    metadata = read_json(cfg["data"]["manifest_dir"] + "/metadata.json")
    print("Manifests:")
    for split, path in paths.items():
        print(f"  {split}: {path}")
    print(f"Classes: {metadata['selected_classes']}")
    print(f"Counts: {metadata['counts']}")


if __name__ == "__main__":
    main()
