from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_fake_dataset(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    rng = np.random.default_rng(7)
    for class_index, class_name in enumerate(["Tomato_A", "Tomato_B", "Tomato_C"]):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_index in range(24):
            array = rng.normal(90 + class_index * 40, 20, size=(64, 64, 3)).clip(0, 255)
            array[8 + class_index * 8 : 24 + class_index * 8, 8:56, class_index] += 50
            Image.fromarray(array.clip(0, 255).astype(np.uint8)).save(class_dir / f"sample_{image_index:03d}.png")


def main() -> None:
    make_fake_dataset(PROJECT_ROOT / "smoke_data")
    for directory in ["smoke_outputs", "smoke_manifests"]:
        target = PROJECT_ROOT / directory
        if target.exists():
            shutil.rmtree(target)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
        "--config",
        str(PROJECT_ROOT / "configs" / "smoke.yaml"),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    summaries = list((PROJECT_ROOT / "smoke_outputs").rglob("summary_aggregated.csv"))
    if not summaries:
        raise RuntimeError("Smoke test completed without summary_aggregated.csv")
    print(f"Smoke test passed: {summaries[0]}")


if __name__ == "__main__":
    main()
