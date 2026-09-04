from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .io import atomic_json_dump
from .seed import make_worker_init_fn


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SampleRecord:
    path: str
    label: int
    class_name: str
    source_id: str


class ManifestImageDataset(Dataset):
    def __init__(self, manifest_path: str | Path, transform: Any = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.records: list[SampleRecord] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                self.records.append(
                    SampleRecord(
                        path=row["path"],
                        label=int(row["label"]),
                        class_name=row["class_name"],
                        source_id=row["source_id"],
                    )
                )
        if not self.records:
            raise RuntimeError(f"Empty manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record.label, record.path


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    ops: list[Any] = [transforms.Resize((image_size, image_size))]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


def create_manifests(cfg: dict[str, Any], force: bool = False) -> dict[str, Path]:
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"]).expanduser().resolve()
    manifest_dir = Path(data_cfg["manifest_dir"]).expanduser().resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    paths = {split: manifest_dir / f"{split}.csv" for split in ("train", "val", "test")}
    meta_path = manifest_dir / "metadata.json"
    if not force and all(path.exists() for path in paths.values()) and meta_path.exists():
        return paths
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}. Expected ImageFolder layout root/class_name/image.jpg"
        )

    selected = data_cfg.get("selected_classes") or []
    contains = str(data_cfg.get("class_name_contains", "")).strip().lower()
    class_dirs = sorted([path for path in root.iterdir() if path.is_dir()])
    if contains and not selected:
        class_dirs = [path for path in class_dirs if contains in path.name.lower()]
    if selected:
        selected_set = set(selected)
        class_dirs = [path for path in class_dirs if path.name in selected_set]
        missing = selected_set - {path.name for path in class_dirs}
        if missing:
            raise ValueError(f"Selected classes not found under {root}: {sorted(missing)}")
    if len(class_dirs) < 2:
        raise ValueError("At least two class directories are required")

    class_to_idx = {path.name: index for index, path in enumerate(class_dirs)}
    records_by_class: dict[str, list[SampleRecord]] = {}
    for class_dir in class_dirs:
        class_records: list[SampleRecord] = []
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                source_id = _source_identifier(image_path)
                class_records.append(
                    SampleRecord(
                        path=str(image_path.resolve()),
                        label=class_to_idx[class_dir.name],
                        class_name=class_dir.name,
                        source_id=source_id,
                    )
                )
        if not class_records:
            raise RuntimeError(f"No images found in {class_dir}")
        records_by_class[class_dir.name] = class_records

    seed = int(data_cfg["split_seed"])
    ratios = data_cfg["split"]
    train_ratio, val_ratio, test_ratio = map(float, [ratios["train"], ratios["val"], ratios["test"]])
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1")

    split_records: dict[str, list[SampleRecord]] = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for class_name, class_records in records_by_class.items():
        # Group by source_id so pre-generated augmentations sharing a source stem stay in one split.
        grouped: dict[str, list[SampleRecord]] = {}
        for record in class_records:
            grouped.setdefault(record.source_id, []).append(record)
        groups = list(grouped.values())
        rng.shuffle(groups)
        n = len(groups)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(max(n_train, 1), max(n - 2, 1)) if n >= 3 else max(n - 1, 1)
        n_val = min(max(n_val, 1), max(n - n_train - 1, 0)) if n >= 3 else max(n - n_train, 0)
        assignments = {
            "train": groups[:n_train],
            "val": groups[n_train : n_train + n_val],
            "test": groups[n_train + n_val :],
        }
        for split, grouped_records in assignments.items():
            for group in grouped_records:
                split_records[split].extend(group)

    for split, records in split_records.items():
        rng.shuffle(records)
        _write_manifest(paths[split], records)

    metadata = {
        "root": str(root),
        "class_to_idx": class_to_idx,
        "selected_classes": list(class_to_idx.keys()),
        "split_seed": seed,
        "split_ratios": ratios,
        "counts": {split: len(records) for split, records in split_records.items()},
        "counts_by_class": {
            split: {
                class_name: sum(1 for record in records if record.class_name == class_name)
                for class_name in class_to_idx
            }
            for split, records in split_records.items()
        },
    }
    atomic_json_dump(metadata, meta_path)
    return paths


def make_loaders(cfg: dict[str, Any], seed: int) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    paths = create_manifests(cfg, force=False)
    data_cfg = cfg["data"]
    image_size = int(data_cfg["image_size"])
    datasets = {
        "train": ManifestImageDataset(paths["train"], build_transforms(image_size, train=True)),
        "val": ManifestImageDataset(paths["val"], build_transforms(image_size, train=False)),
        "test": ManifestImageDataset(paths["test"], build_transforms(image_size, train=False)),
    }
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": int(data_cfg["batch_size"]),
        "num_workers": int(data_cfg["num_workers"]),
        "pin_memory": bool(data_cfg.get("pin_memory", True) and torch.cuda.is_available()),
        "worker_init_fn": make_worker_init_fn(seed),
        "generator": generator,
        "persistent_workers": bool(data_cfg.get("persistent_workers", False) and int(data_cfg["num_workers"]) > 0),
    }
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, drop_last=False, **common),
        "val": DataLoader(datasets["val"], shuffle=False, drop_last=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, drop_last=False, **common),
    }
    metadata_path = Path(data_cfg["manifest_dir"]) / "metadata.json"
    import json

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return loaders, metadata


def _source_identifier(path: Path) -> str:
    stem = path.stem.lower()
    # Common suffixes used by offline augmentation pipelines.
    for token in ("_aug", "-aug", "_flip", "_rot", "_crop", "_copy"):
        if token in stem:
            stem = stem.split(token, 1)[0]
    digest = hashlib.sha1(str(path.parent.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{digest}:{stem}"


def _write_manifest(path: Path, records: list[SampleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "class_name", "source_id"])
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)
