"""PyTorch Dataset and sampler utilities for BCS regression training.

Expected CSV format (two required columns, header row):

    image_path,bcs_score
    data/bcs_frames/cow_1_frame_420.jpg,2.75
    data/bcs_frames/cow_2_frame_310.jpg,3.50
    ...

``image_path`` may be absolute or relative to the CSV file's parent directory.
``bcs_score`` must be a float in [1.0, 5.0] (Edmonson scale, 0.25 increments).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

# Input resolution expected by EfficientNet-B0
INPUT_SIZE = 224

# BCS valid range
BCS_MIN = 1.0
BCS_MAX = 5.0

# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────

def build_train_transform() -> Callable:
    """Augmentation pipeline for training (manuscript spec + standard practices)."""
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_val_transform() -> Callable:
    """Deterministic transform for validation / inference."""
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class BCSDataset(Dataset):
    """Dataset mapping back-view cow images to BCS scores (1.0–5.0)."""

    def __init__(
        self,
        csv_path: Path,
        transform: Callable | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.csv_dir = self.csv_path.parent
        self.transform = transform or build_val_transform()
        self.samples: list[tuple[Path, float]] = []
        self._load(self.csv_path)

    # ------------------------------------------------------------------
    def _load(self, csv_path: Path) -> None:
        if not csv_path.is_file():
            raise FileNotFoundError(f"BCS CSV not found: {csv_path}")

        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or "image_path" not in reader.fieldnames or "bcs_score" not in reader.fieldnames:
                raise ValueError(
                    f"{csv_path}: CSV must have 'image_path' and 'bcs_score' columns. "
                    f"Got: {reader.fieldnames}"
                )
            for row in reader:
                img_p = Path(row["image_path"].strip())
                if not img_p.is_absolute():
                    img_p = (self.csv_dir / img_p).resolve()
                score = float(row["bcs_score"].strip())
                if not (BCS_MIN <= score <= BCS_MAX):
                    raise ValueError(
                        f"bcs_score {score} out of range [{BCS_MIN}, {BCS_MAX}] "
                        f"in row: {row}"
                    )
                self.samples.append((img_p, score))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, score = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)
        label = torch.tensor(score, dtype=torch.float32)
        return tensor, label

    # ------------------------------------------------------------------
    @property
    def scores(self) -> list[float]:
        return [s for _, s in self.samples]


# ──────────────────────────────────────────────────────────────────────────────
# Weighted sampler
# ──────────────────────────────────────────────────────────────────────────────

def build_weighted_sampler(dataset: BCSDataset, n_bins: int = 17) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler that up-samples rare BCS classes.

    BCS 1.0–5.0 in 0.25 steps → 17 bins.  Bin index = round((score-1)/0.25).
    """
    scores = dataset.scores
    bin_counts: dict[int, int] = {}

    def _bin(s: float) -> int:
        return round((s - BCS_MIN) / 0.25)

    for s in scores:
        b = _bin(s)
        bin_counts[b] = bin_counts.get(b, 0) + 1

    weights = [1.0 / bin_counts[_bin(s)] for s in scores]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Train / val split
# ──────────────────────────────────────────────────────────────────────────────

def split_dataset(
    dataset: BCSDataset,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Stratified-ish split into train / val subsets."""
    from torch.utils.data import Subset

    n = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    n_val = max(1, math.ceil(n * val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
