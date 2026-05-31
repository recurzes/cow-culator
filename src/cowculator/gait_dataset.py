"""PyTorch Dataset and sampler utilities for lameness/gait sequence training.

Expected CSV format (two required columns, header row):

    sequence_path,lameness_score
    data/pose_sequences/cow_1.npy,1
    data/pose_sequences/cow_4.npy,3

``sequence_path`` may be absolute or relative to the CSV file's parent directory.
``lameness_score`` must be an integer in [1, 5] (Sprecher locomotion scale).

Each ``.npy`` file must contain a float32 array of shape ``[T, K, 3]`` where:
- T  = number of tracked frames for that cow
- K  = number of pose keypoints (inferred from the file; must be consistent)
- 3  = (normalised x, normalised y, confidence)

The dataset pads or truncates every sequence to a fixed ``seq_len`` and
returns a flat tensor of shape ``[seq_len, K*3]``.
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, WeightedRandomSampler

# Sprecher locomotion scoring scale
LAMENESS_MIN = 1
LAMENESS_MAX = 5
NUM_CLASSES = LAMENESS_MAX - LAMENESS_MIN + 1  # 5

DEFAULT_SEQ_LEN = 60  # ~2 s at 30 fps


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_sequence(path: Path) -> np.ndarray:
    """Load a ``.npy`` pose sequence as float32 ``[T, K, 3]``."""
    arr = np.load(str(path)).astype(np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"{path}: expected shape [T, K, 3], got {arr.shape}"
        )
    return arr


def _pad_or_truncate(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Return float32 array of shape ``[seq_len, K, 3]``."""
    t = arr.shape[0]
    if t >= seq_len:
        # Take the last seq_len frames (cow exiting = most diagnostic)
        return arr[-seq_len:]
    # Pad at the start with zeros
    pad = np.zeros((seq_len - t, arr.shape[1], arr.shape[2]), dtype=np.float32)
    return np.concatenate([pad, arr], axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Augmentations
# ──────────────────────────────────────────────────────────────────────────────

def augment_temporal_jitter(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Randomly sample a contiguous sub-window of length ``seq_len``."""
    t = arr.shape[0]
    if t <= seq_len:
        return arr
    start = random.randint(0, t - seq_len)
    return arr[start : start + seq_len]


def augment_horizontal_flip(arr: np.ndarray) -> np.ndarray:
    """Mirror all normalised x coordinates: x → (1 - x).

    Confidence and y values are unchanged.
    arr shape: [T, K, 3]
    """
    flipped = arr.copy()
    flipped[:, :, 0] = 1.0 - flipped[:, :, 0]
    return flipped


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class GaitSequenceDataset(Dataset):
    """Dataset mapping per-track pose sequences to Sprecher lameness scores.

    Parameters
    ----------
    csv_path  : Path to the label CSV (``sequence_path``, ``lameness_score``).
    seq_len   : Fixed temporal window length.  Sequences are truncated (taking
                the last frames) or zero-padded at the start to match.
    augment   : If True, apply temporal jitter and random horizontal flip.
    """

    def __init__(
        self,
        csv_path: Path,
        seq_len: int = DEFAULT_SEQ_LEN,
        augment: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.csv_dir = self.csv_path.parent
        self.seq_len = seq_len
        self.augment = augment
        self.samples: list[tuple[Path, int]] = []
        self._load(self.csv_path)
        # Infer K from first sample for downstream use
        first_arr = _load_sequence(self.samples[0][0])
        self.num_keypoints: int = first_arr.shape[1]

    # ------------------------------------------------------------------
    def _load(self, csv_path: Path) -> None:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Lameness CSV not found: {csv_path}")

        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or not {
                "sequence_path",
                "lameness_score",
            }.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"{csv_path}: CSV must have 'sequence_path' and "
                    f"'lameness_score' columns. Got: {reader.fieldnames}"
                )
            for row in reader:
                seq_p = Path(row["sequence_path"].strip())
                if not seq_p.is_absolute():
                    seq_p = (self.csv_dir / seq_p).resolve()
                score = int(float(row["lameness_score"].strip()))
                if not (LAMENESS_MIN <= score <= LAMENESS_MAX):
                    raise ValueError(
                        f"lameness_score {score} out of range "
                        f"[{LAMENESS_MIN}, {LAMENESS_MAX}] in row: {row}"
                    )
                self.samples.append((seq_p, score))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_path, score = self.samples[idx]
        arr = _load_sequence(seq_path)  # [T, K, 3]

        if self.augment:
            arr = augment_temporal_jitter(arr, self.seq_len)
            if random.random() < 0.5:
                arr = augment_horizontal_flip(arr)

        arr = _pad_or_truncate(arr, self.seq_len)  # [seq_len, K, 3]
        # Flatten keypoints: [seq_len, K*3]
        seq_tensor = torch.from_numpy(arr.reshape(self.seq_len, -1))
        # Class label: 0-indexed (score 1 → class 0)
        label = torch.tensor(score - LAMENESS_MIN, dtype=torch.long)
        return seq_tensor, label

    # ------------------------------------------------------------------
    @property
    def scores(self) -> list[int]:
        return [s for _, s in self.samples]

    @property
    def input_size(self) -> int:
        """Flattened feature size per timestep: K * 3."""
        return self.num_keypoints * 3


# ──────────────────────────────────────────────────────────────────────────────
# Weighted sampler
# ──────────────────────────────────────────────────────────────────────────────

def build_weighted_sampler(dataset: GaitSequenceDataset) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler that up-samples rare lameness classes."""
    scores = dataset.scores
    class_counts: dict[int, int] = {}
    for s in scores:
        class_counts[s] = class_counts.get(s, 0) + 1
    weights = [1.0 / class_counts[s] for s in scores]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Train / val split
# ──────────────────────────────────────────────────────────────────────────────

def split_dataset(
    dataset: GaitSequenceDataset,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Random split into train / val subsets."""
    n = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    n_val = max(1, math.ceil(n * val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
