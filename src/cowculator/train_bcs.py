"""Train an EfficientNet-B0 BCS regression model.

Usage (via CLI):
    cowculator train-bcs --csv data/bcs_labels.csv

Usage (direct):
    python -m cowculator.train_bcs --csv data/bcs_labels.csv --epochs 50

The model takes back-view cow images (224×224 RGB) and predicts a BCS score
in [1.0, 5.0] on the Edmonson dairy cattle scale.

Architecture:
  EfficientNet-B0 (ImageNet pretrained)
  └── Replace classifier head → Linear(1280, 1)
  └── Inference: clamp(output, 1.0, 5.0)

Checkpoints are saved to  runs/bcs/exp_N/weights/best.pt  mirroring the
YOLO pose convention used by the rest of this project.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models

from cowculator.bcs_dataset import (
    BCSDataset,
    BCS_MAX,
    BCS_MIN,
    build_train_transform,
    build_val_transform,
    build_weighted_sampler,
    split_dataset,
)
from cowculator.paths import repo_root


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 with a single-output regression head."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    # Replace final classifier (1280 → 1000) with regression head
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def _clamp_score(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, BCS_MIN, BCS_MAX)


def compute_mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (_clamp_score(preds) - targets).abs().mean().item()


def compute_within_half(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of predictions within 0.5 BCS units of the ground truth."""
    correct = ((_clamp_score(preds) - targets).abs() <= 0.5).float()
    return correct.mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# Next experiment directory  (runs/bcs/exp_N/)
# ──────────────────────────────────────────────────────────────────────────────

def _next_exp_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("exp")],
        key=lambda d: d.name,
    )
    n = len(existing) + 1
    return base / f"exp{n}"


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    csv_path: Path,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    val_fraction: float = 0.2,
    seed: int = 42,
    workers: int = 0,
    device_str: str | None = None,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    runs_dir: Path | None = None,
) -> Path:
    """
    Full training run.  Returns path to the saved best checkpoint.

    Parameters
    ----------
    csv_path       : Path to the BCS label CSV (image_path, bcs_score).
    epochs         : Number of training epochs.
    batch_size     : Batch size for train and val loaders.
    lr             : Initial learning rate for AdamW.
    val_fraction   : Fraction of dataset held out for validation.
    seed           : Random seed for reproducibility.
    workers        : DataLoader worker processes (0 = main process).
    device_str     : 'cpu', '0', 'cuda:0', etc. Auto-detected if None.
    pretrained     : Load ImageNet weights for EfficientNet-B0.
    freeze_backbone: Only train the regression head (useful with tiny datasets).
    runs_dir       : Parent of exp_N directories (default: <repo>/runs/bcs).
    """
    # ── device ────────────────────────────────────────────────────────────────
    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ── experiment directory ───────────────────────────────────────────────────
    if runs_dir is None:
        runs_dir = repo_root() / "runs" / "bcs"
    exp_dir = _next_exp_dir(runs_dir)
    weights_dir = exp_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = weights_dir / "best.pt"
    last_ckpt = weights_dir / "last.pt"
    print(f"Experiment dir : {exp_dir}")
    print(f"Device         : {device}")

    # ── datasets ──────────────────────────────────────────────────────────────
    full_train_ds = BCSDataset(csv_path, transform=build_train_transform())
    full_val_ds = BCSDataset(csv_path, transform=build_val_transform())

    train_subset, val_subset = split_dataset(full_train_ds, val_fraction=val_fraction, seed=seed)
    _, val_idx_subset = split_dataset(full_val_ds, val_fraction=val_fraction, seed=seed)

    # Re-index val to use val transform
    val_subset = Subset(full_val_ds, val_idx_subset.indices)  # type: ignore[attr-defined]

    sampler = build_weighted_sampler(
        BCSDataset(csv_path),  # scores only; no transform needed
    )
    # WeightedRandomSampler operates on the full dataset; we apply it only to
    # the training split by limiting num_samples and filtering by index.
    # For simplicity: use sampler on a fresh full dataset sliced to train indices.
    train_base = BCSDataset(csv_path, transform=build_train_transform())
    train_indices: list[int] = train_subset.indices  # type: ignore[attr-defined]

    class _IndexSampler(torch.utils.data.Sampler):
        def __init__(self, indices: list[int], weights: list[float]) -> None:
            self._indices = indices
            sub_w = [weights[i] for i in indices]
            total = sum(sub_w)
            self._probs = [w / total for w in sub_w]

        def __iter__(self):  # type: ignore[override]
            gen = torch.multinomial(
                torch.tensor(self._probs),
                num_samples=len(self._indices),
                replacement=True,
            )
            return iter(self._indices[i] for i in gen.tolist())

        def __len__(self) -> int:
            return len(self._indices)

    all_scores = BCSDataset(csv_path).scores
    from cowculator.bcs_dataset import BCS_MIN as _BCS_MIN

    def _bin(s: float) -> int:
        return round((s - _BCS_MIN) / 0.25)

    bin_counts: dict[int, int] = {}
    for s in all_scores:
        b = _bin(s)
        bin_counts[b] = bin_counts.get(b, 0) + 1
    raw_weights = [1.0 / bin_counts[_bin(s)] for s in all_scores]

    train_sampler = _IndexSampler(train_indices, raw_weights)

    train_loader = DataLoader(
        train_base,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )

    n_train = len(train_indices)
    n_val = len(val_subset)
    print(f"Train samples  : {n_train}  |  Val samples: {n_val}")
    print(f"Batch size     : {batch_size}  |  Epochs: {epochs}  |  LR: {lr}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(pretrained=pretrained).to(device)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("Backbone frozen: only classifier head will be trained.")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_mae = float("inf")
    history: list[dict] = []

    print(
        f"\n{'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>9}  "
        f"{'Val MAE':>8}  {'W/0.5':>6}  {'Time':>6}"
    )
    print("-" * 58)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # -- train
        model.train()
        train_loss_sum = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss = train_loss_sum / n_train

        # -- validate
        model.eval()
        val_loss_sum = 0.0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                preds = model(imgs).squeeze(1)
                val_loss_sum += criterion(preds, labels).item() * imgs.size(0)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        val_loss = val_loss_sum / n_val
        preds_cat = torch.cat(all_preds)
        labels_cat = torch.cat(all_labels)
        val_mae = compute_mae(preds_cat, labels_cat)
        val_w05 = compute_within_half(preds_cat, labels_cat)

        elapsed = time.time() - t0
        print(
            f"{epoch:>5}  {train_loss:>11.4f}  {val_loss:>9.4f}  "
            f"{val_mae:>8.4f}  {val_w05:>5.1%}  {elapsed:>5.1f}s"
        )

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_mae": round(val_mae, 6),
            "val_within_0.5": round(val_w05, 6),
        }
        history.append(row)

        # save last
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": val_mae,
                "val_within_0.5": val_w05,
            },
            last_ckpt,
        )

        # save best
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_mae": val_mae,
                    "val_within_0.5": val_w05,
                },
                best_ckpt,
            )

    # ── persist training history ───────────────────────────────────────────────
    (exp_dir / "results.json").write_text(json.dumps(history, indent=2))

    print(f"\nBest val MAE: {best_val_mae:.4f}")
    print(f"Checkpoint  : {best_ckpt}")
    return best_ckpt


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train EfficientNet-B0 BCS regression model."
    )
    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="CSV with columns 'image_path,bcs_score'",
    )
    p.add_argument("--epochs", type=int, default=100, help="Training epochs (default 100)")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size (default 16)")
    p.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate (default 1e-4)")
    p.add_argument("--val-fraction", type=float, default=0.2, help="Val split fraction (default 0.2)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers (default 0)")
    p.add_argument("--device", default=None, help="Device: cpu, 0, cuda:0, etc.")
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train EfficientNet-B0 from random init (not recommended)",
    )
    p.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone; only train the regression head (useful for tiny datasets)",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Parent directory for experiment folders (default: runs/bcs/)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    try:
        train(
            csv_path=args.csv,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_fraction=args.val_fraction,
            seed=args.seed,
            workers=args.workers,
            device_str=args.device,
            pretrained=not args.no_pretrained,
            freeze_backbone=args.freeze_backbone,
            runs_dir=args.runs_dir,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
