"""Train a bidirectional GRU lameness classifier on pose keypoint sequences.

Usage (via CLI):
    cowculator train-lameness --csv data/lameness_labels.csv

Usage (direct):
    python -m cowculator.train_lameness --csv data/lameness_labels.csv --epochs 80

The model consumes per-cow pose sequences of shape ``[seq_len, K*3]`` produced
by ``cowculator pose-track --save-poses`` and predicts a Sprecher locomotion
score (1 = normal … 5 = severely lame).

Two output modes are supported:
  - Classification (default): CrossEntropyLoss over 5 classes (1–5).
  - Regression (``--regression``): MSELoss, output clamped to [1.0, 5.0].

Feature engineering:
  Pass ``--use-features`` to append derived gait features (spine angle,
  spine curvature, hip drop, hoof x-range, mean confidence) to the raw
  keypoint vector.  The input size grows from K*3 to K*3+5.

  Pass ``--normalize`` to apply per-sequence z-score normalization before
  flattening, which stabilizes GRU training across different camera setups.

Architecture:
  BiGRU(input=K*3[+F], hidden=128, layers=2, dropout=0.3, bidirectional=True)
  └── Linear(256 → num_classes)   [classification]
      Linear(256 → 1)             [regression]

Checkpoints mirror the YOLO/BCS convention: ``runs/lameness/expN/weights/best.pt``.
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

from cowculator.gait_dataset import (
    DEFAULT_SEQ_LEN,
    LAMENESS_MAX,
    LAMENESS_MIN,
    NUM_CLASSES,
    GaitSequenceDataset,
    build_weighted_sampler,
    split_dataset,
)
from cowculator.paths import repo_root


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class LamenessGRU(nn.Module):
    """Bidirectional GRU for lameness classification or regression."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = NUM_CLASSES,
        regression: bool = False,
    ) -> None:
        super().__init__()
        self.regression = regression
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        gru_out_size = hidden_size * 2  # bidirectional
        self.dropout = nn.Dropout(p=dropout)
        if regression:
            self.head = nn.Linear(gru_out_size, 1)
        else:
            self.head = nn.Linear(gru_out_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_size]
        out, _ = self.gru(x)
        # Use the last timestep's output
        last = out[:, -1, :]  # [batch, hidden*2]
        last = self.dropout(last)
        logits = self.head(last)
        if self.regression:
            logits = torch.clamp(logits.squeeze(1), LAMENESS_MIN, LAMENESS_MAX)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of exact class matches (classification mode)."""
    return (preds.argmax(dim=1) == targets).float().mean().item()


def compute_within_one(
    preds: torch.Tensor,
    targets: torch.Tensor,
    regression: bool,
) -> float:
    """Fraction of predictions within 1 Sprecher unit of the ground truth."""
    if regression:
        pred_scores = preds
    else:
        pred_scores = preds.argmax(dim=1).float() + LAMENESS_MIN
    true_scores = targets.float() + LAMENESS_MIN
    return ((pred_scores - true_scores).abs() <= 1.0).float().mean().item()


def compute_mae(
    preds: torch.Tensor,
    targets: torch.Tensor,
    regression: bool,
) -> float:
    if regression:
        pred_scores = preds
    else:
        pred_scores = preds.argmax(dim=1).float() + LAMENESS_MIN
    true_scores = targets.float() + LAMENESS_MIN
    return (pred_scores - true_scores).abs().mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# Next experiment directory  (runs/lameness/expN/)
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
    seq_len: int = DEFAULT_SEQ_LEN,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 42,
    workers: int = 0,
    device_str: str | None = None,
    regression: bool = False,
    use_features: bool = False,
    normalize: bool = False,
    runs_dir: Path | None = None,
) -> Path:
    """
    Full training run. Returns path to the best checkpoint.

    Parameters
    ----------
    csv_path     : Label CSV (``sequence_path``, ``lameness_score``).
    seq_len      : Fixed temporal window length fed to the GRU.
    hidden_size  : GRU hidden units per direction.
    num_layers   : GRU layer count.
    dropout      : Dropout applied between GRU layers and before the head.
    epochs       : Training epochs.
    batch_size   : Batch size for train and val loaders.
    lr           : Initial AdamW learning rate.
    val_fraction : Fraction of dataset held out for validation.
    seed         : Random seed.
    workers      : DataLoader worker processes (0 = main process; safe on Windows).
    device_str   : ``'cpu'``, ``'0'``, ``'cuda:0'``, etc. Auto-detected if None.
    regression   : Use MSE regression instead of cross-entropy classification.
    use_features : Append derived gait features (spine angle, hip drop, etc.)
                   to the raw keypoint vector.  Increases input size by 5.
    normalize    : Apply per-sequence z-score normalization before flattening.
    runs_dir     : Parent of expN directories (default: <repo>/runs/lameness).
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
        runs_dir = repo_root() / "runs" / "lameness"
    exp_dir = _next_exp_dir(runs_dir)
    weights_dir = exp_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = weights_dir / "best.pt"
    last_ckpt = weights_dir / "last.pt"
    print(f"Experiment dir : {exp_dir}")
    print(f"Device         : {device}")

    # ── datasets ──────────────────────────────────────────────────────────────
    from cowculator.gait_features import build_feature_fn, make_layout_from_k

    # Build feature_fn lazily after we know K (from the first .npy file).
    # We construct a temporary dataset just to read K, then rebuild properly.
    _probe_ds = GaitSequenceDataset(csv_path, seq_len=seq_len)
    feature_fn = build_feature_fn(make_layout_from_k(_probe_ds.num_keypoints)) if use_features else None

    train_ds = GaitSequenceDataset(csv_path, seq_len=seq_len, augment=True,
                                   normalize=normalize, feature_fn=feature_fn)
    val_ds   = GaitSequenceDataset(csv_path, seq_len=seq_len, augment=False,
                                   normalize=normalize, feature_fn=feature_fn)

    train_subset, _ = split_dataset(train_ds, val_fraction=val_fraction, seed=seed)
    _, val_subset_base = split_dataset(val_ds, val_fraction=val_fraction, seed=seed)
    val_subset = Subset(val_ds, val_subset_base.indices)  # type: ignore[attr-defined]

    train_indices: list[int] = train_subset.indices  # type: ignore[attr-defined]

    # Weighted sampling for class imbalance — use utility from gait_dataset
    train_base = GaitSequenceDataset(csv_path, seq_len=seq_len, augment=True,
                                     normalize=normalize, feature_fn=feature_fn)
    train_split = Subset(train_base, train_indices)
    train_sampler = build_weighted_sampler(train_split)
    train_loader = DataLoader(
        train_split,
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
    input_size = train_ds.input_size
    print(f"Input size     : {input_size} (K*3 = {train_ds.num_keypoints}×3"
          + (f" + {input_size - train_ds.num_keypoints*3} derived features" if use_features else "")
          + ")")
    print(f"Seq len        : {seq_len}")
    print(f"Train samples  : {n_train}  |  Val samples: {n_val}")
    print(f"Batch size     : {batch_size}  |  Epochs: {epochs}  |  LR: {lr}")
    print(f"Mode           : {'regression' if regression else 'classification'}")
    print(f"Use features   : {use_features}  |  Normalize: {normalize}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = LamenessGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=NUM_CLASSES,
        regression=regression,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion: nn.Module = nn.MSELoss() if regression else nn.CrossEntropyLoss()

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_metric = float("inf")  # MAE for regression; loss for classification
    history: list[dict] = []

    header = (
        f"\n{'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>9}  "
        f"{'Val MAE':>8}  {'W/1':>6}  "
        + (f"{'Acc':>6}  " if not regression else "")
        + f"{'Time':>6}"
    )
    print(header)
    print("-" * (len(header) - 1))

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # -- train
        model.train()
        train_loss_sum = 0.0
        for seqs, labels in train_loader:
            seqs = seqs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(seqs)
            if regression:
                loss = criterion(preds, labels.float() + LAMENESS_MIN)
            else:
                loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * seqs.size(0)

        scheduler.step()
        train_loss = train_loss_sum / n_train

        # -- validate
        model.eval()
        val_loss_sum = 0.0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs = seqs.to(device)
                labels = labels.to(device)
                preds = model(seqs)
                if regression:
                    loss = criterion(preds, labels.float() + LAMENESS_MIN - 1)
                else:
                    loss = criterion(preds, labels)
                val_loss_sum += loss.item() * seqs.size(0)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        val_loss = val_loss_sum / n_val
        preds_cat = torch.cat(all_preds)
        labels_cat = torch.cat(all_labels)

        val_mae = compute_mae(preds_cat, labels_cat, regression)
        val_w1 = compute_within_one(preds_cat, labels_cat, regression)
        val_acc = compute_accuracy(preds_cat, labels_cat) if not regression else None

        elapsed = time.time() - t0
        acc_str = f"  {val_acc:>5.1%}" if val_acc is not None else ""
        print(
            f"{epoch:>5}  {train_loss:>11.4f}  {val_loss:>9.4f}  "
            f"{val_mae:>8.4f}  {val_w1:>5.1%}{acc_str}  {elapsed:>5.1f}s"
        )

        row: dict = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_mae": round(val_mae, 6),
            "val_within_1": round(val_w1, 6),
        }
        if val_acc is not None:
            row["val_accuracy"] = round(val_acc, 6)
        history.append(row)

        # Save last checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": val_mae,
                "val_within_1": val_w1,
                "config": {
                    "input_size": input_size,
                    "seq_len": seq_len,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "num_classes": NUM_CLASSES,
                    "regression": regression,
                    "use_features": use_features,
                    "normalize": normalize,
                },
            },
            last_ckpt,
        )

        # Save best checkpoint (by MAE)
        if val_mae < best_val_metric:
            best_val_metric = val_mae
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_mae": val_mae,
                    "val_within_1": val_w1,
                    "config": {
                        "input_size": input_size,
                        "seq_len": seq_len,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "num_classes": NUM_CLASSES,
                        "regression": regression,
                        "use_features": use_features,
                        "normalize": normalize,
                    },
                },
                best_ckpt,
            )

    # ── persist training history ───────────────────────────────────────────────
    (exp_dir / "results.json").write_text(json.dumps(history, indent=2))

    print(f"\nBest val MAE : {best_val_metric:.4f}")
    print(f"Checkpoint   : {best_ckpt}")
    return best_ckpt


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train bidirectional GRU lameness classifier on pose sequences."
    )
    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="CSV with columns 'sequence_path,lameness_score'",
    )
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                   help=f"Sequence window length in frames (default {DEFAULT_SEQ_LEN})")
    p.add_argument("--hidden", type=int, default=128,
                   help="GRU hidden units per direction (default 128)")
    p.add_argument("--layers", type=int, default=2,
                   help="Number of GRU layers (default 2)")
    p.add_argument("--dropout", type=float, default=0.3,
                   help="Dropout probability (default 0.3)")
    p.add_argument("--epochs", type=int, default=100,
                   help="Training epochs (default 100)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size (default 16)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="AdamW learning rate (default 1e-3)")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="Validation split fraction (default 0.2)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default 42)")
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers (default 0; safe on Windows)")
    p.add_argument("--device", default=None,
                   help="Device: cpu, 0, cuda:0, etc.")
    p.add_argument("--regression", action="store_true",
                   help="Use MSE regression instead of cross-entropy classification")
    p.add_argument("--use-features", action="store_true",
                   help="Append derived gait features (spine angle, hip drop, hoof range, "
                        "spine curvature, mean confidence) to the raw keypoint vector")
    p.add_argument("--normalize", action="store_true",
                   help="Apply per-sequence z-score normalization before flattening")
    p.add_argument("--runs-dir", type=Path, default=None,
                   help="Parent directory for experiment folders (default: runs/lameness/)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    try:
        train(
            csv_path=args.csv,
            seq_len=args.seq_len,
            hidden_size=args.hidden,
            num_layers=args.layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_fraction=args.val_fraction,
            seed=args.seed,
            workers=args.workers,
            device_str=args.device,
            regression=args.regression,
            use_features=args.use_features,
            normalize=args.normalize,
            runs_dir=args.runs_dir,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
