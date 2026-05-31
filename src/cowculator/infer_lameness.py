"""Batch inference for the lameness GRU model.

Usage (via CLI):
    cowculator infer-lameness -- --sequences-dir data/pose_sequences/ --out results/lameness.csv

Usage (direct):
    python -m cowculator.infer_lameness --sequences-dir data/pose_sequences/ --model runs/lameness/exp1/weights/best.pt

Loads a checkpoint produced by ``train_lameness.py`` and infers on every
``.npy`` file found under ``--sequences-dir``.

Output CSV columns:
    sequence_path, predicted_score, confidence

``predicted_score`` is an integer in [1, 5] (Sprecher scale).
``confidence`` is the softmax probability of the predicted class
(classification mode) or ``NaN`` in regression mode.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cowculator.gait_dataset import (
    DEFAULT_SEQ_LEN,
    LAMENESS_MIN,
    NUM_CLASSES,
    _load_sequence,
    _pad_or_truncate,
)
from cowculator.train_lameness import LamenessGRU
from cowculator.paths import repo_root


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def _default_lameness_checkpoint() -> Path | None:
    """Return the most recently modified best.pt under runs/lameness/."""
    runs = repo_root() / "runs" / "lameness"
    if not runs.is_dir():
        return None
    candidates = sorted(
        runs.glob("**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[LamenessGRU, dict]:
    """Load a ``LamenessGRU`` from a checkpoint.  Returns (model, config)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = LamenessGRU(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg.get("dropout", 0.3),
        num_classes=cfg.get("num_classes", NUM_CLASSES),
        regression=cfg.get("regression", False),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def infer(
    sequences_dir: Path,
    checkpoint_path: Path,
    out_csv: Path,
    device_str: str | None = None,
    batch_size: int = 32,
) -> None:
    """Run batch inference and write results to ``out_csv``.

    Parameters
    ----------
    sequences_dir   : Directory containing ``.npy`` pose sequence files.
    checkpoint_path : Path to a lameness model checkpoint.
    out_csv         : Destination CSV path.
    device_str      : ``'cpu'``, ``'0'``, etc. Auto-detected if None.
    batch_size      : Number of sequences processed per forward pass.
    """
    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not sequences_dir.is_dir():
        raise FileNotFoundError(f"Sequences directory not found: {sequences_dir}")

    model, cfg = load_model(checkpoint_path, device)
    seq_len: int = cfg.get("seq_len", DEFAULT_SEQ_LEN)
    regression: bool = cfg.get("regression", False)

    npy_files = sorted(sequences_dir.glob("*.npy"))
    if not npy_files:
        raise ValueError(f"No .npy files found in {sequences_dir}")

    print(
        f"model={checkpoint_path}  seq_len={seq_len}  "
        f"mode={'regression' if regression else 'classification'}  "
        f"sequences={len(npy_files)}"
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # Process in batches
    batch_paths: list[Path] = []
    batch_tensors: list[torch.Tensor] = []

    def _flush(paths: list[Path], tensors: list[torch.Tensor]) -> None:
        if not tensors:
            return
        batch = torch.stack(tensors, dim=0).to(device)  # [B, seq_len, K*3]
        with torch.no_grad():
            logits = model(batch)
        if regression:
            scores = logits.cpu().tolist()
            for p, s in zip(paths, scores):
                rows.append(
                    {
                        "sequence_path": str(p),
                        "predicted_score": round(float(s)),
                        "confidence": "NaN",
                    }
                )
        else:
            probs = F.softmax(logits, dim=1).cpu()
            pred_classes = probs.argmax(dim=1).tolist()
            confidences = probs.max(dim=1).values.tolist()
            for p, cls, conf in zip(paths, pred_classes, confidences):
                rows.append(
                    {
                        "sequence_path": str(p),
                        "predicted_score": int(cls) + LAMENESS_MIN,
                        "confidence": round(float(conf), 4),
                    }
                )

    for npy in npy_files:
        try:
            arr = _load_sequence(npy)
        except Exception as exc:
            print(f"Warning: skipping {npy}: {exc}", file=sys.stderr)
            continue
        arr = _pad_or_truncate(arr, seq_len)  # [seq_len, K, 3]
        flat = torch.from_numpy(arr.reshape(seq_len, -1))  # [seq_len, K*3]
        batch_paths.append(npy)
        batch_tensors.append(flat)

        if len(batch_tensors) >= batch_size:
            _flush(batch_paths, batch_tensors)
            batch_paths = []
            batch_tensors = []

    _flush(batch_paths, batch_tensors)

    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["sequence_path", "predicted_score", "confidence"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} predictions to {out_csv}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    root = repo_root()
    p = argparse.ArgumentParser(
        description="Predict lameness scores from .npy pose sequence files."
    )
    p.add_argument(
        "--sequences-dir",
        type=Path,
        default=root / "data" / "pose_sequences",
        help="Directory containing .npy pose sequence files",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to lameness checkpoint (default: latest runs/lameness/**/weights/best.pt)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=root / "results" / "lameness.csv",
        help="Output CSV path (default: results/lameness.csv)",
    )
    p.add_argument("--device", default=None, help="Device: cpu, 0, cuda:0, etc.")
    p.add_argument("--batch-size", type=int, default=32, help="Inference batch size (default 32)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    model_path = args.model
    if model_path is None:
        model_path = _default_lameness_checkpoint()
    if model_path is None:
        print(
            "Error: no checkpoint under runs/lameness/ and --model not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        infer(
            sequences_dir=args.sequences_dir,
            checkpoint_path=model_path,
            out_csv=args.out,
            device_str=args.device,
            batch_size=args.batch_size,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
