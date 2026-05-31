"""Run BCS inference on a directory of back-view images or a single image.

Loads a trained checkpoint from ``train_bcs.py`` and outputs a CSV with
predicted BCS scores to stdout or a file.

Output CSV columns:
    image_path, cow_id, bcs_score, bcs_rounded

``cow_id`` is parsed from filenames matching the pattern ``cow_N_frame_M.jpg``
produced by ``bcs_frame_selector.py``.  Images that do not match the pattern
receive ``cow_id = ""``.

Usage:
    cowculator infer-bcs --model runs/bcs/exp1/weights/best.pt \\
                         --images data/bcs_frames/ \\
                         --out results/bcs_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch
from PIL import Image

from cowculator.bcs_dataset import BCS_MAX, BCS_MIN, INPUT_SIZE, build_val_transform
from cowculator.train_bcs import build_model
from cowculator.paths import repo_root

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_COW_ID_RE = re.compile(r"cow_(\d+)_frame_", re.IGNORECASE)

_TRANSFORM = build_val_transform()


def _parse_cow_id(name: str) -> str:
    m = _COW_ID_RE.search(name)
    return m.group(1) if m else ""


def _round_bcs(score: float) -> float:
    """Round to nearest 0.25 on the Edmonson scale."""
    return round(round(score / 0.25) * 0.25, 2)


def _collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in _IMG_SUFFIXES else []
    return sorted(
        p for p in path.rglob("*") if p.suffix.lower() in _IMG_SUFFIXES
    )


def _default_bcs_checkpoint() -> Path | None:
    bcs_dir = repo_root() / "runs" / "bcs"
    if not bcs_dir.is_dir():
        return None
    candidates = [c for c in bcs_dir.rglob("weights/best.pt") if c.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(checkpoint_path, map_location=device)
    model = build_model(pretrained=False)
    # Checkpoint may contain 'model_state_dict' (from train_bcs) or raw state dict
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    images: list[Path],
    device: torch.device,
    batch_size: int = 16,
) -> list[dict]:
    """
    Run inference in batches; return list of result dicts.
    """
    results: list[dict] = []

    for i in range(0, len(images), batch_size):
        batch_paths = images[i : i + batch_size]
        tensors = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(_TRANSFORM(img))
            except Exception as e:
                print(f"  Warning: could not load {p}: {e}", file=sys.stderr)
                tensors.append(torch.zeros(3, INPUT_SIZE, INPUT_SIZE))

        batch = torch.stack(tensors).to(device)
        raw = model(batch).squeeze(1)
        scores = torch.clamp(raw, BCS_MIN, BCS_MAX).cpu().tolist()

        for path, score in zip(batch_paths, scores):
            results.append(
                {
                    "image_path": str(path),
                    "cow_id": _parse_cow_id(path.name),
                    "bcs_score": round(float(score), 4),
                    "bcs_rounded": _round_bcs(float(score)),
                }
            )

    return results


def write_csv(results: list[dict], out: Path | None) -> None:
    fieldnames = ["image_path", "cow_id", "bcs_score", "bcs_rounded"]
    if out is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote {len(results)} predictions to {out}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BCS inference: predict Body Condition Scores from back-view images."
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to BCS checkpoint .pt (default: latest runs/bcs/**/weights/best.pt)",
    )
    p.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Image file or directory of images to score",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: stdout)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size (default: 16)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device: cpu, cuda, 0, etc. (default: auto)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    # ── device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ── checkpoint ────────────────────────────────────────────────────────────
    ckpt_path = args.model
    if ckpt_path is None:
        ckpt_path = _default_bcs_checkpoint()
    if ckpt_path is None:
        print(
            "error: no BCS checkpoint found under runs/bcs/. "
            "Train first with 'cowculator train-bcs' or pass --model.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not ckpt_path.is_file():
        print(f"error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Model  : {ckpt_path}")
    print(f"Device : {device}")

    # ── collect images ────────────────────────────────────────────────────────
    images = _collect_images(args.images)
    if not images:
        print(f"error: no images found at {args.images}", file=sys.stderr)
        sys.exit(1)
    print(f"Images : {len(images)}")

    # ── inference ─────────────────────────────────────────────────────────────
    model = load_model(ckpt_path, device)
    results = run_inference(model, images, device, batch_size=args.batch_size)

    # ── output ────────────────────────────────────────────────────────────────
    write_csv(results, args.out)


if __name__ == "__main__":
    main()
