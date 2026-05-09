"""Prepare YOLOv8 pose dataset from annotations/ + yolo_labels/, then optionally train."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path

from cowculator.paths import (
    default_annotations_dir,
    default_labels_dir,
    repo_root,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _glob_images(annotations_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in annotations_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            parts = {x.lower() for x in p.parts}
            if "images" in parts:
                out.append(p)
    return sorted(out)


def _unique_image_name(
    image_path: Path, annotations_dir: Path, dup_lower: set[str]
) -> str:
    """Disambiguate if the same basename appears under different batch folders."""
    name = image_path.name
    if name.lower() not in dup_lower:
        return name
    try:
        rel = image_path.relative_to(annotations_dir)
    except ValueError:
        return name
    parts = rel.parts
    if len(parts) >= 2:
        prefix = parts[0]
        stem = image_path.stem
        suffix = image_path.suffix
        return f"{prefix}_{stem}{suffix}"
    return name


def _parse_pose_line(line: str) -> int:
    """Return keypoint count K for one YOLO pose line (5 + K*3 tokens)."""
    tokens = line.strip().split()
    if not tokens:
        return -1
    n = len(tokens)
    if n < 5:
        raise ValueError(f"Pose line needs at least 5 values, got {n}: {line!r}")
    rest = n - 5
    if rest % 3 != 0:
        raise ValueError(
            f"Expected 5 + 3*K tokens, got {n} (remainder {(n - 5) % 3}): {line!r}"
        )
    return rest // 3


def _validate_label_file(path: Path, expected_k: int) -> None:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            k = _parse_pose_line(s)
            if k != expected_k:
                raise ValueError(
                    f"{path}: line {i} has {k} keypoints, expected {expected_k}"
                )


def _infer_kpt_shape_from_labels(label_paths: list[Path]) -> tuple[int, int]:
    if not label_paths:
        raise ValueError("No label files to infer kpt_shape")
    ks: set[int] = set()
    for lp in label_paths:
        with open(lp, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    ks.add(_parse_pose_line(s))
                    break
            else:
                raise ValueError(f"{lp}: no non-empty lines")
    if len(ks) != 1:
        raise ValueError(f"Inconsistent keypoint counts across files: {sorted(ks)}")
    k = ks.pop()
    return k, 3


def prepare(
    *,
    repo_root_arg: Path,
    annotations_dir: Path,
    labels_dir: Path,
    out_root: Path,
    val_fraction: float,
    seed: int,
    use_symlinks: bool,
) -> Path:
    if not annotations_dir.is_dir():
        raise SystemExit(f"Missing annotations dir: {annotations_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"Missing labels dir: {labels_dir}")
    if not 0.0 < val_fraction < 1.0:
        raise SystemExit("--val-fraction must be in (0, 1)")

    images = _glob_images(annotations_dir)
    dup_lower = {
        k for k, c in Counter(p.name.lower() for p in images).items() if c > 1
    }

    paired: list[tuple[Path, Path, str]] = []
    skipped_no_label: list[Path] = []
    for p in images:
        uniq = _unique_image_name(p, annotations_dir, dup_lower)
        label_path = labels_dir / f"{p.stem}.txt"
        if not label_path.is_file():
            skipped_no_label.append(p)
            continue
        with open(label_path, encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            skipped_no_label.append(p)
            continue
        paired.append((p, label_path, uniq))

    all_label_txts = sorted(labels_dir.glob("*.txt"))
    used_label_stems = {pl.stem for _, pl, _ in paired}
    orphan_labels = [p for p in all_label_txts if p.stem not in used_label_stems]

    if not paired:
        raise SystemExit("No image+label pairs found. Check paths and naming.")

    label_paths = [pl for _, pl, _ in paired]
    k, dim = _infer_kpt_shape_from_labels(label_paths)
    for _, pl, _ in paired:
        _validate_label_file(pl, k)

    rng = random.Random(seed)
    rng.shuffle(paired)
    if len(paired) < 2:
        train_set = list(paired)
        val_set: list[tuple[Path, Path, str]] = []
    else:
        n_val = max(1, int(round(len(paired) * val_fraction)))
        n_val = min(n_val, len(paired) - 1)
        val_set = paired[:n_val]
        train_set = paired[n_val:]

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    def install(split: str, items: list[tuple[Path, Path, str]]) -> None:
        for img_path, lbl_path, uniq in items:
            dst_img = out_root / "images" / split / uniq
            dst_lbl = out_root / "labels" / split / f"{Path(uniq).stem}.txt"
            if use_symlinks:
                if dst_img.exists() or dst_img.is_symlink():
                    dst_img.unlink()
                if dst_lbl.exists() or dst_lbl.is_symlink():
                    dst_lbl.unlink()
                dst_img.symlink_to(img_path.resolve())
                dst_lbl.symlink_to(lbl_path.resolve())
            else:
                shutil.copy2(img_path, dst_img)
                shutil.copy2(lbl_path, dst_lbl)

    install("train", train_set)
    if val_set:
        install("val", val_set)
    else:
        (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    yaml_path = out_root / "dataset.yaml"
    rel_root = out_root.resolve().relative_to(repo_root_arg.resolve())
    val_rel = "images/val" if val_set else "images/train"
    yaml_text = f"""# Auto-generated by cowculator prepare
path: {rel_root.as_posix()}
train: images/train
val: {val_rel}
nc: 1
names:
  0: cow
kpt_shape: [{k}, {dim}]
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")

    print("Prepare summary")
    print(f"  paired:        {len(paired)}")
    print(
        f"  train / val:   {len(train_set)} / {len(val_set)}"
        + ("  (val uses train images)" if not val_set else "")
    )
    print(f"  kpt_shape:     [{k}, {dim}]")
    print(f"  output:        {out_root}")
    print(f"  dataset.yaml:  {yaml_path}")
    if skipped_no_label:
        print(f"  images w/o label or empty txt: {len(skipped_no_label)}")
    if orphan_labels:
        print(f"  orphan labels (no matching image): {len(orphan_labels)}")
    return yaml_path


def train(
    *,
    data_yaml: Path,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str | None,
    project: str,
    name: str,
    workers: int,
) -> None:
    from ultralytics import YOLO

    if not data_yaml.is_file():
        raise SystemExit(f"Missing dataset yaml: {data_yaml}")
    m = YOLO(model)
    m.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        workers=workers,
    )


def train_options_parser() -> argparse.ArgumentParser:
    """Shared dataset + training flags (for ``train_main`` and ``cowculator`` subcommands)."""
    root = repo_root()
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--annotations-dir",
        type=Path,
        default=default_annotations_dir(),
        help="Root folder containing batch*/images/",
    )
    p.add_argument(
        "--labels-dir",
        type=Path,
        default=default_labels_dir(),
        help="Folder with YOLO pose .txt labels",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "yolo_dataset",
        help="YOLO dataset output root",
    )
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--link",
        action="store_true",
        help="Symlink instead of copy images/labels",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="dataset.yaml for training (default: --out-dir/dataset.yaml)",
    )
    p.add_argument("--model", default="yolov8n-pose.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None, help="e.g. 0, cpu, mps")
    p.add_argument("--project", default="runs/pose")
    p.add_argument("--name", default="train")
    p.add_argument("--workers", type=int, default=8)
    return p


def run_prepare_train_args(
    args: Namespace,
    *,
    do_prepare: bool,
    do_train: bool,
) -> None:
    """Run prepare and/or train from a namespace (same fields as ``train_main``)."""
    root = repo_root()
    data_yaml = args.data
    if data_yaml is None:
        data_yaml = args.out_dir / "dataset.yaml"

    if do_prepare:
        prepare(
            repo_root_arg=root,
            annotations_dir=args.annotations_dir.resolve(),
            labels_dir=args.labels_dir.resolve(),
            out_root=args.out_dir.resolve(),
            val_fraction=args.val_fraction,
            seed=args.seed,
            use_symlinks=args.link,
        )

    if do_train:
        train(
            data_yaml=data_yaml.resolve(),
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            workers=args.workers,
        )


def train_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, parents=[train_options_parser()])
    p.add_argument(
        "--prepare",
        action="store_true",
        help="Build yolo_dataset/ and dataset.yaml",
    )
    p.add_argument(
        "--train",
        action="store_true",
        help="Run Ultralytics training (after prepare if combined)",
    )
    p.add_argument(
        "--prepare-and-train",
        action="store_true",
        help="Prepare then train in one run",
    )
    args = p.parse_args(argv)

    do_prepare = args.prepare or args.prepare_and_train
    do_train = args.train or args.prepare_and_train

    if not do_prepare and not do_train:
        p.error("Specify --prepare, --train, or --prepare-and-train")

    run_prepare_train_args(args, do_prepare=do_prepare, do_train=do_train)


def main() -> None:
    try:
        train_main()
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
