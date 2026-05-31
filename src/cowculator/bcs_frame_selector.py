"""Select back-view frames from per-cow crop directories for BCS annotation.

Cows walk through the camera lane; the *last* frames of each tracked crop
sequence are the most likely back-view candidates.  This module takes the
``data/tracked_cows/cow_N/`` directories produced by ``pose_track --save-crops``
and copies the final ``--last-n`` frames (sorted by frame number) into a flat
``data/bcs_frames/`` output directory for manual BCS labelling.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from cowculator.paths import repo_root

_FRAME_RE = re.compile(r"frame_(\d+)\.jpg$", re.IGNORECASE)


def _frame_number(p: Path) -> int:
    m = _FRAME_RE.search(p.name)
    return int(m.group(1)) if m else 0


def _cow_id_from_dir(d: Path) -> int | None:
    m = re.match(r"cow_(\d+)$", d.name)
    return int(m.group(1)) if m else None


def select_frames(
    crops_dir: Path,
    out_dir: Path,
    last_n: int = 5,
    min_frames: int = 1,
    overwrite: bool = False,
) -> dict[int, list[Path]]:
    """
    Walk *crops_dir* for ``cow_N/`` subdirectories, select the last *last_n*
    frames from each, and copy them to *out_dir*.

    Returns a mapping of ``cow_id -> [copied_paths]``.
    """
    if not crops_dir.is_dir():
        raise ValueError(f"crops_dir does not exist: {crops_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, list[Path]] = {}

    cow_dirs = sorted(
        [d for d in crops_dir.iterdir() if d.is_dir() and _cow_id_from_dir(d) is not None],
        key=lambda d: _cow_id_from_dir(d) or 0,
    )

    if not cow_dirs:
        print(f"No cow_N/ subdirectories found under {crops_dir}", file=sys.stderr)
        return results

    for cow_dir in cow_dirs:
        cow_id = _cow_id_from_dir(cow_dir)
        if cow_id is None:
            continue

        frames = sorted(
            [f for f in cow_dir.iterdir() if _FRAME_RE.search(f.name)],
            key=_frame_number,
        )

        if len(frames) < min_frames:
            print(
                f"  cow_{cow_id}: {len(frames)} frame(s) < min_frames={min_frames}, skipping"
            )
            continue

        tail = frames[-last_n:]
        copied: list[Path] = []

        for src in tail:
            dst_name = f"cow_{cow_id}_{src.name}"
            dst = out_dir / dst_name
            if dst.exists() and not overwrite:
                copied.append(dst)
                continue
            shutil.copy2(src, dst)
            copied.append(dst)

        results[cow_id] = copied
        print(f"  cow_{cow_id}: {len(frames)} frame(s) total → copied {len(copied)} back-view candidate(s)")

    return results


def main(argv: list[str] | None = None) -> None:
    root = repo_root()

    p = argparse.ArgumentParser(
        description=(
            "Select back-view frame candidates from per-cow crop directories "
            "for BCS annotation. Copies the last N frames of each tracked cow "
            "into a flat output directory."
        )
    )
    p.add_argument(
        "--crops-dir",
        type=Path,
        default=root / "data" / "tracked_cows",
        help="Directory containing cow_N/ subdirs (default: data/tracked_cows)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data" / "bcs_frames",
        help="Output directory for selected frames (default: data/bcs_frames)",
    )
    p.add_argument(
        "--last-n",
        type=int,
        default=5,
        metavar="N",
        help="Number of tail frames to select per cow (default: 5)",
    )
    p.add_argument(
        "--min-frames",
        type=int,
        default=1,
        metavar="M",
        help="Skip cows with fewer than M total frames (default: 1)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in out-dir",
    )

    args = p.parse_args(argv)

    print(f"crops_dir : {args.crops_dir}")
    print(f"out_dir   : {args.out_dir}")
    print(f"last_n    : {args.last_n}")
    print(f"min_frames: {args.min_frames}")
    print()

    results = select_frames(
        crops_dir=args.crops_dir,
        out_dir=args.out_dir,
        last_n=args.last_n,
        min_frames=args.min_frames,
        overwrite=args.overwrite,
    )

    total = sum(len(v) for v in results.values())
    print(f"\nDone. {len(results)} cow(s), {total} frame(s) copied to {args.out_dir}")
    print(
        f"\nNext step: open {args.out_dir} and create a CSV with columns "
        "'image_path,bcs_score' assigning each frame a BCS value (1.0–5.0, 0.25 steps)."
    )


if __name__ == "__main__":
    main()
