"""Repository root and default paths (env overrides)."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Project root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def _env_path(name: str) -> Path | None:
    v = os.environ.get(name)
    if not v:
        return None
    return Path(v).expanduser()


def default_pose_checkpoint() -> Path | None:
    """Latest ``weights/best.pt`` under ``runs/pose/``, or ``COWCULATOR_MODEL`` if set."""
    p = _env_path("COWCULATOR_MODEL")
    if p is not None:
        if p.is_file():
            return p.resolve()
    root = repo_root()
    pose_runs = root / "runs" / "pose"
    if not pose_runs.is_dir():
        return None
    candidates = [c for c in pose_runs.rglob("weights/best.pt") if c.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.stat().st_mtime)


def default_annotations_dir() -> Path:
    p = _env_path("COWCULATOR_ANNOTATIONS_DIR")
    if p is not None and p.is_dir():
        return p.resolve()
    return repo_root() / "annotations"


def default_labels_dir() -> Path:
    p = _env_path("COWCULATOR_LABELS_DIR")
    if p is not None and p.is_dir():
        return p.resolve()
    return repo_root() / "yolo_labels"
