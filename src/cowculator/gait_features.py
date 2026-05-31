"""Derived gait feature engineering for cow pose keypoint sequences.

This module computes biomechanically meaningful per-frame features from raw
keypoint arrays ``[T, K, 3]`` (normalised x, normalised y, confidence).
These features directly reflect the veterinary indicators described in the
manuscript: spine curvature, hip drop asymmetry, and hoof placement range.

Default skeleton layout (7 keypoints, side-view):
    0  head / nose
    1  neck
    2  withers  (top of shoulder)
    3  mid-back
    4  loin / hip
    5  tail-head
    6  rear ankle / hoof

Extended layout (10 keypoints):
    0  head / nose
    1  neck
    2  withers
    3  mid-back
    4  loin / hip
    5  tail-head
    6  front-left hoof
    7  front-right hoof
    8  rear-left hoof
    9  rear-right hoof

Pass a custom ``CowKeypointLayout`` to ``build_feature_fn`` if your annotation
scheme differs.  All features degrade gracefully: if the required keypoints do
not exist in the array (K too small), the corresponding feature column is
filled with zeros so the tensor shape remains fixed.

Usage::

    import numpy as np
    from cowculator.gait_features import build_feature_fn, DEFAULT_LAYOUT

    feature_fn = build_feature_fn(DEFAULT_LAYOUT)
    arr = np.zeros((60, 7, 3), dtype=np.float32)
    extra = feature_fn(arr)           # [60, 4]  — 4 derived features
    # Concatenate to flat keypoints for the GRU:
    flat_kp = arr.reshape(60, -1)     # [60, 21]
    full    = np.concatenate([flat_kp, extra], axis=1)  # [60, 25]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Skeleton layout
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CowKeypointLayout:
    """Maps semantic body-part names to keypoint indices.

    Any index set to -1 means "not annotated" and the corresponding feature
    will be zero-filled.
    """
    head: int = 0
    neck: int = 1
    withers: int = 2
    mid_back: int = 3
    loin: int = 4
    tail_head: int = 5
    # Hoof / ankle indices (left and right where available)
    front_left_hoof: int = 6
    front_right_hoof: int = -1
    rear_left_hoof: int = -1
    rear_right_hoof: int = -1

    # Convenience: spine chain from neck to tail for curvature
    spine_chain: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])

    # Hip pair for asymmetry (should both be visible in the view)
    # For side-view, loin is the single hip proxy; set both to the same
    # index to disable asymmetry and get a 0 difference.
    hip_left: int = 4
    hip_right: int = 4


DEFAULT_LAYOUT = CowKeypointLayout()

# 10-keypoint extended layout
EXTENDED_LAYOUT = CowKeypointLayout(
    head=0, neck=1, withers=2, mid_back=3, loin=4, tail_head=5,
    front_left_hoof=6, front_right_hoof=7,
    rear_left_hoof=8, rear_right_hoof=9,
    spine_chain=[1, 2, 3, 4, 5],
    hip_left=4, hip_right=4,
)


# ──────────────────────────────────────────────────────────────────────────────
# Per-frame feature helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get(arr: np.ndarray, k: int, idx: int) -> np.ndarray | None:
    """Return frame column [T, 3] for keypoint ``idx``, or None if out of range."""
    if idx < 0 or idx >= k:
        return None
    return arr[:, idx, :]  # [T, 3]


def _angle_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle (radians) of vector (b - a) relative to positive-x axis.

    Parameters
    ----------
    a, b : [T, 2] normalised (x, y) arrays
    Returns [T] float32 array of angles.
    """
    dx = b[:, 0] - a[:, 0]
    dy = b[:, 1] - a[:, 1]
    return np.arctan2(dy, dx).astype(np.float32)


def feature_spine_angle(arr: np.ndarray, layout: CowKeypointLayout) -> np.ndarray:
    """Angle (rad) of the overall spine vector from neck → tail-head, per frame.

    A rising spine angle indicates the characteristic arched-back posture
    associated with higher Sprecher scores.

    Returns ``[T]`` float32, zeros when keypoints are unavailable.
    """
    T, K, _ = arr.shape
    neck = _get(arr, K, layout.neck)
    tail = _get(arr, K, layout.tail_head)
    if neck is None or tail is None:
        return np.zeros(T, dtype=np.float32)
    return _angle_2d(neck[:, :2], tail[:, :2])


def feature_spine_curvature(arr: np.ndarray, layout: CowKeypointLayout) -> np.ndarray:
    """Mean absolute angular deviation along the spine chain, per frame.

    Computed as the average |angle_i+1 - angle_i| over consecutive spine
    segments.  Higher values indicate a more arched back.

    Returns ``[T]`` float32, zeros when fewer than 3 spine keypoints exist.
    """
    T, K, _ = arr.shape
    valid_chain = [i for i in layout.spine_chain if 0 <= i < K]
    if len(valid_chain) < 3:
        return np.zeros(T, dtype=np.float32)

    segments_angles = []
    for a_idx, b_idx in zip(valid_chain[:-1], valid_chain[1:]):
        a = arr[:, a_idx, :2]
        b = arr[:, b_idx, :2]
        segments_angles.append(_angle_2d(a, b))  # [T]

    # Mean absolute difference between consecutive segment angles
    curvature = np.zeros(T, dtype=np.float32)
    for i in range(len(segments_angles) - 1):
        curvature += np.abs(segments_angles[i + 1] - segments_angles[i])
    curvature /= max(len(segments_angles) - 1, 1)
    return curvature


def feature_hip_drop(arr: np.ndarray, layout: CowKeypointLayout) -> np.ndarray:
    """Vertical asymmetry between the left and right hip keypoints, per frame.

    For side-view annotations where only one hip is visible, this returns
    the normalised y-coordinate of the single hip point (useful for tracking
    hip rise/fall across frames).  When two hip keypoints exist, returns their
    absolute y-difference.

    Returns ``[T]`` float32.
    """
    T, K, _ = arr.shape
    l_hip = _get(arr, K, layout.hip_left)
    r_hip = _get(arr, K, layout.hip_right)

    if l_hip is None and r_hip is None:
        return np.zeros(T, dtype=np.float32)

    if l_hip is None:
        return r_hip[:, 1].copy()  # single hip y-coord
    if r_hip is None:
        return l_hip[:, 1].copy()

    if layout.hip_left == layout.hip_right:
        return l_hip[:, 1].copy()  # same index — return y-coord

    return np.abs(l_hip[:, 1] - r_hip[:, 1]).astype(np.float32)


def feature_hoof_x_range(arr: np.ndarray, layout: CowKeypointLayout) -> np.ndarray:
    """Horizontal range of all annotated hoof/ankle keypoints, per frame.

    A larger x-range per frame indicates a longer stride.  The value is the
    max-x minus min-x across all available hoof keypoints.

    Returns ``[T]`` float32, zeros when no hoof keypoints exist.
    """
    T, K, _ = arr.shape
    hoof_indices = [
        layout.front_left_hoof, layout.front_right_hoof,
        layout.rear_left_hoof, layout.rear_right_hoof,
    ]
    available = [arr[:, i, 0] for i in hoof_indices if 0 <= i < K]  # list of [T]
    if not available:
        return np.zeros(T, dtype=np.float32)
    stacked = np.stack(available, axis=1)  # [T, n_hooves]
    return (stacked.max(axis=1) - stacked.min(axis=1)).astype(np.float32)


def feature_mean_confidence(arr: np.ndarray, layout: CowKeypointLayout) -> np.ndarray:
    """Mean keypoint detection confidence across all keypoints, per frame.

    Acts as a quality gate: low-confidence frames have near-zero feature
    magnitude and contribute less signal to the GRU.

    Returns ``[T]`` float32 in [0, 1].
    """
    return arr[:, :, 2].mean(axis=1).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Feature function factory
# ──────────────────────────────────────────────────────────────────────────────

# Registry: name → function(arr, layout) → [T]
_FEATURE_REGISTRY: dict[str, Callable] = {
    "spine_angle": feature_spine_angle,
    "spine_curvature": feature_spine_curvature,
    "hip_drop": feature_hip_drop,
    "hoof_x_range": feature_hoof_x_range,
    "mean_confidence": feature_mean_confidence,
}

DEFAULT_FEATURES = list(_FEATURE_REGISTRY.keys())
NUM_DEFAULT_FEATURES = len(DEFAULT_FEATURES)


def build_feature_fn(
    layout: CowKeypointLayout = DEFAULT_LAYOUT,
    feature_names: list[str] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable ``feature_fn(arr) -> [T, F]``.

    Parameters
    ----------
    layout        : Skeleton keypoint index mapping.
    feature_names : Subset of features to compute (default: all 5).
                    Order determines column order in the output.

    Returns
    -------
    Callable that accepts ``arr: [T, K, 3]`` and returns ``[T, F]`` float32.
    """
    names = feature_names or DEFAULT_FEATURES
    unknown = set(names) - set(_FEATURE_REGISTRY)
    if unknown:
        raise ValueError(f"Unknown feature names: {sorted(unknown)}.  "
                         f"Valid: {sorted(_FEATURE_REGISTRY)}")

    fns = [_FEATURE_REGISTRY[n] for n in names]

    def _compute(arr: np.ndarray) -> np.ndarray:
        cols = [fn(arr, layout) for fn in fns]  # each [T]
        return np.stack(cols, axis=1)            # [T, F]

    _compute.feature_names = names  # type: ignore[attr-defined]
    _compute.num_features = len(names)  # type: ignore[attr-defined]
    return _compute


def make_layout_from_k(k: int) -> CowKeypointLayout:
    """Infer a best-effort layout from the keypoint count K.

    K=7   → DEFAULT_LAYOUT (7-keypoint side-view)
    K=10  → EXTENDED_LAYOUT (10-keypoint with 4 hooves)
    other → DEFAULT_LAYOUT with out-of-range indices silenced (zero-filled)
    """
    if k == 10:
        return EXTENDED_LAYOUT
    return DEFAULT_LAYOUT
