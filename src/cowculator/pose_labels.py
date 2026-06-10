"""Shared YOLO pose label conventions (cow skeleton keypoint count)."""

from __future__ import annotations

from pathlib import Path

EXPECTED_KEYPOINTS = 21
KPT_DIM = 3


def parse_pose_line_keypoint_count(line: str) -> int:
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


def keypoint_count_from_parts(kpt_parts: list[str]) -> int:
    return len(kpt_parts)


def filter_pose_label_lines(
    path: Path,
    *,
    expected_k: int = EXPECTED_KEYPOINTS,
) -> tuple[list[str], list[str]]:
    """Return (valid lines, issue messages) for non-empty lines in *path*."""
    valid: list[str] = []
    issues: list[str] = []
    with open(path, encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s:
                continue
            try:
                k = parse_pose_line_keypoint_count(s)
            except ValueError as exc:
                issues.append(f"{path.name}:{i}: {exc}")
                continue
            if k == expected_k:
                valid.append(s)
            else:
                issues.append(
                    f"{path.name}:{i}: {k} keypoints, expected {expected_k}"
                )
    return valid, issues
