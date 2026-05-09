#!/usr/bin/env python3
"""Convert YOLOv8-pose .txt pseudo-labels to COCO keypoints JSON for CVAT.

JPEG and PNG dimensions are read without extra dependencies. For WebP, BMP, etc.,
install Pillow: ``pip install pillow``.

CVAT
----
1. Create a task that includes the same image files the JSON will reference. Set
   ``--file-name-style`` so ``file_name`` matches what CVAT uses for the frame
   (``basename`` = ``frame_0.jpg``; ``stem`` = ``frame_0`` when CVAT’s item id has
   no extension; ``relative`` / ``relative_stem`` with subfolders).
2. In Raw / labels, add a *skeleton* with the same number of points as
   ``--num-keypoints`` (default 21). The **skeleton label's name in CVAT** must
   match ``--category-name`` (COCO ``categories[0].name``; default ``cow``). If
   your task uses e.g. ``cow_skeleton`` as the skeleton label, pass
   ``--category-name cow_skeleton`` when building the JSON. Name each
   sub-point to match ``categories[0].keypoints`` in the JSON in order. Default
   sub-point names are ``kpt_00`` … ``kpt_20``; use ``--keypoint-names`` to
   supply your own (one per line).
3. Upload annotations: format **COCO Keypoints 1.0** and select this ``.json`` (or
   a ``.zip`` from ``--zip`` with ``annotations/person_keypoints_<subset>.json``).

By default, ``-o`` is omitted and the JSON (and optional ``.zip``) is written under
``--images-root``: at the root of the tree, or in the subfolder of the image when
using ``--only`` (e.g. ``cow_1/person_keypoints_coco.json``).

If import fails, export one hand-drawn COCO Keypoints from CVAT and compare
``categories[0].keypoints`` and ``file_name`` to your file.

Troubleshooting: if CVAT reports **Could not match item id** (e.g. ``cow_1/frame_0``),
your task's frame names do not match the ``file_name`` fields in the JSON. Use
``--file-name-style basename`` for flat names with extension, ``stem`` for
**``frame_0``**-style (no ``.jpg``) when the task or export shows the frame
without a suffix, ``relative`` / ``relative_stem`` for paths with or without
extension on the last component.

If CVAT reports **Label '...' is not registered for this task** (or “can't import
annotation (skeleton)”), the category name in the JSON does not match any label
on the task. Re-export with ``--category-name`` set to the **exact** skeleton
label name in the task, or add a skeleton label in CVAT whose name matches
``--category-name`` (default ``cow``).

Visibility: YOLO uses ``0`` and ``2`` in labels (not present / visible). These map
1:1 to COCO (``1`` = occluded is not emitted by the converter).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zipfile
from pathlib import Path

from cowculator.paths import repo_root

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    return sorted(out)


def _parse_pose_keypoint_count(line: str) -> int:
    """Return K for one YOLO pose line (5 + 3*K tokens)."""
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


def _parse_pose_line(line: str) -> tuple[int, float, float, float, float, list[tuple[float, float, int]]]:
    """Parse a single YOLO pose line into class, normalized box, and keypoints."""
    tokens = line.strip().split()
    if not tokens:
        raise ValueError("empty line")
    n = len(tokens)
    if n < 5:
        raise ValueError(f"Pose line needs at least 5 values, got {n}")
    rest = n - 5
    if rest % 3 != 0:
        raise ValueError("Expected 5 + 3*K tokens")
    k = rest // 3
    cls = int(float(tokens[0]))
    cx, cy, w, h = (float(t) for t in tokens[1:5])
    kpts: list[tuple[float, float, int]] = []
    for i in range(k):
        o = 5 + i * 3
        kx, ky, kv = float(tokens[o]), float(tokens[o + 1]), int(float(tokens[o + 2]))
        kpts.append((kx, ky, kv))
    return cls, cx, cy, w, h, kpts


def _label_path_for_image(img: Path, images_root: Path, labels_root: Path | None) -> Path:
    if labels_root is None:
        return img.with_suffix(".txt")
    rel = img.relative_to(images_root)
    return (labels_root / rel).with_suffix(".txt")


def _file_name_for_cvat(img: Path, images_root: Path, style: str) -> str:
    if style == "basename":
        return img.name
    if style == "stem":
        return img.stem
    if style == "relative":
        return str(img.relative_to(images_root).as_posix())
    if style == "relative_stem":
        rel = img.relative_to(images_root)
        if rel.parent == Path("."):
            return rel.stem
        return f"{rel.parent.as_posix()}/{rel.stem}"
    raise ValueError(f"unknown file_name style: {style!r}")


def _yolo_to_coco_box_and_keypoints(
    img_w: int,
    img_h: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    kpts: list[tuple[float, float, int]],
) -> tuple[list[float], list[float], int]:
    """Return COCO bbox [x,y,w,h] pixels, flat keypoints, num_keypoints v>0."""
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    bw = w * img_w
    bh = h * img_h
    # clamp to non-negative
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    bbox = [x1, y1, bw, bh]
    flat: list[float] = []
    n_vis = 0
    for kx, ky, v in kpts:
        # YOLO stores normalized 0-1; COCO in pixels, int visibility
        xi = kx * img_w
        yi = ky * img_h
        vi = int(v) if v in (0, 1, 2) else 2 if v > 0 else 0
        if vi > 0:
            n_vis += 1
        flat.extend([round(xi, 2), round(yi, 2), vi])
    return bbox, flat, n_vis


def _default_keypoint_names(n: int) -> list[str]:
    return [f"kpt_{i:02d}" for i in range(n)]


def _image_size_png(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        if f.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("not png")
        _len = int.from_bytes(f.read(4), "big")
        if f.read(4) != b"IHDR":
            raise ValueError("png: expected IHDR")
        w = int.from_bytes(f.read(4), "big")
        h = int.from_bytes(f.read(4), "big")
    return w, h


def _image_size_jpeg(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        if f.read(2) != b"\xff\xd8":
            raise ValueError("not jpeg")
        while True:
            b = f.read(1)
            if not b:
                raise ValueError("jpeg: eof before SOF")
            if b != b"\xff":
                continue
            while True:
                b = f.read(1)
                if not b:
                    raise ValueError("jpeg: eof")
                if b != b"\xff":
                    break
            marker = b[0]
            if 0xD0 <= marker <= 0xD7 or marker in (0xD8, 0xD9, 0x01):
                continue
            length = struct.unpack(">H", f.read(2))[0]
            if length < 2:
                raise ValueError("jpeg: bad segment length")
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7):
                f.read(1)  # precision
                h, w = struct.unpack(">HH", f.read(4))
                return w, h
            f.read(length - 2)


def _image_size(path: Path) -> tuple[int, int]:
    suf = path.suffix.lower()
    if suf == ".png":
        try:
            return _image_size_png(path)
        except (ValueError, OSError, struct.error):
            pass
    if suf in (".jpg", ".jpeg", ".jpe"):
        try:
            return _image_size_jpeg(path)
        except (ValueError, OSError, struct.error):
            pass
    try:
        from PIL import Image
    except ImportError:
        print(
            "error: unsupported or corrupt image; install Pillow: pip install pillow",
            file=sys.stderr,
        )
        sys.exit(1)
    with Image.open(path) as im:
        return im.size


def _load_keypoint_names(path: Path, expected: int) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) != expected:
        raise ValueError(
            f"{path}: expected {expected} non-empty keypoint name lines, got {len(lines)}"
        )
    return lines


def main(argv: list[str] | None = None) -> None:
    root = repo_root()
    p = argparse.ArgumentParser(
        description=__doc__.split("CVAT")[0].strip(),
    )
    p.add_argument(
        "--images-root",
        type=Path,
        default=root / "data" / "tracked_cows",
        help="Root folder to scan for images (recursively)",
    )
    p.add_argument(
        "--labels-root",
        type=Path,
        default=None,
        help="If set, labels are labels_root + relpath from images-root with .txt",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output COCO keypoints .json (default: <images-root>/person_keypoints_coco.json, "
        "or <images-root>/<dir>/ when using --only with a path in a subfolder)",
    )
    p.add_argument(
        "--file-name-style",
        choices=("basename", "stem", "relative", "relative_stem"),
        default="relative",
        help="basename=file.jpg; stem=file; relative=path/file.jpg; relative_stem=path/file (match CVAT frame id)",
    )
    p.add_argument(
        "--keypoint-names",
        type=Path,
        default=None,
        help="Text file, one keypoint name per line (default: kpt_00 .. kpt_N)",
    )
    p.add_argument(
        "--num-keypoints",
        type=int,
        default=21,
        help="Expected keypoints K per instance (default 21 for this project)",
    )
    p.add_argument(
        "--category-id",
        type=int,
        default=1,
        help="COCO category id (default 1)",
    )
    p.add_argument(
        "--category-name",
        default="cow",
        help="COCO categories[].name; must match a skeleton label on the CVAT task (exact string)",
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Only this image, path relative to --images-root (e.g. cow_1/frame_0.jpg)",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If >0, keep at most this many images after sort (for testing)",
    )
    p.add_argument(
        "--zip",
        action="store_true",
        help="Also write a zip with annotations/person_keypoints_<subset>.json",
    )
    p.add_argument(
        "--subset",
        default="default",
        help="Subset name in zip path (default: default)",
    )
    p.add_argument(
        "--skeleton-json",
        type=Path,
        default=None,
        help="JSON file: list of [i,j] 1-based keypoint index pairs; else skeleton []",
    )
    args = p.parse_args(argv)

    images_root = args.images_root.resolve()
    if not images_root.is_dir():
        print(f"error: --images-root is not a directory: {images_root}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        if args.only:
            rel = Path(args.only)
            out_dir = (
                images_root
                if rel.parent == Path(".")
                else (images_root / rel.parent).resolve()
            )
        else:
            out_dir = images_root
        args.output = out_dir / "person_keypoints_coco.json"

    labels_root = args.labels_root.resolve() if args.labels_root else None
    if labels_root and not labels_root.is_dir():
        print(f"error: --labels-root is not a directory: {labels_root}", file=sys.stderr)
        sys.exit(1)

    n_kpt = args.num_keypoints
    if args.keypoint_names:
        keypoint_names = _load_keypoint_names(args.keypoint_names.resolve(), n_kpt)
    else:
        keypoint_names = _default_keypoint_names(n_kpt)

    skeleton: list[list[int]] = []
    if args.skeleton_json:
        raw = args.skeleton_json.read_text(encoding="utf-8")
        skeleton = json.loads(raw)
        if not isinstance(skeleton, list):
            raise SystemExit("skeleton-json must be a JSON array of [i,j] pairs")

    images_paths = _collect_images(images_root)
    if not images_paths:
        print(f"error: no images under {images_root}", file=sys.stderr)
        sys.exit(1)

    if args.only:
        only_path = (images_root / args.only).resolve()
        if not only_path.is_file():
            print(f"error: --only does not resolve to a file: {args.only!r} -> {only_path}", file=sys.stderr)
            sys.exit(1)
        images_paths = [only_path]
    if args.max_images and args.max_images > 0:
        images_paths = images_paths[: args.max_images]

    coco: dict = {
        "info": {
            "description": "yolo_pose_to_coco_keypoints",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": args.category_id,
                "name": args.category_name,
                "supercategory": "animal",
                "keypoints": keypoint_names,
                "skeleton": skeleton,
            }
        ],
    }

    next_ann_id = 1
    for img_id, img_path in enumerate(images_paths, start=1):
        label_path = _label_path_for_image(img_path, images_root, labels_root)
        if not label_path.is_file():
            print(f"error: missing label for {img_path}: {label_path}", file=sys.stderr)
            sys.exit(1)
        w, h = _image_size(img_path)
        file_name = _file_name_for_cvat(img_path, images_root, args.file_name_style)
        coco["images"].append(
            {
                "id": img_id,
                "file_name": file_name,
                "width": w,
                "height": h,
            }
        )
        with open(label_path, encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        for line in lines:
            k_count = _parse_pose_keypoint_count(line)
            if k_count != n_kpt:
                print(
                    f"error: {label_path}: expected {n_kpt} keypoints, got {k_count}",
                    file=sys.stderr,
                )
                sys.exit(1)
            _, cx, cy, bw, bh, kpts = _parse_pose_line(line)
            if len(kpts) != n_kpt:
                print(f"error: {label_path}: parsed {len(kpts)} keypoints, expected {n_kpt}", file=sys.stderr)
                sys.exit(1)
            bbox, k_flat, n_vis = _yolo_to_coco_box_and_keypoints(w, h, cx, cy, bw, bh, kpts)
            area = max(0.0, float(bbox[2] * bbox[3]))
            coco["annotations"].append(
                {
                    "id": next_ann_id,
                    "image_id": img_id,
                    "category_id": args.category_id,
                    "bbox": bbox,
                    "area": area,
                    "keypoints": k_flat,
                    "num_keypoints": n_vis,
                    "iscrowd": 0,
                }
            )
            next_ann_id += 1

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if args.zip:
        zip_name = f"person_keypoints_{args.subset}.json"
        inner = f"annotations/{zip_name}"
        zip_path = out.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(out, arcname=inner)
        print(f"wrote {out}")
        print(f"wrote {zip_path} ({inner})")
    else:
        print(f"wrote {out}")

    print(
        f"summary: images={len(coco['images'])} annotations={len(coco['annotations'])}"
    )


if __name__ == "__main__":
    main()
