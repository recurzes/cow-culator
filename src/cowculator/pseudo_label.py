"""Run a YOLOv8-pose checkpoint on images; save pseudo-labels as YOLO pose .txt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cowculator.paths import default_pose_checkpoint, repo_root

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    return sorted(out)


def _kpt_v(conf: float, thr: float) -> int:
    return 2 if conf >= thr else 0


def _pose_lines_from_result(
    r,
    *,
    kpt_conf_thr: float,
) -> list[str]:
    if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
        return []
    n = len(r.boxes)
    box_conf = r.boxes.conf
    if box_conf is None:
        return []
    order = sorted(range(n), key=lambda i: float(box_conf[i].item()), reverse=True)
    xywhn = r.boxes.xywhn
    xyn = r.keypoints.xyn
    kptc = r.keypoints.conf
    if xyn is None or kptc is None:
        return []
    lines: list[str] = []
    for j in order:
        cx, cy, w, h = (float(t) for t in xywhn[j])
        kpts_parts: list[str] = []
        for ki in range(xyn.shape[1]):
            x = float(xyn[j, ki, 0])
            y = float(xyn[j, ki, 1])
            kc = float(kptc[j, ki])
            kpts_parts.append(f"{x:.6f} {y:.6f} {_kpt_v(kc, kpt_conf_thr)}")
        yolo = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} " + " ".join(kpts_parts)
        lines.append(yolo)
    return lines


def _resolve_model_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    ckpt = default_pose_checkpoint()
    if ckpt is not None:
        return ckpt
    print(
        "error: no pose weights found under runs/pose/ and COWCULATOR_MODEL unset; "
        "pass --model",
        file=sys.stderr,
    )
    sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    root = repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--images-root",
        type=Path,
        default=root / "data" / "tracked_cows",
        help="Root folder to scan for images (recursively)",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="YOLOv8-pose .pt (default: latest runs/pose/**/weights/best.pt or COWCULATOR_MODEL)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, write .txt under this path mirroring paths under --images-root",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Box confidence threshold")
    p.add_argument(
        "--kpt-conf",
        type=float,
        default=0.25,
        help="Keypoint conf >= this -> v=2 in label, else v=0",
    )
    p.add_argument("--device", default=None, help="e.g. 0, cpu, mps")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .txt (default: skip if label exists)",
    )
    p.add_argument(
        "--max-dets",
        type=int,
        default=0,
        help="If >0, keep at most this many detections per image (highest conf first)",
    )
    p.add_argument(
        "--write-empty",
        action="store_true",
        help="When no detections, write an empty .txt; default: do not create a file",
    )
    args = p.parse_args(argv)

    images_root = args.images_root.resolve()
    if not images_root.is_dir():
        print(f"error: --images-root is not a directory: {images_root}", file=sys.stderr)
        sys.exit(1)
    out_base = args.output_dir.resolve() if args.output_dir else None

    model_path = _resolve_model_path(args.model)
    if not model_path.is_file():
        print(f"error: model is not a file: {model_path}", file=sys.stderr)
        sys.exit(1)

    images = _collect_images(images_root)
    if not images:
        print(f"error: no images under {images_root}", file=sys.stderr)
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if getattr(model, "task", None) != "pose":
        print(
            f"error: model task is {getattr(model, 'task', None)!r}, expected 'pose'",
            file=sys.stderr,
        )
        sys.exit(1)

    n_written = 0
    n_skipped_existing = 0
    n_no_detection = 0
    n_failed = 0

    for img_path in images:
        if out_base is not None:
            rel = img_path.relative_to(images_root)
            txt_path = (out_base / rel).with_suffix(".txt")
        else:
            txt_path = img_path.with_suffix(".txt")
        if txt_path.is_file() and not args.force:
            n_skipped_existing += 1
            continue
        if out_base is not None:
            txt_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            results = model.predict(
                str(img_path),
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
        except Exception as e:
            n_failed += 1
            print(f"error: predict failed {img_path}: {e}", file=sys.stderr)
            continue
        r = results[0]
        lines = _pose_lines_from_result(r, kpt_conf_thr=args.kpt_conf)
        if args.max_dets and args.max_dets > 0:
            lines = lines[: args.max_dets]
        if not lines:
            n_no_detection += 1
            if args.write_empty:
                txt_path.write_text("", encoding="utf-8")
                n_written += 1
            continue
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_written += 1

    print("pseudo-label summary")
    print(f"  model:            {model_path}")
    print(f"  images total:     {len(images)}")
    print(f"  written:          {n_written}")
    print(f"  skipped existing: {n_skipped_existing}")
    print(f"  no detections:    {n_no_detection}")
    print(f"  failed:           {n_failed}")


if __name__ == "__main__":
    main()
