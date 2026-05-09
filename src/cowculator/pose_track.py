#!/usr/bin/env python3
"""
Run a trained YOLO *pose* model on a video: boxes, skeleton (via Ultralytics plot),
and track IDs on the output MP4.

Class indices:
- Custom single-class (nc:1) cow models use class 0. Default ``--classes`` is ``0``.
- COCO pretrained ``yolov8n.pt``-style models use class 19 for cow; class 0 would
  not be "cow" on those weights.

Cropping (``--save-crops``) is optional; the default path is visualize + track only.
``tools/legacy/cow_tracker.py`` remains the legacy box-only + always-on crop flow.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from cowculator.paths import default_pose_checkpoint, repo_root

VIDEO_PATH: str | None = None

DEFAULT_OUT_DIR = "outputs/tracked_videos"
CROP_BASE_DIR = "data/tracked_cows"
PADDING = 0.10


def parse_classes(s: str | None) -> list[int] | None:
    if s is None:
        return [0]
    t = s.strip()
    if not t or t.lower() == "all":
        return None
    return [int(x.strip()) for x in t.split(",") if x.strip()]


def class_filter_summary(classes: list[int] | None) -> str:
    if classes is None:
        return "all"
    return str(classes)


def ensure_pose_model(model: YOLO) -> None:
    task = getattr(model, "task", None)
    if task != "pose":
        print(
            f"Error: this script expects a pose model (task 'pose'). "
            f"Loaded model task is {task!r}. Use a .pt from pose training or a pose "
            f"weights file.",
            file=sys.stderr,
        )
        sys.exit(1)


def overlay_track_ids(annotated: np.ndarray, results: Any) -> np.ndarray:
    r0 = results[0]
    boxes = r0.boxes
    if boxes is None or len(boxes) == 0:
        return annotated
    xyxy = boxes.xyxy
    n = len(boxes)
    ids = boxes.id
    if ids is None:
        return annotated
    for i in range(n):
        tid_val = ids[i]
        if tid_val is None:
            continue
        try:
            t = int(tid_val)
        except (TypeError, ValueError):
            continue
        x1, y1, x2, y2 = map(int, xyxy[i].tolist())
        label = f"ID {t}"
        y_text = max(y1 - 4, 12)
        cv2.putText(
            annotated,
            label,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )
    return annotated


def save_crops_for_frame(
    frame_bgr: np.ndarray,
    results: Any,
    width: int,
    height: int,
    frame_id: int,
    crop_dir: str,
) -> None:
    r0 = results[0]
    boxes = r0.boxes
    if boxes is None or len(boxes) == 0 or boxes.id is None:
        return
    for box, track_id in zip(boxes.xyxy, boxes.id):
        if track_id is None:
            continue
        cow_id = int(track_id)
        x1, y1, x2, y2 = map(int, box)
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = int(box_w * PADDING)
        pad_y = int(box_h * PADDING)
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(width, x2 + pad_x)
        crop_y2 = min(height, y2 + pad_y)
        cow_folder = os.path.join(crop_dir, f"cow_{cow_id}")
        os.makedirs(cow_folder, exist_ok=True)
        crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size > 0:
            out_path = os.path.join(cow_folder, f"frame_{frame_id}.jpg")
            cv2.imwrite(out_path, crop)


def _resolve_model_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    ckpt = default_pose_checkpoint()
    if ckpt is not None:
        return str(ckpt)
    print(
        "Error: no pose weights under runs/pose/ and COWCULATOR_MODEL unset; pass --model.",
        file=sys.stderr,
    )
    sys.exit(1)


def build_argparser() -> argparse.ArgumentParser:
    root = repo_root()
    p = argparse.ArgumentParser(
        description="Pose track + skeleton visualization on video (Ultralytics)."
    )
    p.add_argument(
        "--video",
        default=None,
        help="Input video path. If omitted, uses VIDEO_PATH in cowculator.pose_track.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Path to pose .pt weights (default: latest runs/pose/**/weights/best.pt)",
    )
    p.add_argument(
        "--out",
        default=None,
        help=f"Output MP4 path. Default: {DEFAULT_OUT_DIR}/<stem>_pose_tracked.mp4",
    )
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.7, help="IoU (NMS / tracker).")
    p.add_argument(
        "--device",
        default=None,
        help="e.g. cpu, 0, 0,1. Default: Ultralytics auto.",
    )
    p.add_argument(
        "--classes",
        default="0",
        metavar="CLASSES",
        help="Comma-separated class ids, or 'all' for no class filter. "
        "Default: 0 (single-class cow). COCO cow is 19 for pretrained COCO only.",
    )
    p.add_argument(
        "--save-crops",
        action="store_true",
        help=f"Save padded crops to {CROP_BASE_DIR}/cow_{{id}}/frame_{{n}}.jpg",
    )
    p.add_argument(
        "--crop-dir",
        default=str(root / CROP_BASE_DIR),
        help="Base directory for crops when --save-crops.",
    )
    p.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        help="Save crops every N frames; default: ~2 crops/sec from FPS.",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an imshow window (headless).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    video = args.video or VIDEO_PATH
    if not video:
        print(
            "Error: provide --video or set VIDEO_PATH in cowculator.pose_track.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isfile(video):
        print(f"Error: video not found: {video}", file=sys.stderr)
        sys.exit(1)

    model_path = _resolve_model_path(args.model)
    if not os.path.isfile(model_path):
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    classes = parse_classes(args.classes)
    device_kw: dict[str, Any] = {}
    if args.device is not None and str(args.device).strip():
        device_kw["device"] = args.device

    track_kwargs: dict[str, Any] = {
        "persist": True,
        "conf": args.conf,
        "iou": args.iou,
        "verbose": False,
    }
    track_kwargs.update(device_kw)
    if classes is not None:
        track_kwargs["classes"] = classes

    model = YOLO(model_path)
    ensure_pose_model(model)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"Error: could not open video: {video}", file=sys.stderr)
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    stem = os.path.splitext(os.path.basename(video))[0]
    root = repo_root()
    out_dir_default = root / DEFAULT_OUT_DIR
    if args.out:
        out_path = args.out
    else:
        out_dir_default.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir_default / f"{stem}_pose_tracked.mp4")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    frame_skip = args.frame_skip
    if args.save_crops and frame_skip is None:
        frame_skip = int(fps / 2) if fps > 0 else 15
    if frame_skip is None:
        frame_skip = 1

    print(
        f"model={model_path!s} task={getattr(model, 'task', None)!r} "
        f"classes={class_filter_summary(classes)} resolution={w}x{h} fps={fps:.2f} "
        f"-> {out_path}"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer: cv2.VideoWriter | None = None

    window_name = "Pose track"
    if not args.no_show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, w, h)

    frame_id = 0
    if args.save_crops:
        os.makedirs(args.crop_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, **track_kwargs)
        annotated = results[0].plot()
        if isinstance(annotated, np.ndarray) and annotated.dtype != np.uint8:
            annotated = np.ascontiguousarray(annotated, dtype=np.uint8)
        annotated = overlay_track_ids(annotated, results)

        if writer is None:
            oh, ow = annotated.shape[0], annotated.shape[1]
            writer = cv2.VideoWriter(out_path, fourcc, fps, (ow, oh))

        if not args.no_show:
            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer is not None:
            writer.write(annotated)

        if args.save_crops and results[0].boxes is not None and frame_id % frame_skip == 0:
            save_crops_for_frame(
                frame,
                results,
                w,
                h,
                frame_id,
                args.crop_dir,
            )

        frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_show:
        cv2.destroyAllWindows()
    print(f"Done. Wrote {out_path} ({frame_id} frames).")


if __name__ == "__main__":
    main()
