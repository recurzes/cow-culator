"""Unified CLI: ``cowculator <subcommand>``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cowculator.paths import (
    default_annotations_dir,
    default_labels_dir,
    default_pose_checkpoint,
    repo_root,
)
from cowculator.train_dataset import run_prepare_train_args, train_options_parser


def _passthrough_argv(remainder: list[str]) -> list[str] | None:
    if not remainder:
        return None
    if remainder[0] == "--":
        rest = remainder[1:]
        return rest if rest else None
    return remainder


def _cmd_extract_frames(args: argparse.Namespace) -> None:
    from cowculator.extract_frames import main as ef_main

    ef_main([str(args.directory)])


def _cmd_xml_to_yolo(args: argparse.Namespace) -> None:
    from cowculator.xml_to_yolo import main as x2y_main

    argv: list[str] = ["--labels-out", str(args.labels_out)]
    if args.xml is not None:
        argv.extend(["--xml", str(args.xml)])
    else:
        argv.extend(["--xml-dir", str(args.xml_dir)])
    if args.overwrite:
        argv.append("--overwrite")
    x2y_main(argv)


def _cmd_prepare(args: argparse.Namespace) -> None:
    run_prepare_train_args(args, do_prepare=True, do_train=False)


def _cmd_train(args: argparse.Namespace) -> None:
    run_prepare_train_args(args, do_prepare=False, do_train=True)


def _cmd_prepare_train(args: argparse.Namespace) -> None:
    run_prepare_train_args(args, do_prepare=True, do_train=True)


def _cmd_pseudo_label(args: argparse.Namespace) -> None:
    from cowculator.pseudo_label import main as pl_main

    pl_main(_passthrough_argv(list(args.remainder)))


def _cmd_yolo_to_coco(args: argparse.Namespace) -> None:
    from cowculator.coco_keypoints import main as coco_main

    coco_main(_passthrough_argv(list(args.remainder)))


def _cmd_pose_track(args: argparse.Namespace) -> None:
    from cowculator.pose_track import main as pt_main

    pt_main(_passthrough_argv(list(args.remainder)))


def _cmd_select_bcs_frames(args: argparse.Namespace) -> None:
    from cowculator.bcs_frame_selector import main as sbf_main

    sbf_main(_passthrough_argv(list(args.remainder)))


def _cmd_train_bcs(args: argparse.Namespace) -> None:
    from cowculator.train_bcs import main as tb_main

    tb_main(_passthrough_argv(list(args.remainder)))


def _cmd_infer_bcs(args: argparse.Namespace) -> None:
    from cowculator.infer_bcs import main as ib_main

    ib_main(_passthrough_argv(list(args.remainder)))


def _cmd_train_lameness(args: argparse.Namespace) -> None:
    from cowculator.train_lameness import main as tl_main

    tl_main(_passthrough_argv(list(args.remainder)))


def _cmd_infer_lameness(args: argparse.Namespace) -> None:
    from cowculator.infer_lameness import main as il_main

    il_main(_passthrough_argv(list(args.remainder)))


def _cmd_doctor(_: argparse.Namespace) -> None:
    root = repo_root()
    ds = root / "yolo_dataset" / "dataset.yaml"
    ckpt = default_pose_checkpoint()
    print(f"repo_root:              {root}")
    print(f"yolo_dataset/dataset.yaml exists: {ds.is_file()} ({ds})")
    print(f"default_pose_checkpoint: {ckpt}")
    print(f"COWCULATOR_MODEL:        {os.environ.get('COWCULATOR_MODEL', '')}")
    print(f"COWCULATOR_ANNOTATIONS_DIR: {os.environ.get('COWCULATOR_ANNOTATIONS_DIR', '')}")
    print(f"COWCULATOR_LABELS_DIR:   {os.environ.get('COWCULATOR_LABELS_DIR', '')}")
    print(f"default annotations dir: {default_annotations_dir()}")
    print(f"default labels dir:      {default_labels_dir()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cowculator",
        description="Cow pose annotation pipeline (CVAT XML, YOLO pose, train, COCO export).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("extract-frames", help="Extract ~2 fps JPEGs from videos in a directory")
    s.add_argument("directory", type=Path, help="Directory containing video files")
    s.set_defaults(_handler=_cmd_extract_frames)

    s = sub.add_parser("xml-to-yolo", help="CVAT XML for Images 1.1 → YOLO pose .txt")
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("--xml", type=Path, help="Single CVAT annotations.xml")
    src.add_argument(
        "--xml-dir",
        type=Path,
        help="Directory of CVAT .xml files (all *.xml, non-recursive)",
    )
    s.add_argument("--labels-out", type=Path, required=True)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(_handler=_cmd_xml_to_yolo)

    for name, handler, help_ in (
        ("prepare", _cmd_prepare, "Build yolo_dataset/ from images + flat .txt labels"),
        ("train", _cmd_train, "Run Ultralytics pose training"),
        ("prepare-train", _cmd_prepare_train, "Prepare yolo_dataset then train"),
    ):
        s = sub.add_parser(name, parents=[train_options_parser()], help=help_)
        s.set_defaults(_handler=handler)

    s = sub.add_parser(
        "pseudo-label",
        help="Run pose model on images; write YOLO .txt (pass-through args)",
    )
    s.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments for pseudo_label (e.g. --images-root ./data/tracked_cows)",
    )
    s.set_defaults(_handler=_cmd_pseudo_label)

    s = sub.add_parser(
        "yolo-to-coco",
        help="YOLO pose .txt → COCO Keypoints JSON for CVAT (pass-through args)",
    )
    s.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments for coco_keypoints module",
    )
    s.set_defaults(_handler=_cmd_yolo_to_coco)

    s = sub.add_parser(
        "pose-track",
        help="Pose + track on a video (pass-through args, e.g. --video path.mp4)",
    )
    s.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments for pose_track",
    )
    s.set_defaults(_handler=_cmd_pose_track)

    s = sub.add_parser(
        "select-bcs-frames",
        help="Copy last-N frames per cow track as back-view BCS candidates (pass-through args)",
    )
    s.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments for bcs_frame_selector (e.g. --crops-dir data/tracked_cows --last-n 5)",
    )
    s.set_defaults(_handler=_cmd_select_bcs_frames)

    s = sub.add_parser(
        "train-bcs",
        help="Train EfficientNet-B0 BCS regression model (pass-through args)",
    )
    s.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments for train_bcs (e.g. --csv data/bcs_labels.csv --epochs 100)",
    )
    s.set_defaults(_handler=_cmd_train_bcs)

    s = sub.add_parser(
        "infer-bcs",
        help="Predict BCS scores from back-view images (pass-through args)",
    )
    s.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="Arguments for infer_bcs (e.g. --images data/bcs_frames/ --out results.csv)",
    )
    s.set_defaults(_handler=_cmd_infer_bcs)

    s = sub.add_parser(
        "train-lameness",
        help="Train bidirectional GRU lameness classifier on pose sequences (pass-through args)",
    )
    s.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments for train_lameness "
            "(e.g. --csv data/lameness_labels.csv --epochs 100 --seq-len 60)"
        ),
    )
    s.set_defaults(_handler=_cmd_train_lameness)

    s = sub.add_parser(
        "infer-lameness",
        help="Predict lameness scores from .npy pose sequences (pass-through args)",
    )
    s.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments for infer_lameness "
            "(e.g. --sequences-dir data/pose_sequences/ --out results/lameness.csv)"
        ),
    )
    s.set_defaults(_handler=_cmd_infer_lameness)

    s = sub.add_parser("doctor", help="Print resolved default paths and env")
    s.set_defaults(_handler=_cmd_doctor)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args._handler(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
