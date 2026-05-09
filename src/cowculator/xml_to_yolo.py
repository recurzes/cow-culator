"""CVAT XML for Images 1.1 → YOLO pose .txt labels."""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_cvat_xml_to_yolo_pose(
    xml_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> int:
    """Write one .txt per image in the XML. Returns number of label files written."""
    xml_path = Path(xml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    n_written = 0
    for image in root.findall("image"):
        img_name = image.get("name")
        img_width = float(image.get("width"))
        img_height = float(image.get("height"))

        txt_filename = os.path.splitext(img_name)[0] + ".txt"
        txt_path = output_dir / txt_filename

        yolo_lines: list[str] = []

        boxes = image.findall("box[@label='cow']")
        if not boxes:
            boxes = image.findall("box[@label='cow_skeleton']")
        points_elems = image.findall("points[@label='cow_skeleton']")

        for box, points_elem in zip(boxes, points_elems):
            xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
            xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))

            dw, dh = 1.0 / img_width, 1.0 / img_height
            xc = (xtl + xbr) / 2.0 * dw
            yc = (ytl + ybr) / 2.0 * dh
            w = (xbr - xtl) * dw
            h = (ybr - ytl) * dh

            raw_points = points_elem.get("points").split(";")
            yolo_kpts = []
            for p in raw_points:
                px, py = map(float, p.split(","))
                yolo_kpts.append(f"{px * dw:.6f} {py * dh:.6f} 2")

            class_id = 0
            yolo_line = (
                f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} " + " ".join(yolo_kpts)
            )
            yolo_lines.append(yolo_line)

        if not yolo_lines:
            continue
        if txt_path.is_file() and not overwrite:
            continue
        txt_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        n_written += 1
    return n_written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml", type=Path, required=True, help="CVAT annotations.xml")
    p.add_argument(
        "--labels-out",
        type=Path,
        required=True,
        help="Directory for YOLO pose .txt files",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt (default: skip existing)",
    )
    args = p.parse_args(argv)
    xml_path = args.xml.resolve()
    if not xml_path.is_file():
        print(f"error: --xml is not a file: {xml_path}", file=sys.stderr)
        sys.exit(1)
    out = args.labels_out.resolve()
    n = convert_cvat_xml_to_yolo_pose(
        xml_path, out, overwrite=args.overwrite
    )
    print(f"xml-to-yolo: wrote {n} label file(s) under {out}")


if __name__ == "__main__":
    main()
