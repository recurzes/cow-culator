"""CVAT XML for Images 1.1 → YOLO pose .txt labels."""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from cowculator.pose_labels import EXPECTED_KEYPOINTS, parse_pose_line_keypoint_count


def _parse_skeleton_orders(root: ET.Element) -> dict[str, list[str]]:
    """CVAT meta: ordered sub-label names per skeleton parent label."""
    orders: dict[str, list[str]] = {}
    for label in root.findall("./meta/task/labels/label"):
        parent = label.findtext("parent")
        name = label.findtext("name")
        if parent and name:
            orders.setdefault(parent, []).append(name)
    return orders


def _point_visibility(elem: ET.Element) -> int:
    if elem.get("outside") == "1":
        return 0
    if elem.get("occluded") == "1":
        return 1
    return 2


def _bbox_from_coords(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    return min(xs), min(ys), max(xs), max(ys)


def _format_yolo_pose_line(
    xtl: float,
    ytl: float,
    xbr: float,
    ybr: float,
    kpt_parts: list[str],
    img_width: float,
    img_height: float,
    *,
    class_id: int = 0,
) -> str:
    dw, dh = 1.0 / img_width, 1.0 / img_height
    xc = (xtl + xbr) / 2.0 * dw
    yc = (ytl + ybr) / 2.0 * dh
    w = (xbr - xtl) * dw
    h = (ybr - ytl) * dh
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} " + " ".join(kpt_parts)


def _line_from_flat_points(
    points_elem: ET.Element, img_width: float, img_height: float
) -> str | None:
    raw = points_elem.get("points", "")
    if not raw:
        return None
    pairs = []
    for p in raw.split(";"):
        p = p.strip()
        if not p or "," not in p:
            continue
        px, py = map(float, p.split(",", 1))
        pairs.append((px, py))
    if not pairs:
        return None
    vis = _point_visibility(points_elem)
    dw, dh = 1.0 / img_width, 1.0 / img_height
    xs, ys = zip(*pairs)
    xtl, ytl, xbr, ybr = _bbox_from_coords(list(xs), list(ys))
    kpt_parts = [f"{px * dw:.6f} {py * dh:.6f} {vis}" for px, py in pairs]
    return _format_yolo_pose_line(xtl, ytl, xbr, ybr, kpt_parts, img_width, img_height)


def _line_from_box_and_points(
    box: ET.Element, points_elem: ET.Element, img_width: float, img_height: float
) -> str:
    xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
    xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
    dw, dh = 1.0 / img_width, 1.0 / img_height
    raw_points = points_elem.get("points", "").split(";")
    kpt_parts = []
    for p in raw_points:
        px, py = map(float, p.split(","))
        kpt_parts.append(f"{px * dw:.6f} {py * dh:.6f} 2")
    return _format_yolo_pose_line(xtl, ytl, xbr, ybr, kpt_parts, img_width, img_height)


def _line_from_skeleton(
    skeleton: ET.Element,
    img_width: float,
    img_height: float,
    skeleton_orders: dict[str, list[str]],
) -> str | None:
    skel_label = skeleton.get("label") or ""
    order = skeleton_orders.get(skel_label)
    by_label: dict[str, ET.Element] = {
        pt.get("label", ""): pt for pt in skeleton.findall("points")
    }
    if not by_label:
        return None
    if not order:
        order = [str(i) for i in range(1, EXPECTED_KEYPOINTS + 1)]

    dw, dh = 1.0 / img_width, 1.0 / img_height
    kpt_parts: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    for name in order:
        pt = by_label.get(name)
        if pt is None:
            kpt_parts.append("0 0 0")
            continue
        coords = pt.get("points", "").strip()
        if not coords or "," not in coords:
            kpt_parts.append("0 0 0")
            continue
        px, py = map(float, coords.split(",", 1))
        vis = _point_visibility(pt)
        if vis > 0:
            xs.append(px)
            ys.append(py)
        kpt_parts.append(f"{px * dw:.6f} {py * dh:.6f} {vis}")

    if not xs:
        return None
    while len(kpt_parts) < EXPECTED_KEYPOINTS:
        kpt_parts.append("0 0 0")
    if len(kpt_parts) > EXPECTED_KEYPOINTS:
        kpt_parts = kpt_parts[:EXPECTED_KEYPOINTS]
    xtl, ytl, xbr, ybr = _bbox_from_coords(xs, ys)
    return _format_yolo_pose_line(xtl, ytl, xbr, ybr, kpt_parts, img_width, img_height)


def _accept_pose_line(
    line: str | None, *, context: str
) -> tuple[str | None, str | None]:
    if line is None:
        return None, None
    k = parse_pose_line_keypoint_count(line)
    if k == EXPECTED_KEYPOINTS:
        return line, None
    return None, f"{context}: {k} keypoints, expected {EXPECTED_KEYPOINTS}"


def convert_cvat_xml_to_yolo_pose(
    xml_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> tuple[int, list[str]]:
    """Write one .txt per image in the XML. Returns (files written, skipped-instance messages)."""
    xml_path = Path(xml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    skeleton_orders = _parse_skeleton_orders(root)

    n_written = 0
    skipped: list[str] = []
    for image in root.findall("image"):
        img_name = image.get("name")
        img_width = float(image.get("width"))
        img_height = float(image.get("height"))

        txt_filename = os.path.splitext(img_name)[0] + ".txt"
        txt_path = output_dir / txt_filename

        yolo_lines: list[str] = []

        for skeleton in image.findall("skeleton"):
            raw = _line_from_skeleton(
                skeleton, img_width, img_height, skeleton_orders
            )
            line, issue = _accept_pose_line(
                raw, context=f"{img_name} skeleton#{len(yolo_lines) + 1}"
            )
            if line:
                yolo_lines.append(line)
            elif issue:
                skipped.append(issue)

        if not yolo_lines:
            boxes = image.findall("box[@label='cow']")
            if not boxes:
                boxes = image.findall("box[@label='cow_skeleton']")
            points_elems = image.findall("points[@label='cow_skeleton']")
            for i, (box, points_elem) in enumerate(zip(boxes, points_elems)):
                raw = _line_from_box_and_points(
                    box, points_elem, img_width, img_height
                )
                line, issue = _accept_pose_line(
                    raw, context=f"{img_name} box+points#{i + 1}"
                )
                if line:
                    yolo_lines.append(line)
                elif issue:
                    skipped.append(issue)

        if not yolo_lines:
            for i, points_elem in enumerate(
                image.findall("points[@label='cow_skeleton']")
            ):
                raw = _line_from_flat_points(points_elem, img_width, img_height)
                line, issue = _accept_pose_line(
                    raw, context=f"{img_name} points#{i + 1}"
                )
                if line:
                    yolo_lines.append(line)
                elif issue:
                    skipped.append(issue)

        if not yolo_lines:
            continue
        if txt_path.is_file() and not overwrite:
            continue
        txt_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        n_written += 1
    return n_written, skipped


def convert_cvat_xml_dir_to_yolo_pose(
    xml_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> tuple[int, list[tuple[Path, int]], list[str]]:
    """Convert every ``*.xml`` in *xml_dir* (non-recursive). Returns total, per-file counts, skips."""
    xml_dir = Path(xml_dir)
    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        return 0, [], []

    per_file: list[tuple[Path, int]] = []
    skipped: list[str] = []
    total = 0
    for xml_path in xml_files:
        n, sk = convert_cvat_xml_to_yolo_pose(
            xml_path, output_dir, overwrite=overwrite
        )
        per_file.append((xml_path, n))
        skipped.extend(sk)
        total += n
    return total, per_file, skipped


def _print_skipped_instances(skipped: list[str], *, limit: int = 40) -> None:
    if not skipped:
        return
    print(f"  skipped instances (not {EXPECTED_KEYPOINTS} keypoints): {len(skipped)}")
    for msg in skipped[:limit]:
        print(f"    {msg}")
    if len(skipped) > limit:
        print(f"    ... and {len(skipped) - limit} more")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--xml", type=Path, help="CVAT annotations.xml")
    src.add_argument(
        "--xml-dir",
        type=Path,
        help="Directory of CVAT .xml files (all *.xml in directory, non-recursive)",
    )
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
    out = args.labels_out.resolve()

    if args.xml is not None:
        xml_path = args.xml.resolve()
        if not xml_path.is_file():
            print(f"error: --xml is not a file: {xml_path}", file=sys.stderr)
            sys.exit(1)
        n, skipped = convert_cvat_xml_to_yolo_pose(
            xml_path, out, overwrite=args.overwrite
        )
        print(f"xml-to-yolo: wrote {n} label file(s) under {out}")
        _print_skipped_instances(skipped)
        return

    xml_dir = args.xml_dir.resolve()
    if not xml_dir.is_dir():
        print(f"error: --xml-dir is not a directory: {xml_dir}", file=sys.stderr)
        sys.exit(1)
    total, per_file, skipped = convert_cvat_xml_dir_to_yolo_pose(
        xml_dir, out, overwrite=args.overwrite
    )
    if not per_file:
        print(f"error: no .xml files found in {xml_dir}", file=sys.stderr)
        sys.exit(1)
    for xml_path, n in per_file:
        print(f"  {xml_path.name} -> {n} labels")
    n_xml = len(per_file)
    print(
        f"xml-to-yolo: wrote {total} label file(s) across {n_xml} XML file(s) under {out}"
    )
    _print_skipped_instances(skipped)


if __name__ == "__main__":
    main()
