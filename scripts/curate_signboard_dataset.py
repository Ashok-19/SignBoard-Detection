#!/usr/bin/env python3
"""Curate a single-class traffic signboard dataset for lightweight YOLO training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imagehash
import yaml
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class Record:
    source: str
    source_rel_id: str
    image_path: Path
    boxes: list[tuple[float, float, float, float]]
    group: str


def load_config(config_path: Path) -> list[dict]:
    data = yaml.safe_load(config_path.read_text())
    return data.get("sources", [])


def find_image_by_stem(base_path: Path) -> Path | None:
    for ext in IMAGE_EXTS:
        candidates = (base_path.with_suffix(ext), Path(f"{base_path}{ext}"))
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def parse_yolo_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for raw in label_path.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            x, y, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        if w <= 0.0 or h <= 0.0:
            continue
        boxes.append((x, y, w, h))
    return boxes


def load_github_abhay(source_name: str, source_path: Path) -> tuple[list[Record], int]:
    labels_root = source_path / "labels"
    images_root = source_path / "JPEGImages"
    if not labels_root.exists() or not images_root.exists():
        return [], 0

    records: list[Record] = []
    missing_images = 0
    for label_path in labels_root.rglob("*.txt"):
        rel = label_path.relative_to(labels_root).with_suffix("")
        image_path = find_image_by_stem(images_root / rel)
        if not image_path:
            missing_images += 1
            continue
        boxes = parse_yolo_boxes(label_path)
        if not boxes:
            continue
        group = f"{source_name}/{rel.parts[0]}/{rel.parts[1] if len(rel.parts) > 1 else 'root'}"
        records.append(
            Record(
                source=source_name,
                source_rel_id=str(rel),
                image_path=image_path,
                boxes=boxes,
                group=group,
            )
        )
    return records, missing_images


def load_yolo_export(source_name: str, source_path: Path) -> tuple[list[Record], int]:
    records: list[Record] = []
    missing_images = 0
    for split in ("train", "valid", "val", "test"):
        labels_root = source_path / split / "labels"
        images_root = source_path / split / "images"
        if not labels_root.exists() or not images_root.exists():
            continue
        for label_path in labels_root.rglob("*.txt"):
            rel = label_path.relative_to(labels_root).with_suffix("")
            image_path = find_image_by_stem(images_root / rel)
            if not image_path:
                missing_images += 1
                continue
            boxes = parse_yolo_boxes(label_path)
            if not boxes:
                continue
            base_group = rel.name.split(".rf.")[0]
            group = f"{source_name}/{split}/{base_group}"
            records.append(
                Record(
                    source=source_name,
                    source_rel_id=f"{split}/{rel}",
                    image_path=image_path,
                    boxes=boxes,
                    group=group,
                )
            )
    return records, missing_images


def load_kaggle_csv(source_name: str, source_path: Path) -> tuple[list[Record], int]:
    csv_files = list(source_path.rglob("*.csv"))
    if not csv_files:
        return [], 0

    field_alias = {
        "filename": ("filename", "image", "img", "file"),
        "width": ("width", "img_width", "w"),
        "height": ("height", "img_height", "h"),
        "xmin": ("xmin", "x_min", "left"),
        "ymin": ("ymin", "y_min", "top"),
        "xmax": ("xmax", "x_max", "right"),
        "ymax": ("ymax", "y_max", "bottom"),
    }

    def resolve_key(keys: Iterable[str], row_keys: list[str]) -> str | None:
        row_key_set = {k.lower(): k for k in row_keys}
        for k in keys:
            if k.lower() in row_key_set:
                return row_key_set[k.lower()]
        return None

    rows_by_image: dict[str, list[tuple[float, float, float, float]]] = {}
    missing_images = 0
    csv_path = csv_files[0]
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        k_filename = resolve_key(field_alias["filename"], fieldnames)
        k_width = resolve_key(field_alias["width"], fieldnames)
        k_height = resolve_key(field_alias["height"], fieldnames)
        k_xmin = resolve_key(field_alias["xmin"], fieldnames)
        k_ymin = resolve_key(field_alias["ymin"], fieldnames)
        k_xmax = resolve_key(field_alias["xmax"], fieldnames)
        k_ymax = resolve_key(field_alias["ymax"], fieldnames)
        req = [k_filename, k_width, k_height, k_xmin, k_ymin, k_xmax, k_ymax]
        if any(k is None for k in req):
            return [], 0

        for row in reader:
            try:
                fname = row[k_filename].strip()
                width = float(row[k_width])
                height = float(row[k_height])
                xmin = float(row[k_xmin])
                ymin = float(row[k_ymin])
                xmax = float(row[k_xmax])
                ymax = float(row[k_ymax])
            except Exception:
                continue
            if width <= 0 or height <= 0 or xmax <= xmin or ymax <= ymin:
                continue
            x = ((xmin + xmax) / 2.0) / width
            y = ((ymin + ymax) / 2.0) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            rows_by_image.setdefault(fname, []).append((x, y, w, h))

    records: list[Record] = []
    for fname, boxes in rows_by_image.items():
        candidates = list(source_path.rglob(fname))
        if not candidates:
            missing_images += 1
            continue
        image_path = candidates[0]
        group = f"{source_name}/{image_path.parent.name}"
        records.append(
            Record(
                source=source_name,
                source_rel_id=fname,
                image_path=image_path,
                boxes=boxes,
                group=group,
            )
        )
    return records, missing_images


def load_pascal_voc(source_name: str, source_path: Path) -> tuple[list[Record], int]:
    xml_files = list(source_path.rglob("*.xml"))
    if not xml_files:
        return [], 0

    records: list[Record] = []
    missing_images = 0
    for xml_file in xml_files:
        try:
            root = ET.parse(xml_file).getroot()
            filename = (root.findtext("filename") or "").strip()
            if not filename:
                continue
            w = float(root.findtext("size/width") or 0.0)
            h = float(root.findtext("size/height") or 0.0)
            if w <= 0 or h <= 0:
                continue
            image_candidates = list(source_path.rglob(filename))
            if not image_candidates:
                missing_images += 1
                continue
            image_path = image_candidates[0]
            boxes: list[tuple[float, float, float, float]] = []
            for obj in root.findall("object"):
                xmin = float(obj.findtext("bndbox/xmin") or 0.0)
                ymin = float(obj.findtext("bndbox/ymin") or 0.0)
                xmax = float(obj.findtext("bndbox/xmax") or 0.0)
                ymax = float(obj.findtext("bndbox/ymax") or 0.0)
                if xmax <= xmin or ymax <= ymin:
                    continue
                x = ((xmin + xmax) / 2.0) / w
                y = ((ymin + ymax) / 2.0) / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                boxes.append((x, y, bw, bh))
            if not boxes:
                continue
            records.append(
                Record(
                    source=source_name,
                    source_rel_id=filename,
                    image_path=image_path,
                    boxes=boxes,
                    group=f"{source_name}/{image_path.parent.name}",
                )
            )
        except Exception:
            continue
    return records, missing_images


def deduplicate_records(records: list[Record], threshold: int) -> tuple[list[Record], int]:
    if threshold < 0:
        return records, 0

    kept: list[Record] = []
    duplicates = 0
    seen_exact: dict[str, int] = {}
    seen_by_prefix: dict[str, list[tuple[imagehash.ImageHash, int]]] = {}

    for rec in records:
        with Image.open(rec.image_path) as img:
            ph = imagehash.phash(img, hash_size=16)
        ph_hex = str(ph)
        if threshold == 0:
            if ph_hex in seen_exact:
                duplicates += 1
                continue
            seen_exact[ph_hex] = 1
            kept.append(rec)
            continue

        prefix = ph_hex[:8]
        is_dup = False
        for existing_hash, _ in seen_by_prefix.get(prefix, []):
            if ph - existing_hash <= threshold:
                duplicates += 1
                is_dup = True
                break
        if is_dup:
            continue
        seen_by_prefix.setdefault(prefix, []).append((ph, 1))
        kept.append(rec)
    return kept, duplicates


def filter_boxes(
    records: list[Record],
    min_box_area: float,
    max_box_area: float,
    max_boxes_per_image: int,
) -> tuple[list[Record], dict]:
    kept: list[Record] = []
    dropped_no_boxes = 0
    dropped_tiny_boxes = 0
    dropped_large_boxes = 0
    for rec in records:
        valid = []
        for x, y, w, h in rec.boxes:
            area = w * h
            if area < min_box_area:
                dropped_tiny_boxes += 1
                continue
            if area > max_box_area:
                dropped_large_boxes += 1
                continue
            valid.append((x, y, w, h))
        if not valid:
            dropped_no_boxes += 1
            continue
        if len(valid) > max_boxes_per_image:
            valid = valid[:max_boxes_per_image]
        kept.append(
            Record(
                source=rec.source,
                source_rel_id=rec.source_rel_id,
                image_path=rec.image_path,
                boxes=valid,
                group=rec.group,
            )
        )
    return kept, {
        "dropped_no_boxes": dropped_no_boxes,
        "dropped_tiny_boxes": dropped_tiny_boxes,
        "dropped_large_boxes": dropped_large_boxes,
    }


def sample_per_source(
    records: list[Record],
    max_images_per_source: int,
    seed: int,
    per_source_limits: dict[str, int] | None = None,
) -> list[Record]:
    per_source_limits = per_source_limits or {}
    if max_images_per_source <= 0 and not per_source_limits:
        return records
    rng = random.Random(seed)
    by_source: dict[str, list[Record]] = {}
    for rec in records:
        by_source.setdefault(rec.source, []).append(rec)
    sampled: list[Record] = []
    for src, items in by_source.items():
        source_cap = per_source_limits.get(src, max_images_per_source)
        if source_cap <= 0 or len(items) <= source_cap:
            sampled.extend(items)
            continue
        rng.shuffle(items)
        sampled.extend(items[:source_cap])
        print(f"[sample] {src}: {len(items)} -> {source_cap}")
    return sampled


def split_grouped(records: list[Record], val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[Record]]:
    groups: dict[str, list[Record]] = {}
    for rec in records:
        groups.setdefault(rec.group, []).append(rec)

    group_items = list(groups.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)

    total = len(records)
    target_test = max(1, int(total * test_ratio))
    target_val = max(1, int(total * val_ratio))

    split_map = {"train": [], "val": [], "test": []}
    test_count = 0
    val_count = 0

    for _, items in group_items:
        if test_count < target_test:
            split_map["test"].extend(items)
            test_count += len(items)
        elif val_count < target_val:
            split_map["val"].extend(items)
            val_count += len(items)
        else:
            split_map["train"].extend(items)

    if not split_map["train"] and split_map["val"]:
        split_map["train"].extend(split_map["val"][: max(1, len(split_map["val"]) // 2)])
        split_map["val"] = split_map["val"][len(split_map["train"]) :]
    if not split_map["train"] and split_map["test"]:
        split_map["train"].extend(split_map["test"][: max(1, len(split_map["test"]) // 2)])
        split_map["test"] = split_map["test"][len(split_map["train"]) :]

    return split_map


def write_dataset(
    output_dir: Path,
    split_map: dict[str, list[Record]],
) -> dict:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    stats = {
        "totals": {"images": 0, "boxes": 0},
        "splits": {},
        "sources": {},
    }

    for split, items in split_map.items():
        split_images = 0
        split_boxes = 0
        for rec in items:
            uid = hashlib.sha1(f"{rec.source}:{rec.source_rel_id}".encode("utf-8")).hexdigest()[:12]
            safe_id = rec.source_rel_id.replace("/", "__").replace("\\", "__")
            safe_id = safe_id.replace(" ", "_")
            stem = f"{rec.source}__{safe_id}__{uid}"
            ext = rec.image_path.suffix.lower()
            img_dst = output_dir / "images" / split / f"{stem}{ext}"
            lbl_dst = output_dir / "labels" / split / f"{stem}.txt"

            shutil.copy2(rec.image_path, img_dst)
            with lbl_dst.open("w") as f:
                for x, y, w, h in rec.boxes:
                    f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            box_count = len(rec.boxes)
            split_images += 1
            split_boxes += box_count
            stats["sources"].setdefault(rec.source, {"images": 0, "boxes": 0})
            stats["sources"][rec.source]["images"] += 1
            stats["sources"][rec.source]["boxes"] += box_count
            manifest_rows.append(
                {
                    "split": split,
                    "source": rec.source,
                    "source_rel_id": rec.source_rel_id,
                    "group": rec.group,
                    "image_path": str(rec.image_path),
                    "dest_image": str(img_dst),
                    "num_boxes": box_count,
                }
            )

        stats["splits"][split] = {"images": split_images, "boxes": split_boxes}
        stats["totals"]["images"] += split_images
        stats["totals"]["boxes"] += split_boxes

    with (output_dir / "dataset.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "path": str(output_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "traffic_sign_board"},
                "nc": 1,
            },
            f,
            sort_keys=False,
        )

    with (output_dir / "reports" / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    with (output_dir / "reports" / "manifest.csv").open("w", newline="") as f:
        fieldnames = [
            "split",
            "source",
            "source_rel_id",
            "group",
            "image_path",
            "dest_image",
            "num_boxes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return stats


def collect_records(sources: list[dict]) -> tuple[list[Record], list[dict], dict]:
    all_records: list[Record] = []
    source_report: list[dict] = []
    missing_report = {"missing_paths": [], "missing_images": {}}

    for source in sources:
        name = source.get("name", "unnamed")
        if not source.get("enabled", True):
            source_report.append({"source": name, "status": "disabled", "records": 0})
            continue

        path = Path(source["path"]).resolve()
        if not path.exists():
            source_report.append({"source": name, "status": "path_missing", "records": 0})
            if source.get("required", False):
                missing_report["missing_paths"].append(name)
            continue

        kind = source.get("kind")
        if kind == "github_abhay":
            records, missing_images = load_github_abhay(name, path)
        elif kind == "yolo_export":
            records, missing_images = load_yolo_export(name, path)
        elif kind == "kaggle_csv":
            records, missing_images = load_kaggle_csv(name, path)
            if not records:
                records, missing_images = load_pascal_voc(name, path)
        else:
            source_report.append({"source": name, "status": f"unsupported_kind:{kind}", "records": 0})
            continue

        all_records.extend(records)
        source_report.append({"source": name, "status": "loaded", "records": len(records)})
        if missing_images:
            missing_report["missing_images"][name] = missing_images
    return all_records, source_report, missing_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a curated signboard train/val/test dataset.")
    parser.add_argument("--sources-config", default="configs/sources.yaml", help="YAML source config path.")
    parser.add_argument(
        "--output-dir",
        default="data/curated/signboard_yolo26_lite",
        help="Output directory for curated dataset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--min-box-area", type=float, default=0.0005)
    parser.add_argument("--max-box-area", type=float, default=0.90)
    parser.add_argument("--max-boxes-per-image", type=int, default=64)
    parser.add_argument(
        "--dedup-threshold",
        type=int,
        default=0,
        help="Perceptual hash threshold. 0 = exact hash dedup only; -1 = disable dedup.",
    )
    parser.add_argument(
        "--max-images-per-source",
        type=int,
        default=0,
        help="Cap images per source (0 disables cap).",
    )
    args = parser.parse_args()

    if args.val_ratio <= 0 or args.test_ratio <= 0 or (args.val_ratio + args.test_ratio) >= 0.8:
        raise ValueError("Use reasonable split ratios, e.g., val=0.1 and test=0.1.")

    sources = load_config(Path(args.sources_config).resolve())
    records, source_report, missing_report = collect_records(sources)
    if not records:
        raise RuntimeError("No records loaded from sources. Check source paths and formats.")

    print("[load] source summary:")
    for row in source_report:
        print(f"  - {row['source']}: {row['status']} ({row['records']} records)")

    records, filter_info = filter_boxes(
        records,
        args.min_box_area,
        args.max_box_area,
        args.max_boxes_per_image,
    )
    source_caps = {src.get("name"): int(src.get("max_images", 0) or 0) for src in sources}
    records = sample_per_source(records, args.max_images_per_source, args.seed, source_caps)
    records, deduped = deduplicate_records(records, args.dedup_threshold)
    if not records:
        raise RuntimeError("No records left after filtering/sampling. Relax thresholds.")
    split_map = split_grouped(records, args.val_ratio, args.test_ratio, args.seed)
    stats = write_dataset(Path(args.output_dir).resolve(), split_map)

    report_path = Path(args.output_dir).resolve() / "reports" / "curation_report.json"
    report = {
        "config": {
            "sources_config": str(Path(args.sources_config).resolve()),
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "min_box_area": args.min_box_area,
            "max_box_area": args.max_box_area,
            "max_boxes_per_image": args.max_boxes_per_image,
            "dedup_threshold": args.dedup_threshold,
            "max_images_per_source": args.max_images_per_source,
            "source_caps": source_caps,
        },
        "source_report": source_report,
        "missing_report": missing_report,
        "filter_info": filter_info,
        "deduplicated_records": deduped,
        "stats": stats,
    }
    report_path.write_text(json.dumps(report, indent=2))

    print("[done] curated dataset created")
    print(f"  output: {Path(args.output_dir).resolve()}")
    print(f"  train: {stats['splits']['train']['images']} images")
    print(f"  val:   {stats['splits']['val']['images']} images")
    print(f"  test:  {stats['splits']['test']['images']} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
