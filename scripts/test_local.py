#!/usr/bin/env python3
"""Evaluate and run inference for a trained signboard YOLO model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def to_float(value) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def detect_device(requested_device: str) -> str:
    """Auto-detect best device. Fall back to CPU if CUDA unavailable."""
    if requested_device == "0" or requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[warn] CUDA not available; falling back to CPU")
            return "cpu"
    return requested_device


def main() -> int:
    parser = argparse.ArgumentParser(description="Test/evaluate trained YOLO signboard model.")
    parser.add_argument("--weights", required=True, help="Path to trained weights (.pt).")
    parser.add_argument("--data", default="data/curated/signboard_yolo26_lite/dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="Device to use. Set to 'cpu' for CPU-only or '0' for GPU. Auto-falls back to CPU if CUDA unavailable.")
    parser.add_argument("--project", default="runs/signboard_eval")
    parser.add_argument("--name", default="test")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--predict-source", default="", help="Optional image dir/video for inference.")
    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    data = Path(args.data).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")

    device = detect_device(args.device)
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
    )

    summary = {
        "weights": str(weights),
        "data": str(data),
        "metrics": {
            "map50": to_float(getattr(metrics.box, "map50", None)),
            "map50_95": to_float(getattr(metrics.box, "map", None)),
            "precision": to_float(getattr(metrics.box, "mp", None)),
            "recall": to_float(getattr(metrics.box, "mr", None)),
        },
    }
    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if args.predict_source:
        model.predict(
            source=args.predict_source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            save=True,
            project=args.project,
            name=f"{args.name}_pred",
        )
        print(f"[done] prediction outputs: {Path(args.project) / (args.name + '_pred')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
