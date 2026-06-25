#!/usr/bin/env python3
"""Export detector and classifier weights to optimized NCNN directories."""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    ("detect", ROOT / "weights/detector/best.pt"),
    ("classify", ROOT / "weights/classifier/best.pt"),
)
PROFILES = {
    "accuracy": (640, False, "best_ncnn_model_640"),
    "balanced": (320, True, "best_ncnn_model_320_fp16"),
    "realtime": (256, True, "best_ncnn_model_256_fp16"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=[*PROFILES, "all"], default="all")
    return parser.parse_args()


def export_profile(task: str, weights: Path, profile: str) -> None:
    imgsz, half, directory_name = PROFILES[profile]
    generated = weights.with_name(f"{weights.stem}_ncnn_model")
    destination = weights.with_name(directory_name)
    if generated.exists():
        shutil.rmtree(generated)
    if destination.exists():
        shutil.rmtree(destination)
    print(f"[export] {task} {profile}: {weights} -> {destination.name}")
    model = YOLO(str(weights), task=task)
    output = Path(model.export(format="ncnn", imgsz=imgsz, half=half, simplify=True))
    output.rename(destination)
    print(f"[done] {task} {profile}: {destination}")


def main() -> None:
    args = parse_args()
    profiles = PROFILES if args.profile == "all" else (args.profile,)
    for task, weights in MODELS:
        for profile in profiles:
            export_profile(task, weights, profile)


if __name__ == "__main__":
    main()
