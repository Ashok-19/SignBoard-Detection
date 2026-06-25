#!/usr/bin/env python3
"""Download source datasets listed in configs/sources.yaml."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import yaml


def load_sources(config_path: Path) -> list[dict]:
    data = yaml.safe_load(config_path.read_text())
    return data.get("sources", [])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_github_zip(url: str, target_dir: Path) -> None:
    ensure_dir(target_dir)
    with tempfile.TemporaryDirectory(prefix="signboard_dl_") as td:
        archive_path = Path(td) / "archive.zip"
        with urlopen(url) as response, archive_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(td)
        extracted_dirs = [p for p in Path(td).iterdir() if p.is_dir()]
        if not extracted_dirs:
            raise RuntimeError("No extracted directory found in GitHub zip.")
        extracted_root = extracted_dirs[0]
        for child in extracted_root.iterdir():
            dest = target_dir / child.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            if child.is_dir():
                shutil.copytree(child, dest)
            else:
                shutil.copy2(child, dest)


def download_roboflow(download_cfg: dict, target_dir: Path, api_key: str) -> None:
    from roboflow import Roboflow

    ensure_dir(target_dir)
    workspace = download_cfg["workspace"]
    project = download_cfg["project"]
    version = int(download_cfg.get("version", 1))
    fmt = download_cfg.get("format", "yolov8")
    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download(fmt, location=str(target_dir))
    print(f"Downloaded Roboflow source into: {ds.location}")


def download_kaggle(download_cfg: dict, target_dir: Path) -> None:
    ensure_dir(target_dir)
    dataset = download_cfg["dataset"]
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "--unzip",
        "-p",
        str(target_dir),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download signboard sources from config.")
    parser.add_argument("--config", default="configs/sources.yaml", help="Path to sources YAML config.")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional source names to download (space-separated).",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    sources = load_sources(config_path)
    only = set(args.only or [])
    roboflow_key = os.getenv("ROBOFLOW_API_KEY", "").strip()

    for source in sources:
        name = source.get("name", "unnamed_source")
        if only and name not in only:
            continue
        if not source.get("enabled", True):
            print(f"[skip] {name}: disabled")
            continue
        download_cfg = source.get("download")
        if not download_cfg:
            print(f"[skip] {name}: no download config")
            continue

        kind = download_cfg.get("kind")
        target = Path(source["path"]).resolve()
        print(f"[start] {name} ({kind}) -> {target}")
        try:
            if kind == "github_zip":
                download_github_zip(download_cfg["url"], target)
            elif kind == "roboflow":
                if not roboflow_key:
                    print(f"[skip] {name}: ROBOFLOW_API_KEY not set")
                    continue
                download_roboflow(download_cfg, target, roboflow_key)
            elif kind == "kaggle":
                download_kaggle(download_cfg, target)
            else:
                print(f"[skip] {name}: unsupported download kind '{kind}'")
                continue
            print(f"[done] {name}")
        except Exception as exc:
            print(f"[error] {name}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
