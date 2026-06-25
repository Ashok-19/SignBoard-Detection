"""SAHI-style tiled/sliced inference helpers for small-object recovery.

Key techniques implemented (based on SAHI, ASAHI, and multi-scale pyramid research):
  - Multi-scale tile pyramid: run at multiple tile sizes for different object scales
  - Adaptive ROI: focus tiles on the horizon/upper region where distant signs appear
  - CLAHE preprocessing: enhance contrast of tile crops for better feature extraction
  - Soft-NMS style merge: gentler suppression to keep nearby-but-distinct detections

Provides:
  - iter_tiles(): generate overlapping tile regions from a frame
  - iter_multiscale_tiles(): generate tiles at multiple scales (pyramid)
  - preprocess_tile(): CLAHE + optional sharpening for distant sign recovery
  - run_detector_on_tiles(): run YOLO detector on each tile, remap boxes
  - merge_detections_nms(): class-aware NMS to merge full-frame + tile boxes
"""

from __future__ import annotations

from dataclasses import dataclass
import threading

import cv2
import numpy as np


_TLS = threading.local()


def _get_clahe_4x4():
    clahe = getattr(_TLS, "clahe_4x4", None)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        _TLS.clahe_4x4 = clahe
    return clahe


@dataclass
class RawDetection:
    """A single detection in full-frame coordinates."""
    xyxy: tuple[float, float, float, float]
    conf: float
    cls_id: int
    cls_name: str
    source: str  # "full" or "tile"
    fresh: bool = True


@dataclass
class TileRegion:
    """A tile crop region in full-frame coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    scale_label: str = ""  # e.g. "640", "384" — for debug


def iter_tiles(
    frame_h: int,
    frame_w: int,
    tile_size: int = 640,
    overlap: float = 0.25,
    roi_mode: str = "upper",
) -> list[TileRegion]:
    """Generate overlapping tile regions.

    Args:
        roi_mode:
            "upper"  = top 75% of frame (default, good for road-facing cameras)
            "full"   = entire frame
            "center" = center 70% crop
            "horizon"= top 60% of frame (tighter, better for distant signs)
    """
    if roi_mode == "upper":
        y_start, y_end = 0, int(frame_h * 0.75)
        x_start, x_end = 0, frame_w
    elif roi_mode == "horizon":
        # Tighter focus on where distant signs appear: top 60%
        y_start, y_end = 0, int(frame_h * 0.60)
        x_start, x_end = 0, frame_w
    elif roi_mode == "center":
        cx, cy = frame_w // 2, frame_h // 2
        rw, rh = int(frame_w * 0.35), int(frame_h * 0.35)
        y_start, y_end = max(0, cy - rh), min(frame_h, cy + rh)
        x_start, x_end = max(0, cx - rw), min(frame_w, cx + rw)
    else:  # full
        y_start, y_end = 0, frame_h
        x_start, x_end = 0, frame_w

    roi_h = y_end - y_start
    roi_w = x_end - x_start
    if roi_h <= 0 or roi_w <= 0:
        return []

    step = max(1, int(tile_size * (1.0 - overlap)))
    tiles: list[TileRegion] = []

    y = y_start
    while y < y_end:
        ty2 = min(y + tile_size, y_end)
        ty1 = max(y_start, ty2 - tile_size)
        # Skip tiles that are too small (less than half tile_size)
        if ty2 - ty1 < tile_size // 2:
            break
        x = x_start
        while x < x_end:
            tx2 = min(x + tile_size, x_end)
            tx1 = max(x_start, tx2 - tile_size)
            if tx2 - tx1 < tile_size // 2:
                break
            tiles.append(TileRegion(tx1, ty1, tx2, ty2, scale_label=str(tile_size)))
            if tx2 >= x_end:
                break
            x += step
        if ty2 >= y_end:
            break
        y += step

    return tiles


def iter_multiscale_tiles(
    frame_h: int,
    frame_w: int,
    tile_sizes: list[int],
    overlap: float = 0.25,
    roi_mode: str = "upper",
) -> list[TileRegion]:
    """Generate tiles at multiple scales (tile-pyramid).

    Larger tiles cover more area with context; smaller tiles provide more
    zoom into distant/small objects. De-duplicates identical tile regions.

    Research basis: Tile-pyramid approach from multi-scale SAHI literature.
    Each scale level independently generates tiles, then all are combined.
    """
    all_tiles: list[TileRegion] = []
    seen: set[tuple[int, int, int, int]] = set()

    for ts in tile_sizes:
        # For smaller tiles, use tighter ROI (horizon) to focus on distance
        if ts <= 384 and roi_mode in ("upper", "horizon"):
            effective_roi = "horizon"
        else:
            effective_roi = roi_mode
        tiles = iter_tiles(frame_h, frame_w, ts, overlap, effective_roi)
        for t in tiles:
            key = (t.x1, t.y1, t.x2, t.y2)
            if key not in seen:
                seen.add(key)
                all_tiles.append(t)

    return all_tiles


def preprocess_tile(crop: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    """Enhance tile crop for better small-object detection.

    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to
    improve contrast of distant/low-contrast signs. Research shows CLAHE
    improves recall for small objects that blend into backgrounds.

    Does NOT upscale — YOLO's own letterboxing handles that.
    """
    if not use_clahe:
        return crop
    # Convert to LAB, apply CLAHE to L channel
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = _get_clahe_4x4()
    l_ch = clahe.apply(l_ch)
    enhanced = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def run_detector_on_tiles(
    detector,
    frame: np.ndarray,
    tiles: list[TileRegion],
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    half: bool,
    max_det: int,
    min_box_px: int = 6,
    use_clahe: bool = True,
) -> list[RawDetection]:
    """Run detector on tiles using BATCHED inference for maximum GPU throughput.

    Instead of calling detector.predict() per tile (N sequential GPU calls),
    this passes ALL tile crops as a list to a single predict() call.
    Ultralytics natively supports list[np.ndarray] batch inference, which
    amortizes CUDA kernel launch overhead and saturates GPU cores.

    Pipeline:
      1. Crop all tiles from frame + optional CLAHE preprocessing (CPU)
      2. Single batched detector.predict(all_crops) call (GPU)
      3. Remap per-tile boxes to full-frame coordinates (CPU)
    """
    if not tiles:
        return []

    # --- Phase 1: CPU — crop & preprocess all tiles ---
    valid_tiles: list[TileRegion] = []
    crops: list[np.ndarray] = []

    for tile in tiles:
        crop = frame[tile.y1:tile.y2, tile.x1:tile.x2]
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            continue
        processed = preprocess_tile(crop, use_clahe=use_clahe)
        valid_tiles.append(tile)
        crops.append(processed)

    if not crops:
        return []

    # --- Phase 2: GPU — single batched inference call ---
    batch_results = detector.predict(
        crops,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        half=half,
        max_det=max_det,
        verbose=False,
    )

    # --- Phase 3: CPU — remap boxes to full-frame coordinates ---
    detections: list[RawDetection] = []

    for tile, res in zip(valid_tiles, batch_results):
        if res.boxes is None or len(res.boxes) == 0:
            continue

        tile_w = tile.x2 - tile.x1
        tile_h = tile.y2 - tile.y1

        for box in res.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().tolist()
            bw = bx2 - bx1
            bh = by2 - by1
            if bw < min_box_px or bh < min_box_px:
                continue

            # Discard detections that span almost the entire tile
            # (likely background false positives)
            if bw > tile_w * 0.95 and bh > tile_h * 0.95:
                continue

            # Remap to full-frame coordinates
            fx1 = bx1 + tile.x1
            fy1 = by1 + tile.y1
            fx2 = bx2 + tile.x1
            fy2 = by2 + tile.y1
            cls_id = int(box.cls[0])
            detections.append(RawDetection(
                xyxy=(fx1, fy1, fx2, fy2),
                conf=float(box.conf[0]),
                cls_id=cls_id,
                cls_name=res.names.get(cls_id, f"det_cls_{cls_id}"),
                source="tile",
            ))

    return detections


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    ab = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + ab - inter + 1e-9)


def merge_detections_nms(
    full_dets: list[RawDetection],
    tile_dets: list[RawDetection],
    iou_threshold: float = 0.40,
) -> list[RawDetection]:
    """Class-aware NMS merge. Full-frame detections take priority.

    Uses a slightly lower default IoU threshold (0.40) compared to typical
    0.45-0.5, because tiled inference can produce boxes at slightly different
    scales than full-frame, and we want to suppress true duplicates without
    losing genuinely distinct nearby signs.
    """
    all_dets = full_dets + tile_dets
    if not all_dets:
        return []

    # Group by class
    by_cls: dict[int, list[RawDetection]] = {}
    for d in all_dets:
        by_cls.setdefault(d.cls_id, []).append(d)

    merged: list[RawDetection] = []
    for cls_id, dets in by_cls.items():
        # Sort: full-frame first (priority), then by confidence descending
        dets.sort(key=lambda d: (d.source == "full", d.conf), reverse=True)
        keep: list[RawDetection] = []
        for d in dets:
            suppress = False
            for k in keep:
                if _iou(d.xyxy, k.xyxy) > iou_threshold:
                    suppress = True
                    break
            if not suppress:
                keep.append(d)
        merged.extend(keep)

    return merged
