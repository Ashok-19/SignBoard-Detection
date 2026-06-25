#!/usr/bin/env python3
"""Live webcam testing for the HearSight two-stage detector + classifier pipeline.

Pipeline:
  1. Run the generic detector on each frame.
  2. Track detector boxes with ByteTrack/BoT-SORT by default.
  3. Crop each tracked detector box with context padding.
  4. Run the crop classifier on a per-track cadence and cache labels between frames.
  5. Draw classifier-confirmed targets by default.

Controls: [q] quit  [p] pause  [r] show/hide rejected detector proposals  [f] fullscreen
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from tile_inference import (
    RawDetection,
    TileRegion,
    iter_multiscale_tiles,
    iter_tiles,
    merge_detections_nms,
    preprocess_tile,
    run_detector_on_tiles,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "hearsight-two-stage/results-v1/runs/hearsight_two_stage"
DEFAULT_DETECTOR = DEFAULT_RESULTS / "generic_detector/weights/best.pt"
DEFAULT_CLASSIFIER = DEFAULT_RESULTS / "crop_classifier/weights/best.pt"
DEFAULT_OUTPUT = REPO_ROOT / "hearsight-two-stage/live_two_stage_preview.mp4"


@dataclass
class TwoStageDetection:
    track_id: str | None
    box: tuple[int, int, int, int]
    detector_name: str
    detector_conf: float
    classifier_name: str
    classifier_conf: float
    accepted: bool


@dataclass
class RuntimeState:
    frame_index: int = 0
    cls_cache: dict[str, tuple[int, str, float]] | None = None
    last_tile_dets: list[RawDetection] | None = None
    # Spatial tracking for tile detections: maps (cls_id, grid_x, grid_y) -> stable track_id
    tile_spatial_tracks: dict[tuple[int, int, int], str] | None = None
    _tile_id_counter: int = 0

    def __post_init__(self) -> None:
        if self.cls_cache is None:
            self.cls_cache = {}
        if self.last_tile_dets is None:
            self.last_tile_dets = []
        if self.tile_spatial_tracks is None:
            self.tile_spatial_tracks = {}

    def get_stable_tile_id(self, cls_id: int, cx: float, cy: float, grid_cell: int = 50) -> str:
        """Get a stable track ID for a tile detection based on spatial position.

        Quantizes the detection center to a grid and reuses the same ID for
        detections in the same cell across frames. This enables classifier
        caching and consistent visual labels for tile-only detections.

        Args:
            grid_cell: size of spatial grid cells in pixels. Detections within
                       the same cell get the same track ID.
        """
        gx = int(cx) // grid_cell
        gy = int(cy) // grid_cell
        key = (cls_id, gx, gy)
        if key not in self.tile_spatial_tracks:
            self._tile_id_counter += 1
            self.tile_spatial_tracks[key] = f"t{self._tile_id_counter}"
        return self.tile_spatial_tracks[key]


class TileWorker:
    """Async background tile inference with cooperative GPU scheduling.

    Runs on a daemon thread with its own detector model instance.
    Uses a shared gpu_lock to prevent GPU contention with the main thread.
    Processes tiles ONE AT A TIME, releasing the lock between each tile
    so the main thread can always jump in within ~20ms (one tile's time).
    This eliminates stutter from GPU contention.
    """

    def __init__(
        self,
        detector_path: str,
        args: argparse.Namespace,
        device: str,
        gpu_lock: threading.Lock,
    ):
        self._args = args
        self._device = device
        self._gpu_lock = gpu_lock
        self._frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._latest_dets: list[RawDetection] = []
        self._result_time: float = 0.0  # monotonic timestamp of last result
        self._stopped = False
        self._tile_fps = 0.0

        # Load a SEPARATE detector instance for the tile thread
        self._detector = YOLO(detector_path, task="detect")
        print(f"[tile-worker] loaded tile detector on {device}")

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def submit_frame(self, frame: np.ndarray) -> None:
        """Submit latest frame for tile processing (non-blocking, overwrites old)."""
        with self._frame_lock:
            self._frame = frame

    def get_detections(self) -> tuple[list[RawDetection], float]:
        """Get latest tile detections and their timestamp (non-blocking)."""
        with self._result_lock:
            return list(self._latest_dets), self._result_time

    @property
    def tile_fps(self) -> float:
        return self._tile_fps

    def _run_loop(self) -> None:
        args = self._args
        tile_overlap = getattr(args, "tile_overlap", 0.25)
        tile_conf = getattr(args, "tile_conf", 0.08)
        tile_max_det = getattr(args, "tile_max_det", 40)
        tile_min_box = getattr(args, "tile_min_box", 6)
        use_clahe = getattr(args, "tile_clahe", True)

        tile_roi = getattr(args, "tile_roi", "auto")
        zoom_mode = getattr(args, "zoom_mode", "off")
        if tile_roi == "auto":
            roi_mode = "center" if zoom_mode == "center" else "upper"
        else:
            roi_mode = tile_roi

        while not self._stopped:
            with self._frame_lock:
                frame = self._frame
                self._frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            t0 = time.monotonic()
            height, width = frame.shape[:2]

            # Generate tiles
            tile_scales = getattr(args, "tile_scales", None)
            if tile_scales:
                tiles = iter_multiscale_tiles(height, width, tile_scales, tile_overlap, roi_mode)
            else:
                tile_size = getattr(args, "tile_size", 640)
                tiles = iter_tiles(height, width, tile_size, tile_overlap, roi_mode)

            if not tiles:
                continue

            # --- Mini-batch tile inference with GPU lock yielding ---
            # Process tiles in small batches of 3. Between each mini-batch,
            # the lock is released so the main thread can jump in.
            # Mini-batching is 3x faster than single-tile while keeping
            # main thread wait under ~60ms.
            MINI_BATCH = 3
            all_dets: list[RawDetection] = []

            for i in range(0, len(tiles), MINI_BATCH):
                if self._stopped:
                    break

                # Prepare mini-batch crops (CPU work, no lock needed)
                batch_tiles: list[TileRegion] = []
                batch_crops: list[np.ndarray] = []
                for tile in tiles[i:i + MINI_BATCH]:
                    crop = frame[tile.y1:tile.y2, tile.x1:tile.x2]
                    if crop.shape[0] < 32 or crop.shape[1] < 32:
                        continue
                    processed = preprocess_tile(crop, use_clahe=use_clahe)
                    batch_tiles.append(tile)
                    batch_crops.append(processed)

                if not batch_crops:
                    continue

                # Acquire GPU lock for mini-batch inference, then release
                with self._gpu_lock:
                    batch_results = self._detector.predict(
                        batch_crops,
                        imgsz=args.det_imgsz,
                        conf=tile_conf,
                        iou=args.det_iou,
                        device=self._device,
                        half=args.half,
                        max_det=tile_max_det,
                        verbose=False,
                    )
                # Lock released — main thread can use GPU now

                for tile, res in zip(batch_tiles, batch_results):
                    if res.boxes is None or len(res.boxes) == 0:
                        continue

                    tile_w = tile.x2 - tile.x1
                    tile_h = tile.y2 - tile.y1

                    for box in res.boxes:
                        bx1, by1, bx2, by2 = box.xyxy[0].cpu().tolist()
                        bw, bh = bx2 - bx1, by2 - by1
                        if bw < tile_min_box or bh < tile_min_box:
                            continue
                        if bw > tile_w * 0.95 and bh > tile_h * 0.95:
                            continue
                        cls_id = int(box.cls[0])
                        all_dets.append(RawDetection(
                            xyxy=(bx1 + tile.x1, by1 + tile.y1, bx2 + tile.x1, by2 + tile.y1),
                            conf=float(box.conf[0]),
                            cls_id=cls_id,
                            cls_name=res.names.get(cls_id, f"det_cls_{cls_id}"),
                            source="tile",
                        ))

            with self._result_lock:
                self._latest_dets = all_dets
                self._result_time = time.monotonic()

            elapsed = time.monotonic() - t0
            self._tile_fps = 1.0 / max(elapsed, 0.001)

    def stop(self) -> None:
        self._stopped = True
        self._thread.join(timeout=3.0)


def detect_device(requested: str) -> str:
    if requested == "0" or requested.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[warn] CUDA not available; using CPU")
            return "cpu"
    return requested


def clip_xyxy(frame: np.ndarray, box) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    return (
        max(0, min(width - 1, int(x1))),
        max(0, min(height - 1, int(y1))),
        max(0, min(width, int(x2))),
        max(0, min(height, int(y2))),
    )


def crop_xyxy(frame: np.ndarray, box, pad: float, min_side: int) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    half_w = max(bw * (0.5 + pad), min_side * 0.5)
    half_h = max(bh * (0.5 + pad), min_side * 0.5)
    ix1 = max(0, int(cx - half_w))
    iy1 = max(0, int(cy - half_h))
    ix2 = min(width, int(cx + half_w))
    iy2 = min(height, int(cy + half_h))

    if ix2 <= ix1 + 2 or iy2 <= iy1 + 2:
        return None
    crop = frame[iy1:iy2, ix1:ix2]
    if min(crop.shape[:2]) < 10:
        return None
    return crop


def pretty_name(name: str) -> str:
    return name.replace("___", " - ").replace("__", "_").replace("_", " ")


def classify_crops(classifier: YOLO, crops: list[np.ndarray], imgsz: int, device: str, half: bool):
    if not crops:
        return []
    return classifier.predict(crops, imgsz=imgsz, device=device, half=half, verbose=False)


def apply_clahe_color(frame: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_ch = clahe.apply(l_ch)
    enhanced = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def simulate_ir_camera(frame: np.ndarray, clip_limit: float = 3.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    gray = clahe.apply(gray)
    
    # Slight sensor noise
    noise = np.random.normal(0, 4, gray.shape).astype(np.int16)
    gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def simulate_ir_color(frame: np.ndarray, clip_limit: float = 3.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    gray = clahe.apply(gray)
    
    # Slight sensor noise
    noise = np.random.normal(0, 4, gray.shape).astype(np.int16)
    gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    try:
        color_mapped = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    except AttributeError:
        try:
            color_mapped = cv2.applyColorMap(gray, 20)  # 20 is TURBO
        except Exception:
            color_mapped = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return color_mapped


def preprocess_vision_mode(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "clahe":
        return apply_clahe_color(frame)
    elif mode == "ir":
        return simulate_ir_camera(frame)
    elif mode == "ir-color":
        return simulate_ir_color(frame)
    return frame


def should_refresh(cache_frame: int | None, frame_index: int, every: int) -> bool:
    return cache_frame is None or every <= 1 or (frame_index - cache_frame) >= every


def _extract_full_frame_raw(det_result) -> list[RawDetection]:
    """Convert Ultralytics detection result to list of RawDetection."""
    raw: list[RawDetection] = []
    if det_result.boxes is None:
        return raw
    for box in det_result.boxes:
        cls_id = int(box.cls[0])
        raw.append(RawDetection(
            xyxy=tuple(box.xyxy[0].cpu().tolist()),
            conf=float(box.conf[0]),
            cls_id=cls_id,
            cls_name=det_result.names.get(cls_id, f"det_cls_{cls_id}"),
            source="full",
        ))
    return raw


def run_two_stage(
    frame: np.ndarray,
    detector: YOLO,
    classifier: YOLO,
    state: RuntimeState,
    args: argparse.Namespace,
    device: str,
    tile_worker: TileWorker | None = None,
) -> list[TwoStageDetection]:
    state.frame_index += 1
    infer_kwargs = dict(
        imgsz=args.det_imgsz,
        conf=args.det_conf,
        iou=args.det_iou,
        device=device,
        half=args.half,
        max_det=args.max_det,
        verbose=False,
    )

    # --- Stage 1: full-frame detector/tracker (main thread only) ---
    if args.no_track:
        det_results = detector.predict(frame, **infer_kwargs)
    else:
        det_results = detector.track(frame, persist=True, tracker=args.tracker, **infer_kwargs)

    det_result = det_results[0] if det_results else None
    has_ids = (
        det_result is not None
        and hasattr(det_result.boxes, "id")
        and det_result.boxes.id is not None
    )

    # --- Get tile detections from async worker (non-blocking) ---
    zoom_mode = getattr(args, "zoom_mode", "off")
    if tile_worker is not None and zoom_mode != "off":
        tile_dets, tile_time = tile_worker.get_detections()
        tile_age = time.monotonic() - tile_time if tile_time > 0 else 999.0
        # Discard stale tile detections (> 2 seconds old).
        # Stale tiles from a previous camera position can suppress
        # current full-frame detections during NMS merge.
        if tile_age < 2.0:
            state.last_tile_dets = tile_dets
        else:
            state.last_tile_dets = []

    # --- Merge full-frame + tile detections ---
    full_raw = _extract_full_frame_raw(det_result) if det_result else []

    if zoom_mode != "off" and state.last_tile_dets:
        merged_raw = merge_detections_nms(full_raw, state.last_tile_dets, iou_threshold=0.40)
    else:
        merged_raw = full_raw

    if not merged_raw:
        return []

    # --- Build metadata + classify ---
    # For tracked boxes from full-frame, use their track IDs.
    # For tile-only boxes, assign synthetic IDs.
    full_id_map: dict[int, int] = {}  # box index in det_result -> track_id
    if has_ids and det_result is not None:
        for idx in range(len(det_result.boxes)):
            full_id_map[idx] = int(det_result.boxes.id[idx])

    pending_cls: list[tuple[int, np.ndarray]] = []
    metadata: list[dict] = []

    for raw_idx, raw in enumerate(merged_raw):
        xyxy = raw.xyxy
        crop = crop_xyxy(frame, xyxy, args.crop_pad, args.min_crop_side)
        if crop is None:
            continue

        det_name = raw.cls_name

        # Try to match to a tracked full-frame box for track ID
        track_id: str | None = None
        if raw.source == "full" and has_ids and det_result is not None:
            best_iou = 0.0
            best_tid = None
            for fidx, tid in full_id_map.items():
                fbox = det_result.boxes.xyxy[fidx].cpu().tolist()
                from tile_inference import _iou
                cur_iou = _iou(raw.xyxy, tuple(fbox))
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_tid = tid
            if best_tid is not None and best_iou > 0.5:
                track_id = str(best_tid)

        if track_id is None and raw.source == "tile":
            cx = (raw.xyxy[0] + raw.xyxy[2]) * 0.5
            cy = (raw.xyxy[1] + raw.xyxy[3]) * 0.5
            track_id = state.get_stable_tile_id(raw.cls_id, cx, cy)

        cache_key = (
            f"{det_name}:{track_id}"
            if track_id is not None
            else f"frame{state.frame_index}:{raw_idx}"
        )

        cls_cached = state.cls_cache.get(cache_key)
        if cls_cached is None or should_refresh(cls_cached[0], state.frame_index, args.classify_every):
            pending_cls.append((len(metadata), crop))

        metadata.append(
            {
                "track_id": track_id,
                "cache_key": cache_key,
                "box": clip_xyxy(frame, xyxy),
                "det_name": det_name,
                "det_conf": raw.conf,
            }
        )

    if pending_cls:
        cls_results = classify_crops(classifier, [crop for _, crop in pending_cls], args.cls_imgsz, device, args.half)
        for (meta_idx, _), cls_result in zip(pending_cls, cls_results):
            probs = cls_result.probs
            if probs is None:
                continue
            cls_id = int(probs.top1)
            cls_conf = float(probs.top1conf)
            cls_name = cls_result.names.get(cls_id, f"cls_{cls_id}")
            state.cls_cache[metadata[meta_idx]["cache_key"]] = (state.frame_index, cls_name, cls_conf)

    detections: list[TwoStageDetection] = []
    for item in metadata:
        cls_cached = state.cls_cache.get(item["cache_key"])
        if cls_cached is None:
            continue
        _, cls_name, cls_conf = cls_cached
        accepted = cls_name != args.reject_class and cls_conf >= args.cls_conf
        detections.append(
            TwoStageDetection(
                track_id=item["track_id"],
                box=item["box"],
                detector_name=item["det_name"],
                detector_conf=item["det_conf"],
                classifier_name=pretty_name(cls_name),
                classifier_conf=cls_conf,
                accepted=accepted,
            )
        )

    return detections


def draw_detections(frame: np.ndarray, detections: list[TwoStageDetection], show_rejected: bool, line_width: int):
    out = frame.copy()
    for det in detections:
        if not det.accepted and not show_rejected:
            continue
        x1, y1, x2, y2 = det.box
        color = (0, 220, 0) if det.accepted else (90, 90, 90)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_width)

        tid = f"#{det.track_id} " if det.track_id is not None else ""
        if det.accepted:
            label = f"{tid}{det.classifier_name} cls {det.classifier_conf:.2f} det {det.detector_conf:.2f}"
        else:
            label = f"{tid}rejected: {det.classifier_name} {det.classifier_conf:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = y1 - 6 if y1 - th - 8 > 0 else y1 + th + 8
        cv2.rectangle(out, (x1, ly - th - 5), (x1 + tw + 6, ly + 3), color, -1)
        cv2.putText(out, label, (x1 + 3, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def open_capture(source: str, width: int, height: int) -> cv2.VideoCapture:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


class FrameGrabber:
    """Threaded camera capture — keeps the latest frame always ready.

    Decouples webcam I/O from the inference loop so the GPU never waits
    for cap.read(). A daemon thread continuously reads and overwrites
    the latest frame. The main thread grabs it instantly via .read().
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._frame: np.ndarray | None = None
        self._ok = False
        self._lock = threading.Lock()
        self._stopped = False
        # Read one frame synchronously to initialize
        self._ok, self._frame = self._cap.read()
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _grab_loop(self) -> None:
        while not self._stopped:
            ok, frame = self._cap.read()
            with self._lock:
                self._ok = ok
                self._frame = frame
            if not ok:
                break

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            return self._ok, self._frame

    def release(self) -> None:
        self._stopped = True
        self._thread.join(timeout=2.0)
        self._cap.release()


class PreviewSink:
    WINDOW_NAME = "HearSight Two-Stage Live Test"

    def __init__(self, mode: str, output: Path, fps: float, size: tuple[int, int]):
        self.mode = mode
        self.output = output
        self.gui_disabled = mode in {"file", "none"}
        self.writer: cv2.VideoWriter | None = None
        self.tk_root = None
        self.tk_label = None
        self.tk_photo = None
        self.pending_key = 255
        self._cv2_failed = False
        self._cv2_window_created = False
        self._fullscreen = False
        self._window_size = size  # (w, h)

        if mode in {"auto", "file"}:
            output.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(output), fourcc, max(1.0, fps), size)
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                print(f"[warn] could not open video writer: {output}")
            elif mode == "file":
                print(f"[info] writing preview video to: {output}")

    def _on_tk_key(self, event) -> None:
        char = getattr(event, "char", "") or ""
        if char:
            self.pending_key = ord(char.lower())

    def _init_tk(self) -> bool:
        if self.tk_root is not None:
            return True
        try:
            import tkinter as tk

            self.tk_root = tk.Tk()
            self.tk_root.title(self.WINDOW_NAME)
            self.tk_root.protocol("WM_DELETE_WINDOW", lambda: setattr(self, "pending_key", ord("q")))
            self.tk_root.bind("<Key>", self._on_tk_key)
            self.tk_label = tk.Label(self.tk_root)
            self.tk_label.pack()
            print("[info] using Tk live preview because OpenCV imshow is unavailable")
            return True
        except Exception as e:
            self.tk_root = None
            self.tk_label = None
            print(f"[warn] Tk live preview unavailable: {e}")
            return False

    def _show_tk(self, frame: np.ndarray) -> int:
        if not self._init_tk():
            self.gui_disabled = True
            return 255
        try:
            from PIL import Image, ImageTk

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            self.tk_photo = ImageTk.PhotoImage(image=image)
            self.tk_label.configure(image=self.tk_photo)
            self.tk_root.update_idletasks()
            self.tk_root.update()
            key = self.pending_key
            self.pending_key = 255
            return key
        except Exception as e:
            self.gui_disabled = True
            print(f"[warn] Tk live preview failed: {e}")
            return 255

    def _create_cv2_window(self) -> None:
        """Create a resizable OpenCV window (once)."""
        if self._cv2_window_created:
            return
        # WINDOW_NORMAL = resizable by user, WINDOW_GUI_EXPANDED = proper GUI
        try:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        except Exception:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        # Start with window at frame size (user can resize/maximize)
        cv2.resizeWindow(self.WINDOW_NAME, self._window_size[0], self._window_size[1])
        self._cv2_window_created = True

    def toggle_fullscreen(self) -> None:
        """Toggle between fullscreen and normal windowed mode."""
        if self._cv2_failed or self.gui_disabled:
            return
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    def show(self, frame: np.ndarray) -> int:
        if self.writer is not None:
            self.writer.write(frame)
        if self.gui_disabled:
            return 255
        if self.mode == "tk" or self._cv2_failed:
            return self._show_tk(frame)
        try:
            self._create_cv2_window()
            cv2.imshow(self.WINDOW_NAME, frame)
            return cv2.waitKey(1) & 0xFF
        except cv2.error as e:
            self._cv2_failed = True
            print("[warn] OpenCV GUI display is unavailable; switching to Tk live preview.")
            key = self._show_tk(frame)
            if self.gui_disabled and self.writer is not None:
                print(f"[info] preview video is being written to: {self.output}")
            elif self.gui_disabled:
                print(f"[warn] no live preview available. OpenCV error: {e}")
            return key

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            print(f"[done] preview video saved to: {self.output}")
        if self.tk_root is not None:
            try:
                self.tk_root.destroy()
            except Exception:
                pass
        if not self.gui_disabled and not self._cv2_failed:
            cv2.destroyAllWindows()


def _detect_cameras() -> dict[str, str]:
    """Auto-detect available cameras via v4l2-ctl."""
    import subprocess
    cams: dict[str, str] = {}
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--list-devices"], stderr=subprocess.DEVNULL, text=True,
        )
        current_name = ""
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("/dev/video"):
                # Only take the first /dev/video per device (capture node)
                if current_name and current_name not in cams:
                    cams[current_name] = line
            else:
                current_name = line.split("(")[0].strip().lower()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return cams


def _resolve_camera(camera: str, source: str) -> str:
    """Resolve --camera name to a /dev/video path or index."""
    if camera == "auto":
        return source  # use --source as-is

    known = _detect_cameras()

    if camera == "laptop":
        # Match common built-in webcam names
        for name, dev in known.items():
            if any(kw in name for kw in ["uvc", "integrated", "built-in", "hd webcam"]):
                print(f"[camera] laptop → {dev} ({name})")
                return dev
        # Fallback: first device (usually built-in)
        if known:
            first_name, first_dev = next(iter(known.items()))
            print(f"[camera] laptop (fallback) → {first_dev} ({first_name})")
            return first_dev
        return "0"

    if camera == "brio":
        for name, dev in known.items():
            if "brio" in name:
                print(f"[camera] brio → {dev} ({name})")
                return dev
        print("[warn] Brio camera not found. Available:")
        for name, dev in known.items():
            print(f"  {dev}: {name}")
        return source

    # Treat as a device name substring search
    for name, dev in known.items():
        if camera.lower() in name:
            print(f"[camera] '{camera}' → {dev} ({name})")
            return dev
    print(f"[warn] camera '{camera}' not found. Available:")
    for name, dev in known.items():
        print(f"  {dev}: {name}")
    return source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live two-stage HearSight webcam tester.")
    parser.add_argument("--detector", default=str(DEFAULT_DETECTOR), help="Path to generic detector best.pt")
    parser.add_argument("--classifier", default=str(DEFAULT_CLASSIFIER), help="Path to crop classifier best.pt")
    parser.add_argument("--camera", default="auto",
                        help="Camera shortcut: 'laptop', 'brio', 'auto' (use --source), "
                             "or any substring to match a connected camera name")
    parser.add_argument("--source", default="0", help="Camera index or video file path (used when --camera=auto)")
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0, cuda:0, or cpu")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--cls-imgsz", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.55)
    parser.add_argument("--cls-conf", type=float, default=0.80)
    parser.add_argument("--reject-class", default="not_target")
    parser.add_argument("--classify-every", type=int, default=3, help="Run crop classifier once every N frames per track")
    parser.add_argument("--no-track", action="store_true", help="Disable detector tracking and use raw detection")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker YAML: bytetrack.yaml or botsort.yaml")
    parser.add_argument("--crop-pad", type=float, default=0.18)
    parser.add_argument("--min-crop-side", type=int, default=48)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA (auto-enabled with --auto-half)")
    parser.add_argument("--auto-half", action="store_true", default=True,
                        help="Auto-enable FP16 when CUDA is available (default: on)")
    parser.add_argument("--no-auto-half", action="store_true", help="Disable auto FP16")
    parser.add_argument("--show-rejected", action="store_true", help="Draw rejected detector proposals")
    parser.add_argument("--display", choices=["auto", "window", "tk", "file", "none"], default="auto")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Preview video path for --display auto/file")
    # --- Tiled / SAHI-style inference flags ---
    parser.add_argument("--zoom-mode", choices=["off", "center", "tiles", "hybrid"], default="off",
                        help="Tile inference mode: off, center, tiles, or hybrid")
    parser.add_argument("--tile-size", type=int, default=240, help="Tile crop size (single-scale fallback)")
    parser.add_argument("--tile-scales", type=int, nargs="+", default=None,
                        help="Multi-scale tile pyramid sizes, e.g. --tile-scales 640 384. "
                             "Smaller tiles = more zoom for distant signs. Overrides --tile-size.")
    parser.add_argument("--tile-overlap", type=float, default=0.30,
                        help="Overlap fraction between tiles (0.30 = 72px overlap on 240px tiles)")
    parser.add_argument("--tile-every", type=int, default=1, help="Run tile inference every N frames")
    parser.add_argument("--tile-conf", type=float, default=0.08, help="Detector confidence for tile inference")
    parser.add_argument("--tile-max-det", type=int, default=40, help="Max detections per tile")
    parser.add_argument("--tile-min-box", type=int, default=6, help="Min box side in pixels (tile-local)")
    parser.add_argument("--tile-roi", choices=["auto", "upper", "horizon", "center", "full"], default="auto",
                        help="ROI for tile placement: auto, upper (75%%), horizon (60%%), center, full")
    parser.add_argument("--no-tile-clahe", action="store_true",
                        help="Disable CLAHE contrast enhancement on tile crops")
    
    # --- Night vision / Low-light simulation flags ---
    parser.add_argument(
        "--simulate-nightvision",
        action="store_true",
        help="Convert webcam frames to grayscale + CLAHE to simulate IR night vision (shortcut for --vision-mode ir)"
    )
    parser.add_argument(
        "--vision-mode",
        choices=["normal", "clahe", "ir", "ir-color"],
        default="normal",
        help="Vision enhancement/simulation mode: normal, clahe (low-light CLAHE), ir (grayscale + CLAHE + noise), ir-color (grayscale + CLAHE + noise + TURBO colormap)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detector_path = Path(args.detector).expanduser().resolve()
    classifier_path = Path(args.classifier).expanduser().resolve()
    if not detector_path.exists():
        raise FileNotFoundError(f"Detector weights not found: {detector_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier weights not found: {classifier_path}")

    # Post-process args
    args.tile_clahe = not args.no_tile_clahe
    if args.simulate_nightvision and args.vision_mode == "normal":
        args.vision_mode = "ir"
    device = detect_device(args.device)

    # Resolve --camera shortcut to actual device
    args.source = _resolve_camera(args.camera, args.source)

    # Auto-enable FP16 on CUDA for ~2x speedup (RTX tensor cores)
    if args.auto_half and not args.no_auto_half and not args.half:
        if device != "cpu" and torch.cuda.is_available():
            args.half = True
            print("[info] auto-enabled FP16 (--half) for CUDA. Use --no-auto-half to disable.")

    detector = YOLO(str(detector_path), task="detect")
    classifier = YOLO(str(classifier_path), task="classify")
    cap = open_capture(args.source, args.width, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 1:
        source_fps = 20.0
    preview = PreviewSink(args.display, Path(args.output).expanduser().resolve(), source_fps, (actual_w, actual_h))
    print(f"Detector:   {detector_path}")
    print(f"Classifier: {classifier_path}")
    print(f"Tracking:   {'off' if args.no_track else args.tracker}")
    scales_info = f"scales={args.tile_scales}" if args.tile_scales else f"tile_size={args.tile_size}"
    print(f"Zoom mode:  {args.zoom_mode} ({scales_info}, overlap={args.tile_overlap}, every={args.tile_every}, roi={args.tile_roi}, clahe={args.tile_clahe})")
    print(f"Source:     {args.source} ({actual_w}x{actual_h})")
    print("Controls:   [q] quit  [p] pause  [r] show/hide rejected  [f] fullscreen")

    # Shared GPU lock — prevents main thread and tile worker from
    # competing for GPU simultaneously (which causes frame stutter)
    gpu_lock = threading.Lock()

    # Start async tile worker if tiling is enabled
    tile_worker: TileWorker | None = None
    if args.zoom_mode != "off":
        tile_worker = TileWorker(str(detector_path), args, device, gpu_lock)
        print("[info] async tile worker started (cooperative GPU scheduling)")

    paused = False
    show_rejected = args.show_rejected
    last_frame = None
    fps_display = 0.0
    frames = 0
    timer = time.time()
    state = RuntimeState()

    # Use threaded frame grabber for async camera I/O
    grabber = FrameGrabber(cap)
    print("[info] threaded camera capture started")

    while True:
        if not paused:
            ok, frame = grabber.read()
            if not ok or frame is None:
                print("[warn] failed to read frame")
                break

            if args.vision_mode != "normal":
                frame = preprocess_vision_mode(frame, args.vision_mode)

            # Submit frame to tile worker (non-blocking)
            if tile_worker is not None:
                tile_worker.submit_frame(frame)

            # Acquire GPU lock so tile worker yields during our detect+classify
            with gpu_lock:
                detections = run_two_stage(frame, detector, classifier, state, args, device, tile_worker)
            shown = sum(1 for d in detections if d.accepted)
            annotated = draw_detections(frame, detections, show_rejected, args.line_width)

            frames += 1
            elapsed = time.time() - timer
            if elapsed >= 1.0:
                fps_display = frames / elapsed
                frames = 0
                timer = time.time()

            cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"accepted: {shown} / proposals: {len(detections)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            zoom_info = ""
            if args.zoom_mode != "off" and tile_worker is not None:
                zoom_info = f" tiles={args.zoom_mode}@{tile_worker.tile_fps:.1f}hz"
            vision_info = f" vision={args.vision_mode}" if args.vision_mode != "normal" else ""
            cv2.putText(
                annotated,
                f"det_conf={args.det_conf:.2f} cls_conf={args.cls_conf:.2f} rejected={'on' if show_rejected else 'off'}{zoom_info}{vision_info}",
                (10, actual_h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            last_frame = annotated

        key = 255
        if last_frame is not None and args.display != "none":
            key = preview.show(last_frame)

        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("r"):
            show_rejected = not show_rejected
            print(f"[info] rejected proposals {'shown' if show_rejected else 'hidden'}")
        if key == ord("f"):
            preview.toggle_fullscreen()

    if tile_worker is not None:
        tile_worker.stop()
    grabber.release()
    preview.close()
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
