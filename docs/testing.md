# Testing Pipeline

## Overview

The testing pipeline runs a **two-stage** (detector → tracker → classifier) inference system with **SAHI-style tiled inference** for detecting small and distant signs. It currently runs on a laptop with a USB webcam.

**Script:** `testing/webcam_test_two_stage.py`  
**Helpers:** `testing/tile_inference.py`

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Main Thread                         │
│                                                         │
│  Camera Frame ──→ Full-Frame Detector ──→ Tracker       │
│       │                    │                  │          │
│       │           ┌───────────────────┐       │          │
│       │           │  Merge (NMS)      │◄──────┘          │
│       │           │  full + tile dets  │                  │
│       │           └───────┬───────────┘                  │
│       │                   │                              │
│       │           Crop Classifier (cached per track)     │
│       │                   │                              │
│       │              Draw + Display                      │
│       │                                                  │
│       ▼                                                  │
│  ┌────────────────────┐                                  │
│  │   Tile Worker       │  (background thread)            │
│  │   - Own YOLO model  │                                 │
│  │   - GPU lock coord  │                                 │
│  │   - Mini-batch ×3   │                                 │
│  └────────────────────┘                                  │
└─────────────────────────────────────────────────────────┘
```

## Why Tiled Inference?

YOLO models trained at 1024px struggle to detect signs that are small in the frame (< 30px). At 50+ meters distance, a signboard might only be 15-20 pixels wide in a 1280×720 frame.

**Tiled inference (SAHI-style)** solves this by:
1. Dividing the frame into overlapping 240×240 tiles
2. Running the detector on each tile at full 1024px resolution
3. The sign that was 15px in the full frame becomes ~60px in the tile — detectable!
4. Tile detections are remapped to full-frame coordinates and merged via NMS

### Async Tile Worker

Tiles run on a **background thread** with a separate detector model instance. This uses **cooperative GPU scheduling** (shared lock) so tile inference never causes display stutter:

- Main thread holds GPU lock during detect + classify (~70ms)
- Tile worker processes 3 tiles per lock acquisition
- Main thread never waits more than ~60ms for tile worker to release

## Performance

| Configuration | FPS | Device |
|---------------|----:|--------|
| Full-frame only (no tiling) | 30+ | RTX 3050 Laptop |
| Hybrid tiling (240px tiles) | 30+ | RTX 3050 Laptop |

## Quick Start

```bash
# With Logitech Brio webcam + tiling
python testing/webcam_test_two_stage.py \
  --camera brio \
  --zoom-mode hybrid --tile-size 240

# With laptop webcam, no tiling
python testing/webcam_test_two_stage.py \
  --camera laptop

# Custom weights
python testing/webcam_test_two_stage.py \
  --detector weights/detector/best.pt \
  --classifier weights/classifier/best.pt \
  --camera brio --zoom-mode hybrid --tile-size 240
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause / resume |
| `r` | Show / hide rejected proposals |
| `f` | Toggle fullscreen |

## CLI Reference

### Camera & Display
| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `auto` | Camera shortcut: `laptop`, `brio`, `auto`, or any name substring |
| `--source` | `0` | Camera index or video file path (used when `--camera=auto`) |
| `--display` | `auto` | Display mode: `auto`, `window`, `tk`, `file`, `none` |
| `--width` | `1280` | Camera resolution width |
| `--height` | `720` | Camera resolution height |

### Detection & Classification
| Flag | Default | Description |
|------|---------|-------------|
| `--detector` | auto | Path to detector `best.pt` |
| `--classifier` | auto | Path to classifier `best.pt` |
| `--det-imgsz` | `1024` | Detector inference resolution |
| `--cls-imgsz` | `640` | Classifier inference resolution |
| `--det-conf` | `0.25` | Detector confidence threshold |
| `--cls-conf` | `0.80` | Classifier confidence threshold |
| `--classify-every` | `3` | Re-classify each track every N frames |
| `--half` | auto | FP16 inference (auto-enabled on CUDA) |

### Tiling
| Flag | Default | Description |
|------|---------|-------------|
| `--zoom-mode` | `off` | Tile mode: `off`, `hybrid`, `tiles`, `center` |
| `--tile-size` | `640` | Tile crop size in pixels |
| `--tile-scales` | none | Multi-scale pyramid, e.g. `--tile-scales 640 384 240` |
| `--tile-overlap` | `0.30` | Overlap fraction between tiles |
| `--tile-conf` | `0.08` | Detector confidence for tiles (lower = more recall) |
| `--tile-roi` | `auto` | ROI: `auto`, `upper`, `horizon`, `center`, `full` |
| `--no-tile-clahe` | off | Disable CLAHE contrast enhancement on tiles |

### Tracking
| Flag | Default | Description |
|------|---------|-------------|
| `--tracker` | `bytetrack.yaml` | Tracker: `bytetrack.yaml` or `botsort.yaml` |
| `--no-track` | off | Disable tracking, use raw detections |

## TODO

- [ ] **Raspberry Pi 5 testing pipeline** — port inference to RPi 5 with camera module, optimize for ARM/NPU
- [ ] **Audio feedback system** — text-to-speech for detected sign types, priority-based alerting
- [ ] **ONNX/TensorRT export** — model optimization for edge deployment
- [ ] **Video file input** — batch processing of dashcam footage for evaluation
