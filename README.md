# HearSight — Traffic Signboard Detection & Classification

A two-stage computer vision system that detects and classifies Indian traffic signboards in real-time using a webcam. Built for the [HearSight](https://github.com/Ashok-19) accessibility project.

## How It Works

```
Camera → Full-Frame Detector → Tracker → Crop Classifier → Audio Alert (TODO)
              │                                    │
              └── Tiled Inference ─────────────────┘
                  (for distant signs)
```

**Stage 1 — Generic Detector:** A YOLO26n model finds all signboards in the frame (3 classes: `road_sign`, `facility_sign`, and `medical_sign`). Uses SAHI-style tiled inference to catch small, distant signs that would otherwise be missed.

**Stage 2 — Crop Classifier:** Each detected sign is cropped, padded, and classified into one of 28 active classification classes (including a background/negative `not_target` class). Results are cached per tracked sign to minimize redundant computation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with webcam (Logitech Brio)
python testing/webcam_test_two_stage.py \
  --camera brio \
  --zoom-mode hybrid --tile-size 240

# Run with laptop webcam
python testing/webcam_test_two_stage.py --camera laptop
```

### Raspberry Pi 5 + Waveshare IR-Cut Camera

The Pi runtime uses Picamera2, the `ov5647_noir.json` tuning file, and optimized
NCNN exports. Tiling runs on every processed frame.

```bash
# Pi environment and optimized NCNN exports
python3 -m venv --system-site-packages .venv-rpi
.venv-rpi/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv-rpi/bin/python -m pip install ultralytics ncnn pnnx
.venv-rpi/bin/python scripts/export_ncnn.py

# Realtime profile: one full-image tile per frame at 320x320 with a 256 NCNN graph
.venv-rpi/bin/python testing/rpi_webcam_test_two_stage_ncnn.py

# Balanced profile: one full-image tile per frame with a 320 NCNN graph
.venv-rpi/bin/python testing/rpi_webcam_test_two_stage_ncnn.py --preset balanced

# Higher-recall distant-sign mode: six overlapping tiles over 640x480 every frame
.venv-rpi/bin/python testing/rpi_webcam_test_two_stage_ncnn.py --preset full-coverage
```

Use `--preset accuracy` for the FP32 `640` reference exports. The realtime
`256` profile should be validated on the held-out dataset before treating its
accuracy as equivalent to the training-resolution reference.

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause |
| `r` | Toggle rejected proposals |
| `f` | Toggle fullscreen |

## Project Structure

```
├── weights/                    # Trained model weights (v3)
│   ├── detector/best.pt        #   Generic detector (YOLO26n, 1 class)
│   └── classifier/best.pt     #   Crop classifier (YOLO26n, 29 classes)
├── testing/                    # Real-time inference pipeline
│   ├── webcam_test_two_stage.py
│   └── tile_inference.py
├── training/                   # Kaggle training notebooks
│   ├── kaggle-hearsight-ts-training.ipynb
│   └── kaggle_dataset_builder.ipynb
├── training_logs/              # Training metrics, curves, confusion matrices
│   ├── detector/               #   Detector results.csv, PR curves, etc.
│   └── classifier/             #   Classifier results.csv, confusion matrix, etc.
├── scripts/                    # Dataset curation & local training
│   ├── curate_signboard_dataset.py
│   ├── fetch_sources.py
│   ├── train_local.py
│   └── test_local.py
├── configs/
│   └── sources.yaml            # Dataset source registry
└── documentation/              # Detailed documentation
    ├── dataset.md              #   Dataset sources & curation
    ├── training.md             #   Training strategy & workflow
    └── testing.md              #   Testing pipeline & CLI reference
```

## Dataset

The v3 dataset is available for download:

- **Kaggle:** [hearsight-ts-dataset-v3](https://www.kaggle.com/datasets/ashok205/hearsight-ts-dataset-v3)
- **Google Drive:** [hearsight-ts-dataset-v3.zip](https://drive.google.com/file/d/19oFK3CY9Yzd3qLxyoRPjkrLxBSxtyVhs/view?usp=sharing)

See [documentation/dataset.md](documentation/dataset.md) for sources, curation pipeline, and dataset structure.

## Documentation

| Document | Description |
|----------|-------------|
| [documentation/dataset.md](documentation/dataset.md) | Dataset collection, sources, curation pipeline |
| [documentation/training.md](documentation/training.md) | Two-stage training strategy, Kaggle workflow |
| [documentation/testing.md](documentation/testing.md) | Testing pipeline, CLI reference, performance |

## Performance

### Model Scores

| Model | Metric | Score |
|-------|--------|------:|
| Detector | mAP@50 | **85.96%** |
| Detector | mAP@50-95 | **71.27%** |
| Detector | Precision / Recall | 88.5% / 78.3% |
| Classifier | Top-1 Accuracy | **95.50%** |
| Classifier | Top-5 Accuracy | **99.56%** |

See [documentation/training.md](documentation/training.md) for full training progression and logs.

### Inference

| Metric | Value |
|--------|-------|
| FPS (RTX 3050, tiling on) | 30+ |
| Detector input | 640×640 |
| Tile size | 240px, 30% overlap |
| Classifier input | 640x640 |

## Roadmap

- [x] Multi-class detector (road, facility, medical signs) for generic signboard localization
- [x] Two-stage pipeline (detector + classifier) with 28 active classification classes (including negative background class)
- [x] SAHI-style tiled inference for small/distant sign detection
- [x] Async tile processing with cooperative GPU scheduling
- [x] Real-time tracker integration (ByteTrack/BoT-SORT)
- [x] Audio feedback system for sign type announcements
- [x] Port to Raspberry Pi 5 for on-device inference
- [x] NCNN/ONNX/OpenVINO model export for edge optimization
- [x] Priority-based alerting (high-priority signs first)
- [x] Webcam video batch processing mode
