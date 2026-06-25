# Documentation — HearSight / SignBoard Detection

This folder contains the complete embedded firmware & interfacing documentation for the
**HearSight Assistive Signboard Detection System** running on **Raspberry Pi 5**.

---

## 📄 Documents

| File | Description |
|------|-------------|
| [`Embedded_Firmware_Code_Documentation.docx`](Embedded_Firmware_Code_Documentation.docx) | **Primary submission document.** Contains complete source code of both firmware files with detailed line-by-line comments, logic explanations, libraries table, pin configuration table, and runtime arguments reference. |
| [`HearSight_Laptop_Code_Documentation.docx`](HearSight_Laptop_Code_Documentation.docx) | Laptop code. |
| [`dataset.md`](dataset.md) | Dataset collection, sources, curation pipeline. |
| [`training.md`](training.md) | Two-stage training strategy, Kaggle workflow. |
| [`testing.md`](testing.md) | Testing pipeline, CLI reference, performance. |

---

## 🗂️ Documented Source Files

| Source File | Location | Description |
|-------------|----------|-------------|
| `rpi_webcam_test_two_stage_ncnn.py` | `testing/` | Main embedded Python firmware — camera capture, YOLO NCNN inference, tracking, audio guidance, GPIO, display |
| `start_hearsight.sh` | `testing/` | Bash startup script — virtual environment activation, Bluetooth audio setup, Python script launch |

> **Note:** The files are located in `testing/` (not `scripts/`). The startup script auto-resolves its path relative to its own location, so it works correctly regardless of where it is called from.

---

## 🔧 System Summary

| Property | Value |
|----------|-------|
| **Platform** | Raspberry Pi 5 |
| **Camera Interface** | 22-pin MIPI CSI-2 connector (NOT GPIO) |
| **Power Supply** | 22.5W USB-C power bank |
| **Languages** | Python 3.13, Bash |
| **Inference Backend** | NCNN (CPU-only, Ultralytics YOLOv8) |
| **Audio Output** | ffplay (via subprocess), Bluetooth or wired earphones |
| **Bluetooth Device** | Soundcore R50i VI (MAC: 18:9C:2C:4E:46:C8) |
| **GPIO (optional)** | Runtime-configurable via `--ir-gpio <pin>` (gpiozero.LED) |
| **Repository** | https://github.com/Ashok-19/SignBoard-Detection |

---

## 📦 Library Versions (Detected on Raspberry Pi 5)

| Library | Version |
|---------|---------|
| Python | 3.13.5 |
| OpenCV (cv2) | 4.10.0 |
| NumPy | 2.2.4 |
| Ultralytics YOLO | 8.4.59 |
| Picamera2 | version not exposed by package |
| gpiozero | version not available from runtime |
| FFmpeg / ffplay | 7.1.3-0+deb13u1+rpt1 |
| bluetoothctl | 5.82 |

---

## 🚀 Quick Start

```bash
# 1. Make the startup script executable (first time only)
chmod +x testing/start_hearsight.sh

# 2. Run the system
./testing/start_hearsight.sh
```

Or manually:

```bash
source .venv-rpi/bin/activate
python testing/rpi_webcam_test_two_stage_ncnn.py \
    --preset accuracy \
    --display window \
    --camera-rotation 270 \
    --sharpness 1.4 --contrast 1.08 \
    --threads 3 --main-every 6 \
    --result-ttl 2.0 --persist-ttl 4.0 \
    --tile-budget 1 --tile-cache-ttl 2.5 --tile-cache-sweeps 1.5 \
    --tile-scan-order center --tile-priority-every 0 \
    --det-conf 0.35 --tile-conf 0.28 --cls-conf 0.90 \
    --classify-every 1 --max-proposals 6 --max-classify-per-cycle 3 \
    --min-box-frac 0.018 --track-ttl 45 --track-ttl-sec 3.5 --track-smooth-alpha 0.45 \
    --audio-cls-gate 0.92 --audio-det-gate 0.38 --audio-strong-cls-gate 0.97 \
    --audio-strong-det-gate 0.55 --audio-stability 2 --audio-confirm-gap 1.25 --audio-debounce 0.35
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause / unpause |
| `r` | Toggle rejected proposals |
| `f` | Toggle fullscreen |
| `l` | Switch audio language (English ↔ Tamil) |

---

## 📋 Submission Fields

**Programming Languages:**
Python, Shell Script, OpenCV

**Code Documentation Document:**
`documentation/Embedded_Firmware_Code_Documentation.docx`

**Firmware Repository Link:**
`SignBoard-Detection-Firmware.zip` (contains raw source files + documentation)

```bash
# Create the ZIP (run from parent directory of repo)
zip -r SignBoard-Detection-Firmware.zip SignBoard-Detection-main/ \
    --exclude 'SignBoard-Detection-main/.venv-rpi/*' \
    --exclude 'SignBoard-Detection-main/.git/*'
```
