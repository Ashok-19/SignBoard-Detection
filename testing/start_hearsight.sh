#!/bin/bash
# Do NOT use 'set -e' — bluetooth commands may fail early at boot and must not abort the script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Small delay to let display, camera, and bluetooth services initialise after login
sleep 5

# Ensure bluetooth is powered on (non-fatal if BT adapter not ready)
bluetoothctl power on || echo "[bluetooth] Failed to power on — continuing without BT"

# Attempt to connect to soundcore R50i VI in the background so it doesn't block startup
(
    echo "Connecting to Soundcore R50i VI (18:9C:2C:4E:46:C8) in the background..."
    for i in {1..10}; do
        # Check if already connected
        if bluetoothctl info 18:9C:2C:4E:46:C8 | grep -q "Connected: yes"; then
            echo "[bluetooth] Soundcore R50i VI is connected!"
            break
        fi
        if bluetoothctl connect 18:9C:2C:4E:46:C8; then
            echo "[bluetooth] Successfully connected to Soundcore R50i VI!"
            break
        fi
        sleep 2
    done
) &

# Activate virtualenv and run hearsight (starts instantly for wired earphones / overall flow)
source .venv-rpi/bin/activate
python "${SCRIPT_DIR}/rpi_webcam_test_two_stage_ncnn.py" \
    --preset accuracy \
    --display window \
    --camera-rotation 270 \
    --sharpness 1.4 \
    --contrast 1.08 \
    --threads 3 \
    --main-every 6 \
    --result-ttl 2.0 \
    --persist-ttl 4.0 \
    --tile-budget 1 \
    --tile-cache-ttl 2.5 \
    --tile-cache-sweeps 1.5 \
    --tile-scan-order center \
    --tile-priority-every 0 \
    --det-conf 0.35 \
    --tile-conf 0.28 \
    --cls-conf 0.90 \
    --classify-every 1 \
    --max-proposals 6 \
    --max-classify-per-cycle 3 \
    --min-box-frac 0.018 \
    --track-ttl 45 \
    --track-ttl-sec 3.5 \
    --track-smooth-alpha 0.45 \
    --audio-cls-gate 0.92 \
    --audio-det-gate 0.38 \
    --audio-strong-cls-gate 0.97 \
    --audio-strong-det-gate 0.55 \
    --audio-stability 2 \
    --audio-confirm-gap 1.25 \
    --audio-debounce 0.35
