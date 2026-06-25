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
    --main-every 1 \
    --result-ttl 2.0 \
    --persist-ttl 4.0 \
    --tile-budget 1 \
    --tile-cache-ttl 0.65 \
    --tile-scan-order center \
    --tile-priority-every 4 \
    --det-conf 0.30 \
    --tile-conf 0.20 \
    --cls-conf 0.85 \
    --audio-cls-gate 0.7 \
    --audio-stability 1 \
    --audio-debounce 0.10
