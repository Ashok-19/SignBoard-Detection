#!/usr/bin/env python3
"""Train YOLO26 on the merged 8-class signboard dataset — regression-optimized.

Compared to the original training recipe:

Augmentation changes (signboards are rigid, structured objects with text):
  mosaic:       1.0 → 0.5   (less synthetic image stitching)
  mixup:        0.15 → 0.0  (removed — hurts structured object regression)
  copy_paste:   0.3 → 0.0   (removed — same reason)
  erasing:      0.4 → 0.1   (minimal random occlusion only)
  degrees:      15 → 10     (signboards aren't heavily rotated)
  shear:        5 → 2       (less geometric warping)
  perspective:  0.0005 → 0.0 (removed)

Training regimen changes:
  epochs:       80/100 → 150  (more time for regression convergence)
  lr0:          0.005 → 0.003 (more stable convergence)
  box loss:     7.5 → 10.0    (higher weight on bounding box regression)
  patience:     40 → 50

After training, exports to ONNX, NCNN, and PyTorch formats.
ONNX  → general-purpose deployment (TensorRT, OpenVINO, ONNX Runtime)
NCNN  → best for Raspberry Pi 4/5 and ARM edge devices
PT    → PyTorch (.pt) for desktop/GPU inference
"""