# Dataset Collection & Curation

## Overview

The HearSight traffic signboard detection system uses a curated multi-source dataset for training. The final production dataset is **hearsight-ts-dataset-v3**, a two-stage dataset with separate splits for a generic detector (single-class: `traffic_sign_board`) and a 29-class crop classifier.

## Dataset Downloads

| Version | Source | Link |
|---------|--------|------|
| hearsight-ts-dataset-v3 | Kaggle | [kaggle.com/datasets/ashok205/hearsight-ts-dataset-v3](https://www.kaggle.com/datasets/ashok205/hearsight-ts-dataset-v3) |
| hearsight-ts-dataset-v3 | Google Drive | [drive.google.com (zip)](https://drive.google.com/file/d/19oFK3CY9Yzd3qLxyoRPjkrLxBSxtyVhs/view?usp=sharing) |

## Data Sources

The dataset was assembled from multiple public sources and curated using `scripts/curate_signboard_dataset.py`. Source configuration is in `configs/sources.yaml`.

### Primary Sources

| Source | Type | Images | Description |
|--------|------|-------:|-------------|
| [AbhayVAshokan/Traffic-Sign-Detection-Dataset](https://github.com/AbhayVAshokan/Traffic-Sign-Detection-Dataset) | GitHub | ~1,800 | Indian traffic sign images with YOLO annotations |
| Roboflow: Traffic Sign (YOLO26 export) | Roboflow | ~500 | Mixed traffic sign images, 2 classes |
| Roboflow: Indian Traffic Sign (YOLO26 export) | Roboflow | ~900 | Indian-specific signs, 85 classes |

### Additional Sources (used in earlier iterations)

| Source | Type | Notes |
|--------|------|-------|
| Roboflow: India Traffic Sign (`college-opfvn/india-traffic-sign`) | Roboflow | Used in v1/v2, disabled in v3 |
| Roboflow: Traffic and Road Signs (`usmanchaudhry622-gmail-com/traffic-and-road-signs`) | Roboflow | Evaluated, not used in final |
| Kaggle: Indian Sign Board (`dataclusterlabs/indian-sign-board-image-dataset`) | Kaggle CSV | Evaluated, not used in final |

## Curation Pipeline

The curation script (`scripts/curate_signboard_dataset.py`) performs:

1. **Source ingestion** — reads images + YOLO labels from each configured source
2. **Class unification** — maps all source classes to a single `traffic_sign_board` class (for the detector)
3. **Box filtering** — drops tiny boxes (`< 0.0005` normalized area) and oversized crop-like boxes (`> 0.90`)
4. **Perceptual deduplication** — exact hash-based dedup to remove duplicate images across sources
5. **Stratified split** — 80/10/10 train/val/test split with source-balanced sampling
6. **Report generation** — produces `curation_report.json`, `stats.json`, and `manifest.csv`

### Rebuild Command

```bash
python scripts/curate_signboard_dataset.py \
  --sources-config configs/sources.yaml \
  --output-dir data/curated/signboard_yolo26_lite \
  --val-ratio 0.1 --test-ratio 0.1 --seed 42 \
  --min-box-area 0.0005 --max-box-area 0.90 \
  --dedup-threshold 0
```

## V3 Dataset Structure

The v3 dataset is a **two-stage** dataset:

```
hearsight-ts-dataset-v3/
├── detector_dataset_v3/       # Generic detector (1 class: traffic_sign_board)
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml
└── classifier_dataset_v3/     # Crop classifier (29 sign type classes)
    ├── train/
    │   ├── stop_sign/
    │   ├── zebra_crossing/
    │   ├── speed_limit_40/
    │   └── ...
    ├── val/
    └── test/
```

## Kaggle Dataset Builder

The dataset builder notebook (`training/kaggle_dataset_builder.ipynb`) automates:
- Downloading and merging source datasets
- Running the curation pipeline
- Splitting into detector and classifier subsets
- Packaging for Kaggle upload

See [training.md](training.md) for how the dataset is used during model training.
