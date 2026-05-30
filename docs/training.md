# Training Strategy

## Why Two-Stage Detection?

Traditional single-model approaches (one YOLO model with 29+ classes) struggle with:
- **Class imbalance** — some sign types have very few training examples
- **Scale variation** — signs range from 10px (distant) to 400px (close-up)
- **Confusion between similar signs** — detector needs to tell apart 29 visually similar classes

The two-stage approach solves this by splitting the task:

```
Stage 1: Generic Detector (YOLO)
  Input:  Full frame -> (640x640)
  Output: Bounding boxes for "traffic_sign_board" (3 classes - "road_sign", "facility_sign", "medical_sign")
  Goal:   High recall — find ALL signs, even small/distant ones

Stage 2: Crop Classifier (YOLO classify)
  Input:  Cropped sign region (padded)
  Output: Sign type classification (20+ classes)
  Goal:   High precision — correctly identify each sign type
```

### Benefits
- **Detector** only needs to learn one class → higher recall, simpler task
- **Classifier** sees zoomed-in crops → easier to distinguish similar signs
- **Independent training** — can improve detector or classifier separately
- **Caching** — classifier results are cached per tracked sign, reducing GPU load

## Model Architecture

| Component | Base Model | Input Size | Classes | Target Weights |
|-----------|-----------|-----------|---------|----------------|
| **Detector** | YOLO26n (detect) | 640×640 | 3 (`road_sign`, `facility_sign`, `medical_sign`) | `weights/detector/best.pt` |
| **Classifier** | YOLO26n-cls (classify) | 640×640 | 28 (27 target sign types + 1 `not_target` catch-all) | `weights/classifier/best.pt` |

## Training Configuration

### Runtime Training Dataset Policy

To optimize the models for high precision and target-specific detection, the Kaggle training script applies a dynamic runtime dataset policy:
1. **Disabled Detector Classes**: Non-signboard labels such as `traffic_light` are completely purged from the detector labels to avoid diluting the generic sign detection capability. This leaves exactly **3 detection classes**: `road_sign`, `facility_sign`, and `medical_sign`.
2. **Disabled Classifier Classes**: Background/utility classes like `Exit`, `Fire extinguisher`, `Traffic light red`, `Traffic light yellow`, and `Traffic light green` are removed from their separate directories and merged into a single `not_target` folder. This acts as a robust negative/catch-all class during inference.
3. **Active Classifier Count**: Dynamic scanning yields exactly **28 active classification directories** (27 target categories + `not_target`).

### Augmentation Policy

- **Detector**: 
  - Reduced mosaic augmentation (`mosaic=0.15`) to prevent tiny objects from becoming completely unrecognizable during multi-image stitching.
  - Mosaic is fully disabled during the final 10 epochs (`close_mosaic=10`) for fine-tuning.
  - Geometric flipping and auto-augmentation are fully disabled to protect orientation-sensitive signs.
- **Classifier**: 
  - Uses standard classification augmentations but restricts any vertical flips.

### Hyperparameters

The following exact hyperparameters are defined in `kaggle-hearsight-ts-training.ipynb` for multi-GPU Tesla T4 execution:

```yaml
# Detector (Stage 1)
epochs: 100
imgsz: 640
batch: 196
device: [0, 1]              # DDP Multi-GPU on dual Tesla T4s
optimizer: AdamW
lr0: 0.0005                 # Controlled learning rate for stable backbone fine-tuning
lrf: 0.01
warmup_epochs: 4
close_mosaic: 10
mosaic: 0.15
patience: 20
cos_lr: true
amp: true
workers: 2

# Classifier (Stage 2)
epochs: 100                 # Early stopped at epoch 96
imgsz: 640
batch: 196
device: [0, 1]              # DDP Multi-GPU
optimizer: AdamW
lr0: 0.001
lrf: 0.01
patience: 20
cos_lr: true
amp: true
workers: 1                  # Restricted to 1 worker to optimize CPU host memory utilization
```

## Training Results

### Detector (Generic Sign Detector)

Trained for 100 epochs on the v3 detector dataset (3 classes: `road_sign`, `facility_sign`, `medical_sign`).

| Metric | Best (Epoch 60) | Final (Epoch 100) |
|--------|----------------:|------------------:|
| **Precision** | 0.8851 | 0.8600 |
| **Recall** | 0.7832 | 0.8031 |
| **mAP@50** | **0.8596** | 0.8510 |
| **mAP@50-95** | 0.7126 | **0.7127** |

Training progression (key milestones):

| Epoch | Precision | Recall | mAP@50 | mAP@50-95 |
|------:|----------:|-------:|-------:|----------:|
| 10 | 0.7827 | 0.6847 | 0.7705 | 0.5929 |
| 25 | 0.8463 | 0.7558 | 0.8463 | 0.6804 |
| 50 | 0.8421 | 0.7761 | 0.8502 | 0.6993 |
| 75 | 0.8780 | 0.7810 | 0.8501 | 0.7107 |
| 100 | 0.8600 | 0.8031 | 0.8510 | 0.7127 |

### Classifier (Crop Sign Classifier)

Trained for 96 epochs on the v3 classifier dataset (28 classes: 27 target categories + 1 `not_target` background class). Early stopped at epoch 96 (patience=20).

| Metric | Best (Epoch 76) | Final (Epoch 96) |
|--------|----------------:|------------------:|
| **Top-1 Accuracy** | **95.50%** | 95.25% |
| **Top-5 Accuracy** | **99.56%** | 99.56% |
| **Val Loss** | 0.278 | 0.282 |

Training progression (key milestones):

| Epoch | Top-1 Acc | Top-5 Acc | Val Loss |
|------:|----------:|----------:|---------:|
| 10 | 84.97% | 97.66% | 0.672 |
| 20 | 93.16% | 99.61% | 0.246 |
| 40 | 93.58% | 99.40% | 0.298 |
| 60 | 95.09% | 99.52% | 0.257 |
| 76 | **95.50%** | **99.56%** | 0.278 |
| 96 | 95.25% | 99.56% | 0.282 |

### Training Logs & Visualizations

Full training logs, curves, and confusion matrices are available in `training_logs/`:

```
training_logs/
├── detector/
│   ├── results.csv              # Per-epoch metrics
│   ├── BoxF1_curve.png          # F1 score curve
│   ├── BoxPR_curve.png          # Precision-Recall curve
│   ├── BoxP_curve.png           # Precision curve
│   ├── BoxR_curve.png           # Recall curve
│   ├── confusion_matrix.png     # Confusion matrix
│   ├── confusion_matrix_normalized.png
│   ├── val_batch*_labels.jpg    # Validation ground truth
│   ├── val_batch*_pred.jpg      # Validation predictions
│   └── train_detector.log       # Full training log
└── classifier/
    ├── results.csv              # Per-epoch metrics
    ├── args.yaml                # Training configuration
    ├── confusion_matrix.png     # 28-class confusion matrix
    ├── confusion_matrix_normalized.png
    ├── val_batch*_labels.jpg    # Validation ground truth
    ├── val_batch*_pred.jpg      # Validation predictions
    └── train_classifier.log     # Full training log
```

## Training Workflow

Training is done on **Kaggle** using free **dual Tesla T4 GPU** instances:

1. Upload dataset to Kaggle as `hearsight-ts-dataset-v3`
2. Open `training/kaggle-hearsight-ts-training.ipynb` on Kaggle
3. Attach the dataset
4. Configure training mode (`TRAIN_MODE`):
   - `"detector"` — train only the generic detector (utilizing both GPUs via DDP)
   - `"classifier"` — train only the crop classifier (utilizing both GPUs)
   - `"both"` — launch both processes in parallel (executing detector on GPU 0 and classifier on GPU 1 simultaneously)
5. Run all cells
6. Download `best.pt` weights from the output directory (`/kaggle/working/runs/hearsight_two_stage/...`)

### Local Training (Optional)

For local GPU training:

```bash
python scripts/train_local.py \
  --data data/curated/signboard_yolo26_lite/dataset.yaml \
  --model yolo26n.pt \
  --epochs 80 --imgsz 1024 --batch 8 --device 0
```

## Training Notebooks

| Notebook | Purpose |
|----------|---------|
| `training/kaggle_dataset_builder.ipynb` | Build and upload the v3 dataset to Kaggle |
| `training/kaggle-hearsight-ts-training.ipynb` | Train detector and/or classifier on Kaggle |

