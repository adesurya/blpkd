# Custom Model Training Guide

## Overview

The platform supports three types of custom model training,
each extending the base capability of the pipeline:

| Model | Purpose | Base Model | Training Data Needed |
|-------|---------|-----------|---------------------|
| Clothing Classifier | Warna baju, jenis pakaian | YOLOv8n-cls | 200+ images per class |
| Activity Recognizer | Berjalan, berlari, duduk, dll | YOLOv8s-cls | 300+ clips per class |
| Custom Face Tuning | Improve accuracy for local faces | InsightFace ArcFace | 10+ faces per identity |

---

## 1. Clothing Color Classifier

**Use case:** Classify upper/lower body clothing into specific color categories
or clothing types (shirt, jacket, hoodie, uniform).

### Step 1: Collect Training Data

Person crops are automatically saved to MinIO during normal operation.
Download them for labeling:

```bash
# Download recent person crops from MinIO
docker compose exec worker-gpu python scripts/training/collect_crops_from_minio.py \
  --output /data/raw_crops

# Or use your own labeled dataset
```

### Step 2: Label Data

Use any labeling tool:
- **Roboflow** (recommended — web-based, easy to use): roboflow.com
- **Label Studio**: open source, self-hosted
- **CVAT**: open source, advanced

Label each person crop with its dominant clothing color/type.

### Step 3: Organize Dataset

```
/data/clothing/
    train/
        red_shirt/
            img001.jpg   ← full person crop, upper body is red
            img002.jpg
        blue_shirt/
            img001.jpg
        white_shirt/
        black_shirt/
        uniform_gray/    ← security/hotel uniform
    val/                 ← 20% of data goes here
        red_shirt/
        blue_shirt/
        ...
```

Minimum: **200 images per class**, recommended: 500+

### Step 4: Train

```bash
docker compose exec worker-gpu python scripts/training/train_clothing_classifier.py \
  --data /data/clothing \
  --epochs 100 \
  --model yolov8n-cls.pt \
  --name clothing_v1 \
  --device 0 \
  --deploy
```

### Step 5: Verify & Restart

```bash
# Check model was deployed
docker compose exec worker-gpu ls /app/models/

# Restart GPU worker to load new model
docker compose restart worker-gpu

# Test prediction
docker compose exec worker-gpu python3 -c "
from ultralytics import YOLO
import cv2
model = YOLO('/app/models/custom_attribute.pt')
img = cv2.imread('/path/to/test_crop.jpg')
result = model(img)
print(result[0].probs.top5)
"
```

---

## 2. Activity Recognizer

**Use case:** Classify what each person is doing:
standing, walking, running, sitting, crouching, falling, fighting.

The activity recognizer enables **conditional counting**:
- "Count only walking people"
- "Alert when someone is running in restricted zone"
- "Detect falls in elderly care areas"

### Training Data Requirements

Each class needs **300+ person crop images** showing that activity.
Crops must show the full body (or at least torso + legs).

```
/data/activities/
    train/
        standing/    ← person standing still
        walking/     ← person mid-stride
        running/     ← person running (arms/legs extended)
        sitting/     ← person seated
        crouching/   ← person crouching/squatting
        falling/     ← person falling (rare class, needs augmentation)
    val/
        standing/
        walking/
        ...
```

### Train

```bash
docker compose exec worker-gpu python scripts/training/train_activity_recognizer.py \
  --data /data/activities \
  --epochs 150 \
  --model yolov8s-cls.pt \
  --name activity_v1 \
  --deploy
```

### Use Conditional Counting

Once deployed, use `filter_criteria` in stream start or video upload:

```bash
# Count ONLY running people
curl -X POST http://localhost:8000/api/v1/streams/CAM_ID/start \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "CAM_ID",
    "analyze_attributes": true,
    "filter_criteria": {"activity": "running"}
  }'

# Count ONLY blue-shirt standing people
curl -X POST http://localhost:8000/api/v1/videos/upload \
  ... \
  -F "filter_criteria={\"upper_color\":\"blue\",\"activity\":\"standing\"}"
```

---

## 3. Custom InsightFace Fine-tuning (Advanced)

**Use case:** Improve face recognition accuracy for a specific population
(e.g., Indonesian faces, uniform-wearing employees, low-light CCTV conditions).

### When to use

The `buffalo_l` model is trained on global datasets.
Fine-tuning on your specific data can improve accuracy by 5–15%.
Recommended when:
- Recognition rate drops below 70% in production
- Your environment has specific lighting conditions
- Your subjects wear uniforms/hijab that occludes part of the face

### Data Requirements

- Minimum 10 photos per identity
- Varied angles, lighting, expressions
- At least 50 different identities for fine-tuning

### Training Pipeline

```bash
# 1. Prepare dataset in ArcFace format
# Each identity gets a folder numbered 0, 1, 2, ...
mkdir -p /data/faces/train
# /data/faces/train/0/img1.jpg (person A)
# /data/faces/train/1/img1.jpg (person B)
# ...

# 2. Generate image list
python scripts/training/prepare_insightface_dataset.py \
  --input /data/faces \
  --output /data/faces_packed

# 3. Fine-tune (requires insightface training toolkit)
# See: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch

# 4. Export to ONNX and drop in models folder
cp finetuned_model.onnx /app/models/buffalo_custom/
```

---

## 4. Training Best Practices

### Data Quality > Quantity

- Sharp, well-lit images outperform large blurry datasets
- Balance classes: don't let one class have 10x more images than others
- Augmentation in training compensates for small datasets

### Validation Strategy

Always hold out 20% of data for validation.
Monitor validation accuracy — stop if it diverges from training accuracy (overfitting).

```
Epoch 50/100: train_acc=0.94, val_acc=0.91  ← good
Epoch 80/100: train_acc=0.98, val_acc=0.87  ← slight overfit, add more augmentation
Epoch 100/100: train_acc=0.99, val_acc=0.72 ← severe overfit, reduce epochs or add data
```

### A/B Testing Models

Deploy new model to one worker only and compare results:

```bash
# In docker-compose.scale.yml, add second GPU worker with different model path
services:
  worker-gpu-b:
    environment:
      - YOLO_MODEL=yolov8m.pt
      - MODELS_PATH=/app/models_v2
```

### Recommended Training Hardware

| Dataset Size | GPU | Estimated Time |
|-------------|-----|----------------|
| <1000 images | RTX 3060 (12GB) | 30 min |
| 1000–10K images | RTX 3060 (12GB) | 2–4 hours |
| 10K–100K images | RTX 3090/4090 | 4–12 hours |
| >100K images | A100 (40GB) | 12–48 hours |

Train inside docker for consistent environment:
```bash
docker compose exec worker-gpu bash
# Then run training scripts
```
