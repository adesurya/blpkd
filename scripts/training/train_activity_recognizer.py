"""
scripts/training/train_activity_recognizer.py

Train a YOLOv8 classification model for human activity recognition.
Activities: standing, walking, running, sitting, crouching, fighting, falling

This is trained on person CROPS (not full frame), so it can classify
what each individual person is doing — enabling:
  - Count only walking people
  - Alert when someone falls
  - Detect running in restricted areas

Dataset structure:
    /data/activities/
        train/
            standing/
                person_crop_001.jpg
                person_crop_002.jpg
            walking/
            running/
            sitting/
            falling/
        val/
            standing/
            walking/
            ...

Usage:
    # Collect crops first (they're saved to MinIO during normal operation)
    python scripts/training/collect_crops_from_minio.py --output /data/raw_crops

    # Label them manually or use Label Studio / Roboflow
    # Then organize into train/val structure above

    # Train
    python scripts/training/train_activity_recognizer.py \
        --data /data/activities \
        --epochs 150 \
        --name activity_v1 \
        --deploy
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def train_activity_recognizer(
    data_path: str,
    model_base: str = "yolov8s-cls.pt",   # small = good balance for activity
    epochs: int = 150,
    batch_size: int = 64,
    image_size: int = 224,
    name: str = "activity_recognizer",
    device: str = "0",
) -> Path:
    from ultralytics import YOLO

    model = YOLO(model_base)

    results = model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        name=name,
        device=device,
        # Activity-specific augmentation
        augment=True,
        fliplr=0.5,
        flipud=0.0,      # don't flip upside down for activities
        degrees=15.0,    # more rotation than clothing (poses vary)
        translate=0.15,
        scale=0.3,
        shear=5.0,       # slight shear for angle variation
        perspective=0.001,
        # Temporal consistency: use mosaic carefully
        mosaic=0.5,
        mixup=0.1,
        # Training
        patience=30,
        save=True,
        plots=True,
        verbose=True,
        # Class weights if imbalanced (e.g., fewer "falling" samples)
        # Specify in data.yaml: class_weights: [1.0, 1.0, 1.0, 1.0, 5.0]  # falling=5x
    )

    best = Path(f"runs/classify/{name}/weights/best.pt")
    print(f"✅ Training complete. Best weights: {best}")
    return best


def collect_crops_from_minio(output_dir: str) -> None:
    """
    Helper: download person crops from MinIO for labeling.
    Crops are stored under MinIO bucket 'frames' with prefix 'sessions/*/persons/'
    """
    from core.abstractions.storage import get_storage
    import asyncio, os

    async def _download():
        storage = get_storage()
        keys = await storage.list_objects("frames", prefix="")
        person_keys = [k for k in keys if "/persons/" in k]
        print(f"Found {len(person_keys)} person crops in MinIO")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for key in person_keys[:500]:  # limit for manageable labeling
            data = await storage.download("frames", key)
            filename = key.replace("/", "_")
            (out / filename).write_bytes(data)
        print(f"Downloaded {min(len(person_keys), 500)} crops to {output_dir}")

    asyncio.run(_download())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolov8s-cls.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--name", default="activity_recognizer")
    parser.add_argument("--device", default="0")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--collect-crops", metavar="OUTPUT_DIR",
                        help="Download person crops from MinIO for labeling")
    args = parser.parse_args()

    if args.collect_crops:
        collect_crops_from_minio(args.collect_crops)
    else:
        weights = train_activity_recognizer(
            data_path=args.data,
            model_base=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            name=args.name,
            device=args.device,
        )
        if args.deploy:
            dest = Path("/app/models/activity_recognizer.pt")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(weights, dest)
            print(f"✅ Deployed to {dest}. Restart GPU workers to load.")
