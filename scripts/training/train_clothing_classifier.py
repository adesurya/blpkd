"""
scripts/training/train_clothing_classifier.py

Train a custom YOLOv8 classification model for:
  - Clothing color categories (beyond basic HSV)
  - Clothing type: shirt, jacket, hoodie, dress, uniform
  - Activity recognition: standing, walking, running, sitting

This model integrates directly into the detection pipeline.
Once trained, place it at /app/models/custom_attribute.pt

Usage:
    # 1. Prepare your dataset
    python scripts/training/prepare_dataset.py --source /path/to/images --output /data/clothing

    # 2. Train
    python scripts/training/train_clothing_classifier.py \
        --data /data/clothing \
        --epochs 100 \
        --model yolov8n-cls.pt \
        --name clothing_classifier_v1

    # 3. Deploy
    cp runs/classify/clothing_classifier_v1/weights/best.pt /app/models/custom_attribute.pt

Dataset structure:
    /data/clothing/
        train/
            red_shirt/        ← class name = folder name
                img001.jpg
                img002.jpg
            blue_shirt/
                img001.jpg
            white_shirt/
            walking/
            running/
            standing/
        val/
            red_shirt/
            blue_shirt/
            ...
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


def train_classifier(
    data_path: str,
    model_base: str = "yolov8n-cls.pt",
    epochs: int = 100,
    batch_size: int = 32,
    image_size: int = 224,
    name: str = "custom_attribute",
    device: str = "0",  # GPU id or "cpu"
) -> Path:
    """
    Train YOLOv8 classification model.

    Args:
        data_path: path to dataset with train/ and val/ subdirs
        model_base: base model (yolov8n-cls.pt = nano/fastest)
        epochs: training epochs
        batch_size: batch size
        image_size: input image size
        name: run name
        device: GPU id ("0", "0,1") or "cpu"

    Returns:
        Path to best weights file
    """
    from ultralytics import YOLO

    model = YOLO(model_base)

    log.info(
        "training.start",
        data=data_path,
        model=model_base,
        epochs=epochs,
        device=device,
    )

    results = model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        name=name,
        device=device,
        # Augmentation settings
        augment=True,
        flipud=0.0,          # don't flip upside down (people are upright)
        fliplr=0.5,          # horizontal flip ok
        degrees=10.0,        # small rotation
        translate=0.1,
        scale=0.2,
        hsv_h=0.015,         # hue shift (clothing color variation)
        hsv_s=0.7,           # saturation shift
        hsv_v=0.4,           # value/brightness shift
        # Training settings
        patience=20,         # early stopping
        save=True,
        plots=True,
        verbose=True,
    )

    best_weights = Path(f"runs/classify/{name}/weights/best.pt")
    log.info("training.complete", best_weights=str(best_weights))
    return best_weights


def deploy_model(weights_path: Path, deploy_path: str = "/app/models/custom_attribute.pt") -> None:
    """Copy trained model to deployment location."""
    deploy = Path(deploy_path)
    deploy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights_path, deploy)
    log.info("training.deployed", path=deploy_path)


def prepare_dataset_from_detections(
    detections_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
) -> None:
    """
    Prepare training dataset from saved detection crops.

    If you have a labeled dataset of person crops (organized by class),
    this script splits them into train/val sets.

    Expected input structure:
        detections_dir/
            red_shirt/  ← label from manual annotation
            blue_shirt/
            running/

    Output structure:
        output_dir/
            train/
                red_shirt/
                blue_shirt/
            val/
                red_shirt/
                blue_shirt/
    """
    import random
    import os

    src = Path(detections_dir)
    dst = Path(output_dir)

    for class_dir in src.iterdir():
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(images)

        split = int(len(images) * train_ratio)
        train_imgs = images[:split]
        val_imgs = images[split:]

        for split_name, imgs in [("train", train_imgs), ("val", val_imgs)]:
            out_dir = dst / split_name / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, out_dir / img.name)

    log.info("dataset.prepared", output=str(dst))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom attribute classifier")
    parser.add_argument("--data", required=True, help="Dataset directory")
    parser.add_argument("--model", default="yolov8n-cls.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--name", default="custom_attribute")
    parser.add_argument("--device", default="0")
    parser.add_argument("--deploy", action="store_true", help="Auto-deploy after training")
    args = parser.parse_args()

    weights = train_classifier(
        data_path=args.data,
        model_base=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        name=args.name,
        device=args.device,
    )

    if args.deploy:
        deploy_model(weights)
        print(f"✅ Model deployed. Restart workers to load new model.")
