"""
YOLO Object Detection Training Script (Ultralytics)
====================================================
Fine-tune or train a YOLO model on a custom dataset.

Dataset structure (YOLO format):
    dataset/
        images/
            train/  *.jpg
            val/    *.jpg
        labels/
            train/  *.txt   (class_id cx cy w h, normalized)
            val/    *.txt
        data.yaml

data.yaml example:
    path: /abs/path/to/dataset
    train: images/train
    val:   images/val
    nc:    3
    names: [cat, dog, bird]

Usage:
    python train_yolo.py --data dataset/data.yaml --model yolo11n.pt --epochs 50
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO object detection model")
    parser.add_argument("--data",    type=str, required=True,       help="Path to data.yaml")
    parser.add_argument("--model",   type=str, default="yolo11n.pt", help="Pretrained model or .yaml config")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--lr0",     type=float, default=0.01,      help="Initial learning rate")
    parser.add_argument("--device",  type=str, default="",          help="cuda device, e.g. 0 or 0,1 or cpu")
    parser.add_argument("--project", type=str, default="runs/yolo", help="Output project directory")
    parser.add_argument("--name",    type=str, default="train",     help="Experiment name")
    parser.add_argument("--resume",  action="store_true",           help="Resume from last checkpoint")
    parser.add_argument("--freeze",  type=int, default=0,           help="Freeze first N layers (0 = none)")
    return parser.parse_args()


def train(args):
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        resume=args.resume,
        freeze=args.freeze,
        # Augmentation defaults (tune as needed)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )

    # Evaluate on val split
    metrics = model.val()
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    # Export best weights to ONNX for deployment
    best = Path(args.project) / args.name / "weights" / "best.pt"
    if best.exists():
        export_model = YOLO(str(best))
        export_model.export(format="onnx", imgsz=args.imgsz)
        print(f"Exported ONNX model to {best.with_suffix('.onnx')}")

    return results


if __name__ == "__main__":
    args = parse_args()
    train(args)
