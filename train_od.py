#!/usr/bin/env python3
"""
Object Detection Training Script using Ultralytics YOLO
Usage: python train_od.py --data <dataset.yaml> --model <yolo_model> --epochs <epochs>
"""
import argparse
import sys
from pathlib import Path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train YOLO object detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset YAML file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='YOLO model variant (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device to use (e.g., 0 or 0,1,2,3 or cpu)'
    )
    
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='Project directory to save results'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='train',
        help='Experiment name'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience (epochs)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of worker threads for data loading'
    )
    
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=True,
        help='Use pretrained weights'
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Check dependencies
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Install it with: pip install ultralytics")
        sys.exit(1)
    
    print("=" * 60)
    print("Object Detection Training with YOLO")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print("=" * 60)
    
    # Load YOLO model
    model = YOLO(args.model)
    
    # Train the model
    # Ultralytics YOLO has built-in progress bars, so no need for tqdm
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        pretrained=args.pretrained,
    )
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Results saved to: {Path(args.project) / args.name}")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
