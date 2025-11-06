#!/usr/bin/env python3
"""
Image Classification Training Script using timm (PyTorch Image Models)
Usage: python train_cls.py --data <data_dir> --model <model_name> --epochs <epochs>
"""
import argparse
import sys
from pathlib import Path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train image classification models with timm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset directory (should contain train/ and val/ subdirectories)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='resnet50',
        help='Model architecture from timm (e.g., resnet50, efficientnet_b0, vit_base_patch16_224)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=224,
        help='Input image size'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device to use (cuda or cpu, auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Disable pretrained weights (not recommended)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='runs/classify',
        help='Directory to save model checkpoints'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='train',
        help='Experiment name'
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Check dependencies after parsing args so --help works
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        import timm
        from tqdm import tqdm
    except ImportError as e:
        print(f"Error: Required package not installed. {e}")
        print("Install with: pip install torch torchvision timm tqdm")
        sys.exit(1)
    
    # Auto-detect device if not specified
    if not args.device:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Image Classification Training with timm")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print(f"Pretrained: {not args.no_pretrained}")
    print("=" * 60)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get data loaders
    print("\nLoading datasets...")
    train_dataset = datasets.ImageFolder(
        root=Path(args.data) / 'train',
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=Path(args.data) / 'val',
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Use pretrained weights by default (recommended)
    use_pretrained = not args.no_pretrained
    
    # Create model
    print(f"\nCreating model: {args.model}")
    print(f"Using pretrained weights: {use_pretrained}")
    model = timm.create_model(
        args.model,
        pretrained=use_pretrained,
        num_classes=num_classes
    )
    model = model.to(args.device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("\nStarting training...")
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print('=' * 60)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{running_loss / (pbar.n + 1):.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
