"""
Image Classification Fine-tuning Script (PyTorch + timm)
========================================================
Fine-tune any timm pretrained model (ResNet / ViT / Swin / ConvNeXT / EfficientNet ...)
on a custom dataset.

Dataset structure (ImageFolder):
    dataset/
        train/
            class_a/  *.jpg
            class_b/  *.jpg
        val/
            class_a/  *.jpg
            class_b/  *.jpg

Usage:
    python train_image_classification.py \\
        --data_dir dataset/ \\
        --model_name resnet50 \\
        --epochs 10 --batch_size 32 --lr 2e-4

List available models:
    python -c "import timm; print(timm.list_models('*vit*', pretrained=True))"
"""

import argparse
import json
import math
import os
from pathlib import Path

import timm
import torch
import torch.nn as nn
from timm.data import create_transform, resolve_model_data_config
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune image classifier (timm)")
    parser.add_argument("--data_dir",   type=str, required=True,
                        help="Root directory with train/ and val/ subfolders")
    parser.add_argument("--model_name", type=str, default="resnet50",
                        help="timm model name (e.g. resnet50, vit_base_patch16_224, "
                             "convnext_tiny, efficientnet_b0)")
    parser.add_argument("--output_dir", type=str, default="runs/classifier",
                        help="Where to save checkpoints and final model")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Fraction of total steps used for LR warmup")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone, only train classifier head")
    parser.add_argument("--device",     type=str, default="",
                        help="cuda / cpu (auto-detected if empty)")
    return parser.parse_args()


def build_transforms(model):
    """Build train / val transforms from the model's pretrained data config."""
    cfg = resolve_model_data_config(model)
    train_tf = create_transform(**cfg, is_training=True)
    val_tf   = create_transform(**cfg, is_training=False)
    return train_tf, val_tf


def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """LR schedule: linear warmup then cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = total_acc = 0.0
    for images, labels in tqdm(loader, desc=f"Epoch {epoch} [train]"):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc  += accuracy(logits, labels)

    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_acc = 0.0
    for images, labels in tqdm(loader, desc="Eval"):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        total_acc  += accuracy(logits, labels)
    n = len(loader)
    return total_loss / n, total_acc / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_checkpoint(model, id2label, model_name, dst):
    dst.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dst / "model.pth")
    with open(dst / "config.json", "w") as f:
        json.dump({
            "model_name": model_name,
            "num_labels": len(id2label),
            "id2label": id2label,
        }, f, indent=2)


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Peek at classes first so we can build the model with the right head
    train_root = os.path.join(args.data_dir, "train")
    class_names = sorted(entry.name for entry in os.scandir(train_root) if entry.is_dir())
    num_labels  = len(class_names)
    id2label    = {i: c for i, c in enumerate(class_names)}
    print(f"Classes ({num_labels}): {class_names}")

    # Model (timm pretrained)
    model = timm.create_model(
        args.model_name,
        pretrained=True,
        num_classes=num_labels,
    ).to(device)

    if args.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True
        print("Backbone frozen — training classifier head only")

    # Transforms derived from the model's pretrained config
    train_tf, val_tf = build_transforms(model)

    # Datasets & loaders
    train_ds = datasets.ImageFolder(train_root, transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Loss, optimizer & scheduler
    criterion     = nn.CrossEntropyLoss()
    total_steps   = len(train_loader) * args.epochs
    warmup_steps  = int(total_steps * args.warmup_ratio)
    optimizer     = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = cosine_warmup_scheduler(optimizer, warmup_steps, total_steps)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:3d} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, id2label, args.model_name, output_dir / "best")
            print(f"  -> Saved best model (val_acc={best_val_acc:.4f})")

    # Save final
    save_checkpoint(model, id2label, args.model_name, output_dir / "final")
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
