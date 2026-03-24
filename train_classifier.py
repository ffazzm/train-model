"""
Image Classification Fine-tuning Script (PyTorch + HuggingFace)
================================================================
Fine-tune any HuggingFace ViT / Swin / ConvNeXT model on a custom dataset.

Dataset structure (ImageFolder):
    dataset/
        train/
            class_a/  *.jpg
            class_b/  *.jpg
        val/
            class_a/  *.jpg
            class_b/  *.jpg

Usage:
    python train_classifier.py \\
        --data_dir dataset/ \\
        --model_name google/vit-base-patch16-224 \\
        --epochs 10 --batch_size 32 --lr 2e-4
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune image classifier (HuggingFace)")
    parser.add_argument("--data_dir",   type=str, required=True,
                        help="Root directory with train/ and val/ subfolders")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224",
                        help="HuggingFace model hub ID or local path")
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


def build_transforms(feature_extractor):
    size = feature_extractor.size
    if isinstance(size, dict):
        h = size.get("height", size.get("shortest_edge", 224))
        w = size.get("width", h)
    else:
        h = w = size

    mean = feature_extractor.image_mean
    std  = feature_extractor.image_std

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.CenterCrop((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf


def accuracy(outputs, labels):
    preds = outputs.logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = total_acc = 0.0
    for batch in tqdm(loader, desc=f"Epoch {epoch} [train]"):
        pixel_values = batch[0].to(device)
        labels       = batch[1].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss    = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc  += accuracy(outputs, labels)

    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = total_acc = 0.0
    for batch in tqdm(loader, desc="Eval"):
        pixel_values = batch[0].to(device)
        labels       = batch[1].to(device)
        outputs = model(pixel_values=pixel_values, labels=labels)
        total_loss += outputs.loss.item()
        total_acc  += accuracy(outputs, labels)
    n = len(loader)
    return total_loss / n, total_acc / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Feature extractor & transforms
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    train_tf, val_tf  = build_transforms(feature_extractor)

    # Datasets
    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, "val"),   transform=val_tf)

    id2label = {i: c for c, i in train_ds.class_to_idx.items()}
    label2id = {c: i for c, i in train_ds.class_to_idx.items()}
    num_labels = len(id2label)
    print(f"Classes ({num_labels}): {list(id2label.values())}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Model
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("Backbone frozen — training head only")

    # Optimizer & scheduler
    total_steps   = len(train_loader) * args.epochs
    warmup_steps  = int(total_steps * args.warmup_ratio)
    optimizer     = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss,   val_acc   = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:3d} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(output_dir / "best")
            feature_extractor.save_pretrained(output_dir / "best")
            print(f"  -> Saved best model (val_acc={best_val_acc:.4f})")

    # Save final
    model.save_pretrained(output_dir / "final")
    feature_extractor.save_pretrained(output_dir / "final")
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
