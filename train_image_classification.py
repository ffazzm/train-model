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

Two-phase transfer learning (recommended for small datasets, e.g. ~800/class):
    python train_image_classification.py \\
        --data_dir dataset/ \\
        --model_name efficientnet_b0 \\
        --epochs 25 --batch_size 32 \\
        --freeze_epochs 5 --lr 1e-3 --finetune_lr 1e-4 \\
        --drop_rate 0.3 --drop_path_rate 0.2 --label_smoothing 0.1

List available models:
    python -c "import timm; print(timm.list_models('*vit*', pretrained=True))"
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
from timm.data import create_transform
# timm >=0.8 exposes resolve_model_data_config(model); timm 0.6.x uses
# resolve_data_config({}, model=model). Support both so the script runs on
# whatever timm version is installed.
try:
    from timm.data import resolve_model_data_config as _resolve_model_data_config

    def resolve_data_cfg(model):
        return _resolve_model_data_config(model)
except ImportError:
    from timm.data import resolve_data_config as _resolve_data_config

    def resolve_data_cfg(model):
        return _resolve_data_config({}, model=model)
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune image classifier (timm)")
    parser.add_argument("--data_dir",   type=str, required=True,
                        help="Root directory with train/ and val/ subfolders")
    parser.add_argument("--model_name", type=str, default="tf_efficientnetv2_b0.in1k",
                        help="timm model name (e.g. tf_efficientnetv2_b0.in1k, resnet50, "
                             "vit_base_patch16_224, convnext_tiny, efficientnet_b0)")
    parser.add_argument("--output_dir", type=str, default="runs/classifier",
                        help="Where to save checkpoints and final model")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Fraction of total steps used for LR warmup")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping: stop if val_acc doesn't improve for N "
                             "consecutive epochs. 0 disables. In two-phase mode the "
                             "counter resets when the backbone unfreezes, and early "
                             "stopping never triggers before phase 2 begins.")
    parser.add_argument("--min_delta", type=float, default=0.0,
                        help="Minimum val_acc gain over the best-so-far to count as an "
                             "improvement (resets the early-stopping counter).")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone for the whole run (head only). "
                             "For two-phase fine-tuning use --freeze_epochs instead.")
    parser.add_argument("--freeze_epochs", type=int, default=0,
                        help="Two-phase fine-tuning: train head only for the first "
                             "N epochs, then unfreeze the backbone. 0 disables.")
    parser.add_argument("--finetune_lr", type=float, default=0.0,
                        help="LR for phase 2 (after unfreezing). 0 = reuse --lr. "
                             "Recommended lower than --lr, e.g. lr=1e-3, finetune_lr=1e-4.")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing for CrossEntropyLoss (0 disables)")
    parser.add_argument("--drop_rate", type=float, default=0.0,
                        help="Dropout before classifier head (regularization)")
    parser.add_argument("--drop_path_rate", type=float, default=0.0,
                        help="Stochastic depth rate in backbone (regularization)")
    parser.add_argument("--num_sample_preds", type=int, default=16,
                        help="Save a montage of N sample predictions after training "
                             "(misclassifications prioritized). 0 disables.")
    parser.add_argument("--device",     type=str, default="",
                        help="cuda / cpu (auto-detected if empty)")
    return parser.parse_args()  


def build_augmentation(input_size, mean, std):
    """Hardcoded training augmentation, tuned for accident-vs-normal vehicle images.

    The accident signal (crumpled panels, debris, deformed geometry) is often
    *localized*, so this pipeline stays conservative on cropping and erasing to
    avoid removing the very cues that define the label. `input_size` is the
    model's expected square crop; `mean` / `std` come from the pretrained config
    so normalization stays consistent with the val transform.
    """
    # return T.Compose([
    #     # Crop augmentation disabled for now: keep the whole frame so localized
    #     # accident cues are never cropped away. Plain (square) resize to the
    #     # model's input size instead. Re-enable RandomResizedCrop later if wanted.
    #     # NOTE: this forces a square aspect ratio (mild distortion) — acceptable
    #     # while we isolate the crop's effect.
    #     T.RandomResizedCrop(input_size, scale=(0.9, 1.0)),
    #     # T.Resize((input_size, input_size)),
    #     T.RandomHorizontalFlip(p=0.5),                 # accidents are mirror-symmetric
    #     # Camera-angle variance (CCTV / dashcam tilt). No vertical flip: an
    #     # upside-down car would confuse the "normal" class.
    #     # T.RandomApply([T.RandomAffine(degrees=12, translate=(0.05, 0.05),
    #     #                               scale=(0.95, 1.05), shear=5)], p=0.5),
    #     # Weather / lighting / day-night variance; small hue shift only.
    #     # T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3,
    #     #                              saturation=0.3, hue=0.05)], p=0.7),
    #     T.RandomPerspective(distortion_scale=0.15, p=0.4),
    #     T.RandomApply([T.RandomAffine(degrees=10, translate=(0.02, 0.02), scale=(0.98, 1.02))], p=0.4),
    #     T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
    #     T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    #     # T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
    #     T.ToTensor(),
    #     T.Normalize(mean=mean, std=std),
    #     # Low prob / small area so it rarely erases the damage evidence.
    #     T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    # ])
    return T.Compose([
        # 1. Pertahankan aspect ratio asli objek crop (bisa pakai padding ke square sebelum resize)
        # Di bawah ini alternatif simpel jika tidak pakai custom padding:
        T.Resize((input_size, input_size)), 
        
        # 2. Geometri & Distorsi Kamera
        T.RandomHorizontalFlip(p=0.5),
        T.RandomPerspective(distortion_scale=0.15, p=0.3),
        T.RandomApply([T.RandomAffine(degrees=10, translate=(0.02, 0.02), scale=(0.95, 1.05))], p=0.3),
        
        # 3. Variasi Cuaca, Cahaya, & Sensor CCTV
        T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03)], p=0.6),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
        
        # 4. Normalisasi & Tensor
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
        
        # 5. Regularisasi (Kecilkan probabilitas agar tidak menutupi defect utama)
        T.RandomErasing(p=0.1, scale=(0.01, 0.05), ratio=(0.3, 3.3)),
    ])


def build_transforms(model):
    """Build train / val transforms from the model's pretrained data config."""
    cfg = resolve_data_cfg(model)
    # Val transform comes straight from the pretrained config.
    val_tf = create_transform(**cfg, is_training=False)

    # Train transform uses our hardcoded augmentation, keeping the model's
    # expected input size and normalization stats.
    input_size = cfg["input_size"][-1]  # (C, H, W) -> H/W
    train_tf = build_augmentation(input_size, cfg["mean"], cfg["std"])
    return train_tf, val_tf


def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """LR schedule: linear warmup then cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def forward_logits(model, images):
    """Run the model and return a single logits tensor.

    Distillation architectures (e.g. LeViT) return a (head, head_dist) tuple
    while training; average them so the rest of the pipeline sees one tensor.
    In eval mode timm already averages the heads and returns a single tensor.
    """
    out = model(images)
    if isinstance(out, (tuple, list)):
        out = sum(out) / len(out)
    return out


def accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def confusion_matrix(preds, labels, num_classes):
    """Build an [num_classes x num_classes] matrix; rows = true, cols = pred."""
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(labels.view(-1), preds.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


def classification_metrics(cm, class_names):
    """Per-class precision / recall / F1 + macro averages from a confusion matrix."""
    cm = cm.float()
    tp = torch.diag(cm)
    support = cm.sum(dim=1)                 # true count per class
    pred_pos = cm.sum(dim=0)                # predicted count per class
    precision = tp / pred_pos.clamp(min=1)
    recall    = tp / support.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

    per_class = {
        class_names[i]: {
            "precision": precision[i].item(),
            "recall":    recall[i].item(),
            "f1":        f1[i].item(),
            "support":   int(support[i].item()),
        }
        for i in range(len(class_names))
    }
    macro = {
        "precision": precision.mean().item(),
        "recall":    recall.mean().item(),
        "f1":        f1.mean().item(),
    }
    return per_class, macro


def format_confusion_matrix(cm, class_names):
    """Render the confusion matrix as an aligned text table."""
    width = max(9, max(len(c) for c in class_names) + 2)
    header = " " * width + "".join(f"{('p:' + c):>{width}}" for c in class_names)
    lines = ["Confusion matrix (rows = true, cols = predicted):", header]
    for i, c in enumerate(class_names):
        row = f"{('t:' + c):>{width}}" + "".join(
            f"{int(cm[i, j].item()):>{width}}" for j in range(len(class_names)))
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device,
                    epoch, epochs):
    model.train()
    total_loss = total_acc = 0.0
    bar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [train]",
               leave=False, dynamic_ncols=True)
    for step, (images, labels) in enumerate(bar, start=1):
        images = images.to(device)
        labels = labels.to(device)

        logits = forward_logits(model, images)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc  += accuracy(logits, labels)
        bar.set_postfix(loss=f"{total_loss / step:.4f}",
                        acc=f"{total_acc / step:.3f}",
                        lr=f"{scheduler.get_last_lr()[0]:.1e}")

    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Eval"):
    model.eval()
    total_loss = total_acc = 0.0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        images = images.to(device)
        labels = labels.to(device)
        logits = forward_logits(model, images)
        total_loss += criterion(logits, labels).item()
        total_acc  += accuracy(logits, labels)
        all_preds.append(logits.argmax(dim=-1).cpu())
        all_labels.append(labels.cpu())
    n = len(loader)
    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return total_loss / n, total_acc / n, preds, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_eval_report(preds, labels, class_names, val_loss, val_acc, epoch, dst,
                      show=True):
    """Save (and optionally print) a confusion matrix + per-class P/R/F1 report."""
    num_classes = len(class_names)
    cm = confusion_matrix(preds, labels, num_classes)
    per_class, macro = classification_metrics(cm, class_names)
    cm_text = format_confusion_matrix(cm, class_names)

    lines = [
        f"Epoch {epoch} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
        "",
        cm_text,
        "",
        f"{'class':<20}{'precision':>12}{'recall':>12}{'f1':>12}{'support':>10}",
    ]
    for name, m in per_class.items():
        lines.append(f"{name:<20}{m['precision']:>12.4f}{m['recall']:>12.4f}"
                     f"{m['f1']:>12.4f}{m['support']:>10d}")
    lines.append(f"{'macro avg':<20}{macro['precision']:>12.4f}"
                 f"{macro['recall']:>12.4f}{macro['f1']:>12.4f}")
    report = "\n".join(lines)
    if show:
        print(report)

    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "eval_report.txt", "w") as f:
        f.write(report + "\n")
    with open(dst / "metrics.json", "w") as f:
        json.dump({
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
            "per_class": per_class,
            "macro_avg": macro,
        }, f, indent=2)


@torch.no_grad()
def save_prediction_grid(model, loader, class_names, mean, std, device, dst,
                         num_samples=16):
    """Save a single stacked image of sample predictions.

    Green border = correct, red = wrong; caption shows p:<pred> t:<true>.
    Misclassifications are collected first so the montage is informative.
    """
    if num_samples <= 0:
        return
    model.eval()
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t  = torch.tensor(std).view(3, 1, 1)

    wrong, correct = [], []
    for images, labels in loader:
        preds = forward_logits(model, images.to(device)).argmax(dim=-1).cpu()
        for img, pred, true in zip(images, preds, labels):
            denorm = (img * std_t + mean_t).clamp(0, 1)  # undo Normalize for display
            item = (denorm, int(pred), int(true))
            if pred != true and len(wrong) < num_samples:
                wrong.append(item)
            elif len(correct) < num_samples:
                correct.append(item)
        if len(wrong) >= num_samples and len(correct) >= num_samples:
            break

    samples = (wrong + correct)[:num_samples]  # prioritize misclassifications
    if not samples:
        return

    cols = int(math.ceil(math.sqrt(len(samples))))
    rows = int(math.ceil(len(samples) / cols))
    _, H, W = samples[0][0].shape
    pad, cap_h = 4, 16
    cell_w, cell_h = W + 2 * pad, H + cap_h + 2 * pad

    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, (img, pred, true) in enumerate(samples):
        r, c = divmod(idx, cols)
        x0, y0 = c * cell_w, r * cell_h
        color = (0, 160, 0) if pred == true else (210, 0, 0)
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1],
                       outline=color, width=2)
        grid.paste(TF.to_pil_image(img), (x0 + pad, y0 + pad))
        draw.text((x0 + pad, y0 + pad + H + 1),
                  f"p:{class_names[pred]} t:{class_names[true]}",
                  fill=color, font=font)

    dst.mkdir(parents=True, exist_ok=True)
    out = dst / "sample_predictions.png"
    grid.save(out)
    n_wrong = sum(1 for _, p, t in samples if p != t)
    print(f"Saved prediction montage ({len(samples)} samples, "
          f"{n_wrong} misclassified) to {out}")


def set_backbone_frozen(model, frozen):
    """Freeze/unfreeze all params, always keeping the classifier head trainable."""
    for param in model.parameters():
        param.requires_grad = not frozen
    # get_classifier() returns a single module for most models, but a tuple of
    # heads for distillation architectures (e.g. LeViT -> (head, head_dist)).
    head = model.get_classifier()
    heads = head if isinstance(head, (tuple, list)) else (head,)
    for h in heads:
        for param in h.parameters():
            param.requires_grad = True


def build_optimizer_scheduler(model, lr, weight_decay, epochs_in_phase,
                              steps_per_epoch, warmup_ratio):
    """Fresh AdamW + cosine-warmup schedule over the trainable params of a phase."""
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    total_steps  = steps_per_epoch * max(1, epochs_in_phase)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = cosine_warmup_scheduler(optimizer, warmup_steps, total_steps)
    return optimizer, scheduler


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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Peek at classes first so we can build the model with the right head
    train_root = os.path.join(args.data_dir, "train")
    class_names = sorted(entry.name for entry in os.scandir(train_root) if entry.is_dir())
    num_labels  = len(class_names)
    id2label    = {i: c for i, c in enumerate(class_names)}

    # Model (timm pretrained). drop_rate / drop_path_rate add regularization,
    # which matters a lot on small datasets (~800/class).
    print(f"Loading pretrained '{args.model_name}' ({num_labels} classes)...")
    model = timm.create_model(
        args.model_name,
        pretrained=True,
        num_classes=num_labels,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    # Transforms derived from the model's pretrained config
    train_tf, val_tf = build_transforms(model)
    data_cfg = resolve_data_cfg(model)  # mean/std for de-normalizing previews

    # Datasets & loaders
    train_ds = datasets.ImageFolder(train_root, transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Loss (label smoothing helps generalization on small datasets)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Two-phase fine-tuning setup.
    #   Phase 1 (epochs 1..freeze_epochs): backbone frozen, train head only.
    #   Phase 2 (remaining epochs):        backbone unfrozen, fine-tune all.
    # --freeze_backbone freezes for the whole run (no phase 2).
    steps_per_epoch = len(train_loader)
    freeze_epochs = args.epochs if args.freeze_backbone else min(args.freeze_epochs, args.epochs)
    finetune_lr = args.finetune_lr if args.finetune_lr > 0 else args.lr
    two_phase = 0 < freeze_epochs < args.epochs

    # --- Run header ---------------------------------------------------------
    if args.freeze_backbone:
        mode = "head-only (backbone frozen)"
    elif two_phase:
        mode = f"two-phase (freeze {freeze_epochs} ep -> finetune @ lr {finetune_lr:g})"
    else:
        mode = "full fine-tune"
    rule = "=" * 68
    print("\n" + rule)
    print(f" Model    : {args.model_name}  |  {num_labels} classes: {class_names}")
    print(f" Device   : {device}  |  output: {output_dir}")
    print(f" Data     : {len(train_ds)} train  /  {len(val_ds)} val images")
    print(f" Schedule : {args.epochs} epochs, batch {args.batch_size}, "
          f"lr {args.lr:g}, wd {args.weight_decay:g}")
    print(f" Mode     : {mode}")
    if args.patience > 0:
        print(f" Early stop: patience {args.patience}, min_delta {args.min_delta:g}")
    print(rule)

    optimizer = scheduler = None
    best_val_acc, best_epoch = 0.0, 0
    epochs_no_improve = 0
    phase_tag = "full"
    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        # --- Phase transitions: (re)build optimizer/scheduler for the phase ---
        if epoch == 1:
            if freeze_epochs > 0:
                set_backbone_frozen(model, True)
                phase_tag = "head" if args.freeze_backbone else "P1 head"
                phase_epochs = freeze_epochs
                print(f"\n> Phase 1: backbone frozen, head only @ lr {args.lr:g}")
            else:
                set_backbone_frozen(model, False)
                phase_tag = "full"
                phase_epochs = args.epochs
            optimizer, scheduler = build_optimizer_scheduler(
                model, args.lr, args.weight_decay, phase_epochs,
                steps_per_epoch, args.warmup_ratio)
        elif epoch == freeze_epochs + 1 and two_phase:
            set_backbone_frozen(model, False)
            phase_tag = "P2 full"
            phase_epochs = args.epochs - freeze_epochs
            print(f"\n> Phase 2: backbone unfrozen, all layers @ lr {finetune_lr:g}")
            optimizer, scheduler = build_optimizer_scheduler(
                model, finetune_lr, args.weight_decay, phase_epochs,
                steps_per_epoch, args.warmup_ratio)
            # Fresh phase, different LR/schedule: reset the patience counter so a
            # plateau in phase 1 doesn't immediately end phase 2.
            epochs_no_improve = 0

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            epoch, args.epochs)
        val_loss, val_acc, preds, labels = evaluate(
            model, val_loader, criterion, device,
            desc=f"Epoch {epoch}/{args.epochs} [val]")

        improved = val_acc > best_val_acc + args.min_delta
        if improved:
            best_val_acc, best_epoch = val_acc, epoch
            epochs_no_improve = 0
            save_checkpoint(model, id2label, args.model_name, output_dir / "best")
            write_eval_report(preds, labels, class_names, val_loss, val_acc,
                              epoch, output_dir / "best", show=False)
        else:
            epochs_no_improve += 1

        dt = time.perf_counter() - t0
        mark = "  * best" if improved else ""
        print(f"Epoch {epoch:>3}/{args.epochs} [{phase_tag:<7}] {dt:5.1f}s | "
              f"train {train_loss:.4f} / {train_acc * 100:5.1f}% | "
              f"val {val_loss:.4f} / {val_acc * 100:5.1f}%{mark}")

        # Early stopping. Blocked while a backbone unfreeze is still pending, so we
        # never stop the run before phase 2 has had a chance to train.
        pending_unfreeze = two_phase and epoch <= freeze_epochs
        if args.patience > 0 and not pending_unfreeze \
                and epochs_no_improve >= args.patience:
            stopped_early = True
            print(f"\n[early stop] epoch {epoch}: val_acc not improved for "
                  f"{epochs_no_improve} epoch(s) (patience {args.patience}).")
            break

    # Save final (last-epoch) checkpoint; report saved to disk (best is what you deploy).
    save_checkpoint(model, id2label, args.model_name, output_dir / "final")
    write_eval_report(preds, labels, class_names, val_loss, val_acc,
                      epoch, output_dir / "final", show=False)

    print("\n" + rule)
    print(f" Training {'early-stopped' if stopped_early else 'completed'} "
          f"at epoch {epoch}/{args.epochs}")
    print(f" Best val acc : {best_val_acc * 100:.2f}%  (epoch {best_epoch})")
    print(f" Checkpoints  : {output_dir}/best (deploy)  |  {output_dir}/final")
    print(rule)

    # -----------------------------------------------------------------------
    # Post-training inference on a held-out split (the BEST checkpoint).
    # Prefer test/ for an unbiased estimate; fall back to val/ with a warning
    # (val/ was used for model selection, so its numbers are optimistic).
    # -----------------------------------------------------------------------
    test_root = os.path.join(args.data_dir, "test")
    if os.path.isdir(test_root):
        eval_root, eval_split = test_root, "test"
    else:
        eval_root, eval_split = os.path.join(args.data_dir, "val"), "val"
        print("\n[warn] No test/ folder found — running post-training inference on "
              "val/. These numbers are optimistic since val/ drove model selection.")

    print(f"\nPost-training inference on '{eval_split}' split (best checkpoint):")
    best_state = torch.load(output_dir / "best" / "model.pth", map_location=device)
    model.load_state_dict(best_state)

    eval_ds = datasets.ImageFolder(eval_root, transform=val_tf)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    eval_loss, eval_acc, eval_preds, eval_labels = evaluate(
        model, eval_loader, criterion, device, desc=f"{eval_split} inference")
    print()  # separate the full report from the (transient) progress bar
    write_eval_report(eval_preds, eval_labels, class_names, eval_loss, eval_acc,
                      epoch=epoch, dst=output_dir / f"{eval_split}_inference", show=True)
    print(f"\n{eval_split} accuracy: {eval_acc * 100:.2f}%  ->  full report + montage in "
          f"{output_dir / f'{eval_split}_inference'}")

    # Stacked montage of sample predictions for a quick visual sanity check
    save_prediction_grid(model, eval_loader, class_names,
                         data_cfg["mean"], data_cfg["std"], device,
                         dst=output_dir / f"{eval_split}_inference",
                         num_samples=args.num_sample_preds)


if __name__ == "__main__":
    main()
