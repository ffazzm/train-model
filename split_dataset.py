"""
Train / Val / Test Split (structure-aware)
==========================================
Split a dataset into train/val/(test) while PRESERVING the layout expected by
the matching training script. The dataset type is auto-detected from the source
structure (or forced with --type):

  classification  (ImageFolder, used by train_image_classification.py)
      source/                         ->   output/
          class_a/ *.jpg                       train/class_a/ ...  val/class_a/ ...
          class_b/ *.jpg                       (test/class_*/  if --test_ratio > 0)

  yolo            (Ultralytics, used by train_yolo.py)
      source/                         ->   output/
          images/ *.jpg                        images/train/ ...  images/val/ ...
          labels/ *.txt                        labels/train/ ...  labels/val/ ...
          data.yaml (optional)                 data.yaml  (path/train/val rewritten)

  paddle          (PaddleOCR det/rec, used by train_paddleocr.py)
      source/                         ->   output/
          <images...>                          train_label.txt  val_label.txt
          label.txt (path<TAB>gt)              (test_label.txt if --test_ratio > 0)
      Only the label file is split (image files stay in place); point the
      training --data_dir at the source images root.

Usage:
    # Auto-detect
    python split_dataset.py --source datasets/raw --output datasets/split --val_ratio 0.15

    # YOLO with a test split
    python split_dataset.py --source datasets/yolo --output datasets/yolo_split \\
        --type yolo --val_ratio 0.1 --test_ratio 0.1

    # PaddleOCR recognition/detection
    python split_dataset.py --source datasets/rec --output datasets/rec \\
        --type paddle --label_file all_label.txt --val_ratio 0.15

Splits are reproducible (--seed); classification is stratified per class.
"""

import argparse
import os
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Structure-aware train/val/test split")
    parser.add_argument("--source", type=str, required=True,
                        help="Source dataset directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Destination directory for the split dataset")
    parser.add_argument("--type", choices=["auto", "classification", "yolo", "paddle"],
                        default="auto", help="Dataset type (auto-detected by default)")
    parser.add_argument("--val_ratio",  type=float, default=0.15,
                        help="Fraction used for validation")
    parser.add_argument("--test_ratio", type=float, default=0.0,
                        help="Fraction used for test (0 = no test split)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible shuffling")
    parser.add_argument("--move", action="store_true",
                        help="Move files instead of copying (classification/yolo only)")
    parser.add_argument("--label_file", type=str, default="",
                        help="PaddleOCR label file (path relative to --source or absolute)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def split_indices(n, val_ratio, test_ratio):
    """Return (n_train, n_val, n_test) counts for a shuffled list of size n."""
    n_val  = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def active_splits(test_ratio):
    return ["train", "val"] + (["test"] if test_ratio > 0 else [])


def bucketize(items, val_ratio, test_ratio):
    """Slice an already-shuffled list into train/val/test buckets."""
    n_train, n_val, _ = split_indices(len(items), val_ratio, test_ratio)
    buckets = {
        "train": items[:n_train],
        "val":   items[n_train:n_train + n_val],
    }
    if test_ratio > 0:
        buckets["test"] = items[n_train + n_val:]
    return buckets


def is_image(name):
    return os.path.splitext(name)[1].lower() in IMG_EXTS


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------

def detect_type(source, label_file):
    if label_file:
        return "paddle"
    if (source / "images").is_dir() and (source / "labels").is_dir():
        return "yolo"
    # A root-level label-list txt (path<TAB>gt) => PaddleOCR
    for txt in source.glob("*.txt"):
        try:
            with open(txt, encoding="utf-8") as f:
                first = f.readline()
            if "\t" in first:
                return "paddle"
        except (UnicodeDecodeError, OSError):
            continue
    # Subfolders that directly contain images => classification
    subdirs = [e for e in os.scandir(source) if e.is_dir()]
    if subdirs and any(
        any(is_image(f) for f in os.listdir(e.path)) for e in subdirs
    ):
        return "classification"
    raise ValueError(
        f"Could not auto-detect dataset type from {source}. "
        f"Use --type {{classification,yolo,paddle}}."
    )


# ---------------------------------------------------------------------------
# Classification (ImageFolder)
# ---------------------------------------------------------------------------

def split_classification(source, output, args, rng, transfer):
    class_names = sorted(e.name for e in os.scandir(source) if e.is_dir())
    if not class_names:
        raise ValueError(f"No class subfolders found in {source}")

    splits = active_splits(args.test_ratio)
    totals = {s: 0 for s in splits}
    print(f"[classification] classes ({len(class_names)}): {class_names}\n")

    for cls in class_names:
        images = sorted(f for f in os.listdir(source / cls)
                        if is_image(f) and (source / cls / f).is_file())
        if not images:
            print(f"  [skip] {cls}: no images")
            continue
        rng.shuffle(images)
        buckets = bucketize(images, args.val_ratio, args.test_ratio)

        for split, files in buckets.items():
            dst_dir = output / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for fname in files:
                transfer(str(source / cls / fname), str(dst_dir / fname))
            totals[split] += len(files)

        counts = "  ".join(f"{s}={len(buckets[s])}" for s in splits)
        print(f"  {cls:20s} total={len(images):5d} | {counts}")

    return totals, splits


# ---------------------------------------------------------------------------
# YOLO (images/ + labels/)
# ---------------------------------------------------------------------------

def split_yolo(source, output, args, rng, transfer):
    img_root = source / "images"
    lbl_root = source / "labels"
    if not img_root.is_dir():
        raise ValueError(f"YOLO layout expected {img_root} (images/) to exist")

    # Collect images (recursively) with their relative path under images/
    images = sorted(p.relative_to(img_root) for p in img_root.rglob("*")
                    if p.is_file() and is_image(p.name))
    if not images:
        raise ValueError(f"No images found under {img_root}")

    rng.shuffle(images)
    buckets = bucketize(images, args.val_ratio, args.test_ratio)
    splits = active_splits(args.test_ratio)
    totals = {s: 0 for s in splits}
    missing_labels = 0

    for split, rels in buckets.items():
        for rel in rels:
            # image
            dst_img = output / "images" / split / rel
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            transfer(str(img_root / rel), str(dst_img))
            # matching label (same relative path, .txt)
            rel_txt = rel.with_suffix(".txt")
            src_lbl = lbl_root / rel_txt
            if src_lbl.is_file():
                dst_lbl = output / "labels" / split / rel_txt
                dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                transfer(str(src_lbl), str(dst_lbl))
            else:
                missing_labels += 1
        totals[split] = len(rels)
        print(f"  {split:5s}: {len(rels)} images")

    if missing_labels:
        print(f"  [warn] {missing_labels} image(s) had no matching label .txt "
              f"(kept as background/unlabeled)")

    _write_yolo_yaml(source, output, splits)
    return totals, splits


def _write_yolo_yaml(source, output, splits):
    """Copy data.yaml if present and rewrite path/train/val/test to the split."""
    src_yaml = next((source / n for n in ("data.yaml", "dataset.yaml")
                     if (source / n).is_file()), None)
    try:
        import yaml
    except ImportError:
        if src_yaml:
            print("  [warn] PyYAML not installed; data.yaml not rewritten")
        return

    cfg = {}
    if src_yaml:
        with open(src_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    cfg["path"] = str(output.resolve())
    cfg["train"] = "images/train"
    cfg["val"] = "images/val"
    if "test" in splits:
        cfg["test"] = "images/test"
    elif "test" in cfg:
        del cfg["test"]

    with open(output / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  data.yaml written to {output / 'data.yaml'}"
          + ("" if src_yaml else " (nc/names not set — fill these in)"))


# ---------------------------------------------------------------------------
# PaddleOCR (split the label list)
# ---------------------------------------------------------------------------

def split_paddle(source, output, args, rng):
    if not args.label_file:
        raise ValueError("--label_file is required for PaddleOCR datasets "
                          "(e.g. --label_file all_label.txt)")
    label_path = Path(args.label_file)
    if not label_path.is_absolute():
        label_path = source / label_path
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    with open(label_path, encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Label file is empty: {label_path}")

    rng.shuffle(lines)
    buckets = bucketize(lines, args.val_ratio, args.test_ratio)
    splits = active_splits(args.test_ratio)
    totals = {s: 0 for s in splits}

    output.mkdir(parents=True, exist_ok=True)
    for split, entries in buckets.items():
        out_file = output / f"{split}_label.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(entries) + "\n")
        totals[split] = len(entries)
        print(f"  {split:5s}: {len(entries)} entries -> {out_file}")

    print("  [note] image files were not moved; point training --data_dir at "
          "the images root (paths in the label file are relative to it).")
    return totals, splits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not 0 <= args.val_ratio < 1 or not 0 <= args.test_ratio < 1:
        raise ValueError("val_ratio and test_ratio must each be in [0, 1)")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be < 1 (need data left for train)")

    source = Path(args.source)
    output = Path(args.output)
    if not source.is_dir():
        raise FileNotFoundError(f"Source dir not found: {source}")

    ds_type = args.type if args.type != "auto" else detect_type(source, args.label_file)
    rng = random.Random(args.seed)
    transfer = shutil.move if args.move else shutil.copy2

    print(f"Detected type : {ds_type}")
    print(f"Ratios        : val={args.val_ratio}  test={args.test_ratio}  "
          f"train={1 - args.val_ratio - args.test_ratio:.2f}")
    print(f"Mode          : {'MOVE' if args.move else 'COPY'}\n")

    if ds_type == "classification":
        totals, splits = split_classification(source, output, args, rng, transfer)
    elif ds_type == "yolo":
        totals, splits = split_yolo(source, output, args, rng, transfer)
    else:  # paddle
        totals, splits = split_paddle(source, output, args, rng)

    print("\nDone. Totals:")
    for s in splits:
        print(f"  {s:5s}: {totals[s]}")
    print(f"\nOutput written to: {output}")


if __name__ == "__main__":
    main()
