"""
Near / Exact Duplicate Image Finder
===================================
Find visually identical or near-identical images in one or more folders. Built
for two jobs on the accident dataset:

  1. Leakage detection — the same source frame (or near-identical crops of it)
     landing in more than one split (train/val/test). Leaked val/test images
     inflate your metrics and hide real-world failure (e.g. video false positives).
  2. De-duplication / label-conflict — exact duplicates wasting capacity, or the
     same image filed under two different class folders (label noise).

How it works (no extra dependencies — PIL + numpy only):
  * dHash (difference hash) gives a compact perceptual fingerprint. Two images
    are "near-duplicate" if their hashes differ in <= --threshold bits.
  * SHA-1 of the raw file bytes catches byte-for-byte exact duplicates.
  * Images are grouped with union-find; each group is checked for whether it
    spans multiple splits (LEAKAGE) or multiple classes (LABEL CONFLICT).

Usage:
    # Scan the split dataset and report cross-split leakage
    python find_similar_images.py datasets/dataset_accident --threshold 5

    # Only show groups that span more than one split (the leakage)
    python find_similar_images.py datasets/dataset_accident --leakage-only

    # Compare a folder of new video frames against the training set
    python find_similar_images.py datasets/dataset_accident/train new_frames/

    # Copy all similar images into a new dir (one subfolder per group) to review
    python find_similar_images.py datasets/dataset_accident \\
        --threshold 5 --copy-dir datasets/_similar

    # Same, but only the cross-split (leakage) groups
    python find_similar_images.py datasets/dataset_accident \\
        --leakage-only --copy-dir datasets/_leakage

    # Remove leaked/near-dup images from val/test only (train never touched).
    # Preview first with --dry-run, then run for real:
    python find_similar_images.py datasets/dataset_accident \\
        --threshold 5 --remove-eval-dupes --dry-run
    python find_similar_images.py datasets/dataset_accident \\
        --threshold 5 --remove-eval-dupes

    # Save a full JSON report, and quarantine redundant near-dupes (keeps one
    # copy per group, preferring the train/ copy so val/test leaks get moved out)
    python find_similar_images.py datasets/dataset_accident \\
        --threshold 5 --output dup_report.json \\
        --quarantine-dir datasets/_quarantine

Threshold guide (64-bit hash, hash-size 8):
    0      = perceptually identical (same frame, maybe re-encoded/resized)
    1-5    = near-identical (adjacent video frames, tiny crops/shifts)  [default 5]
    6-10   = similar scene; expect some false matches
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:                       # tqdm is optional
    def tqdm(x, **kwargs):
        return x

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = {"train", "val", "validation", "valid", "eval", "test"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Find near/exact duplicate images (leakage + de-dup)")
    p.add_argument("paths", nargs="+",
                   help="One or more directories to scan (searched recursively)")
    p.add_argument("--threshold", type=int, default=5,
                   help="Max Hamming distance (bits) to treat two images as "
                        "near-duplicate. 0 = perceptually identical only (default 5)")
    p.add_argument("--hash-size", type=int, default=8,
                   help="dHash size; bits = hash_size^2 (default 8 => 64-bit)")
    p.add_argument("--output", type=str, default="",
                   help="Write a full JSON report to this path")
    p.add_argument("--limit", type=int, default=40,
                   help="Max number of groups to print to console (default 40)")
    p.add_argument("--leakage-only", action="store_true",
                   help="Only report groups that span more than one split")
    p.add_argument("--copy-dir", type=str, default="",
                   help="COPY every image in each near-dup group into this dir "
                        "(one subfolder per group, originals untouched) — good for "
                        "eyeballing the matches. Respects --leakage-only.")
    p.add_argument("--quarantine-dir", type=str, default="",
                   help="MOVE redundant near-dupes here (keeps one per group). "
                        "Prefers keeping the train/ copy so val/test leaks move out.")
    p.add_argument("--remove-eval-dupes", action="store_true",
                   help="DELETE near-dup images that live in val/test, keeping the "
                        "train copy (or one eval copy if the group has no train "
                        "member). Never deletes train images. Respects --leakage-only. "
                        "Use --dry-run to preview first.")
    p.add_argument("--dry-run", action="store_true",
                   help="With --remove-eval-dupes: print what would be deleted "
                        "without actually deleting.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def dhash_bits(path, hash_size=8):
    """Difference hash -> flat bool array of length hash_size*hash_size."""
    img = Image.open(path).convert("L").resize(
        (hash_size + 1, hash_size), Image.LANCZOS)
    px = np.asarray(img, dtype=np.int16)
    diff = px[:, 1:] > px[:, :-1]          # compare adjacent columns
    return diff.flatten()


def sha1_of_file(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def infer_split_class(path, root):
    """Derive (split, class) from a path relative to the scan root.

    Layout <root>/<split>/<class>/img -> ("train", "accident").
    Layout <root>/<class>/img         -> (None, "accident").
    """
    try:
        parts = Path(path).relative_to(root).parts
    except ValueError:
        parts = Path(path).parts
    split = parts[0].lower() if len(parts) >= 2 and parts[0].lower() in SPLIT_NAMES else None
    cls = parts[-2] if len(parts) >= 2 else None
    return split, cls


# ---------------------------------------------------------------------------
# Union-find
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_images(paths):
    """Return list of (abs_path, root) for every image under the given roots."""
    items = []
    for raw in paths:
        root = Path(raw).resolve()
        if not root.is_dir():
            print(f"[warn] not a directory, skipping: {root}")
            continue
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                    items.append((os.path.join(dirpath, fn), root))
    return items


def main():
    args = parse_args()

    items = collect_images(args.paths)
    if not items:
        print("No images found.")
        return
    print(f"Scanning {len(items)} images from {len(args.paths)} path(s)...")

    # Fingerprint every image; drop unreadable ones.
    records, bits = [], []
    for path, root in tqdm(items, desc="Hashing"):
        try:
            b = dhash_bits(path, args.hash_size)
        except Exception as e:                       # corrupt / unreadable
            print(f"[skip] {path}: {e}")
            continue
        split, cls = infer_split_class(path, root)
        records.append({"path": path, "root": str(root),
                        "split": split, "class": cls,
                        "sha1": sha1_of_file(path)})
        bits.append(b)

    n = len(records)
    if n < 2:
        print("Fewer than 2 readable images — nothing to compare.")
        return
    H = np.asarray(bits, dtype=np.uint8)             # (n, bits)

    # Pairwise near-duplicate grouping (row-by-row to keep memory light).
    uf = UnionFind(n)
    for i in tqdm(range(n), desc="Comparing"):
        dist = (H[i] != H).sum(axis=1)               # Hamming distance to all
        for j in np.nonzero(dist <= args.threshold)[0]:
            if j > i:
                uf.union(i, int(j))

    # Assemble groups of size >= 2.
    groups = {}
    for idx in range(n):
        groups.setdefault(uf.find(idx), []).append(idx)
    groups = [g for g in groups.values() if len(g) > 1]

    def summarize(members):
        splits = sorted({records[i]["split"] for i in members if records[i]["split"]})
        classes = sorted({records[i]["class"] for i in members if records[i]["class"]})
        sha1s = {records[i]["sha1"] for i in members}
        return {
            "size": len(members),
            "cross_split": len(splits) > 1,
            "cross_class": len(classes) > 1,
            "byte_identical": len(sha1s) == 1,
            "splits": splits,
            "classes": classes,
            "members": [{
                "path": records[i]["path"],
                "split": records[i]["split"],
                "class": records[i]["class"],
                "sha1": records[i]["sha1"][:10],
            } for i in members],
        }

    summaries = [summarize(g) for g in groups]
    # Leakage groups first, then bigger groups.
    summaries.sort(key=lambda s: (not s["cross_split"], not s["cross_class"], -s["size"]))

    leak = [s for s in summaries if s["cross_split"]]
    conflict = [s for s in summaries if s["cross_class"]]
    n_in_groups = sum(s["size"] for s in summaries)

    print("\n" + "=" * 70)
    print(f"Images scanned          : {n}")
    print(f"Duplicate/near-dup groups: {len(summaries)}  "
          f"({n_in_groups} images, {n_in_groups - len(summaries)} redundant)")
    print(f"  -> CROSS-SPLIT (LEAKAGE): {len(leak)}")
    print(f"  -> CROSS-CLASS (CONFLICT): {len(conflict)}")
    print(f"Threshold (Hamming bits)  : {args.threshold}")
    print("=" * 70)

    shown = leak if args.leakage_only else summaries
    for s in shown[:args.limit]:
        tags = []
        if s["cross_split"]:
            tags.append("LEAKAGE")
        if s["cross_class"]:
            tags.append("LABEL-CONFLICT")
        if s["byte_identical"]:
            tags.append("EXACT")
        tag = ("  [" + ", ".join(tags) + "]") if tags else ""
        print(f"\nGroup of {s['size']}  splits={s['splits']} classes={s['classes']}{tag}")
        for mem in s["members"]:
            rel = mem["path"]
            print(f"    {mem['split'] or '-':>10} / {mem['class'] or '-':<10}  {rel}")
    if len(shown) > args.limit:
        print(f"\n... {len(shown) - args.limit} more group(s) not shown "
              f"(raise --limit or use --output).")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "threshold": args.threshold,
                "hash_size": args.hash_size,
                "num_images": n,
                "num_groups": len(summaries),
                "num_leakage_groups": len(leak),
                "num_label_conflict_groups": len(conflict),
                "groups": summaries,
            }, f, indent=2)
        print(f"\nFull JSON report written to {args.output}")

    if args.copy_dir:
        copy_groups(shown, args.copy_dir)

    if args.remove_eval_dupes:
        remove_eval_dupes(shown, args.dry_run)

    if args.quarantine_dir:
        quarantine(summaries, args.quarantine_dir)


def _group_tags(s):
    """Short tag list (LEAKAGE / CONFLICT / EXACT) used in folder names."""
    tags = []
    if s["cross_split"]:
        tags.append("LEAKAGE")
    if s["cross_class"]:
        tags.append("CONFLICT")
    if s["byte_identical"]:
        tags.append("EXACT")
    return tags


def copy_groups(summaries, copy_dir):
    """Copy every image in each near-dup group into copy_dir, one subfolder per
    group, so matching images sit side by side for visual review.

    Non-destructive: originals are left in place. Copied filenames are prefixed
    with <split>_<class>_ so provenance is visible at a glance.
    """
    croot = Path(copy_dir)
    copied = 0
    for gi, s in enumerate(summaries):
        tags = _group_tags(s)
        suffix = ("_" + "_".join(tags)) if tags else ""
        gdir = croot / f"group_{gi:04d}{suffix}"
        gdir.mkdir(parents=True, exist_ok=True)
        for m in s["members"]:
            src = Path(m["path"])
            if not src.exists():
                continue
            name = f"{m['split'] or 'na'}_{m['class'] or 'na'}_{src.name}"
            dst = gdir / name
            if dst.exists():                       # collision -> disambiguate
                dst = gdir / f"{src.stem}_{m['sha1']}{src.suffix}"
            shutil.copy2(src, dst)
            copied += 1
    print(f"\nCopied {copied} image(s) from {len(summaries)} group(s) to {croot} "
          f"(one subfolder per group; originals untouched).")


def remove_eval_dupes(summaries, dry_run=False):
    """Delete near-dup images that live in val/test, keeping the train copy.

    Policy per group:
      * Group has a train member  -> remove ALL its val/test members (they are
        near-duplicates of a training image, i.e. leakage).
      * Group has no train member -> keep one eval copy, remove the rest
        (de-duplicate within val/test).
    Train images are never deleted. Use dry_run=True to preview.
    """
    EVAL = {"val", "validation", "valid", "eval", "test"}
    removed = 0
    for s in summaries:
        members = s["members"]
        rep = next((m for m in members if m["split"] == "train"), members[0])
        for m in members:
            if m is rep or (m["split"] or "") not in EVAL:
                continue
            path = m["path"]
            if dry_run:
                print(f"[dry-run] would remove: {path}")
            else:
                try:
                    os.remove(path)
                    print(f"removed: {path}")
                except FileNotFoundError:
                    pass
            removed += 1
    verb = "Would remove" if dry_run else "Removed"
    print(f"\n{verb} {removed} val/test near-duplicate image(s) from "
          f"{len(summaries)} group(s). Train images untouched.")


def quarantine(summaries, quarantine_dir):
    """Move redundant near-dupes out, keeping one representative per group.

    Representative preference: keep a train/ copy if the group has one (so the
    leaked val/test copies are the ones moved out); otherwise keep the first.
    Original folder structure is mirrored under the quarantine dir.
    """
    qroot = Path(quarantine_dir)
    moved = 0
    for s in summaries:
        members = s["members"]
        keep = next((m for m in members if m["split"] == "train"), members[0])
        for m in members:
            if m is keep:
                continue
            src = Path(m["path"])
            if not src.exists():
                continue
            dst = qroot / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            # avoid clobbering same-name files from different folders
            if dst.exists():
                dst = qroot / f"{src.stem}_{m['sha1']}{src.suffix}"
            os.rename(src, dst)
            moved += 1
    print(f"\nQuarantined {moved} redundant image(s) to {qroot} "
          f"(kept 1 per group, preferred train/).")


if __name__ == "__main__":
    main()
