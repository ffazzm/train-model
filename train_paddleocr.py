"""
PaddleOCR Fine-tuning Script
============================
Fine-tune PaddleOCR components:
  - Text Detection  (DB / EAST)
  - Text Recognition (CRNN / SVTR)

This script wraps PaddlePaddle's tools/ CLI with Python for convenience.
It downloads a pretrained model, patches the config, and launches training.

Prerequisites:
    pip install paddlepaddle-gpu paddleocr

Dataset structure
-----------------
Detection (icdar-style):
    det_dataset/
        train_images/   *.jpg
        train_label.txt        # path\tJSON-polygon-list
        val_images/     *.jpg
        val_label.txt

Recognition:
    rec_dataset/
        train/   *.jpg
        train_label.txt        # relative_path\tlabel
        val/     *.jpg
        val_label.txt

Usage:
    # Detection fine-tune
    python train_paddleocr.py det \\
        --data_dir det_dataset/ \\
        --pretrained_model en_PP-OCRv4_det

    # Recognition fine-tune
    python train_paddleocr.py rec \\
        --data_dir rec_dataset/ \\
        --pretrained_model en_PP-OCRv4_rec \\
        --char_dict_path dict/en_dict.txt
"""

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Pretrained model registry (PaddleOCR model hub URLs)
# ---------------------------------------------------------------------------
PRETRAINED_CONFIGS = {
    # Detection
    "en_PP-OCRv4_det": {
        "type": "det",
        "config": "configs/det/PP-OCRv4/en_PP-OCRv4_det_student.yml",
        "pretrain_url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_det_train.tar",
    },
    "ch_PP-OCRv4_det": {
        "type": "det",
        "config": "configs/det/PP-OCRv4/ch_PP-OCRv4_det_student.yml",
        "pretrain_url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar",
    },
    # Recognition
    "en_PP-OCRv4_rec": {
        "type": "rec",
        "config": "configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml",
        "pretrain_url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar",
    },
    "ch_PP-OCRv4_rec": {
        "type": "rec",
        "config": "configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml",
        "pretrain_url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_and_extract(url: str, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / Path(url).name
    if not tar_path.exists():
        print(f"Downloading pretrained weights from {url} ...")
        urllib.request.urlretrieve(url, tar_path)
    print(f"Extracting {tar_path} ...")
    subprocess.run(["tar", "-xf", str(tar_path), "-C", str(dest_dir)], check=True)
    # Return extracted directory (first subdir)
    for p in dest_dir.iterdir():
        if p.is_dir():
            return p
    return dest_dir


def get_paddleocr_root() -> Path:
    """Locate the PaddleOCR repo root (needed for config files)."""
    try:
        import paddleocr
        return Path(paddleocr.__file__).parent
    except ImportError:
        sys.exit("paddleocr not installed. Run: pip install paddleocr")


def patch_yaml(cfg_path: Path, overrides: dict) -> Path:
    """Deep-merge overrides into a YAML config and write a new file."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    def deep_set(d, keys, val):
        k = keys[0]
        if len(keys) == 1:
            d[k] = val
        else:
            if k not in d:
                d[k] = {}
            deep_set(d[k], keys[1:], val)

    for dotted_key, val in overrides.items():
        deep_set(cfg, dotted_key.split("."), val)

    out_path = cfg_path.parent / ("patched_" + cfg_path.name)
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return out_path


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def train_det(args):
    registry  = PRETRAINED_CONFIGS[args.pretrained_model]
    ocr_root  = get_paddleocr_root()
    cfg_path  = ocr_root / registry["config"]

    pretrain_dir = Path(args.output_dir) / "pretrained"
    weights_dir  = download_and_extract(registry["pretrain_url"], pretrain_dir)

    overrides = {
        "Global.pretrained_model":          str(weights_dir / "best_accuracy"),
        "Global.save_model_dir":            str(Path(args.output_dir) / "det_output"),
        "Global.epoch_num":                 args.epochs,
        "Global.save_epoch_step":           args.save_epoch_step,
        "Train.dataset.data_dir":           args.data_dir,
        "Train.dataset.label_file_list":    [os.path.join(args.data_dir, "train_label.txt")],
        "Train.loader.batch_size_per_card": args.batch_size,
        "Eval.dataset.data_dir":            args.data_dir,
        "Eval.dataset.label_file_list":     [os.path.join(args.data_dir, "val_label.txt")],
        "Eval.loader.batch_size_per_card":  args.batch_size,
    }
    patched_cfg = patch_yaml(cfg_path, overrides)
    print(f"Patched config written to: {patched_cfg}")

    cmd = [
        sys.executable, "-m", "paddle.distributed.launch",
        "--gpus", args.gpus,
        str(ocr_root / "tools" / "train.py"),
        "-c", str(patched_cfg),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

def train_rec(args):
    registry  = PRETRAINED_CONFIGS[args.pretrained_model]
    ocr_root  = get_paddleocr_root()
    cfg_path  = ocr_root / registry["config"]

    pretrain_dir = Path(args.output_dir) / "pretrained"
    weights_dir  = download_and_extract(registry["pretrain_url"], pretrain_dir)

    overrides = {
        "Global.pretrained_model":                  str(weights_dir / "best_accuracy"),
        "Global.save_model_dir":                    str(Path(args.output_dir) / "rec_output"),
        "Global.epoch_num":                         args.epochs,
        "Global.save_epoch_step":                   args.save_epoch_step,
        "Train.dataset.data_dir":                   args.data_dir,
        "Train.dataset.label_file_list":            [os.path.join(args.data_dir, "train_label.txt")],
        "Train.loader.batch_size_per_card":         args.batch_size,
        "Eval.dataset.data_dir":                    args.data_dir,
        "Eval.dataset.label_file_list":             [os.path.join(args.data_dir, "val_label.txt")],
        "Eval.loader.batch_size_per_card":          args.batch_size,
    }
    if args.char_dict_path:
        overrides["Global.character_dict_path"] = args.char_dict_path

    patched_cfg = patch_yaml(cfg_path, overrides)
    print(f"Patched config written to: {patched_cfg}")

    cmd = [
        sys.executable, "-m", "paddle.distributed.launch",
        "--gpus", args.gpus,
        str(ocr_root / "tools" / "train.py"),
        "-c", str(patched_cfg),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune PaddleOCR models")
    sub = parser.add_subparsers(dest="task", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data_dir",         type=str, required=True)
    common.add_argument("--pretrained_model",  type=str, required=True,
                        choices=list(PRETRAINED_CONFIGS.keys()))
    common.add_argument("--output_dir",        type=str, default="runs/paddleocr")
    common.add_argument("--epochs",            type=int, default=100)
    common.add_argument("--batch_size",        type=int, default=8)
    common.add_argument("--save_epoch_step",   type=int, default=10)
    common.add_argument("--gpus",              type=str, default="0",
                        help="GPU IDs for distributed launch, e.g. '0,1'")

    det_p = sub.add_parser("det", parents=[common], help="Text detection fine-tune")

    rec_p = sub.add_parser("rec", parents=[common], help="Text recognition fine-tune")
    rec_p.add_argument("--char_dict_path", type=str, default="",
                       help="Path to character dictionary file")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task == "det":
        train_det(args)
    elif args.task == "rec":
        train_rec(args)
