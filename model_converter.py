"""
Convert a trained timm classifier checkpoint to ONNX.
=====================================================
Reads a checkpoint directory produced by train_image_classification.py:

    <checkpoint_dir>/
        model.pth      # state_dict
        config.json    # {"model_name", "num_labels", "id2label"}

and exports an ONNX model with the correct architecture, head size, and
input resolution (pulled from the model's pretrained data config).

Usage:
    python model_converter.py --checkpoint_dir runs/accident_effb0/best
    python model_converter.py --checkpoint_dir runs/accident_effb0/best \\
        --output model.onnx --opset 15

    # Also verify PyTorch-vs-ONNX drift on a held-out split:
    python model_converter.py --checkpoint_dir runs/accident_effb0/best \\
        --val_dir dataset/val
"""

import argparse
import json
from pathlib import Path

import timm
import torch


def get_data_config(model):
    """Return the model's pretrained data config (input_size, mean, std), across timm versions."""
    try:  # timm >= 1.0
        from timm.data import resolve_model_data_config
        return resolve_model_data_config(model)
    except ImportError:  # timm < 1.0
        from timm.data import resolve_data_config
        return resolve_data_config({}, model=model)


def get_input_size(cfg):
    return cfg["input_size"][-1]  # (C, H, W) -> H/W


def parse_args():
    p = argparse.ArgumentParser(description="Convert trained timm checkpoint to ONNX")
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="Dir with model.pth and config.json (e.g. runs/xxx/best)")
    p.add_argument("--output", type=str, default="",
                   help="Output .onnx path (default: <checkpoint_dir>/model.onnx)")
    p.add_argument("--opset", type=int, default=15, help="ONNX opset version")
    p.add_argument("--img_size", type=int, default=0,
                   help="Override input size (default: from model's data config)")
    p.add_argument("--device", type=str, default="cpu", help="cpu / cuda")
    p.add_argument("--val_dir", type=str, default="",
                   help="Optional folder (ImageFolder layout) to verify PyTorch-vs-ONNX "
                        "drift on, e.g. dataset/val or dataset/test")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for drift verify")
    p.add_argument("--verify_batches", type=int, default=0,
                   help="Limit drift verify to first N batches (0 = whole set)")
    return p.parse_args()


def verify_drift(torch_model, onnx_path, val_dir, cfg, device,
                 batch_size=32, max_batches=0, num_workers=4):
    """Run PyTorch and ONNX on the same images; report numerical + accuracy drift.

    Confirms the export is faithful: logit difference should be ~1e-4 or smaller,
    argmax agreement ~100%, and ONNX accuracy should match the PyTorch model.
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("(install 'onnxruntime' to run drift verification)")
        return
    from timm.data import create_transform
    from torch.utils.data import DataLoader
    from torchvision import datasets

    val_tf = create_transform(**cfg, is_training=False)  # same preprocessing as training/val
    ds = datasets.ImageFolder(val_dir, transform=val_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    torch_model.eval()
    n = agree = torch_correct = onnx_correct = 0
    max_abs = 0.0
    with torch.no_grad():
        for bi, (images, labels) in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            t_logits = torch_model(images.to(device)).cpu()
            o_logits = torch.from_numpy(
                sess.run(None, {in_name: images.numpy().astype(np.float32)})[0])

            max_abs = max(max_abs, (t_logits - o_logits).abs().max().item())
            t_pred, o_pred = t_logits.argmax(1), o_logits.argmax(1)
            agree         += (t_pred == o_pred).sum().item()
            torch_correct += (t_pred == labels).sum().item()
            onnx_correct  += (o_pred == labels).sum().item()
            n += labels.size(0)

    if n == 0:
        print(f"[warn] No images found in {val_dir}")
        return
    print(f"\nDrift verification on {n} images ({val_dir}):")
    print(f"  max |logit diff| (torch vs onnx): {max_abs:.3e}   (want < 1e-3)")
    print(f"  argmax agreement:                 {agree}/{n} = {agree / n:.4f}")
    print(f"  PyTorch accuracy:                 {torch_correct / n:.4f}")
    print(f"  ONNX accuracy:                    {onnx_correct / n:.4f}")
    print(f"  accuracy drift (onnx - torch):    {(onnx_correct - torch_correct) / n:+.4f}")


def main():
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    cfg_path = ckpt_dir / "config.json"
    weights_path = ckpt_dir / "model.pth"

    if not cfg_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Expected {cfg_path} and {weights_path}. "
            "Point --checkpoint_dir at a 'best'/'final' folder from training.")

    with open(cfg_path) as f:
        cfg = json.load(f)
    model_name = cfg["model_name"]
    num_labels = cfg["num_labels"]
    id2label   = cfg.get("id2label", {})
    print(f"Loading '{model_name}' with {num_labels} classes: "
          f"{list(id2label.values()) if id2label else '(labels unavailable)'}")

    device = torch.device(args.device)

    # 1. Rebuild the exact architecture (pretrained=False — we load our weights).
    model = timm.create_model(model_name, pretrained=False, num_classes=num_labels)
    try:  # weights_only=True is safe (state_dict is pure tensors); older torch lacks the arg
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # 2. Dummy input matching the model's expected resolution.
    data_cfg = get_data_config(model)
    size = args.img_size if args.img_size > 0 else get_input_size(data_cfg)
    dummy_input = torch.randn(1, 3, size, size, device=device)
    print(f"Exporting with input shape (1, 3, {size}, {size})")

    # 3. Export to ONNX (dynamic batch dimension).
    #    dynamo=False forces the legacy TorchScript exporter, which honors the
    #    requested opset + dynamic_axes cleanly. Newer torch defaults to the dynamo
    #    exporter (opset 18) and fails to down-convert to lower opsets. Fall back if
    #    this torch version doesn't accept the 'dynamo' kwarg.
    out_path = Path(args.output) if args.output else ckpt_dir / "model.onnx"
    export_kwargs = dict(
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    try:
        torch.onnx.export(model, dummy_input, str(out_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy_input, str(out_path), **export_kwargs)
    print(f"Saved ONNX model to: {out_path}")

    # 4. Sanity check the exported graph if onnx is installed.
    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(out_path)))
        print("ONNX model check passed.")
    except ImportError:
        print("(install 'onnx' to validate the exported graph)")

    # 5. Optional drift verification against the original PyTorch model.
    if args.val_dir:
        verify_drift(model, out_path, args.val_dir, data_cfg, device,
                     batch_size=args.batch_size, max_batches=args.verify_batches)


if __name__ == "__main__":
    main()
