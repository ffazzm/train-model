# Training Script Examples

This document provides detailed examples for using the training scripts.

## Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
```

## 1. Object Detection with YOLO (train_od.py)

### Basic Training
```bash
python train_od.py --data dataset.yaml --model yolov8n.pt --epochs 100
```

### Advanced Training
```bash
# Train YOLOv8 medium with custom settings
python train_od.py \
    --data ./datasets/coco.yaml \
    --model yolov8m.pt \
    --epochs 150 \
    --batch-size 32 \
    --imgsz 640 \
    --device 0 \
    --project runs/custom_yolo \
    --name exp1 \
    --patience 30 \
    --workers 8
```

### Multi-GPU Training
```bash
# Use multiple GPUs (0, 1, 2)
python train_od.py \
    --data dataset.yaml \
    --model yolov8l.pt \
    --device 0,1,2 \
    --batch-size 48
```

### Resume Training
Ultralytics YOLO automatically saves checkpoints. To resume:
```bash
python train_od.py --data dataset.yaml --model runs/detect/train/weights/last.pt
```

### Dataset Format (YAML)
```yaml
path: ./datasets/my_dataset
train: images/train
val: images/val
nc: 80
names: ['person', 'bicycle', 'car', ...]
```

## 2. Image Classification with timm (train_cls.py)

### Basic Training
```bash
python train_cls.py --data ./data --model resnet50 --epochs 30
```

### Train Different Models
```bash
# ResNet-50
python train_cls.py --data ./data --model resnet50 --epochs 50 --batch-size 64

# EfficientNet-B0
python train_cls.py --data ./data --model efficientnet_b0 --epochs 100 --lr 0.0001

# Vision Transformer
python train_cls.py --data ./data --model vit_base_patch16_224 --epochs 100 --batch-size 32

# MobileNet V3
python train_cls.py --data ./data --model mobilenetv3_large_100 --epochs 50
```

### Advanced Training
```bash
python train_cls.py \
    --data ./datasets/imagenet \
    --model resnet101 \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.01 \
    --imgsz 224 \
    --device cuda \
    --num-workers 8 \
    --output-dir runs/classification \
    --name resnet101_exp1 \
    --pretrained
```

### Dataset Format
Directory structure should be:
```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
└── val/
    ├── class1/
    │   └── ...
    └── class2/
        └── ...
```

## 3. General Deep Learning Models (train_general.py)

### Multi-Layer Perceptron (MLP)
```bash
python train_general.py \
    --model-type mlp \
    --input-size 784 \
    --hidden-size 256 \
    --num-layers 3 \
    --num-classes 10 \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.001
```

### Convolutional Neural Network (CNN)
```bash
python train_general.py \
    --model-type cnn \
    --num-classes 100 \
    --epochs 100 \
    --batch-size 32 \
    --dropout 0.3
```

### Recurrent Neural Network (RNN)
```bash
python train_general.py \
    --model-type rnn \
    --input-size 128 \
    --hidden-size 256 \
    --num-layers 2 \
    --num-classes 5 \
    --epochs 30
```

### Long Short-Term Memory (LSTM)
```bash
python train_general.py \
    --model-type lstm \
    --input-size 300 \
    --hidden-size 512 \
    --num-layers 3 \
    --num-classes 2 \
    --epochs 50 \
    --dropout 0.5
```

## Tips and Best Practices

### Learning Rate
- Start with default (0.001) and adjust based on training behavior
- Use lower learning rates for fine-tuning pretrained models (0.0001)
- Consider learning rate schedulers for better convergence

### Batch Size
- Larger batch sizes: faster training but may need more memory
- Smaller batch sizes: better generalization but slower training
- Adjust based on GPU memory availability

### Epochs
- Object Detection: 100-300 epochs
- Image Classification: 30-100 epochs
- Start with fewer epochs and increase if underfitting

### Device Selection
- `--device cuda` or `--device 0`: Single GPU
- `--device 0,1,2,3`: Multi-GPU
- `--device cpu`: CPU-only (slower)

### Progress Monitoring
All scripts use tqdm to display:
- Current epoch and batch progress
- Training/validation loss
- Training/validation accuracy
- Estimated time remaining

### Model Checkpoints
- train_od.py: Saves to `runs/detect/train/weights/`
- train_cls.py: Saves to `runs/classify/train/best_model.pth`
- train_general.py: Saves to `runs/general/train/`

## Common Issues

### Out of Memory
- Reduce batch size
- Use smaller model variant
- Reduce image size

### Slow Training
- Increase num_workers for data loading
- Use GPU if available
- Reduce image size if appropriate

### Poor Performance
- Increase epochs
- Adjust learning rate
- Use pretrained models when available
- Check data quality and augmentation

## Available Models

### YOLO (Ultralytics)
- yolov8n.pt (nano)
- yolov8s.pt (small)
- yolov8m.pt (medium)
- yolov8l.pt (large)
- yolov8x.pt (xlarge)

### timm Models (700+)
Common architectures:
- ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
- EfficientNet: efficientnet_b0 to efficientnet_b7
- Vision Transformer: vit_tiny_patch16_224, vit_base_patch16_224
- DenseNet: densenet121, densenet169, densenet201
- MobileNet: mobilenetv2_100, mobilenetv3_large_100
- ConvNeXt: convnext_tiny, convnext_small, convnext_base
- Swin Transformer: swin_tiny_patch4_window7_224

Use `timm.list_models()` in Python to see all available models.
