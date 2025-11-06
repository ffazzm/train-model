# Train Model Examples

This repository provides examples for training Deep Learning models with CLI interface.

## Features

- **Object Detection Training** (`train_od.py`) - YOLO models using Ultralytics
- **Image Classification Training** (`train_cls.py`) - Models using timm (PyTorch Image Models)
- **General DL Training** (`train_general.py`) - Template for custom deep learning models
- Progress bars with tqdm for better training visibility
- Command-line interface for easy usage

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Object Detection (YOLO)

Train YOLO object detection models:

```bash
python train_od.py --data dataset.yaml --model yolov8n.pt --epochs 100 --batch-size 16
```

**Arguments:**
- `--data`: Path to dataset YAML file (required)
- `--model`: YOLO model variant (default: yolov8n.pt)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--imgsz`: Input image size (default: 640)
- `--device`: Device to use (e.g., 0, cpu)
- `--project`: Project directory (default: runs/detect)
- `--name`: Experiment name (default: train)

**Example:**
```bash
# Train YOLOv8 nano on custom dataset
python train_od.py --data ./data/coco.yaml --model yolov8n.pt --epochs 50 --batch-size 32

# Train YOLOv8 medium with GPU 0
python train_od.py --data ./data/coco.yaml --model yolov8m.pt --device 0 --epochs 100
```

### Image Classification

Train image classification models using timm:

```bash
python train_cls.py --data ./data --model resnet50 --epochs 30 --batch-size 32
```

**Arguments:**
- `--data`: Path to dataset directory with train/ and val/ subdirectories (required)
- `--model`: Model architecture from timm (default: resnet50)
- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--imgsz`: Input image size (default: 224)
- `--device`: Device to use (default: auto)
- `--output-dir`: Output directory (default: runs/classify)
- `--name`: Experiment name (default: train)

**Example:**
```bash
# Train ResNet50 on custom dataset
python train_cls.py --data ./datasets/imagenet --model resnet50 --epochs 50

# Train EfficientNet-B0
python train_cls.py --data ./datasets/flowers --model efficientnet_b0 --epochs 30 --lr 0.0001

# Train Vision Transformer
python train_cls.py --data ./datasets/cifar10 --model vit_base_patch16_224 --epochs 100
```

### General Deep Learning Models

Template for training custom deep learning models:

```bash
python train_general.py --model-type mlp --input-size 784 --num-classes 10 --epochs 50
```

**Arguments:**
- `--model-type`: Type of model (mlp, cnn, rnn, lstm, transformer)
- `--input-size`: Input size for the model (default: 784)
- `--hidden-size`: Hidden layer size (default: 128)
- `--num-layers`: Number of hidden layers (default: 2)
- `--num-classes`: Number of output classes (default: 10)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--device`: Device to use (default: auto)

**Example:**
```bash
# Train MLP
python train_general.py --model-type mlp --input-size 784 --num-classes 10

# Train CNN
python train_general.py --model-type cnn --num-classes 100

# Train LSTM
python train_general.py --model-type lstm --input-size 128 --hidden-size 256 --num-layers 3
```

## Supported Models

### Object Detection
- YOLOv8 (nano, small, medium, large, xlarge) via Ultralytics

### Image Classification
- ResNet family (resnet18, resnet34, resnet50, resnet101, resnet152)
- EfficientNet family (efficientnet_b0 to efficientnet_b7)
- Vision Transformers (vit_base_patch16_224, vit_large_patch16_224)
- DenseNet family (densenet121, densenet169, densenet201)
- MobileNet family (mobilenetv2_100, mobilenetv3_large_100)
- And 700+ more models from timm library

### General Models
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)

## Progress Bars

All training scripts use tqdm for progress visualization during training. The scripts automatically display:
- Current epoch progress
- Training/validation loss
- Training/validation accuracy
- Estimated time remaining

## Dataset Format

### Object Detection (YOLO)
Requires a YAML file specifying dataset paths and classes:
```yaml
path: ./datasets/coco
train: train/images
val: val/images
nc: 80
names: ['person', 'bicycle', 'car', ...]
```

### Image Classification
Requires directory structure:
```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       ├── img1.jpg
│       └── img2.jpg
└── val/
    ├── class1/
    └── class2/
```

## License

This repository is licensed under the MIT License.
