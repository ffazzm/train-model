#!/usr/bin/env python3
"""
General Deep Learning Model Training Script
Usage: python train_general.py --config <config.yaml>
"""
import argparse
import sys
import yaml
from pathlib import Path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train general deep learning models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='mlp',
        choices=['mlp', 'cnn', 'rnn', 'lstm', 'transformer'],
        help='Type of model architecture'
    )
    
    parser.add_argument(
        '--input-size',
        type=int,
        default=784,
        help='Input size for the model'
    )
    
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=128,
        help='Hidden layer size'
    )
    
    parser.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='Number of hidden layers'
    )
    
    parser.add_argument(
        '--num-classes',
        type=int,
        default=10,
        help='Number of output classes'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device to use (cuda or cpu, auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='runs/general',
        help='Directory to save model checkpoints'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='train',
        help='Experiment name'
    )
    
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Dropout rate'
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
        from torch.utils.data import DataLoader, Dataset
        from tqdm import tqdm
    except ImportError as e:
        print(f"Error: Required package not installed. {e}")
        print("Install with: pip install torch tqdm pyyaml")
        sys.exit(1)
    
    # Auto-detect device if not specified
    if not args.device:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define model classes
    class SimpleMLP(nn.Module):
        """Simple Multi-Layer Perceptron"""
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
            super(SimpleMLP, self).__init__()
            
            layers = []
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Linear(hidden_size, num_classes))
            
            self.model = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.model(x)

    class SimpleCNN(nn.Module):
        """Simple Convolutional Neural Network"""
        def __init__(self, num_classes, dropout=0.5):
            super(SimpleCNN, self).__init__()
            
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout),
            )
            
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    class SimpleRNN(nn.Module):
        """Simple Recurrent Neural Network"""
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
            super(SimpleRNN, self).__init__()
            
            self.rnn = nn.RNN(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.fc = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            out, _ = self.rnn(x)
            out = self.fc(out[:, -1, :])
            return out

    class SimpleLSTM(nn.Module):
        """Simple Long Short-Term Memory Network"""
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
            super(SimpleLSTM, self).__init__()
            
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.fc = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("General Deep Learning Model Training")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Input size: {args.input_size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Number of layers: {args.num_layers}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Dropout: {args.dropout}")
    print("=" * 60)
    
    # Create model
    print(f"\nCreating {args.model_type.upper()} model...")
    if args.model_type == 'mlp':
        model = SimpleMLP(
            args.input_size,
            args.hidden_size,
            args.num_layers,
            args.num_classes,
            args.dropout
        )
    elif args.model_type == 'cnn':
        model = SimpleCNN(args.num_classes, args.dropout)
    elif args.model_type == 'rnn':
        model = SimpleRNN(
            args.input_size,
            args.hidden_size,
            args.num_layers,
            args.num_classes,
            args.dropout
        )
    elif args.model_type == 'lstm':
        model = SimpleLSTM(
            args.input_size,
            args.hidden_size,
            args.num_layers,
            args.num_classes,
            args.dropout
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    model = model.to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print("\nNote: This is a template training script.")
    print("You need to provide your own data loaders.")
    print("See train_od.py and train_cls.py for complete examples.")
    print("\nExample usage:")
    print("  - For object detection: use train_od.py")
    print("  - For image classification: use train_cls.py")
    print("  - For custom tasks: adapt this script with your data loaders")
    
    # Save model architecture
    model_path = output_dir / 'model_architecture.txt'
    with open(model_path, 'w') as f:
        f.write(str(model))
    print(f"\nModel architecture saved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
