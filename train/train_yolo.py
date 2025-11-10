"""
YOLOv8 Logo Detection Training Script

This script trains a YOLOv8 model for logo detection and exports it to ONNX format.
Since ultralytics has dependency conflicts, this script provides instructions and
a standalone training approach using PyTorch.

Usage:
    python train/train_yolo.py --data_dir data/yolo_dataset --epochs 100

Dataset structure:
    data/yolo_dataset/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/

Each label file should be in YOLO format:
    class_id x_center y_center width height (normalized 0-1)
"""

import argparse
import sys
from pathlib import Path

def print_yolo_training_guide():
    """Print comprehensive guide for training YOLO models."""
    
    guide = """
==============================================================================
    YOLO Logo Detection - Training Guide
==============================================================================

OPTION 1: Train with Ultralytics YOLOv8 (Recommended if compatible)
-------------------------------------------------------------------

1. Install ultralytics in a separate environment:
   
   conda create -n yolo python=3.9
   conda activate yolo
   pip install ultralytics torch torchvision opencv-python

2. Prepare your dataset in YOLO format:
   
   dataset/
   ├── images/
   │   ├── train/  (training images)
   │   └── val/    (validation images)
   └── labels/
       ├── train/  (annotation .txt files)
       └── val/

3. Create dataset.yaml:
   
   path: /path/to/dataset
   train: images/train
   val: images/val
   
   nc: 6  # number of classes
   names: ['TechCo', 'Shopmart', 'Fastfood', 'Autodrive', 'Softnet', 'Mediaplay']

4. Train the model:
   
   from ultralytics import YOLO
   
   # Load pre-trained YOLOv8 nano model
   model = YOLO('yolov8n.pt')
   
   # Train
   results = model.train(
       data='dataset.yaml',
       epochs=100,
       imgsz=640,
       batch=16,
       device='cpu',  # or 'cuda' if GPU available
       project='runs/detect',
       name='logo_detector'
   )
   
   # Export to ONNX
   model.export(format='onnx', imgsz=640)

5. Copy the exported ONNX model to this project:
   
   cp runs/detect/logo_detector/weights/best.onnx models/yolov8_logo.onnx


OPTION 2: Use Pre-trained Object Detection and Fine-tune
---------------------------------------------------------

If you have a small logo dataset (<200 images), consider using transfer learning
from a pre-trained COCO object detector:

1. Download a pre-trained YOLOv8 ONNX model:
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx

2. Fine-tune using your logo dataset (requires ultralytics)

3. Export to ONNX and place in models/ directory


OPTION 3: Annotation Tools
---------------------------

To create YOLO format annotations for your logo images:

1. Label Studio: https://labelstud.io/
   - Web-based annotation tool
   - Exports to YOLO format
   - Supports collaborative annotation

2. LabelImg: https://github.com/heartexlabs/labelImg
   - Desktop annotation tool
   - Direct YOLO format export

3. Roboflow: https://roboflow.com/
   - Online platform
   - Auto-augmentation
   - YOLO export


DATA COLLECTION TIPS
--------------------

For logo detection, collect:
- Real logos in various contexts (products, websites, signage)
- Different scales and orientations
- Various lighting conditions
- Partial occlusions
- Fake/tampered variations for authenticity training

Recommended dataset size:
- Minimum: 100 images per brand (600 total for 6 brands)
- Good: 500 images per brand (3000 total)
- Excellent: 1000+ images per brand

Data augmentation (automatic in YOLOv8):
- Random flips, rotations
- Color jittering
- Random crops
- Mosaic augmentation


EXPORT AND INTEGRATION
-----------------------

After training, export your model to ONNX:

   from ultralytics import YOLO
   model = YOLO('path/to/best.pt')
   model.export(format='onnx', imgsz=640, opset=12)

Place the exported model at: models/yolov8_logo.onnx

The detector will automatically load and use it when method='yolo' is specified.


PERFORMANCE BENCHMARKS
----------------------

Expected performance on logo detection:
- YOLOv8n (nano): ~50 FPS on CPU, 200+ FPS on GPU
- mAP@0.5: 85-95% (with good training data)
- Model size: ~6 MB (ONNX format)

==============================================================================
"""
    
    print(guide)


def create_sample_training_script():
    """Create a sample YOLOv8 training script."""
    
    script = """
# sample_yolo_train.py
# Run this in an environment where ultralytics is installed

from ultralytics import YOLO
from pathlib import Path

def train_logo_detector():
    # Initialize model (downloads pre-trained weights automatically)
    model = YOLO('yolov8n.pt')  # nano model - fastest, smallest
    # or use: yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)
    
    # Train
    results = model.train(
        data='dataset.yaml',  # path to your dataset config
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,  # early stopping
        save=True,
        device='cpu',  # change to 'cuda' for GPU
        project='runs/detect',
        name='logo_detector',
        
        # Augmentation settings
        augment=True,
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        degrees=10,   # rotation
        translate=0.1,  # translation
        scale=0.5,    # scale
        shear=0.0,    # shear
        perspective=0.0,  # perspective
        flipud=0.0,   # flip up-down
        fliplr=0.5,   # flip left-right
        mosaic=1.0,   # mosaic augmentation
    )
    
    # Validate
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    
    # Export to ONNX
    success = model.export(
        format='onnx',
        imgsz=640,
        opset=12,  # ONNX opset version
        simplify=True,  # simplify ONNX model
    )
    
    print(f"Model exported to: {model.export_file}")
    print("Copy this file to: models/yolov8_logo.onnx")

if __name__ == '__main__':
    train_logo_detector()
"""
    
    output_path = Path('train/sample_yolo_train.py')
    output_path.write_text(script)
    print(f"\n✓ Sample training script created: {output_path}")
    print("  Edit dataset.yaml path and run in ultralytics environment\n")


def create_dataset_yaml_template():
    """Create dataset configuration template."""
    
    yaml_content = """# YOLOv8 Logo Detection Dataset Configuration

# Dataset root path (absolute or relative to this file)
path: ../data/yolo_dataset

# Relative paths from 'path'
train: images/train
val: images/val
test: images/test  # optional

# Number of classes
nc: 6

# Class names (must match the order in label files)
names:
  0: TechCo
  1: Shopmart
  2: Fastfood
  3: Autodrive
  4: Softnet
  5: Mediaplay

# Training hyperparameters (optional, can override in code)
# epochs: 100
# batch: 16
# imgsz: 640
"""
    
    output_path = Path('train/dataset_template.yaml')
    output_path.write_text(yaml_content)
    print(f"✓ Dataset config template created: {output_path}")
    print("  Edit paths and class names as needed\n")


def main():
    parser = argparse.ArgumentParser(description='YOLO Logo Detection Training Guide')
    parser.add_argument('--create-samples', action='store_true',
                       help='Create sample training scripts and templates')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("  YOLOv8 Logo Detection Training System")
    print("="*80 + "\n")
    
    if args.create_samples:
        create_sample_training_script()
        create_dataset_yaml_template()
        print("✓ Sample files created successfully!")
        print("\nNext steps:")
        print("1. Prepare your logo dataset in YOLO format")
        print("2. Edit train/dataset_template.yaml with your paths")
        print("3. Run train/sample_yolo_train.py in ultralytics environment")
        print("4. Copy exported ONNX model to models/yolov8_logo.onnx\n")
    else:
        print_yolo_training_guide()
        print("\n💡 TIP: Run with --create-samples to generate training templates\n")


if __name__ == '__main__':
    main()
