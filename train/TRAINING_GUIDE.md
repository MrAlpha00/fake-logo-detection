# Logo Classifier Training Guide

This guide explains how to train a production-ready MobileNetV2 classifier for logo brand authentication.

## Overview

The training pipeline consists of:
1. **Data Collection**: Gather real logo images for each brand
2. **Data Preparation**: Organize and augment the dataset
3. **Model Training**: Train MobileNetV2 with transfer learning
4. **Model Deployment**: Save and integrate trained weights

## Step 1: Data Collection

### Minimum Requirements

For each brand class, collect:
- **Minimum**: 50 high-quality logo images
- **Recommended**: 100-200 images per brand
- **Ideal**: 500+ images per brand

### Data Collection Sources

1. **Official Brand Assets**
   - Company websites and press kits
   - Official social media accounts
   - Brand guidelines documents

2. **Real-World Photos**
   - Product packaging
   - Signage and advertisements
   - Website screenshots
   - Mobile app interfaces

3. **Synthetic Variations**
   - Different backgrounds
   - Various sizes and resolutions
   - Different lighting conditions
   - Rotated and scaled versions

### Dataset Diversity Guidelines

Collect logos with variation in:
- **Size**: Small (50x50px) to large (1000x1000px)
- **Background**: White, colored, transparent, textured
- **Quality**: HD, compressed, low-resolution
- **Angle**: Straight-on, rotated, perspective distortion
- **Lighting**: Bright, dark, shadowed
- **Context**: Isolated logos, logos on products, logos in scenes

### Directory Structure

Organize raw logo images by brand:

```
raw_logos/
├── techco/
│   ├── logo_001.jpg
│   ├── logo_002.png
│   └── ...
├── shopmart/
│   ├── logo_001.jpg
│   └── ...
├── fastfood/
│   └── ...
├── autodrive/
│   └── ...
├── softnet/
│   └── ...
└── mediaplay/
    └── ...
```

## Step 2: Data Preparation & Augmentation

Use the provided script to prepare and augment your dataset:

### Basic Usage

```bash
python train/prepare_training_data.py \
    --input_dir raw_logos \
    --output_dir data/training_dataset \
    --augment_factor 10
```

### Parameters

- `--input_dir`: Path to raw logo images organized by brand
- `--output_dir`: Where to save organized dataset
- `--augment_factor`: Number of augmented versions per original image (default: 10)
- `--skip_organize`: Skip organization step if already organized
- `--skip_augment`: Skip augmentation step

### Augmentation Techniques Applied

The script applies random combinations of:
- Rotation (-15° to +15°)
- Scaling (0.8x to 1.2x)
- Brightness adjustment (0.7x to 1.3x)
- Contrast adjustment (0.7x to 1.3x)
- Saturation adjustment (0.7x to 1.3x)
- Gaussian noise addition
- Random blur
- Perspective transforms
- Random cropping
- JPEG compression artifacts (60-95% quality)

### Expected Output

For 50 original images per brand with augment_factor=10:
- **Before**: 50 images per brand = 300 total (6 brands)
- **After**: 550 images per brand = 3,300 total

## Step 3: Model Training

### Hardware Requirements

- **Minimum**: CPU with 8GB RAM (slow, ~2-3 hours)
- **Recommended**: GPU with 4GB+ VRAM (~15-30 minutes)
- **Ideal**: GPU with 8GB+ VRAM (~10-15 minutes)

### Training Command

```bash
python train/train_classifier.py \
    --data_dir data/training_dataset_augmented \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.001 \
    --val_split 0.2 \
    --output models/classifier_trained.pth \
    --device cuda  # or 'cpu' if no GPU
```

### Training Parameters

- `--data_dir`: Path to prepared dataset
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 32, reduce if out of memory)
- `--lr`: Learning rate (default: 0.001)
- `--val_split`: Validation split ratio (default: 0.2)
- `--output`: Where to save trained model (default: models/classifier.pth)
- `--device`: Training device ('cuda' or 'cpu')

### Training Process

The script will:
1. Load and split data into train/validation sets (80/20 split)
2. Initialize MobileNetV2 with ImageNet pre-trained weights
3. Replace classifier head for your number of classes
4. Train for specified epochs with data augmentation
5. Validate after each epoch
6. Save best model based on validation accuracy
7. Save class names to `models/class_names.txt`

### Expected Training Time

- **CPU (8GB RAM)**: 2-3 hours for 20 epochs
- **GPU (RTX 3060)**: 15-20 minutes for 20 epochs
- **GPU (V100)**: 8-10 minutes for 20 epochs

### Monitoring Training

Watch for:
- **Training accuracy**: Should reach >90% by epoch 10
- **Validation accuracy**: Should reach >85% by epoch 15
- **Loss**: Should decrease steadily
- **Overfitting**: If train acc >> val acc, reduce epochs or add more data

### Early Stopping

Training automatically saves the best model based on validation accuracy. If validation accuracy plateaus for 5+ epochs, you can stop early.

## Step 4: Model Deployment

### Output Files

After training, you'll have:
- `models/classifier_trained.pth`: Model weights
- `models/class_names.txt`: Brand class names

### Integration with Application

1. **Update classifier initialization** in `src/classifier.py`:
   ```python
   def __init__(self, weights_path='models/classifier_trained.pth'):
   ```

2. **Test the trained model**:
   ```bash
   streamlit run src/app_streamlit.py
   ```

3. **Verify production mode**:
   - App should load trained weights instead of running in demo mode
   - Check logs for "Loaded trained model from models/classifier_trained.pth"

## Performance Benchmarks

### Expected Accuracy

With proper data collection and training:
- **Training Accuracy**: 95-98%
- **Validation Accuracy**: 90-95%
- **Real-World Accuracy**: 85-92%

### Troubleshooting Low Accuracy

If validation accuracy < 80%:
1. **Collect more data**: Aim for 200+ images per brand
2. **Improve data quality**: Remove mislabeled or poor-quality images
3. **Increase data diversity**: Add more backgrounds, angles, lighting
4. **Train longer**: Increase epochs to 30-50
5. **Adjust learning rate**: Try 0.0001 (slower) or 0.01 (faster)
6. **Check class balance**: Ensure similar number of images per brand

### Model Size

- **Weights file**: ~9-14 MB
- **RAM usage**: ~100-200 MB during inference
- **Inference speed**: 20-50ms per image (CPU), 5-10ms (GPU)

## Advanced Configuration

### Fine-Tuning Specific Layers

For better performance with limited data, fine-tune only the classifier:

```python
# In train_classifier.py, before training loop:
for param in model.features.parameters():
    param.requires_grad = False  # Freeze feature extractor
```

### Custom Data Augmentation

Edit `prepare_training_data.py` to add domain-specific augmentations:
- Watermarks or overlays
- Background replacement
- Color space shifts
- Resolution degradation

### Transfer Learning from Related Domain

If you have a logo dataset from a different domain:

```bash
# Pre-train on general logo dataset
python train/train_classifier.py --data_dir data/general_logos --output models/pretrained_logos.pth

# Fine-tune on your specific brands
python train/train_classifier.py --data_dir data/training_dataset_augmented --pretrained models/pretrained_logos.pth
```

## Production Checklist

Before deploying to production:

- [ ] Collected 100+ images per brand
- [ ] Applied data augmentation (10x factor)
- [ ] Trained for 20+ epochs
- [ ] Achieved >85% validation accuracy
- [ ] Tested on held-out test set
- [ ] Verified real-world performance with sample images
- [ ] Saved model weights to `models/` directory
- [ ] Updated `class_names.txt` with correct brand names
- [ ] Tested integration in Streamlit app
- [ ] Benchmarked inference speed (should be <50ms per image)

## Example: Training with 6 Brands

```bash
# Step 1: Prepare dataset (assuming you have 100 raw images per brand)
python train/prepare_training_data.py \
    --input_dir raw_logos \
    --output_dir data/training_dataset \
    --augment_factor 10

# Expected output: 1,100 images per brand = 6,600 total

# Step 2: Train model
python train/train_classifier.py \
    --data_dir data/training_dataset_augmented \
    --epochs 25 \
    --batch_size 32 \
    --lr 0.001 \
    --device cuda

# Expected training time: ~20 minutes on RTX 3060
# Expected validation accuracy: ~92%

# Step 3: Test in app
streamlit run src/app_streamlit.py
```

## Continuous Improvement

### Collecting More Data

As you use the system:
1. Save misclassified examples
2. Add them to training set with correct labels
3. Retrain periodically (monthly or quarterly)
4. Track accuracy improvements over time

### Active Learning

Identify challenging cases:
1. Run batch inference on unlabeled logo images
2. Review low-confidence predictions
3. Manually label and add to training set
4. Retrain with expanded dataset

## Support & Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **MobileNetV2 Paper**: https://arxiv.org/abs/1801.04381
- **Transfer Learning Guide**: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- **Data Augmentation Best Practices**: https://pytorch.org/vision/stable/transforms.html

## License Considerations

When collecting training data:
- ✅ **OK**: Official brand assets with permission
- ✅ **OK**: Photos you took yourself
- ✅ **OK**: Public domain images
- ⚠️ **Careful**: Screenshots (may be copyrighted)
- ❌ **Avoid**: Scraped images without permission
- ❌ **Avoid**: Copyrighted marketing materials

Always obtain proper permissions for commercial use.
