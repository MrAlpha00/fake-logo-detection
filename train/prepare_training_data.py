"""
Data preparation and augmentation script for logo classifier training.

This script helps create a training dataset by:
1. Organizing raw logo images into the required directory structure
2. Applying data augmentation to increase dataset size
3. Splitting data into train/validation sets

Usage:
    python train/prepare_training_data.py --input_dir raw_logos --output_dir data/training_dataset --augment_factor 10
"""
import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random


class LogoAugmenter:
    """
    Augment logo images with various transformations to increase training data.
    """
    
    def __init__(self):
        self.augmentations = [
            self.random_rotation,
            self.random_scale,
            self.random_brightness,
            self.random_contrast,
            self.random_saturation,
            self.add_noise,
            self.random_blur,
            self.random_perspective,
            self.random_crop,
            self.add_compression_artifacts
        ]
    
    def random_rotation(self, img, angle_range=(-15, 15)):
        """Rotate image by random angle."""
        angle = random.uniform(*angle_range)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    def random_scale(self, img, scale_range=(0.8, 1.2)):
        """Scale image by random factor."""
        scale = random.uniform(*scale_range)
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(img, (new_w, new_h))
        
        # Pad or crop to original size
        if scale > 1:
            # Crop center
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            return scaled[start_y:start_y+h, start_x:start_x+w]
        else:
            # Pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            return cv2.copyMakeBorder(scaled, pad_h, pad_h, pad_w, pad_w, 
                                     cv2.BORDER_REPLICATE)
    
    def random_brightness(self, img, factor_range=(0.7, 1.3)):
        """Adjust brightness."""
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_img)
        factor = random.uniform(*factor_range)
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def random_contrast(self, img, factor_range=(0.7, 1.3)):
        """Adjust contrast."""
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_img)
        factor = random.uniform(*factor_range)
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def random_saturation(self, img, factor_range=(0.7, 1.3)):
        """Adjust color saturation."""
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Color(pil_img)
        factor = random.uniform(*factor_range)
        enhanced = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
    
    def add_noise(self, img, noise_level=10):
        """Add Gaussian noise."""
        # Convert to float to avoid uint8 wraparound
        img_float = img.astype(np.float32)
        noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
        
        # Add noise and clip to valid range
        noisy = img_float + noise
        noisy = np.clip(noisy, 0, 255)
        
        return noisy.astype(np.uint8)
    
    def random_blur(self, img, kernel_range=(3, 7)):
        """Apply random blur."""
        kernel_size = random.choice(range(kernel_range[0], kernel_range[1], 2))
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def random_perspective(self, img, strength=0.1):
        """Apply random perspective transform."""
        h, w = img.shape[:2]
        
        # Random perspective points
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = pts1 + np.random.uniform(-strength * w, strength * w, pts1.shape).astype(np.float32)
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    def random_crop(self, img, crop_ratio=0.9):
        """Random crop and resize."""
        h, w = img.shape[:2]
        new_h = int(h * crop_ratio)
        new_w = int(w * crop_ratio)
        
        start_y = random.randint(0, h - new_h)
        start_x = random.randint(0, w - new_w)
        
        cropped = img[start_y:start_y+new_h, start_x:start_x+new_w]
        return cv2.resize(cropped, (w, h))
    
    def add_compression_artifacts(self, img, quality_range=(60, 95)):
        """Simulate JPEG compression artifacts."""
        quality = random.randint(*quality_range)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc_img = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
    
    def augment(self, img, num_augmentations=3):
        """
        Apply random augmentations to image.
        
        Args:
            img: Input image (BGR)
            num_augmentations: Number of random augmentations to apply
        
        Returns:
            Augmented image (BGR)
        """
        augmented = img.copy()
        
        # Apply random subset of augmentations
        selected_augs = random.sample(self.augmentations, 
                                     min(num_augmentations, len(self.augmentations)))
        
        for aug_func in selected_augs:
            try:
                augmented = aug_func(augmented)
            except Exception as e:
                print(f"Warning: Augmentation {aug_func.__name__} failed: {e}")
        
        return augmented


def organize_dataset(input_dir, output_dir):
    """
    Organize raw images into training dataset structure.
    
    Expected input structure:
        input_dir/
            brand1/
                *.jpg, *.png
            brand2/
                *.jpg, *.png
    
    Output structure:
        output_dir/
            brand1/
                img_001.jpg
                img_002.jpg
            brand2/
                ...
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Organizing dataset from {input_dir} to {output_dir}")
    
    for brand_dir in input_path.iterdir():
        if brand_dir.is_dir():
            brand_name = brand_dir.name
            output_brand_dir = output_path / brand_name
            output_brand_dir.mkdir(exist_ok=True)
            
            # Copy images
            img_count = 0
            for img_path in list(brand_dir.glob('*.jpg')) + list(brand_dir.glob('*.png')):
                output_img_path = output_brand_dir / f"img_{img_count:04d}{img_path.suffix}"
                shutil.copy2(img_path, output_img_path)
                img_count += 1
            
            print(f"  {brand_name}: {img_count} images")


def augment_dataset(input_dir, output_dir, augment_factor=10):
    """
    Augment existing dataset to increase training samples.
    
    Args:
        input_dir: Directory with organized dataset
        output_dir: Output directory for augmented dataset
        augment_factor: Number of augmented versions per original image
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    augmenter = LogoAugmenter()
    
    print(f"\nAugmenting dataset: {augment_factor}x per image")
    
    total_generated = 0
    
    for brand_dir in input_path.iterdir():
        if brand_dir.is_dir():
            brand_name = brand_dir.name
            output_brand_dir = output_path / brand_name
            output_brand_dir.mkdir(exist_ok=True)
            
            # Get all images
            image_files = list(brand_dir.glob('*.jpg')) + list(brand_dir.glob('*.png'))
            
            print(f"\n{brand_name}: {len(image_files)} original images")
            
            for img_idx, img_path in enumerate(image_files):
                # Copy original
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  Warning: Could not load {img_path}")
                    continue
                
                output_original = output_brand_dir / f"{brand_name}_{img_idx:04d}_orig.jpg"
                cv2.imwrite(str(output_original), img)
                
                # Generate augmented versions
                for aug_idx in range(augment_factor):
                    augmented = augmenter.augment(img, num_augmentations=random.randint(2, 4))
                    output_aug = output_brand_dir / f"{brand_name}_{img_idx:04d}_aug_{aug_idx:03d}.jpg"
                    cv2.imwrite(str(output_aug), augmented)
                    total_generated += 1
            
            total_images = len(list(output_brand_dir.glob('*.jpg')))
            print(f"  Generated {total_images} total images (1 original + {augment_factor} augmented per image)")
    
    print(f"\n✓ Total images generated: {total_generated + len(list(output_path.rglob('*_orig.jpg')))}")


def main():
    parser = argparse.ArgumentParser(description='Prepare training dataset for logo classifier')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory with raw logo images organized by brand')
    parser.add_argument('--output_dir', type=str, default='data/training_dataset',
                       help='Output directory for processed dataset')
    parser.add_argument('--augment_factor', type=int, default=10,
                       help='Number of augmented versions per original image')
    parser.add_argument('--skip_organize', action='store_true',
                       help='Skip organization step (use if already organized)')
    parser.add_argument('--skip_augment', action='store_true',
                       help='Skip augmentation step')
    
    args = parser.parse_args()
    
    # Step 1: Organize dataset
    if not args.skip_organize:
        organize_dataset(args.input_dir, args.output_dir)
    
    # Step 2: Augment dataset
    if not args.skip_augment:
        augment_output = args.output_dir + '_augmented'
        augment_dataset(args.output_dir, augment_output, args.augment_factor)
        print(f"\n✓ Augmented dataset ready at: {augment_output}")
        print(f"\nTo train the classifier, run:")
        print(f"  python train/train_classifier.py --data_dir {augment_output} --epochs 20")
    else:
        print(f"\n✓ Dataset ready at: {args.output_dir}")
        print(f"\nTo train the classifier, run:")
        print(f"  python train/train_classifier.py --data_dir {args.output_dir} --epochs 20")


if __name__ == '__main__':
    main()
