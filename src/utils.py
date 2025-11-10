"""
Utility functions for image handling, preprocessing, and logging.
Provides common functionality used across the fake logo detection suite.
"""
import cv2
import numpy as np
import hashlib
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"fake_logo_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_image(image_path, mode='BGR'):
    """
    Load an image from file path.
    
    Args:
        image_path: Path to image file
        mode: Color mode - 'BGR', 'RGB', or 'GRAY'
    
    Returns:
        numpy.ndarray: Loaded image
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        if mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        logger.info(f"Loaded image: {image_path} with shape {img.shape}")
        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise


def save_image(image, output_path, mode='BGR'):
    """
    Save an image to file.
    
    Args:
        image: Image array to save
        output_path: Destination file path
        mode: Color mode of input image
    """
    try:
        if mode == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(str(output_path), image)
        logger.info(f"Saved image to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {e}")
        raise


def compute_image_hash(image):
    """
    Compute SHA-256 hash of image for deduplication and tracking.
    
    Args:
        image: Image array
    
    Returns:
        str: Hex digest of image hash
    """
    # Convert to bytes and compute hash
    image_bytes = image.tobytes()
    hash_obj = hashlib.sha256(image_bytes)
    return hash_obj.hexdigest()


def resize_image(image, target_size=(224, 224), maintain_aspect=False):
    """
    Resize image to target size.
    
    Args:
        image: Input image array
        target_size: Tuple of (width, height)
        maintain_aspect: If True, maintain aspect ratio with padding
    
    Returns:
        numpy.ndarray: Resized image
    """
    if maintain_aspect:
        # Resize maintaining aspect ratio and pad
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        if len(image.shape) == 3:
            padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        else:
            padded = np.zeros((target_h, target_w), dtype=image.dtype)
        
        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def draw_bounding_box(image, bbox, label, confidence, color=(0, 255, 0), thickness=2):
    """
    Draw bounding box with label on image.
    
    Args:
        image: Image array
        bbox: Bounding box as (x, y, w, h)
        label: Text label
        confidence: Confidence score (0-1)
        color: Box color (BGR)
        thickness: Line thickness
    
    Returns:
        numpy.ndarray: Image with drawn box
    """
    img_copy = image.copy()
    x, y, w, h = [int(v) for v in bbox]
    
    # Draw rectangle
    cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
    
    # Prepare label text
    text = f"{label}: {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    
    # Get text size for background rectangle
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Draw filled rectangle for text background
    cv2.rectangle(img_copy, (x, y - text_h - 10), (x + text_w, y), color, -1)
    
    # Draw text
    cv2.putText(img_copy, text, (x, y - 5), font, font_scale, (255, 255, 255), font_thickness)
    
    return img_copy


def crop_region(image, bbox):
    """
    Crop a region from image using bounding box.
    
    Args:
        image: Source image
        bbox: Bounding box as (x, y, w, h)
    
    Returns:
        numpy.ndarray: Cropped region
    """
    x, y, w, h = [int(v) for v in bbox]
    # Ensure coordinates are within image bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    return image[y:y+h, x:x+w].copy()


def normalize_image(image):
    """
    Normalize image to [0, 1] range for neural network input.
    
    Args:
        image: Input image (0-255)
    
    Returns:
        numpy.ndarray: Normalized image (0-1)
    """
    return image.astype(np.float32) / 255.0


def get_logger(name):
    """Get a logger instance with the given name."""
    return logging.getLogger(name)
