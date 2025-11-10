"""
Tamper detection module using Error Level Analysis (ELA).
Detects image manipulation by analyzing JPEG compression artifacts.
"""
import cv2
import numpy as np
from PIL import Image
import io
from src.utils import get_logger

logger = get_logger(__name__)


def error_level_analysis(image, quality=90):
    """
    Perform Error Level Analysis to detect image tampering.
    
    ELA works by re-saving the image at a known JPEG quality and comparing
    the difference. Edited regions show different error levels than original regions.
    
    Args:
        image: Input image (BGR numpy array)
        quality: JPEG quality for re-compression (default 90)
    
    Returns:
        dict: {
            'ela_image': numpy array (BGR) showing error levels,
            'mean_brightness': float (average error level),
            'is_suspicious': bool (True if high error levels detected),
            'suspiciousness_score': float (0-1)
        }
    """
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Save to bytes buffer with specified JPEG quality
    buffer = io.BytesIO()
    pil_image.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    
    # Load the recompressed image
    recompressed = Image.open(buffer)
    recompressed_array = np.array(recompressed)
    
    # Compute absolute difference (error level)
    original_array = np.array(pil_image)
    
    # Ensure same dimensions
    if original_array.shape != recompressed_array.shape:
        recompressed_array = cv2.resize(recompressed_array, 
                                        (original_array.shape[1], original_array.shape[0]))
    
    # Calculate error level
    error = cv2.absdiff(original_array, recompressed_array)
    
    # Enhance error visibility by scaling
    # Multiply by a factor to make subtle differences more visible
    ela_enhanced = np.clip(error * 10, 0, 255).astype(np.uint8)
    
    # Compute mean brightness of error image (higher = more suspicious)
    mean_brightness = np.mean(error)
    
    # Suspiciousness threshold calibration
    # High error levels suggest tampering, but very low can also indicate multiple resaves
    suspiciousness_score = min(1.0, mean_brightness / 50.0)
    
    # Flag as suspicious if mean error is above threshold
    is_suspicious = mean_brightness > 15.0
    
    # Convert back to BGR for OpenCV compatibility
    ela_bgr = cv2.cvtColor(ela_enhanced, cv2.COLOR_RGB2BGR)
    
    logger.info(f"ELA analysis: mean_brightness={mean_brightness:.2f}, "
                f"suspicious={is_suspicious}, score={suspiciousness_score:.3f}")
    
    return {
        'ela_image': ela_bgr,
        'mean_brightness': float(mean_brightness),
        'is_suspicious': is_suspicious,
        'suspiciousness_score': float(suspiciousness_score)
    }


def detect_clone_regions(image, threshold=0.95):
    """
    Detect potential clone-stamp tampering by finding duplicate regions.
    
    This is a simplified clone detection using template matching.
    For production, consider more sophisticated methods like SIFT-based duplicate detection.
    
    Args:
        image: Input image (BGR)
        threshold: Similarity threshold for clone detection
    
    Returns:
        list: List of suspicious clone region pairs
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clones = []
    
    # Sample small patches and look for duplicates
    patch_size = 32
    step = 16
    
    patches = []
    locations = []
    
    for y in range(0, gray.shape[0] - patch_size, step):
        for x in range(0, gray.shape[1] - patch_size, step):
            patch = gray[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            locations.append((x, y))
    
    # Compare patches (simplified for demo - full implementation would use more efficient matching)
    # This is computationally expensive for large images
    for i in range(min(len(patches), 100)):  # Limit to first 100 patches for speed
        for j in range(i+1, min(len(patches), 100)):
            # Compute normalized cross-correlation
            corr = cv2.matchTemplate(patches[i], patches[j], cv2.TM_CCOEFF_NORMED)[0, 0]
            
            if corr > threshold:
                clones.append({
                    'location1': locations[i],
                    'location2': locations[j],
                    'similarity': float(corr)
                })
    
    logger.info(f"Clone detection found {len(clones)} potential duplicate regions")
    return clones


def analyze_noise_patterns(image):
    """
    Analyze noise patterns to detect inconsistencies suggesting manipulation.
    
    Authentic images should have consistent noise across the image.
    Tampered regions often show different noise characteristics.
    
    Args:
        image: Input image (BGR)
    
    Returns:
        dict: Noise analysis results
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur and compute difference (noise extraction)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = cv2.absdiff(gray, blurred)
    
    # Divide image into regions and compute noise variance
    h, w = gray.shape
    grid_size = 4
    block_h, block_w = h // grid_size, w // grid_size
    
    variances = []
    for i in range(grid_size):
        for j in range(grid_size):
            block = noise[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            variances.append(np.var(block))
    
    # Compute variance of variances (should be low for authentic images)
    variance_consistency = np.var(variances)
    
    # Flag inconsistent noise patterns
    is_inconsistent = variance_consistency > 20.0
    
    return {
        'noise_variance_consistency': float(variance_consistency),
        'is_inconsistent': is_inconsistent,
        'regional_variances': [float(v) for v in variances]
    }
