"""
Severity scoring module combining multiple authenticity metrics.
Computes a 0-100 severity score from SSIM, color distance, and classifier confidence.
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from src.utils import get_logger

logger = get_logger(__name__)


def compute_color_histogram_distance(image1, image2):
    """
    Compute color histogram distance between two images using Bhattacharyya distance.
    
    Args:
        image1: First image (BGR)
        image2: Second image (BGR)
    
    Returns:
        float: Distance metric (0 = identical, 1 = completely different)
    """
    # Resize images to same size for comparison
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    # Compute histograms for each channel
    hist_distance = 0
    for i in range(3):  # BGR channels
        hist1 = cv2.calcHist([image1], [i], None, [256], [0, 256])
        hist2 = cv2.calcHist([image2], [i], None, [256], [0, 256])
        
        # Normalize histograms
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        # Compute Bhattacharyya distance
        distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        hist_distance += distance
    
    # Average across channels and clip to [0, 1]
    return min(1.0, hist_distance / 3.0)


def compute_ssim_score(image1, image2):
    """
    Compute Structural Similarity Index between two images.
    
    Args:
        image1: First image (BGR or grayscale)
        image2: Second image (BGR or grayscale)
    
    Returns:
        float: SSIM score (1 = identical, 0 = completely different)
    """
    # Convert to grayscale if needed
    if len(image1.shape) == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = image1
    
    if len(image2.shape) == 3:
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = image2
    
    # Resize to same dimensions
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    
    # Compute SSIM
    score, _ = ssim(gray1, gray2, full=True)
    
    return float(score)


def compute_severity(logo_crop, reference_logo, classifier_result=None):
    """
    Compute combined severity score indicating likelihood of tampering/fakery.
    
    Score ranges from 0 (authentic) to 100 (highly suspicious fake).
    Combines three metrics:
    1. Classifier fake probability (from neural network)
    2. Structural dissimilarity (1 - SSIM)
    3. Color histogram distance
    
    Args:
        logo_crop: Detected logo region (BGR numpy array)
        reference_logo: Reference template for this brand (BGR numpy array)
        classifier_result: Classification result dict (optional)
    
    Returns:
        dict: {
            'severity_score': int (0-100),
            'breakdown': {
                'classifier_fake_prob': float (0-1),
                'ssim_dissimilarity': float (0-1),
                'color_distance': float (0-1)
            }
        }
    """
    # Initialize breakdown components
    breakdown = {
        'classifier_fake_prob': 0.0,
        'ssim_dissimilarity': 0.0,
        'color_distance': 0.0
    }
    
    # Component 1: Classifier fake probability
    if classifier_result and 'is_fake_prob' in classifier_result:
        breakdown['classifier_fake_prob'] = classifier_result['is_fake_prob']
    else:
        # Default moderate suspicion if no classifier
        breakdown['classifier_fake_prob'] = 0.3
    
    # Component 2: Structural dissimilarity (1 - SSIM)
    try:
        ssim_score = compute_ssim_score(logo_crop, reference_logo)
        breakdown['ssim_dissimilarity'] = 1.0 - ssim_score
    except Exception as e:
        logger.warning(f"Error computing SSIM: {e}")
        breakdown['ssim_dissimilarity'] = 0.5
    
    # Component 3: Color histogram distance
    try:
        color_dist = compute_color_histogram_distance(logo_crop, reference_logo)
        breakdown['color_distance'] = color_dist
    except Exception as e:
        logger.warning(f"Error computing color distance: {e}")
        breakdown['color_distance'] = 0.5
    
    # Weighted combination: classifier is most important, then structural, then color
    weights = {
        'classifier_fake_prob': 0.5,
        'ssim_dissimilarity': 0.3,
        'color_distance': 0.2
    }
    
    weighted_score = (
        breakdown['classifier_fake_prob'] * weights['classifier_fake_prob'] +
        breakdown['ssim_dissimilarity'] * weights['ssim_dissimilarity'] +
        breakdown['color_distance'] * weights['color_distance']
    )
    
    # Convert to 0-100 scale
    severity_score = int(weighted_score * 100)
    
    logger.info(f"Severity score: {severity_score} | Breakdown: {breakdown}")
    
    return {
        'severity_score': severity_score,
        'breakdown': breakdown
    }


def interpret_severity(severity_score):
    """
    Provide human-readable interpretation of severity score.
    
    Args:
        severity_score: Severity score (0-100)
    
    Returns:
        dict: {
            'level': str ('Low', 'Medium', 'High', 'Critical'),
            'description': str (detailed explanation)
        }
    """
    if severity_score < 30:
        return {
            'level': 'Low',
            'description': 'Logo appears authentic with minimal signs of tampering.',
            'color': 'green'
        }
    elif severity_score < 50:
        return {
            'level': 'Medium',
            'description': 'Some inconsistencies detected. Logo may have been modified.',
            'color': 'yellow'
        }
    elif severity_score < 70:
        return {
            'level': 'High',
            'description': 'Significant tampering indicators present. Likely a fake logo.',
            'color': 'orange'
        }
    else:
        return {
            'level': 'Critical',
            'description': 'Strong evidence of forgery. Logo is highly suspicious.',
            'color': 'red'
        }
