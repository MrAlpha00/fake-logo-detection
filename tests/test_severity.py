"""
Unit tests for severity scoring module.
Tests compute_severity function and related utilities.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import cv2
from src.severity import compute_severity, compute_ssim_score, compute_color_histogram_distance


def create_test_image(color=(100, 150, 200), size=(100, 100)):
    """Helper to create a test image with uniform color."""
    return np.full((size[0], size[1], 3), color, dtype=np.uint8)


def test_compute_ssim_identical_images():
    """SSIM should be 1.0 for identical images."""
    img1 = create_test_image()
    img2 = img1.copy()
    
    ssim_score = compute_ssim_score(img1, img2)
    
    assert ssim_score == pytest.approx(1.0, abs=0.01), "Identical images should have SSIM = 1.0"


def test_compute_ssim_different_images():
    """SSIM should be < 1.0 for different images."""
    img1 = create_test_image(color=(100, 100, 100))
    img2 = create_test_image(color=(200, 200, 200))
    
    ssim_score = compute_ssim_score(img1, img2)
    
    assert ssim_score < 1.0, "Different images should have SSIM < 1.0"


def test_compute_color_histogram_identical():
    """Color histogram distance should be 0 for identical images."""
    img1 = create_test_image()
    img2 = img1.copy()
    
    distance = compute_color_histogram_distance(img1, img2)
    
    assert distance == pytest.approx(0.0, abs=0.01), "Identical images should have distance = 0"


def test_compute_color_histogram_different():
    """Color histogram distance should be > 0 for different colors."""
    img1 = create_test_image(color=(255, 0, 0))  # Red
    img2 = create_test_image(color=(0, 255, 0))  # Green
    
    distance = compute_color_histogram_distance(img1, img2)
    
    assert distance > 0, "Different colored images should have distance > 0"


def test_compute_severity_authentic_logo():
    """Authentic logo (identical to reference) should have low severity."""
    logo_crop = create_test_image()
    reference = logo_crop.copy()
    
    classifier_result = {
        'is_fake_prob': 0.1,  # Low fake probability
        'confidence': 0.9
    }
    
    result = compute_severity(logo_crop, reference, classifier_result)
    
    assert 'severity_score' in result
    assert 'breakdown' in result
    assert result['severity_score'] < 40, "Identical logos should have low severity"


def test_compute_severity_fake_logo():
    """Modified logo should have higher severity."""
    logo_crop = create_test_image(color=(100, 100, 100))
    reference = create_test_image(color=(200, 200, 200))
    
    classifier_result = {
        'is_fake_prob': 0.8,  # High fake probability
        'confidence': 0.2
    }
    
    result = compute_severity(logo_crop, reference, classifier_result)
    
    assert result['severity_score'] > 40, "Modified logos should have higher severity"


def test_compute_severity_breakdown_components():
    """Severity breakdown should contain all required components."""
    logo_crop = create_test_image()
    reference = create_test_image()
    
    result = compute_severity(logo_crop, reference)
    
    breakdown = result['breakdown']
    assert 'classifier_fake_prob' in breakdown
    assert 'ssim_dissimilarity' in breakdown
    assert 'color_distance' in breakdown
    
    # All components should be in [0, 1] range
    assert 0 <= breakdown['classifier_fake_prob'] <= 1
    assert 0 <= breakdown['ssim_dissimilarity'] <= 1
    assert 0 <= breakdown['color_distance'] <= 1


def test_compute_severity_score_range():
    """Severity score should always be in [0, 100] range."""
    for _ in range(10):
        # Random images
        logo_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        reference = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = compute_severity(logo_crop, reference)
        
        assert 0 <= result['severity_score'] <= 100, \
            "Severity score must be in [0, 100] range"


def test_compute_severity_without_classifier():
    """Should work even without classifier results."""
    logo_crop = create_test_image()
    reference = create_test_image()
    
    result = compute_severity(logo_crop, reference, classifier_result=None)
    
    assert 'severity_score' in result
    assert 'breakdown' in result
    # Should use default fake probability
    assert result['breakdown']['classifier_fake_prob'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
