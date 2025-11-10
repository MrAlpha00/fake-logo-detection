"""
Unit tests for tamper detection module.
Tests Error Level Analysis and related functions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import cv2
from src.tamper import error_level_analysis, analyze_noise_patterns


def create_test_image(size=(200, 200), pattern='uniform'):
    """Helper to create test images with different patterns."""
    if pattern == 'uniform':
        return np.full((size[0], size[1], 3), 128, dtype=np.uint8)
    elif pattern == 'gradient':
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        for i in range(size[0]):
            img[i, :, :] = int(255 * i / size[0])
        return img
    elif pattern == 'random':
        return np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)


def test_error_level_analysis_returns_correct_keys():
    """ELA should return dict with required keys."""
    image = create_test_image()
    
    result = error_level_analysis(image)
    
    assert 'ela_image' in result
    assert 'mean_brightness' in result
    assert 'is_suspicious' in result
    assert 'suspiciousness_score' in result


def test_error_level_analysis_ela_image_shape():
    """ELA image should have same shape as input."""
    image = create_test_image(size=(150, 200))
    
    result = error_level_analysis(image)
    
    assert result['ela_image'].shape == image.shape, \
        "ELA image should have same dimensions as input"


def test_error_level_analysis_mean_brightness_range():
    """Mean brightness should be non-negative."""
    image = create_test_image()
    
    result = error_level_analysis(image)
    
    assert result['mean_brightness'] >= 0, \
        "Mean brightness should be non-negative"


def test_error_level_analysis_suspiciousness_range():
    """Suspiciousness score should be in [0, 1] range."""
    for _ in range(5):
        image = create_test_image(pattern='random')
        result = error_level_analysis(image)
        
        assert 0 <= result['suspiciousness_score'] <= 1, \
            "Suspiciousness score must be in [0, 1] range"


def test_error_level_analysis_uniform_image():
    """Uniform image should have low error levels."""
    image = create_test_image(pattern='uniform')
    
    result = error_level_analysis(image)
    
    # Uniform images compressed once should have relatively low error
    # (though not zero due to JPEG compression)
    assert result['mean_brightness'] < 50, \
        "Uniform images should have relatively low ELA brightness"


def test_error_level_analysis_different_quality():
    """ELA with different JPEG quality should produce different results."""
    image = create_test_image()
    
    result_high = error_level_analysis(image, quality=95)
    result_low = error_level_analysis(image, quality=50)
    
    # Results should differ
    assert result_high['mean_brightness'] != result_low['mean_brightness']


def test_analyze_noise_patterns_returns_correct_keys():
    """Noise analysis should return dict with required keys."""
    image = create_test_image()
    
    result = analyze_noise_patterns(image)
    
    assert 'noise_variance_consistency' in result
    assert 'is_inconsistent' in result
    assert 'regional_variances' in result


def test_analyze_noise_patterns_regional_variances():
    """Should return variance for each region in grid."""
    image = create_test_image(size=(200, 200))
    
    result = analyze_noise_patterns(image)
    
    # For 4x4 grid, should have 16 regional variances
    assert len(result['regional_variances']) == 16, \
        "Should have 16 regional variances for 4x4 grid"


def test_analyze_noise_patterns_uniform_image():
    """Uniform image should have consistent noise patterns."""
    image = create_test_image(pattern='uniform')
    
    result = analyze_noise_patterns(image)
    
    # Variance of variances should be low for uniform image
    variances = np.array(result['regional_variances'])
    assert np.std(variances) < 50, \
        "Uniform image should have consistent noise"


def test_ela_on_pre_compressed_image():
    """Test ELA on an image that's been compressed multiple times."""
    import io
    from PIL import Image as PILImage
    
    # Create and compress image
    image = create_test_image(pattern='gradient')
    
    # First compression
    pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_img.save(buffer, 'JPEG', quality=85)
    buffer.seek(0)
    
    # Load and convert back
    compressed = PILImage.open(buffer)
    compressed_array = cv2.cvtColor(np.array(compressed), cv2.COLOR_RGB2BGR)
    
    # Run ELA
    result = error_level_analysis(compressed_array)
    
    # Should still produce valid results
    assert result['ela_image'] is not None
    assert isinstance(result['mean_brightness'], float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
