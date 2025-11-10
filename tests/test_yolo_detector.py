"""
Unit tests for YOLO detector post-processing logic.
"""
import numpy as np
import pytest
from src.detector import LogoDetector


def test_yolo_postprocess_shape_handling():
    """Test that YOLO post-processing correctly handles different output shapes."""
    detector = LogoDetector(method='sift')  # Initialize detector
    
    # Simulate YOLOv8 output shape: (1, 10, 8400) -> (features, num_predictions)
    # For 6 classes: 4 coords + 6 classes = 10 features
    num_predictions = 100
    num_features = 10  # 4 box + 6 classes
    
    # Create fake YOLO output
    fake_output = np.random.randn(1, num_features, num_predictions).astype(np.float32)
    
    # Add a strong detection manually
    # Format: [x_center, y_center, width, height, class_0, ..., class_5]
    fake_output[0, 0, 0] = 320  # x_center (middle of 640x640)
    fake_output[0, 1, 0] = 320  # y_center
    fake_output[0, 2, 0] = 100  # width
    fake_output[0, 3, 0] = 100  # height
    # Set high score for class 0
    fake_output[0, 4, 0] = 5.0  # logit for class 0 (will become ~0.99 after sigmoid)
    fake_output[0, 5:, 0] = -5.0  # low scores for other classes
    
    # Test post-processing
    detections = detector._postprocess_yolo(
        fake_output,
        orig_width=640,
        orig_height=640,
        ratio=1.0,
        pad_w=0,
        pad_h=0,
        conf_threshold=0.5
    )
    
    # Should detect at least one box
    assert len(detections) > 0, "Failed to detect any boxes from fake YOLO output"
    
    # Check first detection
    det = detections[0]
    assert 'bbox' in det
    assert 'label' in det
    assert 'confidence' in det
    assert det['confidence'] > 0.5


def test_yolo_postprocess_transposed_shape():
    """Test that transposed shape (num_predictions, features) is handled correctly."""
    detector = LogoDetector(method='sift')
    
    # Create output in (num_predictions, features) format
    num_predictions = 50
    num_features = 10
    
    fake_output = np.random.randn(1, num_predictions, num_features).astype(np.float32)
    
    # Add a strong detection
    fake_output[0, 0, :4] = [320, 320, 80, 80]  # box coords
    fake_output[0, 0, 4] = 5.0  # class 0 score
    fake_output[0, 0, 5:] = -5.0  # other class scores
    
    detections = detector._postprocess_yolo(
        fake_output,
        orig_width=640,
        orig_height=640,
        ratio=1.0,
        pad_w=0,
        pad_h=0,
        conf_threshold=0.5
    )
    
    assert len(detections) > 0


def test_yolo_sigmoid_application():
    """Test that sigmoid is correctly applied to class scores."""
    detector = LogoDetector(method='sift')
    
    # Create output with raw logits (negative and positive values)
    fake_output = np.zeros((1, 10, 5), dtype=np.float32)
    
    # Detection with high positive logit for class 0
    fake_output[0, 0, 0] = 100  # x
    fake_output[0, 1, 0] = 100  # y
    fake_output[0, 2, 0] = 50   # w
    fake_output[0, 3, 0] = 50   # h
    fake_output[0, 4, 0] = 10.0  # very high logit -> should be >0.99 after sigmoid
    fake_output[0, 5:, 0] = -10.0  # very low logits
    
    detections = detector._postprocess_yolo(
        fake_output,
        orig_width=200,
        orig_height=200,
        ratio=1.0,
        pad_w=0,
        pad_h=0,
        conf_threshold=0.9  # High threshold to test sigmoid
    )
    
    # Should still detect because sigmoid(10.0) ≈ 0.9999
    assert len(detections) > 0, "Sigmoid not properly applied to high logit values"


def test_yolo_nms():
    """Test Non-Maximum Suppression removes overlapping boxes."""
    detector = LogoDetector(method='sift')
    
    # Create boxes with significant overlap
    boxes = [
        [10, 10, 50, 50],  # Box 1
        [15, 15, 50, 50],  # Box 2 - overlaps with Box 1
        [100, 100, 30, 30],  # Box 3 - separate
    ]
    scores = [0.9, 0.85, 0.95]
    
    # Apply NMS with IoU threshold 0.3
    kept_indices = detector._nms(boxes, scores, iou_threshold=0.3)
    
    # Should keep Box 1 (higher score than Box 2) and Box 3
    assert len(kept_indices) == 2
    assert 0 in kept_indices  # Box 1 kept (higher score)
    assert 2 in kept_indices  # Box 3 kept (no overlap)


def test_yolo_coordinate_rescaling():
    """Test that coordinates are properly rescaled from padded input."""
    detector = LogoDetector(method='sift')
    
    # Simulate detection in padded/scaled image
    # Original image: 400x300, scaled to 640x640 with padding
    orig_width, orig_height = 400, 300
    target_size = 640
    
    ratio = min(target_size / orig_width, target_size / orig_height)
    new_width = int(orig_width * ratio)
    new_height = int(orig_height * ratio)
    pad_w = (target_size - new_width) // 2
    pad_h = (target_size - new_height) // 2
    
    # Create detection at center of input image (should map back to center of original)
    fake_output = np.zeros((1, 10, 1), dtype=np.float32)
    fake_output[0, 0, 0] = target_size / 2  # x_center in input
    fake_output[0, 1, 0] = target_size / 2  # y_center in input
    fake_output[0, 2, 0] = 100  # width
    fake_output[0, 3, 0] = 100  # height
    fake_output[0, 4, 0] = 5.0  # high class score
    
    detections = detector._postprocess_yolo(
        fake_output,
        orig_width=orig_width,
        orig_height=orig_height,
        ratio=ratio,
        pad_w=pad_w,
        pad_h=pad_h,
        conf_threshold=0.5
    )
    
    assert len(detections) > 0
    bbox = detections[0]['bbox']
    x, y, w, h = bbox
    
    # Center should be roughly in middle of original image
    center_x = x + w / 2
    center_y = y + h / 2
    
    # Allow some tolerance for rounding
    assert abs(center_x - orig_width / 2) < 50
    assert abs(center_y - orig_height / 2) < 50


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
