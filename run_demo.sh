#!/bin/bash
# Automated demo script for Fake Logo Detection Suite
# Runs detection on sample images and generates screenshots

echo "=========================================="
echo "Fake Logo Detection Suite - Demo Runner"
echo "=========================================="
echo ""

# Create output directory
mkdir -p demo_outputs

echo "Running demo detections..."
echo ""

# Run Python demo script
python3 << 'PYTHON_SCRIPT'
import cv2
import numpy as np
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from src.detector import LogoDetector
from src.classifier import BrandClassifier
from src.severity import compute_severity
from src.tamper import error_level_analysis
from src.utils import draw_bounding_box, crop_region
from src.db import DetectionDatabase

print("Initializing models...")
detector = LogoDetector(method='sift')
classifier = BrandClassifier()
db = DetectionDatabase()

print(f"Loaded {len(detector.templates)} logo templates")
print("")

# Load reference templates
templates = {}
for template_path in Path('data/logos_db').glob('*.png'):
    name = template_path.stem.split('_')[-1].capitalize()
    templates[name] = cv2.imread(str(template_path))

def process_and_save(image_path, output_prefix):
    """Process an image and save annotated result."""
    print(f"Processing: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  ERROR: Could not load {image_path}")
        return None
    
    # Detect logos
    start_time = time.time()
    detections = detector.detect(image, confidence_threshold=0.4)
    
    if len(detections) == 0:
        print(f"  No logos detected")
        return None
    
    print(f"  Found {len(detections)} logo(s)")
    
    # Process each detection
    severities = []
    annotated = image.copy()
    
    for i, det in enumerate(detections):
        # Crop and classify
        crop = crop_region(image, det['bbox'])
        classification = classifier.classify(crop)
        
        # Get reference
        ref = templates.get(det['label'].capitalize(), list(templates.values())[0])
        
        # Compute severity
        severity = compute_severity(crop, ref, classification)
        severities.append(severity['severity_score'])
        
        # Choose color based on severity
        if severity['severity_score'] < 30:
            color = (0, 255, 0)  # Green
        elif severity['severity_score'] < 60:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        # Draw bounding box
        annotated = draw_bounding_box(
            annotated, det['bbox'], 
            f"{det['label']} (S:{severity['severity_score']})",
            det['confidence'], color=color
        )
        
        print(f"    Detection {i+1}: {det['label']} | "
              f"Conf: {det['confidence']:.2f} | "
              f"Severity: {severity['severity_score']}/100")
    
    # Run ELA
    ela_result = error_level_analysis(image)
    print(f"  ELA Suspiciousness: {ela_result['suspiciousness_score']:.2%}")
    
    # Save annotated image
    output_path = f"demo_outputs/{output_prefix}_annotated.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"  Saved: {output_path}")
    
    # Save ELA image
    ela_path = f"demo_outputs/{output_prefix}_ela.jpg"
    cv2.imwrite(ela_path, ela_result['ela_image'])
    print(f"  Saved: {ela_path}")
    
    # Log to database
    processing_time = (time.time() - start_time) * 1000
    det_data = [{
        'label': d['label'],
        'confidence': d['confidence'],
        'bbox': d['bbox'],
        'severity': {'severity_score': s}
    } for d, s in zip(detections, severities)]
    
    from src.utils import compute_image_hash
    image_hash = compute_image_hash(image)
    db.log_detection(image_path.name, image_hash, det_data, processing_time)
    
    print(f"  Processing time: {processing_time:.0f}ms")
    print("")
    
    return {
        'avg_severity': np.mean(severities),
        'ela_suspicious': ela_result['is_suspicious']
    }

# Demo 1: Real logo (should have low severity)
print("=" * 60)
print("DEMO 1: Real Logo (Expected: Low Severity < 40)")
print("=" * 60)
result1 = process_and_save(Path('data/samples/real_logo1.jpg'), 'demo1_real')

if result1:
    if result1['avg_severity'] < 40:
        print("✓ PASS: Real logo detected with low severity")
    else:
        print(f"⚠ WARNING: Real logo has higher severity than expected: {result1['avg_severity']:.0f}")

# Demo 2: Fake logo (should have high severity)
print("=" * 60)
print("DEMO 2: Fake Logo (Expected: High Severity > 60)")
print("=" * 60)
result2 = process_and_save(Path('data/samples/fake_logo1_compressed.jpg'), 'demo2_fake')

if result2:
    if result2['avg_severity'] > 60:
        print("✓ PASS: Fake logo detected with high severity")
    else:
        print(f"⚠ NOTE: Fake logo severity lower than expected: {result2['avg_severity']:.0f}")
    
    if result2['ela_suspicious']:
        print("✓ PASS: ELA flagged as suspicious")
    else:
        print("⚠ NOTE: ELA did not flag as suspicious")

print("")
print("=" * 60)
print("Demo Complete!")
print("=" * 60)
print("")
print("Generated outputs:")
print("  - demo_outputs/demo1_real_annotated.jpg")
print("  - demo_outputs/demo1_real_ela.jpg")
print("  - demo_outputs/demo2_fake_annotated.jpg")
print("  - demo_outputs/demo2_fake_ela.jpg")
print("")
print("Database entries logged. Check detections.db")
print("")

# Show statistics
stats = db.get_statistics()
print(f"Database Statistics:")
print(f"  Total detections: {stats.get('total_detections', 0)}")
print(f"  Total logos: {stats.get('total_logos', 0)}")
print(f"  Fake logos: {stats.get('total_fakes', 0)}")
print("")

PYTHON_SCRIPT

echo ""
echo "Demo completed successfully!"
echo ""
echo "To view the Streamlit web app, run:"
echo "  streamlit run src/app_streamlit.py --server.port 5000"
echo ""
