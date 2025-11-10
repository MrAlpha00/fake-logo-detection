"""
Logo detection module using template matching and SIFT-based feature detection.
Provides two detection methods: (A) Fast template/SIFT matching for demo
                                  (B) Placeholder for YOLO-based detection
"""
import cv2
import numpy as np
from pathlib import Path
from src.utils import get_logger

logger = get_logger(__name__)


class LogoDetector:
    """
    Multi-method logo detector supporting template matching and SIFT features.
    """
    
    def __init__(self, templates_dir='data/logos_db', method='sift'):
        """
        Initialize detector with reference logo templates.
        
        Args:
            templates_dir: Directory containing reference logo images
            method: Detection method - 'template', 'sift', or 'yolo' (future)
        """
        self.templates_dir = Path(templates_dir)
        self.method = method
        self.templates = []
        self.template_names = []
        
        # Load reference templates
        self._load_templates()
        
        # Initialize SIFT detector for feature-based matching
        self.sift = cv2.SIFT_create()
        
        # FLANN matcher for efficient keypoint matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        logger.info(f"LogoDetector initialized with {len(self.templates)} templates using method: {method}")
    
    def _load_templates(self):
        """Load all template images from templates directory."""
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return
        
        for template_path in sorted(self.templates_dir.glob('*.png')) + sorted(self.templates_dir.glob('*.jpg')):
            template = cv2.imread(str(template_path))
            if template is not None:
                self.templates.append(template)
                # Extract brand name from filename (e.g., logo_1_techco.png -> TechCo)
                name = template_path.stem.split('_')[-1].capitalize()
                self.template_names.append(name)
                logger.debug(f"Loaded template: {template_path.name} as {name}")
    
    def detect(self, image, confidence_threshold=0.5):
        """
        Detect logos in image using configured method.
        
        Args:
            image: Input image (BGR format)
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            list: List of detections as dicts with keys:
                  'bbox': (x, y, w, h), 'label': brand name, 'confidence': score
        """
        if self.method == 'template':
            return self._detect_template_matching(image, confidence_threshold)
        elif self.method == 'sift':
            return self._detect_sift(image, confidence_threshold)
        elif self.method == 'yolo':
            return self._detect_yolo(image, confidence_threshold)
        else:
            logger.error(f"Unknown detection method: {self.method}")
            return []
    
    def _detect_template_matching(self, image, threshold=0.5):
        """
        Detect logos using simple template matching.
        Fast but less robust to scale/rotation changes.
        """
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for template, name in zip(self.templates, self.template_names):
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            h, w = template_gray.shape
            
            # Multi-scale template matching to handle size variations
            for scale in [0.8, 1.0, 1.2]:
                scaled_template = cv2.resize(template_gray, (int(w*scale), int(h*scale)))
                th, tw = scaled_template.shape
                
                # Skip if template is larger than image
                if th > gray.shape[0] or tw > gray.shape[1]:
                    continue
                
                # Perform template matching
                result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                
                # Find matches above threshold
                locations = np.where(result >= threshold)
                
                for pt in zip(*locations[::-1]):
                    confidence = result[pt[1], pt[0]]
                    bbox = (pt[0], pt[1], tw, th)
                    
                    # Non-maximum suppression check
                    if not self._is_overlapping(bbox, detections):
                        detections.append({
                            'bbox': bbox,
                            'label': name,
                            'confidence': float(confidence)
                        })
        
        logger.info(f"Template matching found {len(detections)} logos")
        return detections
    
    def _detect_sift(self, image, threshold=0.5):
        """
        Detect logos using SIFT features and keypoint matching.
        More robust to transformations than template matching.
        """
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints in query image
        kp_image, desc_image = self.sift.detectAndCompute(gray, None)
        
        if desc_image is None:
            logger.warning("No keypoints detected in image")
            return []
        
        for template, name in zip(self.templates, self.template_names):
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # Detect keypoints in template
            kp_template, desc_template = self.sift.detectAndCompute(template_gray, None)
            
            if desc_template is None or len(kp_template) < 4:
                continue
            
            # Match descriptors using FLANN
            try:
                matches = self.flann.knnMatch(desc_template, desc_image, k=2)
            except:
                continue
            
            # Apply Lowe's ratio test to filter good matches
            good_matches = []
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # Require minimum number of good matches
            if len(good_matches) >= 10:
                # Extract matched keypoint locations
                src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography to locate template in image
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    # Get template corners
                    h, w = template_gray.shape
                    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    
                    # Transform corners to image space
                    try:
                        transformed = cv2.perspectiveTransform(corners, M)
                        
                        # Compute bounding box from transformed corners
                        x_coords = transformed[:, 0, 0]
                        y_coords = transformed[:, 0, 1]
                        x, y = int(min(x_coords)), int(min(y_coords))
                        w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                        
                        # Confidence based on number of inliers
                        inliers = np.sum(mask)
                        confidence = min(1.0, inliers / 30.0)
                        
                        if confidence >= threshold and w > 0 and h > 0:
                            bbox = (x, y, w, h)
                            if not self._is_overlapping(bbox, detections):
                                detections.append({
                                    'bbox': bbox,
                                    'label': name,
                                    'confidence': float(confidence)
                                })
                    except:
                        pass
        
        logger.info(f"SIFT detection found {len(detections)} logos")
        return detections
    
    def _detect_yolo(self, image, threshold=0.5):
        """
        Placeholder for YOLO-based detection.
        
        To use YOLO detection:
        1. Install ultralytics: pip install ultralytics
        2. Train a YOLOv8 model on logo dataset or use pre-trained weights
        3. Load model: self.yolo_model = YOLO('path/to/weights.pt')
        4. Run inference: results = self.yolo_model(image)
        """
        logger.warning("YOLO detection not implemented. Install ultralytics and train a model.")
        return []
    
    def _is_overlapping(self, bbox, detections, iou_threshold=0.3):
        """
        Check if bounding box overlaps significantly with existing detections.
        Used for non-maximum suppression.
        """
        x1, y1, w1, h1 = bbox
        
        for det in detections:
            x2, y2, w2, h2 = det['bbox']
            
            # Compute intersection
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right < x_left or y_bottom < y_top:
                continue
            
            intersection = (x_right - x_left) * (y_bottom - y_top)
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            iou = intersection / union if union > 0 else 0
            
            if iou > iou_threshold:
                return True
        
        return False
