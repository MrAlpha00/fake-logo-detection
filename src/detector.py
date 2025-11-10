"""
Logo detection module using template matching, SIFT-based feature detection, and YOLO.
Provides three detection methods: (A) Fast template/SIFT matching for demo
                                   (B) YOLO-based detection via ONNX models
"""
import cv2
import numpy as np
from pathlib import Path
from src.utils import get_logger

logger = get_logger(__name__)

# Try to import ONNX Runtime for YOLO support
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime not available - YOLO detection disabled")


class LogoDetector:
    """
    Multi-method logo detector supporting template matching and SIFT features.
    """
    
    def __init__(self, templates_dir='data/logos_db', method='sift', yolo_model_path=None):
        """
        Initialize detector with reference logo templates.
        
        Args:
            templates_dir: Directory containing reference logo images
            method: Detection method - 'template', 'sift', or 'yolo'
            yolo_model_path: Path to YOLO ONNX model file (required if method='yolo')
        """
        self.templates_dir = Path(templates_dir)
        self.method = method
        self.templates = []
        self.template_names = []
        self.yolo_session = None
        self.yolo_input_size = 640
        self.class_names = ['TechCo', 'Shopmart', 'Fastfood', 'Autodrive', 'Softnet', 'Mediaplay']
        
        # Load reference templates
        self._load_templates()
        
        # Initialize SIFT detector for feature-based matching
        self.sift = cv2.SIFT_create()
        
        # FLANN matcher for efficient keypoint matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Initialize YOLO model if method is 'yolo'
        if self.method == 'yolo':
            self._load_yolo_model(yolo_model_path)
        
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
    
    def _load_yolo_model(self, model_path):
        """
        Load YOLO model in ONNX format for inference.
        
        Args:
            model_path: Path to YOLO ONNX model file
        """
        if not ONNX_AVAILABLE:
            logger.error("onnxruntime not installed - cannot load YOLO model")
            return
        
        if model_path is None:
            model_path = Path('models/yolov8_logo.onnx')
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            logger.warning(f"YOLO model not found at {model_path}. YOLO detection will not work.")
            logger.info("To train a YOLO model: Use train/train_yolo.py script")
            return
        
        try:
            # Create ONNX Runtime session
            self.yolo_session = ort.InferenceSession(
                str(model_path),
                providers=['CPUExecutionProvider']
            )
            logger.info(f"YOLO model loaded from {model_path}")
            
            # Get input shape from model
            input_shape = self.yolo_session.get_inputs()[0].shape
            if len(input_shape) == 4:
                self.yolo_input_size = input_shape[2]  # Assuming NCHW format
                
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.yolo_session = None
    
    def _detect_yolo(self, image, threshold=0.5):
        """
        Detect logos using YOLO model via ONNX Runtime.
        
        Args:
            image: Input image (BGR format)
            threshold: Confidence threshold for detections
        
        Returns:
            list: List of detections with bbox, label, and confidence
        """
        if self.yolo_session is None:
            logger.warning("YOLO model not loaded. Use method='sift' or 'template' instead.")
            return []
        
        # Preprocess image for YOLO input
        original_height, original_width = image.shape[:2]
        input_image, ratio, (pad_w, pad_h) = self._preprocess_yolo(image)
        
        # Run inference
        try:
            input_name = self.yolo_session.get_inputs()[0].name
            output_names = [output.name for output in self.yolo_session.get_outputs()]
            outputs = self.yolo_session.run(output_names, {input_name: input_image})
            
            # Post-process outputs
            detections = self._postprocess_yolo(
                outputs[0], 
                original_width, 
                original_height,
                ratio,
                pad_w,
                pad_h,
                threshold
            )
            
            logger.info(f"YOLO detection found {len(detections)} logos")
            return detections
            
        except Exception as e:
            logger.error(f"Error during YOLO inference: {e}")
            return []
    
    def _preprocess_yolo(self, image):
        """
        Preprocess image for YOLO input (letterbox resize + normalization).
        
        Returns:
            tuple: (preprocessed_image, ratio, (pad_w, pad_h))
        """
        height, width = image.shape[:2]
        target_size = self.yolo_input_size
        
        # Calculate scaling ratio maintaining aspect ratio
        ratio = min(target_size / width, target_size / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image (letterbox)
        pad_w = (target_size - new_width) // 2
        pad_h = (target_size - new_height) // 2
        
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_height, pad_w:pad_w+new_width] = resized
        
        # Convert BGR to RGB and normalize to [0, 1]
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.astype(np.float32) / 255.0
        
        # Transpose to NCHW format
        padded = padded.transpose(2, 0, 1)
        padded = np.expand_dims(padded, axis=0)
        
        return padded, ratio, (pad_w, pad_h)
    
    def _postprocess_yolo(self, output, orig_width, orig_height, ratio, pad_w, pad_h, conf_threshold=0.5, iou_threshold=0.45):
        """
        Post-process YOLO output to extract bounding boxes.
        
        YOLOv8 ONNX output format: (batch, num_predictions, 84)
        where each prediction is: [x_center, y_center, width, height, objectness, class_0, ..., class_N]
        or in some exports: [x_center, y_center, width, height, class_0, ..., class_N]
        
        Args:
            output: Raw YOLO output (1, num_predictions, 84+) or (1, 84+, num_predictions)
            orig_width, orig_height: Original image dimensions
            ratio: Scaling ratio used in preprocessing
            pad_w, pad_h: Padding added in preprocessing
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        
        Returns:
            list: Detected boxes as dicts with bbox, label, confidence
        """
        # Handle different output formats
        # Common YOLO exports: (1, 84, 8400) or (1, 8400, 84)
        # We need format: (num_predictions, features)
        if len(output.shape) == 3:
            # Remove batch dimension first
            output = output[0]
        
        num_classes = len(self.class_names)
        expected_features = 4 + num_classes  # box coords + classes
        
        # Check if we need to transpose based on which dimension matches expected_features
        # If output is (features, num_predictions) we need to transpose
        if len(output.shape) == 2:
            # Check if second dimension matches expected features (already in correct format)
            if output.shape[1] == expected_features or output.shape[1] == expected_features + 1:
                # Already in (num_predictions, features) format - no transpose needed
                pass
            # Check if first dimension matches expected features (needs transpose)
            elif output.shape[0] == expected_features or output.shape[0] == expected_features + 1:
                # Shape is (features, num_predictions) -> transpose to (num_predictions, features)
                output = output.transpose(1, 0)
            # Fallback heuristic: transpose if first dim is much smaller than second
            elif output.shape[0] < output.shape[1] and output.shape[0] < 100:
                output = output.transpose(1, 0)
        
        # Now output should be (num_predictions, features)
        # Extract boxes, scores, and class predictions
        boxes = []
        scores = []
        class_ids = []
        
        # Determine if output includes objectness score
        # YOLOv8 typically has format without separate objectness: 4 + num_classes
        # YOLOv5 might have: 4 + 1 + num_classes
        has_objectness = output.shape[1] > expected_features
        
        for detection in output:
            # Extract box coordinates (first 4 values)
            x_center, y_center, width, height = detection[:4]
            
            # Extract objectness and class scores
            if has_objectness:
                # Format: [x, y, w, h, objectness, class_0, ..., class_N]
                objectness = 1 / (1 + np.exp(-detection[4]))  # sigmoid
                class_scores = detection[5:5+num_classes]
                class_scores = 1 / (1 + np.exp(-class_scores))  # sigmoid on class logits
            else:
                # Format: [x, y, w, h, class_0, ..., class_N]
                # Class scores already include objectness in YOLOv8
                objectness = 1.0
                class_scores = detection[4:4+num_classes]
                # Apply sigmoid if values are logits (outside [0,1] range)
                if np.any(class_scores < 0) or np.any(class_scores > 1):
                    class_scores = 1 / (1 + np.exp(-class_scores))
            
            # Get class with highest score
            class_id = np.argmax(class_scores)
            class_prob = class_scores[class_id]
            
            # Compute final confidence (objectness * class probability)
            confidence = objectness * class_prob
            
            if confidence >= conf_threshold:
                # Convert from center format to corner format
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                
                # Rescale coordinates back to original image
                x1 = (x1 - pad_w) / ratio
                y1 = (y1 - pad_h) / ratio
                w = width / ratio
                h = height / ratio
                
                # Ensure positive dimensions
                if w > 0 and h > 0:
                    boxes.append([x1, y1, w, h])
                    scores.append(float(confidence))
                    class_ids.append(int(class_id))
        
        # Apply Non-Maximum Suppression
        if len(boxes) == 0:
            return []
        
        indices = self._nms(boxes, scores, iou_threshold)
        
        detections = []
        for idx in indices:
            x, y, w, h = boxes[idx]
            detections.append({
                'bbox': (int(max(0, x)), int(max(0, y)), int(w), int(h)),
                'label': self.class_names[class_ids[idx]] if class_ids[idx] < len(self.class_names) else f'Class_{class_ids[idx]}',
                'confidence': scores[idx]
            })
        
        return detections
    
    def _nms(self, boxes, scores, iou_threshold):
        """
        Perform Non-Maximum Suppression.
        
        Args:
            boxes: List of boxes [x, y, w, h]
            scores: List of confidence scores
            iou_threshold: IoU threshold for suppression
        
        Returns:
            list: Indices of kept boxes
        """
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        areas = boxes[:, 2] * boxes[:, 3]
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
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
