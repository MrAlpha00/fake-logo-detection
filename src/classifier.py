"""
Brand classification module using PyTorch transfer learning.
Provides MobileNetV2-based classifier with deterministic fallback for demo mode.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from src.utils import get_logger

logger = get_logger(__name__)

# Try importing torchvision, fallback to demo mode if not available
try:
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logger.warning("torchvision not available, running in demo-only mode")


class BrandClassifier:
    """
    Multi-class brand classifier based on MobileNetV2 architecture.
    Supports loading trained weights or running in demo mode with deterministic predictions.
    """
    
    def __init__(self, model_path=None, num_classes=6, device='cpu'):
        """
        Initialize classifier model.
        
        Args:
            model_path: Path to trained model weights (None for demo mode)
            num_classes: Number of brand classes
            device: Device for inference ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.model_path = model_path
        
        # Define brand class names (should match training data)
        self.class_names = [
            'TechCo', 'Shopmart', 'Fastfood', 
            'Autodrive', 'Softnet', 'Mediaplay'
        ]
        
        # Build model architecture
        if TORCHVISION_AVAILABLE:
            self.model = self._build_model()
            
            # Load weights if available
            if model_path and Path(model_path).exists():
                self._load_weights(model_path)
                self.demo_mode = False
                logger.info(f"Loaded classifier weights from {model_path}")
            else:
                self.demo_mode = True
                logger.warning("Running in DEMO mode with simulated predictions")
            
            # Define image preprocessing transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.model = None
            self.demo_mode = True
            self.transform = None
            logger.warning("torchvision not available - running in DEMO mode only")
    
    def _build_model(self):
        """Build MobileNetV2 model architecture."""
        if not TORCHVISION_AVAILABLE:
            return None
            
        # Load pre-trained MobileNetV2 and modify final layer
        model = models.mobilenet_v2(pretrained=False)
        
        # Replace classifier head for our number of classes
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, self.num_classes)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _load_weights(self, model_path):
        """Load trained model weights."""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info("Model weights loaded successfully")
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            self.demo_mode = True
    
    def classify(self, image_crop):
        """
        Classify a logo crop to predict brand and authenticity.
        
        Args:
            image_crop: Cropped logo region (BGR numpy array)
        
        Returns:
            dict: Classification results with keys:
                  'brand': predicted brand name
                  'confidence': confidence score (0-1)
                  'probabilities': class probability distribution
                  'is_fake_prob': probability of being fake (1 - confidence)
        """
        if self.demo_mode:
            return self._demo_classify(image_crop)
        else:
            return self._real_classify(image_crop)
    
    def _real_classify(self, image_crop):
        """Run real classification using trained model."""
        if not TORCHVISION_AVAILABLE or self.model is None:
            return self._demo_classify(image_crop)
            
        import cv2
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess image
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Get top prediction
        confidence, pred_idx = torch.max(probabilities, 0)
        brand = self.class_names[pred_idx.item()]
        
        return {
            'brand': brand,
            'confidence': float(confidence),
            'probabilities': probabilities.cpu().numpy().tolist(),
            'is_fake_prob': float(1.0 - confidence)
        }
    
    def _demo_classify(self, image_crop):
        """
        Generate deterministic demo predictions based on image characteristics.
        Uses color histogram and size to produce consistent results.
        """
        # Compute image hash for deterministic results
        mean_color = np.mean(image_crop, axis=(0, 1))
        
        # Map color to brand (deterministic based on dominant color)
        blue_val = mean_color[0]
        green_val = mean_color[1]
        red_val = mean_color[2]
        
        # Simple heuristic mapping
        if blue_val > red_val and blue_val > green_val:
            brand_idx = 0  # TechCo (blue)
            confidence = 0.85
        elif red_val > blue_val and red_val > green_val:
            if red_val > 150:
                brand_idx = 1  # ShopMart (orange/red)
                confidence = 0.82
            else:
                brand_idx = 3  # AutoDrive (dark red)
                confidence = 0.78
        elif green_val > red_val and green_val > blue_val:
            brand_idx = 4  # SoftNet (green)
            confidence = 0.80
        else:
            # Yellow or purple
            if red_val + green_val > 300:
                brand_idx = 2  # FastFood (yellow)
                confidence = 0.83
            else:
                brand_idx = 5  # MediaPlay (purple)
                confidence = 0.77
        
        # Reduce confidence for low quality images (simulating fake detection)
        quality_metric = np.std(image_crop)
        if quality_metric < 30:  # Low variation suggests compression/tampering
            confidence *= 0.7
        
        # Create probability distribution
        probs = np.ones(self.num_classes) * 0.05
        probs[brand_idx] = confidence
        probs = probs / probs.sum()  # Normalize
        
        brand = self.class_names[brand_idx]
        
        logger.debug(f"Demo classification: {brand} with confidence {confidence:.3f}")
        
        return {
            'brand': brand,
            'confidence': float(confidence),
            'probabilities': probs.tolist(),
            'is_fake_prob': float(1.0 - confidence)
        }
    
    def get_feature_map(self, image_crop):
        """
        Extract feature map from model for Grad-CAM visualization.
        
        Args:
            image_crop: Cropped logo region (BGR numpy array)
        
        Returns:
            tuple: (feature_map, predicted_class_idx)
        """
        if self.demo_mode:
            # Return dummy feature map for demo
            return np.random.rand(7, 7, 1280), 0
        
        # Convert BGR to RGB
        import cv2
        image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Extract features from last convolutional layer
        features = None
        def hook_fn(module, input, output):
            nonlocal features
            features = output.detach()
        
        # Register hook on MobileNetV2's last conv layer
        handle = self.model.features[-1].register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_tensor)
            pred_idx = torch.argmax(outputs, dim=1).item()
        
        handle.remove()
        
        if features is not None:
            features = features.cpu().numpy()[0]  # Shape: (C, H, W)
            features = np.transpose(features, (1, 2, 0))  # Shape: (H, W, C)
        
        return features, pred_idx
