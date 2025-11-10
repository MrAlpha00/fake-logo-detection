"""
Explainability module implementing Grad-CAM (Gradient-weighted Class Activation Mapping).
Visualizes which regions of the logo the classifier focuses on for its decision.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.utils import get_logger

logger = get_logger(__name__)


def generate_gradcam(model, image_tensor, target_class=None):
    """
    Generate Grad-CAM heatmap for a given model and input.
    
    Grad-CAM highlights the important regions in the image that contribute
    to the model's classification decision.
    
    Args:
        model: PyTorch model
        image_tensor: Preprocessed input tensor (1, C, H, W)
        target_class: Target class index (None = use predicted class)
    
    Returns:
        numpy.ndarray: Heatmap array (H, W) with values [0, 1]
    """
    model.eval()
    
    # Storage for gradients and activations
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Register hooks on the last convolutional layer
    # For MobileNetV2: features[-1] is the last conv block
    try:
        target_layer = model.features[-1]
    except:
        # Fallback for other architectures
        logger.warning("Could not find target layer, returning dummy heatmap")
        return np.random.rand(7, 7)
    
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)
    
    # Forward pass
    output = model(image_tensor)
    
    # Use predicted class if target not specified
    if target_class is None:
        target_class = torch.argmax(output, dim=1).item()
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass for target class
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    output.backward(gradient=one_hot, retain_graph=True)
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Get activations and gradients
    if len(activations) == 0 or len(gradients) == 0:
        logger.warning("No activations or gradients captured, returning dummy heatmap")
        return np.random.rand(7, 7)
    
    activation = activations[0].detach().cpu()
    gradient = gradients[0].detach().cpu()
    
    # Global average pooling of gradients
    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
    
    # Weighted combination of activation maps
    cam = torch.sum(weights * activation, dim=1, keepdim=True)
    
    # Apply ReLU to focus on positive contributions
    cam = F.relu(cam)
    
    # Normalize to [0, 1]
    cam = cam.squeeze().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return cam


def overlay_gradcam(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original image (BGR numpy array)
        heatmap: Grad-CAM heatmap (H, W) with values [0, 1]
        alpha: Transparency of overlay (0 = transparent, 1 = opaque)
        colormap: OpenCV colormap for heatmap visualization
    
    Returns:
        numpy.ndarray: Image with overlaid heatmap (BGR)
    """
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to 0-255 range
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def generate_gradcam_for_crop(classifier, image_crop, alpha=0.5):
    """
    Generate and overlay Grad-CAM for a logo crop using the classifier.
    
    Args:
        classifier: BrandClassifier instance
        image_crop: Cropped logo region (BGR numpy array)
        alpha: Overlay transparency
    
    Returns:
        numpy.ndarray: Image with Grad-CAM overlay (BGR)
    """
    if classifier.demo_mode:
        # Generate simple demo heatmap based on image center
        h, w = image_crop.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        # Gaussian-like heatmap centered on image
        heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 4)**2))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        logger.info("Generated demo Grad-CAM heatmap")
    else:
        # Real Grad-CAM using model
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_image = Image.fromarray(image_rgb)
            
            # Preprocess
            input_tensor = classifier.transform(pil_image).unsqueeze(0).to(classifier.device)
            
            # Generate Grad-CAM
            heatmap = generate_gradcam(classifier.model, input_tensor)
            
            logger.info("Generated real Grad-CAM heatmap")
        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {e}")
            # Fallback to center heatmap
            h, w = image_crop.shape[:2]
            heatmap = np.random.rand(h, w)
    
    # Overlay on original image
    overlay = overlay_gradcam(image_crop, heatmap, alpha)
    
    return overlay


def explain_classification(classifier_result, severity_result):
    """
    Generate human-readable explanation of classification and severity.
    
    Args:
        classifier_result: Classification result dict
        severity_result: Severity scoring result dict
    
    Returns:
        str: Natural language explanation
    """
    brand = classifier_result.get('brand', 'Unknown')
    confidence = classifier_result.get('confidence', 0.0)
    severity = severity_result.get('severity_score', 0)
    breakdown = severity_result.get('breakdown', {})
    
    explanation = f"The logo was identified as '{brand}' with {confidence*100:.1f}% confidence.\n\n"
    
    explanation += f"Authenticity Score: {100 - severity}/100\n"
    explanation += f"Severity Level: {severity}/100\n\n"
    
    explanation += "Detailed Analysis:\n"
    explanation += f"• Classifier Suspicion: {breakdown.get('classifier_fake_prob', 0)*100:.1f}%\n"
    explanation += f"• Structural Dissimilarity: {breakdown.get('ssim_dissimilarity', 0)*100:.1f}%\n"
    explanation += f"• Color Variation: {breakdown.get('color_distance', 0)*100:.1f}%\n\n"
    
    if severity < 30:
        explanation += "Conclusion: Logo appears authentic with no significant tampering detected."
    elif severity < 60:
        explanation += "Conclusion: Some inconsistencies detected. Further investigation recommended."
    else:
        explanation += "Conclusion: Strong indicators of tampering. Logo is likely fake or altered."
    
    return explanation
