"""
FastAPI REST API for Fake Logo Detection Suite
Provides HTTP endpoints for logo detection, batch processing, and enterprise integration.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
from PIL import Image
import io
import asyncio
from datetime import datetime
from pathlib import Path
import secrets
import hashlib

from src.detector import LogoDetector
from src.classifier import BrandClassifier
from src.severity import compute_severity, interpret_severity
from src.tamper import error_level_analysis
from src.explain import generate_gradcam_for_crop
from src.similarity import SimilaritySearcher
from src.db import DetectionDatabase
from src.utils import load_image, compute_image_hash, crop_region, get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fake Logo Detection API",
    description="REST API for detecting and analyzing logo authenticity with deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (loaded once at startup)
detector_sift = None
detector_template = None
detector_yolo = None
classifier = None
similarity_searcher = None
db = None

# Simple API key authentication (replace with proper auth in production)
API_KEYS = {
    "demo_key_12345": {"name": "Demo User", "tier": "basic"},
    "enterprise_key_67890": {"name": "Enterprise Client", "tier": "premium"}
}


# Pydantic models for request/response
class DetectionRequest(BaseModel):
    """Single image detection request."""
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    detection_method: str = Field(default="sift", pattern="^(sift|template|yolo)$")
    include_gradcam: bool = Field(default=True)
    include_similarity: bool = Field(default=True)
    include_tamper_analysis: bool = Field(default=True)


class BatchDetectionRequest(BaseModel):
    """Batch detection request configuration."""
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    detection_method: str = Field(default="sift")
    include_gradcam: bool = Field(default=False)  # Disabled by default for batch
    include_similarity: bool = Field(default=False)
    include_tamper_analysis: bool = Field(default=True)


class LogoDetectionResult(BaseModel):
    """Single logo detection result."""
    bbox: List[int]
    label: str
    confidence: float
    severity_score: int
    severity_level: str
    is_fake: bool


class DetectionResponse(BaseModel):
    """Detection API response."""
    success: bool
    timestamp: str
    processing_time_ms: float
    image_hash: str
    detections: List[Dict[str, Any]]
    ela_result: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class BatchDetectionResponse(BaseModel):
    """Batch detection API response."""
    success: bool
    timestamp: str
    total_images: int
    processed: int
    failed: int
    results: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    version: str
    models_loaded: bool
    detector_method: str


# Dependency for API key authentication
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key from header."""
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key required")
    
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return API_KEYS[x_api_key]


@app.on_event("startup")
async def startup_event():
    """Load models on app startup."""
    global detector_sift, detector_template, detector_yolo, classifier, similarity_searcher, db
    
    logger.info("Loading detection models...")
    # Load separate detector instances for each method (thread-safe)
    detector_sift = LogoDetector(method='sift')
    detector_template = LogoDetector(method='template')
    detector_yolo = LogoDetector(method='yolo', yolo_model_path='models/yolov8_logo.onnx')
    
    classifier = BrandClassifier()
    similarity_searcher = SimilaritySearcher()
    db = DetectionDatabase()
    logger.info("All models loaded successfully")


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Fake Logo Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "detect": "/api/v1/detect",
            "batch": "/api/v1/batch"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": detector_sift is not None,
        "detector_method": "sift,template,yolo"
    }


@app.post("/api/v1/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_logo(
    file: UploadFile = File(...),
    config: DetectionRequest = Depends(),
    user: dict = Depends(verify_api_key)
):
    """
    Detect and analyze logos in a single image.
    
    Args:
        file: Image file (JPEG, PNG)
        config: Detection configuration
        
    Returns:
        Detection results with severity scores and analysis
    """
    start_time = datetime.now()
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Compute image hash
        image_hash = compute_image_hash(image_np)
        
        # Select appropriate detector based on method (thread-safe - no state mutation)
        if config.detection_method == 'yolo':
            detector = detector_yolo
        elif config.detection_method == 'template':
            detector = detector_template
        else:  # default to sift
            detector = detector_sift
        
        # Detect logos
        detections = detector.detect(image_np, confidence_threshold=config.confidence_threshold)
        
        if len(detections) == 0:
            return DetectionResponse(
                success=True,
                timestamp=start_time.isoformat(),
                processing_time_ms=0,
                image_hash=image_hash,
                detections=[],
                message="No logos detected"
            )
        
        # Load reference templates
        templates = {}
        for template_path in Path('data/logos_db').glob('*.png'):
            name = template_path.stem.split('_')[-1].capitalize()
            templates[name] = cv2.imread(str(template_path))
        
        # Process each detection
        processed_detections = []
        
        for det in detections:
            bbox = det['bbox']
            label = det['label']
            
            # Crop logo region
            logo_crop = crop_region(image_np, bbox)
            
            # Classify brand
            classification = classifier.classify(logo_crop)
            
            # Get reference template
            ref_template = templates.get(label.capitalize(), list(templates.values())[0] if templates else logo_crop)
            
            # Compute severity score
            severity = compute_severity(logo_crop, ref_template, classification)
            severity_level, _, _ = interpret_severity(severity['severity_score'])
            
            # Determine if fake
            is_fake = severity['severity_score'] >= 60
            
            result = {
                'bbox': list(bbox),
                'label': classification['brand'],
                'confidence': float(det['confidence']),
                'severity_score': severity['severity_score'],
                'severity_level': severity_level,
                'severity_breakdown': severity['breakdown'],
                'is_fake': is_fake,
                'classification': classification
            }
            
            # Add Grad-CAM if requested
            if config.include_gradcam:
                gradcam_overlay = generate_gradcam_for_crop(classifier, logo_crop)
                if gradcam_overlay is not None:
                    result['has_gradcam'] = True
            
            # Add similarity search if requested
            if config.include_similarity:
                similar_logos = similarity_searcher.find_similar(logo_crop, top_k=3)
                result['similar_logos'] = [
                    {'image': s['image'], 'distance': float(s['distance'])}
                    for s in similar_logos
                ]
            
            processed_detections.append(result)
        
        # Tamper detection on whole image
        ela_result = None
        if config.include_tamper_analysis:
            ela_result = error_level_analysis(image_np)
            ela_result = {
                'is_suspicious': bool(ela_result['is_suspicious']),
                'suspiciousness_score': float(ela_result['suspiciousness_score']),
                'mean_brightness': float(ela_result['mean_brightness'])
            }
        
        # Log to database
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        db.log_detection(
            filename=file.filename,
            image_hash=image_hash,
            detections=[{
                'bbox': d['bbox'],
                'label': d['label'],
                'confidence': d['confidence'],
                'severity': {'severity_score': d['severity_score'], 'breakdown': d['severity_breakdown']},
                'is_fake': d['is_fake']
            } for d in processed_detections],
            processing_time_ms=processing_time
        )
        
        return DetectionResponse(
            success=True,
            timestamp=start_time.isoformat(),
            processing_time_ms=processing_time,
            image_hash=image_hash,
            detections=processed_detections,
            ela_result=ela_result
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/batch", response_model=BatchDetectionResponse, tags=["Detection"])
async def batch_detect_logos(
    files: List[UploadFile] = File(...),
    config: BatchDetectionRequest = Depends(),
    background_tasks: BackgroundTasks = None,
    user: dict = Depends(verify_api_key)
):
    """
    Detect logos in multiple images (batch processing).
    
    Args:
        files: List of image files
        config: Batch detection configuration
        
    Returns:
        Batch detection results
    """
    start_time = datetime.now()
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    results = []
    processed = 0
    failed = 0
    
    # Select appropriate detector based on method (thread-safe)
    if config.detection_method == 'yolo':
        detector = detector_yolo
    elif config.detection_method == 'template':
        detector = detector_template
    else:  # default to sift
        detector = detector_sift
    
    for file in files:
        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Compute image hash
            image_hash = compute_image_hash(image_np)
            
            # Detect logos
            detections = detector.detect(image_np, confidence_threshold=config.confidence_threshold)
            
            # Quick processing for batch (skip expensive operations)
            simple_detections = []
            for det in detections:
                bbox = det['bbox']
                logo_crop = crop_region(image_np, bbox)
                classification = classifier.classify(logo_crop)
                
                # Simple severity estimate (skip SSIM for speed)
                severity_score = int(classification['fake_probability'] * 100)
                severity_level, _, _ = interpret_severity(severity_score)
                
                simple_detections.append({
                    'bbox': list(bbox),
                    'label': classification['brand'],
                    'confidence': float(det['confidence']),
                    'severity_score': severity_score,
                    'severity_level': severity_level,
                    'is_fake': severity_score >= 60
                })
            
            # ELA if requested
            ela_result = None
            if config.include_tamper_analysis:
                ela = error_level_analysis(image_np)
                ela_result = {
                    'is_suspicious': bool(ela['is_suspicious']),
                    'suspiciousness_score': float(ela['suspiciousness_score'])
                }
            
            results.append({
                'filename': file.filename,
                'success': True,
                'image_hash': image_hash,
                'detections': simple_detections,
                'ela_result': ela_result
            })
            processed += 1
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                'filename': file.filename,
                'success': False,
                'error': str(e)
            })
            failed += 1
    
    return BatchDetectionResponse(
        success=True,
        timestamp=start_time.isoformat(),
        total_images=len(files),
        processed=processed,
        failed=failed,
        results=results
    )


@app.get("/api/v1/stats", tags=["Analytics"])
async def get_statistics(user: dict = Depends(verify_api_key)):
    """Get detection statistics from database."""
    stats = db.get_statistics()
    return {
        "success": True,
        "statistics": stats
    }


@app.get("/api/v1/history", tags=["Analytics"])
async def get_detection_history(
    limit: int = 10,
    user: dict = Depends(verify_api_key)
):
    """Get recent detection history."""
    history = db.get_recent_detections(limit=limit)
    return {
        "success": True,
        "history": history
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
