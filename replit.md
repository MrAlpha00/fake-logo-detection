# Fake Logo Detection & Forensics Suite

## Overview

A comprehensive computer vision and deep learning system for detecting, authenticating, and analyzing logos in images. The system combines multiple detection methods (SIFT feature matching, template matching), deep learning classification (MobileNetV2), forensic analysis (Error Level Analysis), explainability (Grad-CAM), and similarity search to provide a complete logo authenticity assessment pipeline.

The application provides an interactive Streamlit web interface for real-time logo analysis, generating severity scores (0-100), tamper detection reports, visual explanations, and PDF forensic reports. All detections are logged to an SQLite database for audit trails.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Streamlit Web Application** (`src/app_streamlit.py`)
- Single-page wide layout with sidebar controls
- Real-time image upload and webcam capture support
- Interactive visualization of detection results, severity scores, and heatmaps
- Batch processing capabilities for multiple images
- PDF report generation triggers

**FastAPI REST API** (`src/api.py` - **NEW**)
- Production-ready REST API for enterprise integration
- Endpoints: /api/v1/detect (single image), /api/v1/batch (batch processing)
- API key authentication for access control
- Async processing with background tasks
- OpenAPI/Swagger documentation at /docs
- Health monitoring and statistics endpoints
- CORS enabled for cross-origin requests
- Runs on port 8000 alongside Streamlit on port 5000

### Backend Architecture

**Modular Pipeline Design** - Each component operates independently with clear interfaces:

1. **Detection Module** (`src/detector.py`)
   - **YOLO Detection (NEW)**: Production-grade YOLOv8 support via ONNX Runtime for improved accuracy
   - Primary method: SIFT feature matching with FLANN-based matcher for fast keypoint matching
   - Alternative: Template matching using OpenCV for simple logo detection
   - ONNX-based YOLO inference with proper sigmoid/objectness handling
   - Custom NMS (Non-Maximum Suppression) implementation
   - Supports models trained with ultralytics and exported to ONNX format
   - Multi-scale detection with configurable confidence thresholds
   - Returns bounding boxes with confidence scores

2. **Classification Module** (`src/classifier.py`)
   - MobileNetV2 transfer learning architecture (6 brand classes: TechCo, Shopmart, Fastfood, Autodrive, Softnet, Mediaplay)
   - Dual mode: Production mode with trained weights, demo mode with deterministic predictions
   - Feature map extraction for Grad-CAM explainability
   - Graceful fallback when torchvision unavailable

3. **Severity Scoring Module** (`src/severity.py`)
   - Weighted combination scoring:
     - 50% classifier confidence (fake probability)
     - 30% SSIM (Structural Similarity Index) comparison
     - 20% color histogram distance (Bhattacharyya)
   - Output: 0-100 severity score with categorical interpretation (Low/Medium/High/Critical)
   - Breakdown dictionary for transparency

4. **Tamper Detection Module** (`src/tamper.py` - **ENHANCED**)
   - **EXIF Metadata Extraction**: Camera model, GPS location, software edits, timestamps
   - **Tampering Indicator Analysis**: Detects editing software usage, timestamp mismatches, stripped metadata
   - **Error Level Analysis (ELA)**: JPEG recompression artifact analysis with visual heatmaps
   - **Clone Detection**: Feature matching to identify copy-pasted regions
   - **Noise Pattern Analysis**: Inconsistency detection across image regions
   - **GPS Coordinate Parsing**: Full GPS metadata extraction (latitude, longitude, altitude)
   - **Forensic Timeline**: Camera capture vs digitization timestamp comparison

5. **Explainability Module** (`src/explain.py`)
   - Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
   - Hook-based gradient capture on final convolutional layer
   - Heatmap overlay generation on original images
   - Natural language explanation generation
   - Demo fallback with center-focused synthetic heatmaps

6. **Similarity Search Module** (`src/similarity.py`)
   - Image embedding using combined features: color histograms, texture descriptors, and shape features
   - Dual indexing support: FAISS (preferred) and Annoy (fallback)
   - K-nearest neighbor retrieval from reference logo database
   - L2-normalized embeddings for consistent distance metrics

7. **Reporting Module** (`src/report.py`)
   - ReportLab-based PDF generation
   - Structured forensic reports with detection metadata, images, severity breakdown, and recommendations
   - Custom styling with corporate header/footer templates

8. **Database Module** (`src/db.py`)
   - SQLite audit logging for all detections
   - Schema tracks: filename, timestamp, detection boxes, severity scores, classification results, tamper flags
   - Query interface for historical analysis and statistics

9. **Utilities Module** (`src/utils.py`)
   - Image loading, saving, and preprocessing (resize with aspect ratio preservation)
   - Bounding box drawing and region cropping
   - Image hashing (MD5) for deduplication
   - Centralized logging configuration with file and console handlers

### Data Storage

**File-based Storage:**
- Reference logos: `data/logos_db/` (6+ brand logo templates)
- Sample test images: `data/samples/` (real and fake variants for demo)
- Model weights: `models/` (classifier checkpoints, similarity indices)
- Generated reports: `reports/` (PDF outputs)
- Demo outputs: `demo_outputs/` (annotated images)

**Database Schema** (SQLite - `detections.db`):
- Primary table: `detections` with columns for image metadata, bbox coordinates, severity scores, classification results, timestamps
- Supports CRUD operations through DetectionDatabase class

### Training Pipeline

**Classifier Training** (`train/train_classifier.py` - **PRODUCTION-READY**)
- PyTorch-based training loop for MobileNetV2 transfer learning
- Custom LogoDataset class with automatic class discovery
- Training-time augmentation: rotations, flips, color jitter, normalization
- Validation split with separate evaluation pipeline
- Learning rate scheduling with ReduceLROnPlateau
- Automatic best model checkpointing based on validation accuracy
- Class name persistence for production deployment
- Supports CLI arguments for full configuration

**Data Preparation** (`train/prepare_training_data.py` - **NEW**)
- Automated dataset organization from raw logo images
- Comprehensive augmentation pipeline with 10 techniques:
  * Random rotation, scaling, perspective transforms
  * Brightness, contrast, saturation adjustments
  * Gaussian noise, blur, cropping
  * JPEG compression artifact simulation
- Configurable augmentation factor (default: 10x expansion)
- Progress tracking and statistics reporting
- Production-ready for real-world training workflows

**Training Guide** (`train/TRAINING_GUIDE.md` - **NEW**)
- Complete end-to-end training documentation
- Data collection best practices (50-500 images per brand)
- Dataset diversity guidelines (size, background, quality, angle, lighting)
- Step-by-step preparation, augmentation, and training instructions
- Hardware requirements and performance benchmarks
- Troubleshooting guide for low accuracy (<80%)
- Production deployment checklist
- Advanced techniques: fine-tuning, transfer learning, active learning

**YOLO Training** (`train/train_yolo.py` - **NEW**)
- Comprehensive training guide for YOLOv8 logo detection models
- Sample training scripts and dataset templates
- Instructions for ultralytics training and ONNX export
- Annotation tool recommendations (Label Studio, LabelImg, Roboflow)
- Data collection best practices for logo datasets
- Model export to ONNX format for production deployment

### Design Patterns

**Separation of Concerns:** Each module (detection, classification, severity, tamper, etc.) is self-contained with clear input/output contracts

**Graceful Degradation:** System operates in demo mode when heavy dependencies (torchvision, FAISS) are unavailable, using deterministic fallbacks

**Hook-based Architecture:** Grad-CAM uses PyTorch hooks for gradient capture without modifying model architecture

**Factory Pattern:** SimilaritySearcher dynamically selects FAISS or Annoy based on availability

## External Dependencies

### Core Libraries
- **OpenCV (cv2)**: Image processing, SIFT detection, template matching, color space conversions
- **PyTorch**: Deep learning framework for MobileNetV2 classifier and Grad-CAM
- **torchvision**: Pre-trained models and transforms (optional with fallback)
- **NumPy**: Array operations and numerical computing
- **Pillow (PIL)**: Image I/O and JPEG compression for ELA

### Web Framework
- **Streamlit**: Interactive web UI with file upload, webcam support, and visualization
- **FastAPI**: Production REST API framework with async support (NEW)
- **Uvicorn**: ASGI server for FastAPI (NEW)
- **Pydantic**: Data validation and serialization (NEW)

### Computer Vision & ML
- **scikit-image**: SSIM computation for structural similarity
- **FAISS**: High-performance similarity search (optional, falls back to Annoy)
- **Annoy**: Approximate nearest neighbor search (fallback for FAISS)
- **ONNX Runtime**: Inference engine for YOLO models (NEW)

### Reporting & Database
- **ReportLab**: PDF generation with tables, images, and custom styling
- **SQLite3**: Embedded database for audit logging (Python standard library)

### Testing
- **pytest**: Unit testing framework for all modules

### Asset Generation
- **ImageDraw/ImageFont** (Pillow): Synthetic logo generation for demo (`generate_demo_assets.py`)

### Configuration Files
- `requirements.txt`: Full dependency list with version constraints
- `.pytest_cache/`: Test artifacts (not version controlled)

### No External API Dependencies
All processing occurs locally without external API calls for privacy and performance.