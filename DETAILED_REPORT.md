# Fake Logo Detection & Forensics Suite
## Comprehensive Technical Documentation

**Version:** 1.0  
**Date:** November 20, 2025  
**Document Type:** Complete Technical Reference & System Architecture Guide

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Core Components & Modules](#3-core-components--modules)
4. [Detection Pipeline Workflow](#4-detection-pipeline-workflow)
5. [Module-by-Module Deep Dive](#5-module-by-module-deep-dive)
6. [Database Schema & Data Management](#6-database-schema--data-management)
7. [Web Interface & API](#7-web-interface--api)
8. [Training & Model Development](#8-training--model-development)
9. [Deployment & Configuration](#9-deployment--configuration)
10. [Troubleshooting & Maintenance](#10-troubleshooting--maintenance)

---

## 1. Executive Summary

### 1.1 Project Overview

The Fake Logo Detection & Forensics Suite is a comprehensive computer vision and machine learning system designed to detect, authenticate, and analyze logos in images. The system combines multiple state-of-the-art detection methods, deep learning classification, forensic image analysis, and explainability features to provide accurate assessments of logo authenticity.

**Primary Use Cases:**
- Brand protection and counterfeit detection
- E-commerce verification
- Social media content moderation
- Legal evidence gathering
- Trademark enforcement

**Key Features:**
- Multi-method logo detection (SIFT, Template Matching, YOLO)
- Deep learning classification using MobileNetV2
- Forensic analysis (Error Level Analysis, EXIF metadata, clone detection)
- Explainable AI using Grad-CAM visualizations
- Similarity search across logo databases
- Real-time online logo fetching via Brandfetch API
- Comprehensive analytics dashboards
- PDF forensic report generation
- RESTful API for enterprise integration

### 1.2 Technology Stack

**Core Technologies:**
- **Python 3.11** - Primary programming language
- **OpenCV** - Computer vision operations
- **PyTorch** - Deep learning framework
- **Streamlit** - Web interface
- **FastAPI** - REST API framework
- **SQLite** - Database for audit logging
- **ONNX Runtime** - Optimized model inference

**Key Libraries:**
- Computer Vision: opencv-python, scikit-image, Pillow
- Machine Learning: torch, torchvision, onnxruntime
- Data Processing: numpy, pandas
- Visualization: plotly, matplotlib
- Document Generation: reportlab
- Vector Search: FAISS, Annoy

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture

The system follows a modular, layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend Layer                          │
│  ┌──────────────────┐         ┌─────────────────────────┐  │
│  │ Streamlit Web UI │         │   FastAPI REST API       │  │
│  └──────────────────┘         └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Processing Layer                          │
│  ┌───────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ Detector  │  │  Classifier  │  │  Tamper Detector    │ │
│  └───────────┘  └──────────────┘  └─────────────────────┘ │
│  ┌───────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ Severity  │  │  Explainer   │  │  Similarity Search  │ │
│  └───────────┘  └──────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data & Storage Layer                      │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │  SQLite DB   │  │  Logo Cache │  │  Model Storage   │  │
│  └──────────────┘  └─────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Design Patterns

**1. Separation of Concerns**
Each module (detector, classifier, tamper analysis, etc.) is self-contained with clear input/output contracts. This enables:
- Independent testing
- Easy maintenance
- Parallel development
- Module replacement without system-wide changes

**2. Graceful Degradation**
The system operates in demo mode when heavy dependencies are unavailable:
- Missing torchvision → deterministic classification fallback
- Missing FAISS → Annoy-based similarity search
- Missing YOLO model → SIFT/template matching only

**3. Hook-based Architecture**
Grad-CAM uses PyTorch hooks for gradient capture without modifying model architecture, allowing clean integration with any CNN model.

**4. Factory Pattern**
SimilaritySearcher dynamically selects FAISS or Annoy based on availability, providing consistent interface regardless of backend.

---

## 3. Core Components & Modules

### 3.1 Component Overview

| Module | File | Primary Function | Key Dependencies |
|--------|------|------------------|------------------|
| **Detector** | `src/detector.py` | Logo detection using SIFT, Template Matching, YOLO | OpenCV, ONNX Runtime |
| **Classifier** | `src/classifier.py` | Brand classification using MobileNetV2 | PyTorch, torchvision |
| **Severity** | `src/severity.py` | Fake probability scoring | scikit-image, NumPy |
| **Tamper** | `src/tamper.py` | Forensic analysis (ELA, EXIF, cloning) | Pillow, OpenCV |
| **Explainer** | `src/explain.py` | Grad-CAM visualization | PyTorch |
| **Similarity** | `src/similarity.py` | Vector search for logo matching | FAISS/Annoy |
| **Reporter** | `src/report.py` | PDF report generation | ReportLab |
| **Database** | `src/db.py` | SQLite audit logging | sqlite3 |
| **Analytics** | `src/analytics.py` | Metrics and visualizations | Plotly, Pandas |
| **Logo Fetcher** | `src/logo_fetcher.py` | Online logo retrieval | Brandfetch API |
| **Utilities** | `src/utils.py` | Image processing, logging | OpenCV, logging |

---

## 4. Detection Pipeline Workflow

### 4.1 End-to-End Processing Flow

When a user uploads an image for analysis, the system executes the following pipeline:

```
1. IMAGE UPLOAD
   ↓
2. PREPROCESSING
   - Resize to max 1024px (preserve aspect ratio)
   - Convert color space if needed
   - Calculate image hash for deduplication
   ↓
3. LOGO DETECTION
   - Run SIFT feature matching against reference logos
   - Run template matching for simple logos
   - Run YOLO if model available
   - Extract bounding boxes + confidence scores
   ↓
4. REGION EXTRACTION
   - Crop detected logo regions
   - Normalize and preprocess for classification
   ↓
5. CLASSIFICATION
   - Feed each region to MobileNetV2
   - Get brand prediction + fake probability
   - Extract feature maps for Grad-CAM
   ↓
6. SEVERITY SCORING
   - Classifier confidence (50% weight)
   - SSIM structural similarity (30% weight)
   - Color histogram distance (20% weight)
   - Compute final 0-100 severity score
   ↓
7. FORENSIC ANALYSIS
   - Extract EXIF metadata (camera, GPS, timestamps)
   - Perform Error Level Analysis (ELA)
   - Clone detection via feature matching
   - Noise pattern analysis
   ↓
8. EXPLAINABILITY
   - Generate Grad-CAM heatmap
   - Overlay on original image
   - Create natural language explanation
   ↓
9. SIMILARITY SEARCH
   - Compute image embedding
   - Query vector index (FAISS/Annoy)
   - Return top-k similar logos
   ↓
10. ANALYTICS
    - Compute comprehensive metrics
    - Generate visualization charts
    - Create summary statistics
    ↓
11. DATABASE LOGGING
    - Store detection metadata
    - Log tamper analysis results
    - Update statistics
    ↓
12. REPORT GENERATION (optional)
    - Create PDF with all findings
    - Include images, charts, recommendations
```

### 4.2 Processing Time Breakdown

Typical processing times on standard hardware:
- Image preprocessing: 10-50ms
- SIFT detection: 100-300ms per template
- Classification: 50-150ms per detection
- Severity scoring: 20-50ms
- Tamper analysis: 200-500ms
- Total average: 1-2 seconds per image

---

## 5. Module-by-Module Deep Dive

### 5.1 Detection Module (`src/detector.py`)

#### Purpose
The detector module is responsible for identifying logo regions in images using multiple complementary methods.

#### Detection Methods

**1. SIFT (Scale-Invariant Feature Transform)**
```python
class LogoDetector:
    def detect_sift(self, image, min_matches=4, lowe_ratio=0.8):
        """
        SIFT-based detection workflow:
        1. Extract keypoints and descriptors from query image
        2. Match against each reference logo template
        3. Apply Lowe's ratio test to filter good matches
        4. Use RANSAC for homography estimation
        5. Transform bounding box to query image coordinates
        
        Parameters:
        - min_matches: Minimum keypoint matches required (default: 4)
        - lowe_ratio: Lowe's ratio for filtering matches (default: 0.8)
        
        Why these values:
        - min_matches=4: Minimum for homography estimation
        - lowe_ratio=0.8: Balances precision vs recall
        """
```

**How SIFT Works:**
- Detects keypoints invariant to scale, rotation, illumination
- Computes 128-dimensional descriptors for each keypoint
- Uses FLANN-based matcher for fast neighbor search
- Applies geometric verification via homography

**When SIFT Excels:**
- Rotated logos
- Scaled logos (different sizes)
- Partial occlusions
- Lighting variations

**When SIFT Struggles:**
- Very small logos (<50px)
- Heavily compressed images
- Logos with few features (simple shapes/text)

**2. Template Matching**
```python
def detect_template(self, image, threshold=0.6):
    """
    Template matching workflow:
    1. Convert both images to grayscale
    2. Apply multi-scale template matching
    3. Use normalized cross-correlation
    4. Dynamic threshold: base_threshold - (0.1 * scale_reduction)
    5. Non-maximum suppression to remove duplicates
    
    Dynamic threshold logic:
    - Larger scales (close to original): High threshold
    - Smaller scales (distant logos): Lower threshold
    - Prevents false positives while catching small logos
    """
```

**How Template Matching Works:**
- Slides template across image at multiple scales
- Computes correlation coefficient at each position
- Peaks indicate potential matches
- Works best for rigid, non-deformed logos

**When Template Matching Excels:**
- Logos with consistent appearance
- High-quality images
- Frontal/planar views

**When Template Matching Struggles:**
- Rotated logos
- Perspective distortion
- Color-dependent logos

**3. YOLO (You Only Look Once)**
```python
def detect_yolo(self, image, conf_threshold=0.25, iou_threshold=0.45):
    """
    YOLO detection workflow:
    1. Preprocess image (letterbox, normalize, BGR→RGB)
    2. Run ONNX inference
    3. Apply sigmoid to logits (if model outputs raw logits)
    4. Filter by confidence threshold
    5. Apply Non-Maximum Suppression (NMS)
    6. Rescale boxes to original image coordinates
    
    ONNX Optimizations:
    - Compiled model runs 2-3x faster than PyTorch
    - Supports batch inference
    - CPU and GPU compatible
    """
```

**How YOLO Works:**
- Single forward pass through CNN
- Predicts bounding boxes + class probabilities
- Grid-based detection (divides image into SxS grid)
- Much faster than region proposal methods

**When YOLO Excels:**
- Real-time requirements
- Multiple logos in image
- Various scales and orientations
- Requires trained model (see Training section)

**When YOLO Struggles:**
- Very small logos (<32px)
- Rare/unseen brand classes
- Limited training data

#### Detection Thresholds

Current optimized values:
```python
# SIFT parameters
MIN_MATCHES = 4        # Lowered from 6 for better recall
LOWE_RATIO = 0.8      # Increased from 0.75 for more matches

# Template matching
BASE_THRESHOLD = 0.6   # Starting threshold
MIN_THRESHOLD = 0.2    # Floor to prevent garbage

# YOLO
CONF_THRESHOLD = 0.25  # Confidence cutoff
IOU_THRESHOLD = 0.45   # NMS overlap threshold
```

**Tuning Guidelines:**
- **Higher recall needed?** → Lower min_matches, increase lowe_ratio
- **Higher precision needed?** → Raise thresholds, increase min_matches
- **Real-world images?** → Use current tuned values
- **Lab/controlled?** → Can use stricter thresholds

### 5.2 Classification Module (`src/classifier.py`)

#### Purpose
Classifies detected logo regions into brand categories and estimates authenticity.

#### Architecture

**MobileNetV2 Transfer Learning**
```python
Model Architecture:
Input (224x224x3)
    ↓
MobileNetV2 Backbone (pretrained on ImageNet)
    ↓ (features extracted from conv layer before classifier)
Feature Maps (1280 channels)
    ↓
Global Average Pooling
    ↓
Fully Connected (1280 → 512)
    ↓
ReLU + Dropout(0.3)
    ↓
Output Layer (512 → num_classes)
    ↓
Softmax → Class Probabilities
```

**Why MobileNetV2?**
- Lightweight: Only 3.5M parameters
- Fast inference: ~50ms on CPU
- Excellent accuracy/speed tradeoff
- Pretrained features transfer well to logo domain

#### Dual-Mode Operation

**1. Production Mode (Trained Model)**
```python
if os.path.exists(model_path):
    # Load trained weights
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state'])
    class_names = checkpoint['class_names']
    # Run real inference
```

**2. Demo Mode (Fallback)**
```python
# Deterministic predictions for demonstration
fake_prob = 0.15 + (hash(image_bytes) % 70) / 100.0
brand_idx = hash(image_bytes) % len(demo_brands)
```

Demo mode enables:
- Testing without trained model
- Demonstrations without GPU
- Development without full setup

#### Classification Process

```python
def predict(self, logo_region):
    """
    Classification workflow:
    1. Resize to 224x224 (MobileNetV2 input size)
    2. Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    3. Convert to tensor, add batch dimension
    4. Forward pass through model
    5. Apply softmax to logits
    6. Extract top prediction + confidence
    7. Calculate fake probability (1 - confidence)
    
    Returns:
    {
        'brand': str,              # Predicted brand name
        'confidence': float,       # 0-1 confidence
        'fake_probability': float, # 1 - confidence
        'all_probs': dict         # All class probabilities
    }
    """
```

#### Feature Extraction for Grad-CAM

```python
def get_feature_maps(self, logo_region):
    """
    Extracts intermediate feature maps for explainability:
    1. Register forward hook on final conv layer
    2. Run forward pass
    3. Capture activations before pooling
    4. Return feature maps (shape: [1, 1280, 7, 7])
    
    These feature maps show what the network "sees"
    and are used to generate Grad-CAM heatmaps.
    """
```

### 5.3 Severity Scoring Module (`src/severity.py`)

#### Purpose
Combines multiple signals to produce a single, interpretable severity score (0-100).

#### Scoring Algorithm

**Weighted Combination Formula:**
```
severity_score = (0.5 × classifier_score) + 
                 (0.3 × ssim_score) + 
                 (0.2 × color_score)

Where:
- classifier_score = fake_probability × 100
- ssim_score = (1 - SSIM) × 100  
- color_score = color_distance × 100
```

**Component Details:**

**1. Classifier Component (50% weight)**
```python
classifier_score = fake_probability * 100

Rationale:
- Neural network learned from thousands of examples
- Captures complex patterns humans might miss
- Most important signal, hence 50% weight
```

**2. SSIM Component (30% weight)**
```python
# Structural Similarity Index
ssim_value = ssim(reference_logo, detected_logo, multichannel=True)
ssim_dissimilarity = (1 - ssim_value) * 100

What SSIM measures:
- Luminance similarity
- Contrast similarity  
- Structure similarity

Perfect match: SSIM = 1.0 → dissimilarity = 0
Completely different: SSIM = 0.0 → dissimilarity = 100

Why 30% weight:
- Objective structural comparison
- Catches geometric distortions
- Complements classifier
```

**3. Color Component (20% weight)**
```python
# Bhattacharyya distance between color histograms
hist_query = cv2.calcHist([logo_bgr], [0,1,2], None, [8,8,8], [0,256]*3)
hist_ref = cv2.calcHist([template_bgr], [0,1,2], None, [8,8,8], [0,256]*3)
color_distance = cv2.compareHist(hist_query, hist_ref, cv2.HISTCMP_BHATTACHARYYA)

Why color distance:
- Counterfeit logos often have color mismatches
- Simple to compute and interpret
- Works even with shape similarities

Why 20% weight:
- Legitimate logos can have approved color variations
- Less reliable than structure and classifier
```

#### Severity Levels

```python
SEVERITY_LEVELS = {
    'LOW': (0, 30),      # Green - Likely authentic
    'MEDIUM': (30, 60),  # Yellow - Suspicious
    'HIGH': (60, 80),    # Orange - Likely fake
    'CRITICAL': (80, 100) # Red - Almost certainly fake
}
```

#### Example Calculation

```python
# Example: Suspicious Starbucks logo
classifier_fake_prob = 0.72      # 72% fake
ssim = 0.65                       # 65% similar structurally
color_distance = 0.45             # 45% color difference

# Component scores
classifier_component = 0.72 * 100 = 72.0
ssim_component = (1 - 0.65) * 100 = 35.0
color_component = 0.45 * 100 = 45.0

# Weighted combination
severity = (0.5 * 72.0) + (0.3 * 35.0) + (0.2 * 45.0)
         = 36.0 + 10.5 + 9.0
         = 55.5 → MEDIUM severity (suspicious)
```

### 5.4 Tamper Detection Module (`src/tamper.py`)

#### Purpose
Forensic analysis to detect image manipulation and gather metadata evidence.

#### Forensic Techniques

**1. Error Level Analysis (ELA)**
```python
def error_level_analysis(image, quality=95):
    """
    How ELA works:
    1. Save image at known quality (e.g., 95)
    2. Reload and compare with original
    3. Differences indicate compression inconsistencies
    4. Tampered regions often have different error levels
    
    Theory:
    - Original JPEG compressed once at capture time
    - Edited regions recompressed when saved
    - Multiple compressions create detectable artifacts
    - ELA highlights these differences
    
    Output:
    - ELA difference image (grayscale)
    - Mean brightness (higher = more suspicious)
    - Suspicious regions visualization
    """
```

**When ELA detects tampering:**
- Copy-paste from different source
- Clone stamp tool usage
- Digital overlays/watermarks added
- Selective region editing

**ELA Limitations:**
- Doesn't work on PNG (lossless)
- Multiple full recompressions reduce signal
- Modern tools can match compression levels

**2. EXIF Metadata Extraction**
```python
def extract_metadata(image):
    """
    Extracted EXIF fields:
    - Camera make/model
    - Software used (Photoshop, GIMP = red flag)
    - GPS coordinates (latitude, longitude, altitude)
    - Timestamps (DateTimeOriginal, DateTimeDigitized)
    - Image dimensions
    - Orientation
    
    Tampering indicators:
    - Software field present → likely edited
    - Timestamp mismatches → manipulation
    - Missing EXIF → stripped to hide editing
    - GPS inconsistencies → potential fraud
    """
```

**3. Clone Detection**
```python
def detect_cloning(image):
    """
    Clone detection workflow:
    1. Extract SIFT keypoints from image
    2. Match keypoints within same image
    3. Filter matches with distance < threshold
    4. Cluster nearby matches
    5. Identify suspicious repeated patterns
    
    Cloning indicators:
    - High number of self-matches
    - Geometric patterns in matches
    - Repeated regions with slight offsets
    
    Use case:
    - Detect clone stamp tool usage
    - Find duplicated logo elements
    - Identify pattern repetition fraud
    """
```

**4. Noise Pattern Analysis**
```python
def analyze_noise_patterns(image):
    """
    Noise analysis workflow:
    1. Divide image into grid (e.g., 8x8 blocks)
    2. Compute standard deviation of each block
    3. Calculate variance of block variances
    4. High variance = inconsistent noise (suspicious)
    
    Theory:
    - Camera sensors produce consistent noise
    - Spliced regions have different noise characteristics
    - Edited regions show noise inconsistencies
    
    Threshold:
    - Variance < 100: Consistent noise (authentic)
    - Variance > 100: Inconsistent (tampered)
    """
```

### 5.5 Explainability Module (`src/explain.py`)

#### Purpose
Makes AI predictions interpretable through visualization and natural language.

#### Grad-CAM Implementation

**Gradient-weighted Class Activation Mapping**
```python
def generate_gradcam(model, image, target_class):
    """
    Grad-CAM algorithm:
    
    1. Forward Pass:
       - Input image through network
       - Extract feature maps A from final conv layer
       - Get prediction scores
    
    2. Backward Pass:
       - Compute gradient of target class score w.r.t. feature maps
       - ∂y^c / ∂A^k (gradient of class c w.r.t. activation map k)
    
    3. Pooling:
       - Global average pool gradients
       - α_k^c = (1/Z) Σ Σ (∂y^c / ∂A^k)
       - These α values are importance weights
    
    4. Weighted Combination:
       - L^c = ReLU(Σ α_k^c · A^k)
       - Combine weighted feature maps
       - ReLU removes negative influences
    
    5. Upsampling:
       - Resize heatmap to original image size
       - Overlay with colormap (red = high activation)
    
    Output:
    - Heatmap showing which regions influenced prediction
    - Red areas = strong positive influence
    - Blue areas = weak influence
    """
```

**Why Grad-CAM?**
- Class-discriminative (specific to predicted class)
- High-resolution compared to CAM
- Works with any CNN architecture
- No retraining required
- Visually intuitive

#### Natural Language Explanations

```python
def generate_explanation(brand, confidence, severity):
    """
    Explanation generation logic:
    
    1. Assess overall authenticity
    2. Identify key factors (high/low)
    3. Provide specific findings
    4. Generate actionable recommendations
    
    Template structure:
    - What was detected: "Detected {brand} logo..."
    - Confidence level: "with {confidence} confidence"
    - Severity assessment: "Severity score: {score} ({level})"
    - Key factors: "Structural similarity: X%, Color match: Y%"
    - Recommendation: "Recommendation: Likely authentic/Further investigation"
    
    Adapts based on:
    - Severity level (LOW/MEDIUM/HIGH/CRITICAL)
    - Confidence threshold (>80% = confident, <50% = uncertain)
    - Tamper detection findings
    """
```

### 5.6 Similarity Search Module (`src/similarity.py`)

#### Purpose
Finds visually similar logos in database using vector search.

#### Embedding Generation

```python
def compute_embedding(image):
    """
    Multi-feature embedding:
    
    1. Color Histogram (512-dim):
       - RGB histograms (16 bins per channel)
       - HSV histograms (16 bins per channel)
       - Captures color distribution
    
    2. HOG (Histogram of Oriented Gradients) (1764-dim):
       - Edge direction histograms
       - Captures shape/structure
       - Cell size: 8x8, Block size: 2x2
    
    3. Hu Moments (7-dim):
       - Scale/rotation invariant moments
       - Global shape descriptors
    
    Total embedding: 2283 dimensions
    L2-normalized for cosine similarity
    """
```

#### Indexing Strategies

**1. FAISS (Facebook AI Similarity Search)**
```python
# IndexFlatL2: Exact nearest neighbor
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)  # Add all logo embeddings

# Search (returns distances and indices)
distances, indices = index.search(query_embedding, k=5)
```

**Advantages:**
- Exact results (not approximate)
- Blazing fast (optimized C++)
- GPU support for large databases
- Production-ready

**2. Annoy (Approximate Nearest Neighbors)**
```python
# Build tree index
annoy_index = AnnoyIndex(embedding_dim, metric='angular')
for i, emb in enumerate(embeddings):
    annoy_index.add_item(i, emb)
annoy_index.build(n_trees=10)

# Search
indices = annoy_index.get_nns_by_vector(query_embedding, k=5)
```

**Advantages:**
- Memory efficient
- Good for large databases (millions of vectors)
- Approximate but fast
- Fallback when FAISS unavailable

#### Similarity Metrics

```python
# Cosine similarity (used)
similarity = 1 - cosine_distance
similarity = emb1 · emb2 / (||emb1|| ||emb2||)

# Why cosine over L2?
# - Invariant to embedding magnitude
# - Better for high-dimensional sparse vectors
# - More intuitive (0-1 range)
```

---

## 6. Database Schema & Data Management

### 6.1 SQLite Schema

**Table: `detections`**
```sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    image_hash TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    num_detections INTEGER,
    processing_time_ms REAL,
    INDEX idx_timestamp ON detections(timestamp),
    INDEX idx_hash ON detections(image_hash)
);
```

**Table: `logo_detections`**
```sql
CREATE TABLE logo_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL,
    brand TEXT NOT NULL,
    confidence REAL,
    bbox TEXT,  -- JSON: {"x1": 10, "y1": 20, "x2": 100, "y2": 120}
    severity_score REAL,
    breakdown TEXT,  -- JSON: {"classifier": 0.7, "ssim": 0.3, "color": 0.2}
    is_fake INTEGER,  -- 0 or 1
    FOREIGN KEY (detection_id) REFERENCES detections(id),
    INDEX idx_brand ON logo_detections(brand),
    INDEX idx_severity ON logo_detections(severity_score)
);
```

**Table: `tamper_analysis`**
```sql
CREATE TABLE tamper_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_id INTEGER NOT NULL,
    ela_suspicious INTEGER,
    ela_score REAL,
    has_exif INTEGER,
    has_gps INTEGER,
    camera_make TEXT,
    software TEXT,
    clone_detected INTEGER,
    FOREIGN KEY (detection_id) REFERENCES detections(id)
);
```

### 6.2 Data Flow

```
User Upload → Detection → Database Write
     ↓
[detections] record created
     ↓
For each detected logo:
     ↓
[logo_detections] record created
     ↓
[tamper_analysis] record created
```

### 6.3 Query Examples

**Get all high-severity detections:**
```sql
SELECT d.filename, d.timestamp, l.brand, l.severity_score
FROM detections d
JOIN logo_detections l ON d.id = l.detection_id
WHERE l.severity_score >= 70
ORDER BY d.timestamp DESC;
```

**Brand distribution:**
```sql
SELECT brand, COUNT(*) as count, AVG(severity_score) as avg_severity
FROM logo_detections
GROUP BY brand
ORDER BY count DESC;
```

**Daily detection stats:**
```sql
SELECT DATE(timestamp) as date, 
       COUNT(*) as total_detections,
       SUM(num_detections) as total_logos
FROM detections
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

---

## 7. Web Interface & API

### 7.1 Streamlit Web Application

**File:** `src/app_streamlit.py`

#### Main Components

**1. Sidebar Configuration**
```python
Sidebar Elements:
├── Detection Method Selector (SIFT/Template/YOLO)
├── Confidence Threshold Slider (0.0 - 1.0)
├── Show Grad-CAM Heatmaps (checkbox)
├── Show Similar Logos (checkbox)
├── Show Error Level Analysis (checkbox)
├── Online Logo Fetcher
│   ├── Domain/Name Input
│   ├── Fetch Button
│   └── Add to Database Option
└── Statistics
    ├── Total Detections
    ├── Total Logos Found
    └── Fake Logos Detected
```

**2. Main Tabs**
```python
Tabs:
├── Upload & Analyze
│   ├── File Upload (single/batch)
│   ├── Camera Capture
│   ├── Demo Images
│   └── Results Display
├── Analytics Dashboard
│   ├── Severity Distribution Chart
│   ├── Component Breakdown
│   ├── Confidence Scatter Plot
│   └── Risk Pie Chart
├── Detection History
│   └── Past Detections Table
└── About
    └── Project Information
```

**3. Detection Results Display**
```python
Results Layout:
├── Annotated Image (bboxes + labels)
├── Detection Summary
│   ├── Number of logos found
│   ├── Average severity
│   └── Processing time
├── Per-Logo Details
│   ├── Cropped Region
│   ├── Severity Gauge
│   ├── Classification Info
│   ├── Breakdown Scores
│   └── Grad-CAM Heatmap (if enabled)
├── Forensic Analysis
│   ├── EXIF Metadata Table
│   ├── ELA Visualization
│   └── Tampering Indicators
├── Analytics Dashboard
│   ├── Summary Statistics Table
│   └── Visualization Charts (4 charts)
└── Actions
    └── Generate PDF Report Button
```

### 7.2 FastAPI REST API

**File:** `src/api.py`

#### Endpoints

**1. Health Check**
```python
GET /
Response: {
    "status": "healthy",
    "version": "1.0",
    "models_loaded": true
}
```

**2. Single Image Detection**
```python
POST /api/v1/detect
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- method: Detection method [sift|template|yolo] (default: sift)
- confidence_threshold: Float 0-1 (default: 0.3)

Response: {
    "filename": str,
    "num_detections": int,
    "processing_time_ms": float,
    "detections": [
        {
            "label": str,
            "confidence": float,
            "bbox": [x1, y1, x2, y2],
            "severity": {
                "severity_score": float,
                "level": str,
                "breakdown": {...}
            },
            "classification": {...}
        }
    ],
    "ela_result": {...},
    "exif_data": {...}
}
```

**3. Batch Detection**
```python
POST /api/v1/batch
Content-Type: multipart/form-data

Parameters:
- files: List of image files (required)
- method: Detection method (default: sift)
- confidence_threshold: Float 0-1 (default: 0.3)

Response: {
    "total_images": int,
    "results": [
        {result object for each image}
    ],
    "total_processing_time_ms": float
}
```

**4. Statistics**
```python
GET /api/v1/statistics

Response: {
    "total_detections": int,
    "total_logos": int,
    "total_fakes": int,
    "brand_distribution": {...}
}
```

**5. API Documentation**
```
GET /docs  → Swagger UI
GET /redoc → ReDoc UI
```

#### Authentication

```python
# API Key authentication
headers = {
    "X-API-Key": "your-api-key-here"
}

# Set in environment
API_KEY = os.getenv("API_KEY", "default-key-change-in-production")
```

#### Example Usage

**Python:**
```python
import requests

url = "http://localhost:8000/api/v1/detect"
headers = {"X-API-Key": "your-key"}
files = {"file": open("logo.jpg", "rb")}
data = {"method": "sift", "confidence_threshold": 0.3}

response = requests.post(url, headers=headers, files=files, data=data)
result = response.json()

print(f"Found {result['num_detections']} logos")
for det in result['detections']:
    print(f"- {det['label']}: {det['confidence']:.2f} confidence")
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/detect" \\
  -H "X-API-Key: your-key" \\
  -F "file=@logo.jpg" \\
  -F "method=sift" \\
  -F "confidence_threshold=0.3"
```

---

## 8. Training & Model Development

### 8.1 Classifier Training

**File:** `train/train_classifier.py`

#### Training Pipeline

```python
Training Workflow:
1. Data Preparation
   ├── Organize images into folders by brand
   ├── data/training/
   │   ├── Nike/
   │   ├── Adidas/
   │   └── ...
   └── Run prepare_training_data.py for augmentation

2. Dataset Loading
   ├── LogoDataset class
   ├── Train/Val split (80/20)
   └── Data loaders (batch_size=32)

3. Model Setup
   ├── MobileNetV2 backbone (pretrained=True)
   ├── Custom classifier head
   ├── Loss: CrossEntropyLoss
   └── Optimizer: Adam (lr=0.001)

4. Training Loop
   ├── Forward pass
   ├── Loss computation
   ├── Backward pass
   ├── Gradient clipping (max_norm=1.0)
   ├── Weight update
   └── Learning rate scheduling

5. Validation
   ├── Evaluate on validation set each epoch
   ├── Track accuracy and loss
   ├── Save best model checkpoint
   └── Early stopping (patience=5)

6. Model Saving
   └── torch.save({
         'model_state': model.state_dict(),
         'optimizer_state': optimizer.state_dict(),
         'class_names': class_names,
         'epoch': epoch,
         'accuracy': best_accuracy
       }, 'models/logo_classifier.pth')
```

#### Hyperparameters

```python
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
LR_SCHEDULER = "ReduceLROnPlateau"
LR_PATIENCE = 3
LR_FACTOR = 0.5
```

#### Data Augmentation

**Training Augmentation:**
```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

**Validation Augmentation:**
```python
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### 8.2 YOLO Training

**File:** `train/train_yolo.py`

#### Dataset Preparation

```python
YOLO Dataset Structure:
dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── ...
│   └── val/
│       ├── img101.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img001.txt  # One label per image
    │   └── ...
    └── val/
        ├── img101.txt
        └── ...

Label Format (YOLO):
<class_id> <x_center> <y_center> <width> <height>

All values normalized to [0, 1]:
- x_center = (x1 + x2) / 2 / image_width
- y_center = (y1 + y2) / 2 / image_height
- width = (x2 - x1) / image_width
- height = (y2 - y1) / image_height
```

#### Training Command

```bash
# Using ultralytics
yolo detect train \\
  data=logo_dataset.yaml \\
  model=yolov8n.pt \\
  epochs=100 \\
  imgsz=640 \\
  batch=16 \\
  device=0 \\
  project=logo_detection \\
  name=yolov8_logo
```

#### ONNX Export

```python
# Export trained model to ONNX
from ultralytics import YOLO

model = YOLO('runs/detect/yolov8_logo/weights/best.pt')
model.export(format='onnx', opset=12, simplify=True)

# Result: best.onnx (can be used with ONNX Runtime)
```

### 8.3 Data Preparation

**File:** `train/prepare_training_data.py`

#### Augmentation Techniques

```python
Augmentation Pipeline (10x expansion):
1. Random Rotation (-30° to +30°)
2. Random Scaling (0.8x to 1.2x)
3. Perspective Transform (slight distortion)
4. Brightness Adjustment (±30%)
5. Contrast Adjustment (±30%)
6. Saturation Adjustment (±30%)
7. Gaussian Noise (σ=10)
8. Gaussian Blur (kernel=3)
9. Random Cropping (90-100% of image)
10. JPEG Compression (quality 60-95)

Example:
Input: 50 Nike logos
Output: 500 augmented Nike logos
```

#### Usage

```bash
python train/prepare_training_data.py \\
  --input_dir data/raw_logos \\
  --output_dir data/training \\
  --augment_factor 10 \\
  --min_size 100 \\
  --target_size 224
```

---

## 9. Deployment & Configuration

### 9.1 Replit Deployment

**Configuration Files:**
- `.replit` - Replit configuration
- `replit.md` - Project documentation
- `.streamlit/config.toml` - Streamlit settings

**Workflows:**
```yaml
# Streamlit App
streamlit_app:
  command: streamlit run src/app_streamlit.py --server.port 5000
  output_type: webview
  wait_for_port: 5000

# FastAPI Server
api_server:
  command: uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
  output_type: console
  wait_for_port: 8000
```

### 9.2 Render Deployment

**File:** `render.yaml`

```yaml
services:
  - type: web
    name: fake-logo-detection
    env: python
    plan: free
    buildCommand: pip install -r render-requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: SESSION_SECRET
        generateValue: true
```

**Deployment Steps:**
1. Push code to GitHub
2. Connect GitHub repo to Render
3. Render auto-detects `render.yaml`
4. Builds and deploys automatically
5. Access at: `https://fake-logo-detection.onrender.com`

**Configuration Fix (Important):**
```toml
# .streamlit/config.toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
# DO NOT hardcode serverAddress or serverPort
# Let Streamlit auto-detect based on Render's $PORT
```

### 9.3 Docker Deployment

**File:** `Dockerfile` (example)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 5000 8000

# Run Streamlit
CMD ["streamlit", "run", "src/app_streamlit.py", "--server.port", "5000", "--server.address", "0.0.0.0"]
```

**Docker Compose:**
```yaml
version: '3.8'
services:
  streamlit:
    build: .
    ports:
      - "5000:5000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./data:/app/data
      - ./models:/app/models

  api:
    build: .
    command: uvicorn src.api:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    environment:
      - API_KEY=${API_KEY}
    volumes:
      - ./data:/app/data
      - ./models:/app/models
```

### 9.4 Environment Variables

**Required:**
- `PYTHON_VERSION` - Python version (3.11.0)
- `SESSION_SECRET` - For Streamlit session management

**Optional:**
- `BRANDFETCH_API_KEY` - For online logo fetching (500K free requests/month)
- `API_KEY` - For FastAPI authentication
- `LOG_LEVEL` - Logging verbosity (INFO, DEBUG, ERROR)
- `MODEL_PATH` - Custom path to trained models

**Setting Environment Variables:**

**Replit:**
- Use Secrets tab in sidebar
- Or `set_env_vars` tool

**Render:**
- Dashboard → Service → Environment
- Or specify in `render.yaml`

**Local:**
```bash
export BRANDFETCH_API_KEY="your-key-here"
export API_KEY="your-api-key"
python -m streamlit run src/app_streamlit.py
```

---

## 10. Troubleshooting & Maintenance

### 10.1 Common Issues

**Issue 1: "No logos detected" for obvious logos**

**Diagnosis:**
- Detection thresholds too strict
- Logo not in reference database
- Image quality too low
- Logo heavily occluded

**Solutions:**
```python
# Lower detection thresholds
detector = LogoDetector(method='sift')
detections = detector.detect(image, min_matches=3, lowe_ratio=0.85)

# Add logo to database
from src.logo_fetcher import LogoFetcher
fetcher = LogoFetcher()
logo_data = fetcher.fetch_logo_by_domain("nike.com")
fetcher.save_to_templates_db(logo_data)

# Preprocess image
image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
image = cv2.GaussianBlur(image, (3,3), 0)  # Denoise
```

**Issue 2: Analytics dashboard shows errors**

**Diagnosis:**
- Missing detection data
- Malformed severity/classification objects
- Chart rendering failures

**Solutions:**
- Fixed in latest version with comprehensive error handling
- All analytics functions wrapped in try-except
- Graceful fallbacks for missing data
- Check logs: `logs/fake_logo_*.log`

**Issue 3: Render deployment - JavaScript errors**

**Diagnosis:**
- Hardcoded server address in `.streamlit/config.toml`
- CORS misconfiguration
- Port mismatch

**Solutions:**
```toml
# CORRECT .streamlit/config.toml
[server]
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
# DO NOT set serverAddress or serverPort
```

### 10.2 Performance Optimization

**Image Processing:**
```python
# Resize large images before processing
max_dim = 1024
h, w = image.shape[:2]
if max(h, w) > max_dim:
    scale = max_dim / max(h, w)
    image = cv2.resize(image, None, fx=scale, fy=scale)
```

**Batch Processing:**
```python
# Process multiple images in parallel
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_image, images)
```

**Model Inference:**
```python
# Use ONNX for faster inference
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
output = session.run(None, {"input": image_tensor})
```

### 10.3 Database Maintenance

**Cleanup Old Detections:**
```sql
-- Delete detections older than 90 days
DELETE FROM detections
WHERE timestamp < datetime('now', '-90 days');

-- Vacuum to reclaim space
VACUUM;
```

**Backup Database:**
```bash
# SQLite backup
sqlite3 detections.db ".backup detections_backup.db"

# Or copy file
cp detections.db backups/detections_$(date +%Y%m%d).db
```

**Reset Statistics:**
```sql
-- Clear all data (use with caution)
DELETE FROM tamper_analysis;
DELETE FROM logo_detections;
DELETE FROM detections;
VACUUM;
```

### 10.4 Monitoring & Logging

**Log Locations:**
- Application logs: `logs/fake_logo_YYYYMMDD.log`
- Replit logs: Use `refresh_all_logs` tool
- Render logs: Dashboard → Logs tab

**Log Levels:**
```python
import logging

# Set log level
logging.getLogger('src').setLevel(logging.DEBUG)

# Common log messages
logger.info("Detection started")      # INFO
logger.warning("Low confidence")      # WARNING
logger.error("Detection failed")      # ERROR
logger.debug("SIFT matches: 12")     # DEBUG
```

**Metrics to Monitor:**
- Average processing time (should be <2s)
- Detection success rate (>70% for known logos)
- API response times (should be <3s)
- Database size (monitor growth)
- Error rates (should be <5%)

---

## Appendix A: Function Reference

### Core Detection Functions

| Function | Module | Purpose | Returns |
|----------|--------|---------|---------|
| `LogoDetector.detect()` | detector.py | Main detection entry point | List[dict] |
| `detect_sift()` | detector.py | SIFT-based detection | List[dict] |
| `detect_template()` | detector.py | Template matching | List[dict] |
| `detect_yolo()` | detector.py | YOLO detection | List[dict] |
| `LogoClassifier.predict()` | classifier.py | Brand classification | dict |
| `compute_severity()` | severity.py | Calculate severity score | dict |
| `analyze_tamper()` | tamper.py | Forensic analysis | dict |
| `generate_gradcam()` | explain.py | Explainability heatmap | np.ndarray |
| `SimilaritySearcher.search()` | similarity.py | Find similar logos | List[dict] |

### Utility Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `load_image()` | utils.py | Load and preprocess image |
| `save_image()` | utils.py | Save image to disk |
| `draw_boxes()` | utils.py | Draw bounding boxes |
| `crop_region()` | utils.py | Extract region from image |
| `compute_image_hash()` | utils.py | MD5 hash for deduplication |

### Database Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `log_detection()` | db.py | Save detection to database |
| `get_detection_history()` | db.py | Retrieve past detections |
| `get_statistics()` | db.py | Get aggregate statistics |
| `log_tamper_analysis()` | db.py | Save forensic results |

---

## Appendix B: Configuration Reference

### Streamlit Config

**File:** `.streamlit/config.toml`

```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200  # MB

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### Detection Parameters

```python
# SIFT Configuration
SIFT_MIN_MATCHES = 4
SIFT_LOWE_RATIO = 0.8
SIFT_RANSAC_THRESHOLD = 5.0

# Template Matching
TEMPLATE_BASE_THRESHOLD = 0.6
TEMPLATE_MIN_THRESHOLD = 0.2
TEMPLATE_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]

# YOLO
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
YOLO_INPUT_SIZE = 640

# Severity Scoring Weights
CLASSIFIER_WEIGHT = 0.5
SSIM_WEIGHT = 0.3
COLOR_WEIGHT = 0.2
```

---

## Appendix C: API Response Examples

### Detection Response (Success)

```json
{
  "filename": "nike_logo.jpg",
  "num_detections": 1,
  "processing_time_ms": 1247.5,
  "detections": [
    {
      "label": "Nike",
      "confidence": 0.89,
      "bbox": [120, 85, 340, 215],
      "severity": {
        "severity_score": 15.3,
        "level": "LOW",
        "description": "Low severity - likely authentic",
        "breakdown": {
          "classifier": 12.5,
          "classifier_fake_prob": 0.125,
          "ssim_dissimilarity": 0.08,
          "color_distance": 0.15
        }
      },
      "classification": {
        "brand": "Nike",
        "confidence": 0.875,
        "fake_probability": 0.125
      },
      "similar_logos": [
        {
          "name": "nike_template_1.png",
          "similarity": 0.94,
          "path": "data/logos_db/nike_template_1.png"
        }
      ]
    }
  ],
  "ela_result": {
    "is_suspicious": false,
    "mean_brightness": 12.4,
    "suspiciousness_score": 0.18,
    "ela_image_available": true
  },
  "exif_data": {
    "has_exif": true,
    "camera_make": "Canon",
    "camera_model": "EOS R5",
    "software": null,
    "has_gps": false,
    "timestamp_original": "2025:11:18 10:30:45"
  }
}
```

### Error Response

```json
{
  "error": "Invalid image format",
  "detail": "File must be PNG, JPG, or JPEG",
  "status_code": 400
}
```

---

## Conclusion

This comprehensive documentation covers the complete Fake Logo Detection & Forensics Suite architecture, implementation details, and operational procedures. The system combines cutting-edge computer vision, deep learning, and forensic techniques to provide robust logo authentication capabilities.

For further assistance:
- GitHub Issues: [Report bugs or request features]
- Documentation: Keep this guide updated with system changes
- Training: See `train/TRAINING_GUIDE.md` for model development
- API Docs: Visit `/docs` endpoint for interactive API documentation

**Last Updated:** November 20, 2025  
**Version:** 1.0  
**Maintainer:** Replit Agent
