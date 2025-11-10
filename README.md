# Fake Logo Detection & Forensics Suite

A comprehensive Python-based system for detecting, classifying, and analyzing logo authenticity using computer vision, deep learning, and forensic analysis techniques.

## 🎯 Features

- **Multi-method Logo Detection**: SIFT feature matching and template matching
- **Brand Classification**: Deep learning with MobileNetV2 architecture
- **Severity Scoring**: Combined metric (0-100) from SSIM, color analysis, and classifier confidence
- **Tamper Detection**: Error Level Analysis (ELA) for image manipulation detection
- **Explainability**: Grad-CAM heatmaps showing classifier decision regions
- **Visual Similarity Search**: FAISS/Annoy-based nearest neighbor matching
- **Audit Logging**: SQLite database tracking all detections
- **PDF Reports**: Comprehensive forensic reports with ReportLab
- **Interactive Web UI**: Streamlit-based interface with real-time analysis

## 📋 Requirements

- Python 3.9+
- See `requirements.txt` for package dependencies (automatically managed by Replit)

## 🚀 Quick Start

### 1. Installation

All dependencies are automatically installed in the Replit environment. If running locally:

```bash
pip install streamlit opencv-python torch scikit-image Pillow faiss-cpu annoy reportlab numpy scipy matplotlib pytest
```

### 2. Run the Web Application

```bash
streamlit run src/app_streamlit.py --server.port 5000
```

Then open your browser to the provided URL (typically http://localhost:5000 on Replit).

### 3. Run the Automated Demo

```bash
bash run_demo.sh
```

This will:
- Process sample images (real and fake logos)
- Generate annotated outputs in `demo_outputs/`
- Log results to SQLite database
- Display statistics and verification

## 📁 Project Structure

```
fake_logo_suite/
├── src/
│   ├── app_streamlit.py      # Main Streamlit web application
│   ├── utils.py               # Utility functions (image handling, logging)
│   ├── detector.py            # Logo detection (SIFT, template matching)
│   ├── classifier.py          # Brand classification (MobileNetV2)
│   ├── similarity.py          # Visual similarity search (FAISS/Annoy)
│   ├── severity.py            # Severity scoring system
│   ├── tamper.py              # Tamper detection (ELA)
│   ├── explain.py             # Grad-CAM explainability
│   ├── report.py              # PDF report generation
│   └── db.py                  # SQLite database logging
├── train/
│   └── train_classifier.py   # Training script for classifier
├── tests/
│   ├── test_severity.py      # Unit tests for severity module
│   ├── test_tamper.py        # Unit tests for tamper detection
│   └── test_similarity.py    # Unit tests for similarity search
├── data/
│   ├── logos_db/             # Reference logo database (6 logos)
│   └── samples/              # Test samples (10 images: 5 real + 5 fake)
├── models/                   # Model weights directory
├── demo_outputs/             # Demo script outputs
├── reports/                  # Generated PDF reports
├── logs/                     # Application logs
├── run_demo.sh              # Automated demo script
├── README.md                # This file
└── DONE.md                  # Implementation summary
```

## 🖥️ Using the Web Interface

### Upload & Analyze Tab

1. **Upload Image**: Click "Browse files" to upload a logo image, or select a demo image
2. **Configure Settings**: Use sidebar to adjust:
   - Detection method (SIFT or Template Matching)
   - Confidence threshold
   - Enable/disable Grad-CAM heatmaps
   - Enable/disable similarity search
   - Enable/disable ELA analysis
3. **Click Analyze**: Process the image through the full pipeline
4. **Review Results**:
   - View annotated image with bounding boxes color-coded by severity
   - See individual detection details with crops and Grad-CAM overlays
   - Review severity breakdown and tampering indicators
   - Explore similar logos from the reference database
5. **Generate Report**: Export comprehensive PDF report

### Detection History Tab

- View all past detection sessions
- See statistics (number of logos, processing time)
- Access detailed information for each session

### About Tab

- Learn about the system features
- Understand severity levels and interpretations
- Get usage instructions

## 🎨 Demo Walkthrough

### Demo 1: Real Logo (Expected Result: Low Severity)

```bash
# Load: data/samples/real_logo1.jpg
# Expected: 
#   - Successful detection
#   - Severity score < 40 (Green/Low)
#   - ELA: Not suspicious
```

### Demo 2: Fake Logo (Expected Result: High Severity)

```bash
# Load: data/samples/fake_logo1_compressed.jpg
# Expected:
#   - Successful detection  
#   - Severity score > 60 (Orange/Red)
#   - ELA: Suspicious tampering detected
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_severity.py -v
pytest tests/test_tamper.py -v
pytest tests/test_similarity.py -v
```

## 🔧 Training Your Own Classifier

### Prepare Dataset

Organize your logo dataset in the following structure:

```
dataset/
  brand1/
    logo1.jpg
    logo2.jpg
  brand2/
    logo1.jpg
  ...
```

### Train Model

```bash
python train/train_classifier.py \
  --data_dir path/to/dataset \
  --epochs 20 \
  --batch_size 32 \
  --lr 0.001 \
  --output models/classifier.pth
```

### Use Trained Model

The classifier will automatically load weights from `models/demo_classifier.pth` if present. Otherwise, it runs in demo mode with deterministic predictions based on image characteristics.

## 📊 Severity Scoring

The system computes a combined severity score (0-100) from three components:

1. **Classifier Fake Probability** (50% weight): Neural network confidence that logo is fake
2. **Structural Dissimilarity** (30% weight): 1 - SSIM compared to reference
3. **Color Distance** (20% weight): Histogram distance from reference

### Severity Levels

- 🟢 **Low (0-30)**: Logo appears authentic
- 🟡 **Medium (30-50)**: Some inconsistencies detected
- 🟠 **High (50-70)**: Likely tampered or fake
- 🔴 **Critical (70-100)**: Strong evidence of forgery

## 🗄️ Database Schema

### `detections` table
- `id`: Primary key
- `filename`: Image filename
- `image_hash`: SHA-256 hash
- `timestamp`: Detection timestamp
- `num_detections`: Number of logos found
- `processing_time_ms`: Processing time

### `logo_detections` table
- `id`: Primary key
- `detection_id`: Foreign key to detections
- `brand`: Detected brand name
- `confidence`: Detection confidence
- `bbox`: Bounding box coordinates (JSON)
- `severity_score`: Severity score (0-100)
- `breakdown`: Severity breakdown (JSON)
- `is_fake`: Boolean flag

### `tamper_analysis` table
- `id`: Primary key
- `detection_id`: Foreign key to detections
- `ela_mean_brightness`: ELA metric
- `is_suspicious`: Tampering flag
- `suspiciousness_score`: Suspiciousness score (0-1)

## 🔮 Future Enhancements

### Planned Features (Not Yet Implemented)

- **YOLOv5/YOLOv8 Integration**: Replace template matching with trained object detector
- **Full Classifier Training**: Train on extensive brand logo dataset
- **Advanced Tampering Detection**: Clone detection, noise analysis, metadata forensics
- **Batch Processing API**: Enterprise-ready API endpoint
- **Webcam Support**: Real-time logo analysis from video stream
- **Custom Domain Support**: Deploy with custom branding

### How to Add YOLO Detection

1. Install ultralytics: `pip install ultralytics`
2. Train YOLOv8 model on logo dataset
3. Save weights to `models/yolo_logo_detector.pt`
4. In `src/detector.py`, implement `_detect_yolo()` method:

```python
from ultralytics import YOLO

def _detect_yolo(self, image, threshold=0.5):
    model = YOLO('models/yolo_logo_detector.pt')
    results = model(image)
    # Parse results and return detections
```

5. Set `method='yolo'` in UI sidebar

## 📝 Notes

- **Demo Mode**: Currently running with simulated classifier predictions. Severity scores are based on image characteristics (color, quality, structure).
- **Model Weights**: To use trained models, place `.pth` files in `models/` directory.
- **CPU Compatible**: All processing runs on CPU by default. For GPU, modify `device='cuda'` in classifier initialization.
- **Reference Database**: Add more reference logos to `data/logos_db/` and rebuild similarity index.

## 🐛 Troubleshooting

### "No logos detected"
- Lower the confidence threshold in sidebar
- Ensure image contains recognizable logo shapes
- Try different detection method (SIFT vs Template)

### "FAISS not available"
- System will automatically fall back to Annoy
- Both provide similar functionality

### Database locked error
- Close any other processes accessing `detections.db`
- Database uses write-ahead logging for concurrency

## 📄 License

This project is for educational and demonstration purposes.

## 👥 Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.

---

**Built with ❤️ using Streamlit, PyTorch, OpenCV, and modern computer vision techniques.**
