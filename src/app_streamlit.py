"""
Main Streamlit Web Application for Fake Logo Detection & Forensics Suite.
Provides interactive UI for logo detection, classification, severity scoring,
tamper analysis, and comprehensive reporting.
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_image, compute_image_hash, draw_bounding_box, crop_region, get_logger
from src.detector import LogoDetector
from src.classifier import BrandClassifier
from src.severity import compute_severity, interpret_severity
from src.tamper import error_level_analysis
from src.explain import generate_gradcam_for_crop, explain_classification
from src.similarity import SimilaritySearcher
from src.db import DetectionDatabase
from src.report import ReportGenerator

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Fake Logo Detection Suite",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .severity-low { border-left-color: #27ae60; }
    .severity-medium { border-left-color: #f39c12; }
    .severity-high { border-left-color: #e67e22; }
    .severity-critical { border-left-color: #e74c3c; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models(detection_method='sift'):
    """Load all ML models and initialize components (cached)."""
    with st.spinner("Loading models and building indices..."):
        detector = LogoDetector(method=detection_method, yolo_model_path='models/yolov8_logo.onnx')
        classifier = BrandClassifier()  # Demo mode with deterministic predictions
        similarity_searcher = SimilaritySearcher()
        db = DetectionDatabase()
        report_gen = ReportGenerator()
        
        logger.info("All models and components loaded")
        return detector, classifier, similarity_searcher, db, report_gen


def process_image(image, detector, classifier, similarity_searcher, 
                 show_gradcam, show_similarity, confidence_threshold):
    """
    Process uploaded image through the full detection pipeline.
    
    Returns dict with all results.
    """
    start_time = time.time()
    
    # Detect logos
    detections = detector.detect(image, confidence_threshold=confidence_threshold)
    
    if len(detections) == 0:
        return {
            'success': False,
            'message': 'No logos detected in the image',
            'detections': []
        }
    
    # Load reference templates for severity comparison
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
        logo_crop = crop_region(image, bbox)
        
        # Classify brand
        classification = classifier.classify(logo_crop)
        
        # Get reference template
        ref_template = templates.get(label.capitalize())
        if ref_template is None:
            # Use first available template as fallback
            ref_template = list(templates.values())[0] if templates else logo_crop
        
        # Compute severity score
        severity = compute_severity(logo_crop, ref_template, classification)
        
        # Generate Grad-CAM if requested
        gradcam_overlay = None
        if show_gradcam:
            gradcam_overlay = generate_gradcam_for_crop(classifier, logo_crop)
        
        # Find similar logos if requested
        similar_logos = None
        if show_similarity:
            similar_logos = similarity_searcher.search(logo_crop, top_k=5)
        
        processed_detections.append({
            'bbox': bbox,
            'label': label,
            'confidence': det['confidence'],
            'classification': classification,
            'severity': severity,
            'gradcam': gradcam_overlay,
            'similar_logos': similar_logos,
            'crop': logo_crop
        })
    
    # Run tamper detection on whole image
    ela_result = error_level_analysis(image)
    
    processing_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return {
        'success': True,
        'detections': processed_detections,
        'ela_result': ela_result,
        'processing_time_ms': processing_time
    }


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🔍 Fake Logo Detection & Forensics Suite</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    detector, classifier, similarity_searcher, db, report_gen = load_models()
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    detection_method = st.sidebar.selectbox(
        "Detection Method",
        ["SIFT Features", "Template Matching"],
        help="SIFT is more robust to transformations"
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for logo detection"
    )
    
    show_gradcam = st.sidebar.checkbox(
        "Show Grad-CAM Heatmaps",
        value=True,
        help="Visualize classifier decision regions"
    )
    
    show_similarity = st.sidebar.checkbox(
        "Show Similar Logos",
        value=True,
        help="Find visually similar logos from database"
    )
    
    show_ela = st.sidebar.checkbox(
        "Show Error Level Analysis",
        value=True,
        help="Detect image tampering via ELA"
    )
    
    # Update detector method
    detector.method = 'sift' if detection_method == "SIFT Features" else 'template'
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Statistics")
    stats = db.get_statistics()
    st.sidebar.metric("Total Detections", stats.get('total_detections', 0))
    st.sidebar.metric("Total Logos Found", stats.get('total_logos', 0))
    st.sidebar.metric("Fake Logos Detected", stats.get('total_fakes', 0))
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📚 Detection History", "ℹ️ About"])
    
    with tab1:
        st.subheader("Upload Image for Analysis")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image containing logos to analyze"
        )
        
        # Demo image selector
        st.markdown("**Or select a demo image:**")
        demo_images = sorted(Path('data/samples').glob('*.jpg'))
        demo_names = [img.name for img in demo_images]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_demo = st.selectbox("Demo Images", ["None"] + demo_names)
        with col2:
            analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        # Process image
        if analyze_button or uploaded_file is not None:
            # Load image
            if uploaded_file is not None:
                # From upload
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                filename = uploaded_file.name
            elif selected_demo != "None":
                # From demo selection
                demo_path = Path('data/samples') / selected_demo
                image = cv2.imread(str(demo_path))
                filename = selected_demo
            else:
                st.warning("Please upload an image or select a demo image")
                return
            
            if image is None:
                st.error("Error loading image")
                return
            
            # Display original image
            st.markdown("### Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Process image
            with st.spinner("Analyzing image..."):
                result = process_image(
                    image, detector, classifier, similarity_searcher,
                    show_gradcam, show_similarity, confidence_threshold
                )
            
            if not result['success']:
                st.warning(result['message'])
                return
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            
            # Summary metrics
            num_detections = len(result['detections'])
            avg_severity = np.mean([d['severity']['severity_score'] for d in result['detections']])
            processing_time = result['processing_time_ms']
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Logos Detected", num_detections)
            col2.metric("Avg Severity", f"{avg_severity:.1f}/100")
            col3.metric("Processing Time", f"{processing_time:.0f} ms")
            
            # Show ELA results
            if show_ela:
                st.markdown("### 🔬 Tamper Detection (Error Level Analysis)")
                ela_result = result['ela_result']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(ela_result['ela_image'], cv2.COLOR_BGR2RGB),
                            caption="ELA Visualization", use_container_width=True)
                
                with col2:
                    st.metric("Mean Brightness", f"{ela_result['mean_brightness']:.2f}")
                    st.metric("Suspiciousness", f"{ela_result['suspiciousness_score']:.2%}")
                    
                    if ela_result['is_suspicious']:
                        st.error("⚠️ Image shows signs of tampering!")
                    else:
                        st.success("✅ No significant tampering detected")
            
            # Individual logo detections
            st.markdown("### 🏷️ Logo Detections")
            
            # Draw all bounding boxes on image
            annotated_image = image.copy()
            for det in result['detections']:
                severity_score = det['severity']['severity_score']
                # Color based on severity
                if severity_score < 30:
                    color = (0, 255, 0)  # Green
                elif severity_score < 60:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                annotated_image = draw_bounding_box(
                    annotated_image, det['bbox'], det['label'],
                    det['confidence'], color=color
                )
            
            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                    caption="Detected Logos", use_container_width=True)
            
            # Details for each detection
            for i, det in enumerate(result['detections'], 1):
                with st.expander(f"Detection #{i}: {det['label']} (Confidence: {det['confidence']:.2%})"):
                    
                    severity_info = interpret_severity(det['severity']['severity_score'])
                    severity_class = f"severity-{severity_info['level'].lower()}"
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Logo Crop**")
                        st.image(cv2.cvtColor(det['crop'], cv2.COLOR_BGR2RGB),
                                use_container_width=True)
                        
                        if show_gradcam and det['gradcam'] is not None:
                            st.markdown("**Grad-CAM Heatmap**")
                            st.image(cv2.cvtColor(det['gradcam'], cv2.COLOR_BGR2RGB),
                                    use_container_width=True)
                    
                    with col2:
                        st.markdown(f"**Severity: {det['severity']['severity_score']}/100**")
                        st.markdown(f"**Level:** :{severity_info['color']}[{severity_info['level']}]")
                        st.markdown(f"*{severity_info['description']}*")
                        
                        st.markdown("**Breakdown:**")
                        breakdown = det['severity']['breakdown']
                        st.write(f"- Classifier Suspicion: {breakdown['classifier_fake_prob']:.1%}")
                        st.write(f"- Structural Diff: {breakdown['ssim_dissimilarity']:.1%}")
                        st.write(f"- Color Variation: {breakdown['color_distance']:.1%}")
                        
                        st.markdown("**Classification:**")
                        st.write(f"- Brand: {det['classification']['brand']}")
                        st.write(f"- Confidence: {det['classification']['confidence']:.1%}")
                    
                    # Similar logos
                    if show_similarity and det['similar_logos']:
                        st.markdown("**Similar Logos from Database:**")
                        cols = st.columns(min(5, len(det['similar_logos'])))
                        
                        for j, sim in enumerate(det['similar_logos'][:5]):
                            with cols[j]:
                                sim_img = cv2.imread(sim['path'])
                                if sim_img is not None:
                                    st.image(cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB),
                                            caption=f"{sim['name']}\n{sim['similarity']:.2%}",
                                            use_container_width=True)
            
            # Generate PDF report
            st.markdown("---")
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("📄 Generate PDF Report", use_container_width=True):
                    with st.spinner("Generating report..."):
                        report_path = report_gen.generate_report(
                            filename, image, result['detections'], result['ela_result']
                        )
                        
                        if report_path:
                            with open(report_path, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Download Report",
                                    data=f,
                                    file_name=Path(report_path).name,
                                    mime='application/pdf',
                                    use_container_width=True
                                )
                            st.success(f"Report generated: {Path(report_path).name}")
            
            # Log to database
            image_hash = compute_image_hash(image)
            detection_id = db.log_detection(
                filename, image_hash, result['detections'], result['processing_time_ms']
            )
            db.log_tamper_analysis(detection_id, result['ela_result'])
            
            st.info(f"Detection logged to database (ID: {detection_id})")
    
    with tab2:
        st.subheader("Detection History")
        
        recent_detections = db.get_recent_detections(limit=20)
        
        if recent_detections:
            st.markdown(f"**Showing {len(recent_detections)} most recent detections**")
            
            for det in recent_detections:
                with st.expander(f"{det['filename']} - {det['timestamp']}"):
                    col1, col2, col3 = st.columns(3)
                    col1.write(f"**ID:** {det['id']}")
                    col2.write(f"**Detections:** {det['num_detections']}")
                    col3.write(f"**Time:** {det['processing_time_ms']:.0f} ms")
                    
                    details = db.get_detection_details(det['id'])
                    if details:
                        st.write("**Logos:**")
                        for logo in details['logos']:
                            st.write(f"- {logo['brand']}: Severity {logo['severity_score']}/100")
        else:
            st.info("No detection history yet. Upload and analyze some images!")
    
    with tab3:
        st.subheader("About Fake Logo Detection Suite")
        
        st.markdown("""
        This application provides comprehensive fake logo detection and forensic analysis using:
        
        - **Logo Detection:** SIFT feature matching and template matching
        - **Brand Classification:** Deep learning with MobileNetV2
        - **Severity Scoring:** Combined metric from SSIM, color analysis, and classifier confidence
        - **Tamper Detection:** Error Level Analysis (ELA) for image manipulation detection
        - **Explainability:** Grad-CAM heatmaps showing classifier decision regions
        - **Similarity Search:** FAISS-based visual similarity matching
        - **Audit Logging:** SQLite database tracking all detections
        - **PDF Reports:** Comprehensive forensic reports with ReportLab
        
        ### How to Use:
        
        1. **Upload an image** or select a demo sample
        2. **Adjust settings** in the sidebar (detection method, confidence threshold)
        3. **Analyze** to run the full detection pipeline
        4. **Review results** including severity scores, ELA analysis, and similar logos
        5. **Generate PDF report** for comprehensive documentation
        
        ### Severity Levels:
        
        - 🟢 **Low (0-30):** Logo appears authentic
        - 🟡 **Medium (30-50):** Some inconsistencies detected
        - 🟠 **High (50-70):** Likely tampered/fake
        - 🔴 **Critical (70-100):** Strong evidence of forgery
        
        ### Demo Mode:
        
        Currently running in demo mode with simulated classifier predictions.
        To use trained models, place weights in `models/` directory.
        """)


if __name__ == "__main__":
    main()
