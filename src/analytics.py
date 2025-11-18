"""
Analytics Dashboard Module
Comprehensive metrics and visualizations for logo detection results.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from src.utils import get_logger

logger = get_logger(__name__)


class AnalyticsDashboard:
    """Generate comprehensive analytics for detection results."""
    
    def __init__(self):
        """Initialize analytics dashboard."""
        self.metrics = {}
    
    def compute_metrics(self, detections, ela_result=None, exif_data=None, processing_time_ms=0):
        """
        Compute comprehensive metrics from detection results.
        
        Args:
            detections: List of detection dicts with bbox, label, confidence, severity, etc.
            ela_result: Error Level Analysis result dict
            exif_data: EXIF metadata dict
            processing_time_ms: Processing time in milliseconds
        
        Returns:
            dict: Comprehensive metrics
        """
        if not detections or len(detections) == 0:
            return {
                'num_logos': 0,
                'message': 'No logos detected'
            }
        
        # Basic counts
        num_logos = len(detections)
        
        # Severity analysis
        severity_scores = [d['severity']['severity_score'] for d in detections]
        avg_severity = np.mean(severity_scores)
        max_severity = np.max(severity_scores)
        min_severity = np.min(severity_scores)
        
        # Confidence analysis
        detection_confidences = [d['confidence'] for d in detections]
        avg_detection_conf = np.mean(detection_confidences)
        
        # Classification analysis
        fake_probabilities = [d['classification']['fake_probability'] for d in detections]
        avg_fake_prob = np.mean(fake_probabilities)
        
        # Brand distribution
        brand_counts = {}
        for d in detections:
            label = d['label']
            brand_counts[label] = brand_counts.get(label, 0) + 1
        
        # Severity breakdown components
        classifier_scores = [d['severity']['breakdown']['classifier'] for d in detections]
        ssim_scores = [d['severity']['breakdown']['ssim_dissimilarity'] for d in detections]
        color_scores = [d['severity']['breakdown']['color_distance'] for d in detections]
        
        # Quality metrics
        quality_metrics = {
            'avg_classifier_component': np.mean(classifier_scores),
            'avg_ssim_component': np.mean(ssim_scores),
            'avg_color_component': np.mean(color_scores)
        }
        
        # Tamper analysis
        tamper_metrics = {}
        if ela_result:
            tamper_metrics = {
                'ela_mean_brightness': ela_result.get('mean_brightness', 0),
                'ela_suspicious': ela_result.get('is_suspicious', False),
                'suspiciousness_score': ela_result.get('suspiciousness_score', 0)
            }
        
        # EXIF forensics
        exif_metrics = {}
        if exif_data:
            exif_metrics = {
                'has_exif': exif_data.get('has_exif', False),
                'has_gps': exif_data.get('has_gps', False),
                'camera_make': exif_data.get('camera_make', 'Unknown'),
                'software_edited': exif_data.get('software') is not None
            }
        
        # Risk assessment
        high_risk_count = sum(1 for s in severity_scores if s >= 70)
        medium_risk_count = sum(1 for s in severity_scores if 50 <= s < 70)
        low_risk_count = sum(1 for s in severity_scores if s < 50)
        
        return {
            'num_logos': num_logos,
            'processing_time_ms': processing_time_ms,
            'severity': {
                'average': avg_severity,
                'maximum': max_severity,
                'minimum': min_severity,
                'high_risk_count': high_risk_count,
                'medium_risk_count': medium_risk_count,
                'low_risk_count': low_risk_count
            },
            'detection_confidence': {
                'average': avg_detection_conf,
                'minimum': min(detection_confidences),
                'maximum': max(detection_confidences)
            },
            'classification': {
                'avg_fake_probability': avg_fake_prob,
                'max_fake_probability': max(fake_probabilities),
                'min_fake_probability': min(fake_probabilities)
            },
            'brand_distribution': brand_counts,
            'quality_components': quality_metrics,
            'tamper_analysis': tamper_metrics,
            'exif_forensics': exif_metrics
        }
    
    def create_severity_distribution_chart(self, detections):
        """Create bar chart showing severity distribution."""
        severity_scores = [d['severity']['severity_score'] for d in detections]
        labels = [d['label'] for d in detections]
        
        # Color code by severity level
        colors = []
        for score in severity_scores:
            if score < 30:
                colors.append('#27ae60')  # Green
            elif score < 60:
                colors.append('#f39c12')  # Yellow
            elif score < 80:
                colors.append('#e67e22')  # Orange
            else:
                colors.append('#e74c3c')  # Red
        
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=severity_scores,
                marker_color=colors,
                text=[f'{s:.1f}' for s in severity_scores],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Severity Scores by Logo',
            xaxis_title='Brand',
            yaxis_title='Severity Score (0-100)',
            height=300,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_component_breakdown_chart(self, detections):
        """Create stacked bar chart showing severity component breakdown."""
        labels = [d['label'] for d in detections]
        classifier_vals = [d['severity']['breakdown']['classifier'] for d in detections]
        ssim_vals = [d['severity']['breakdown']['ssim_dissimilarity'] for d in detections]
        color_vals = [d['severity']['breakdown']['color_distance'] for d in detections]
        
        fig = go.Figure(data=[
            go.Bar(name='Classifier (50%)', x=labels, y=classifier_vals, marker_color='#3498db'),
            go.Bar(name='SSIM (30%)', x=labels, y=ssim_vals, marker_color='#9b59b6'),
            go.Bar(name='Color (20%)', x=labels, y=color_vals, marker_color='#e67e22')
        ])
        
        fig.update_layout(
            title='Severity Component Breakdown',
            xaxis_title='Brand',
            yaxis_title='Component Score',
            barmode='stack',
            height=300,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_confidence_scatter(self, detections):
        """Create scatter plot of detection confidence vs fake probability."""
        detection_conf = [d['confidence'] for d in detections]
        fake_prob = [d['classification']['fake_probability'] for d in detections]
        labels = [d['label'] for d in detections]
        severity = [d['severity']['severity_score'] for d in detections]
        
        fig = go.Figure(data=go.Scatter(
            x=detection_conf,
            y=fake_prob,
            mode='markers+text',
            marker=dict(
                size=[s/5 for s in severity],  # Size by severity
                color=severity,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Severity"),
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=labels,
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>Detection Conf: %{x:.2f}<br>Fake Prob: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Detection Confidence vs Fake Probability',
            xaxis_title='Detection Confidence',
            yaxis_title='Fake Probability',
            height=350,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_risk_pie_chart(self, metrics):
        """Create pie chart showing risk distribution."""
        severity = metrics['severity']
        
        labels = ['Low Risk (<50)', 'Medium Risk (50-70)', 'High Risk (≥70)']
        values = [severity['low_risk_count'], severity['medium_risk_count'], severity['high_risk_count']]
        colors = ['#27ae60', '#f39c12', '#e74c3c']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title='Risk Distribution',
            height=300,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
    
    def generate_summary_table(self, metrics):
        """Generate summary statistics table."""
        data = {
            'Metric': [
                'Total Logos Detected',
                'Average Severity',
                'Maximum Severity',
                'Average Detection Confidence',
                'Average Fake Probability',
                'Processing Time (ms)',
                'High Risk Logos',
                'Medium Risk Logos',
                'Low Risk Logos'
            ],
            'Value': [
                metrics['num_logos'],
                f"{metrics['severity']['average']:.1f}",
                f"{metrics['severity']['maximum']:.1f}",
                f"{metrics['detection_confidence']['average']:.2f}",
                f"{metrics['classification']['avg_fake_probability']:.2%}",
                f"{metrics['processing_time_ms']:.0f}",
                metrics['severity']['high_risk_count'],
                metrics['severity']['medium_risk_count'],
                metrics['severity']['low_risk_count']
            ]
        }
        
        return pd.DataFrame(data)
