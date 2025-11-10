"""
PDF report generation module using ReportLab.
Creates comprehensive forensic reports for logo detection results.
"""
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
from pathlib import Path
import io
import cv2
import numpy as np
from src.utils import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generate PDF forensic reports for logo detection results.
    """
    
    def __init__(self, output_dir='reports'):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define styles
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495E'),
            spaceAfter=12,
            spaceBefore=12
        )
    
    def generate_report(self, filename, original_image, detections, ela_result, 
                       similarity_results=None, output_filename=None):
        """
        Generate comprehensive PDF report for a logo detection session.
        
        Args:
            filename: Original image filename
            original_image: Original image array (BGR)
            detections: List of detection results with severity and classification
            ela_result: Error Level Analysis result dict
            similarity_results: Visual similarity search results
            output_filename: Custom output filename (None = auto-generate)
        
        Returns:
            str: Path to generated PDF report
        """
        # Generate output filename
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"report_{Path(filename).stem}_{timestamp}.pdf"
        
        report_path = self.output_dir / output_filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build story (content elements)
        story = []
        
        # Title
        story.append(Paragraph("Fake Logo Detection Report", self.title_style))
        story.append(Spacer(1, 0.2 * inch))
        
        # Metadata table
        metadata = [
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Image File:', filename],
            ['Total Detections:', str(len(detections))],
        ]
        
        metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ECF0F1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        story.append(metadata_table)
        story.append(Spacer(1, 0.3 * inch))
        
        # Original image section
        story.append(Paragraph("Original Image", self.heading_style))
        
        # Save original image to temp file for inclusion
        temp_img_path = self._save_temp_image(original_image, 'original')
        if temp_img_path:
            img = Image(str(temp_img_path), width=4*inch, height=3*inch, kind='proportional')
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))
        
        # Tampering Analysis section
        story.append(Paragraph("Tampering Analysis (ELA)", self.heading_style))
        
        ela_data = [
            ['Mean Brightness:', f"{ela_result.get('mean_brightness', 0):.2f}"],
            ['Suspiciousness Score:', f"{ela_result.get('suspiciousness_score', 0):.2%}"],
            ['Is Suspicious:', 'Yes' if ela_result.get('is_suspicious') else 'No']
        ]
        
        ela_table = Table(ela_data, colWidths=[2*inch, 2*inch])
        ela_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ECF0F1')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        story.append(ela_table)
        story.append(Spacer(1, 0.2 * inch))
        
        # Include ELA image
        if 'ela_image' in ela_result:
            ela_img_path = self._save_temp_image(ela_result['ela_image'], 'ela')
            if ela_img_path:
                ela_img = Image(str(ela_img_path), width=4*inch, height=3*inch, kind='proportional')
                story.append(ela_img)
                story.append(Spacer(1, 0.3 * inch))
        
        # Detection Results section
        story.append(Paragraph("Logo Detections", self.heading_style))
        
        for i, det in enumerate(detections, 1):
            story.append(Paragraph(f"Detection #{i}", self.styles['Heading3']))
            
            # Detection details table
            severity = det.get('severity', {})
            breakdown = severity.get('breakdown', {})
            
            det_data = [
                ['Brand:', det.get('label', 'Unknown')],
                ['Confidence:', f"{det.get('confidence', 0):.2%}"],
                ['Severity Score:', f"{severity.get('severity_score', 0)}/100"],
                ['Bounding Box:', str(det.get('bbox', 'N/A'))],
            ]
            
            det_table = Table(det_data, colWidths=[2*inch, 3*inch])
            det_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ECF0F1')),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            
            story.append(det_table)
            story.append(Spacer(1, 0.1 * inch))
            
            # Severity breakdown
            if breakdown:
                breakdown_data = [
                    ['Classifier Fake Prob:', f"{breakdown.get('classifier_fake_prob', 0):.2%}"],
                    ['SSIM Dissimilarity:', f"{breakdown.get('ssim_dissimilarity', 0):.2%}"],
                    ['Color Distance:', f"{breakdown.get('color_distance', 0):.2%}"],
                ]
                
                breakdown_table = Table(breakdown_data, colWidths=[2*inch, 2*inch])
                breakdown_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                
                story.append(Paragraph("Severity Breakdown:", self.styles['BodyText']))
                story.append(breakdown_table)
            
            story.append(Spacer(1, 0.2 * inch))
        
        # Build PDF
        try:
            doc.build(story)
            logger.info(f"Generated PDF report: {report_path}")
            return str(report_path)
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return None
    
    def _save_temp_image(self, image, prefix='temp'):
        """
        Save image to temporary file for PDF inclusion.
        
        Args:
            image: Image array (BGR)
            prefix: Filename prefix
        
        Returns:
            Path: Path to saved temporary image
        """
        try:
            temp_path = self.output_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            cv2.imwrite(str(temp_path), image)
            return temp_path
        except Exception as e:
            logger.error(f"Error saving temp image: {e}")
            return None
    
    def generate_summary_report(self, detections_list, output_filename='summary_report.pdf'):
        """
        Generate summary report covering multiple detection sessions.
        
        Args:
            detections_list: List of detection session results
            output_filename: Output PDF filename
        
        Returns:
            str: Path to generated report
        """
        report_path = self.output_dir / output_filename
        
        doc = SimpleDocTemplate(str(report_path), pagesize=letter)
        story = []
        
        # Title
        story.append(Paragraph("Logo Detection Summary Report", self.title_style))
        story.append(Spacer(1, 0.3 * inch))
        
        # Summary statistics
        total_images = len(detections_list)
        total_logos = sum(len(d.get('detections', [])) for d in detections_list)
        
        summary_data = [
            ['Total Images Processed:', str(total_images)],
            ['Total Logos Detected:', str(total_logos)],
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ECF0F1')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.5 * inch))
        
        # Individual session summaries
        for i, session in enumerate(detections_list, 1):
            story.append(Paragraph(f"Session {i}: {session.get('filename', 'Unknown')}", 
                                 self.styles['Heading3']))
            
            session_data = [
                ['Detections:', str(len(session.get('detections', [])))],
                ['Timestamp:', session.get('timestamp', 'N/A')]
            ]
            
            session_table = Table(session_data, colWidths=[1.5*inch, 3*inch])
            session_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
            ]))
            
            story.append(session_table)
            story.append(Spacer(1, 0.2 * inch))
        
        # Build PDF
        try:
            doc.build(story)
            logger.info(f"Generated summary report: {report_path}")
            return str(report_path)
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return None
