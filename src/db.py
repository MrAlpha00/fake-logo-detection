"""
Database module for audit logging of logo detections.
Uses SQLite to store detection history with timestamps and metadata.
"""
import sqlite3
from datetime import datetime
from pathlib import Path
import json
from src.utils import get_logger

logger = get_logger(__name__)


class DetectionDatabase:
    """
    SQLite database for logging logo detection results.
    """
    
    def __init__(self, db_path='detections.db'):
        """
        Initialize database connection and create tables if needed.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = None
        self.cursor = None
        
        # Initialize database
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        # Main detections table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                image_hash TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                num_detections INTEGER DEFAULT 0,
                processing_time_ms REAL
            )
        ''')
        
        # Individual logo detections table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS logo_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id INTEGER,
                brand TEXT,
                confidence REAL,
                bbox TEXT,
                severity_score INTEGER,
                breakdown TEXT,
                is_fake BOOLEAN,
                FOREIGN KEY (detection_id) REFERENCES detections(id)
            )
        ''')
        
        # Tampering analysis table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tamper_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id INTEGER,
                ela_mean_brightness REAL,
                is_suspicious BOOLEAN,
                suspiciousness_score REAL,
                FOREIGN KEY (detection_id) REFERENCES detections(id)
            )
        ''')
        
        self.conn.commit()
        logger.info("Database tables initialized")
    
    def log_detection(self, filename, image_hash, detections, processing_time_ms=0.0):
        """
        Log a detection session to the database.
        
        Args:
            filename: Name of processed image file
            image_hash: SHA-256 hash of image
            detections: List of detection results
            processing_time_ms: Processing time in milliseconds
        
        Returns:
            int: Detection ID
        """
        try:
            # Insert main detection record
            self.cursor.execute('''
                INSERT INTO detections (filename, image_hash, num_detections, processing_time_ms)
                VALUES (?, ?, ?, ?)
            ''', (filename, image_hash, len(detections), processing_time_ms))
            
            detection_id = self.cursor.lastrowid
            
            # Insert individual logo detections
            for det in detections:
                bbox_str = json.dumps(det.get('bbox', []))
                breakdown_str = json.dumps(det.get('severity', {}).get('breakdown', {}))
                
                self.cursor.execute('''
                    INSERT INTO logo_detections 
                    (detection_id, brand, confidence, bbox, severity_score, breakdown, is_fake)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    detection_id,
                    det.get('label', 'Unknown'),
                    det.get('confidence', 0.0),
                    bbox_str,
                    det.get('severity', {}).get('severity_score', 0),
                    breakdown_str,
                    det.get('severity', {}).get('severity_score', 0) > 60
                ))
            
            self.conn.commit()
            logger.info(f"Logged detection session: ID={detection_id}, file={filename}")
            
            return detection_id
        
        except Exception as e:
            logger.error(f"Error logging detection: {e}")
            self.conn.rollback()
            return None
    
    def log_tamper_analysis(self, detection_id, ela_result):
        """
        Log tampering analysis results.
        
        Args:
            detection_id: Parent detection ID
            ela_result: ELA analysis result dict
        """
        try:
            self.cursor.execute('''
                INSERT INTO tamper_analysis 
                (detection_id, ela_mean_brightness, is_suspicious, suspiciousness_score)
                VALUES (?, ?, ?, ?)
            ''', (
                detection_id,
                ela_result.get('mean_brightness', 0.0),
                ela_result.get('is_suspicious', False),
                ela_result.get('suspiciousness_score', 0.0)
            ))
            
            self.conn.commit()
            logger.info(f"Logged tamper analysis for detection {detection_id}")
        
        except Exception as e:
            logger.error(f"Error logging tamper analysis: {e}")
            self.conn.rollback()
    
    def get_recent_detections(self, limit=10):
        """
        Retrieve recent detection records.
        
        Args:
            limit: Maximum number of records to return
        
        Returns:
            list: List of detection records as dicts
        """
        try:
            # Create new cursor for this query
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT id, filename, timestamp, num_detections, processing_time_ms
                FROM detections
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            cursor.close()
            
            results = []
            for row in rows:
                results.append({
                    'id': row[0],
                    'filename': row[1],
                    'timestamp': row[2],
                    'num_detections': row[3],
                    'processing_time_ms': row[4]
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving detections: {e}")
            return []
    
    def get_detection_details(self, detection_id):
        """
        Get detailed information about a specific detection.
        
        Args:
            detection_id: Detection ID
        
        Returns:
            dict: Detailed detection information
        """
        try:
            # Create new cursor
            cursor = self.conn.cursor()
            
            # Get main detection info
            cursor.execute('''
                SELECT filename, image_hash, timestamp, num_detections
                FROM detections WHERE id = ?
            ''', (detection_id,))
            
            main_row = cursor.fetchone()
            if not main_row:
                cursor.close()
                return None
            
            # Get logo detections
            cursor.execute('''
                SELECT brand, confidence, bbox, severity_score, breakdown, is_fake
                FROM logo_detections WHERE detection_id = ?
            ''', (detection_id,))
            
            logo_rows = cursor.fetchall()
            cursor.close()
            
            logos = []
            for row in logo_rows:
                logos.append({
                    'brand': row[0],
                    'confidence': row[1],
                    'bbox': json.loads(row[2]),
                    'severity_score': row[3],
                    'breakdown': json.loads(row[4]),
                    'is_fake': bool(row[5])
                })
            
            return {
                'filename': main_row[0],
                'image_hash': main_row[1],
                'timestamp': main_row[2],
                'num_detections': main_row[3],
                'logos': logos
            }
        
        except Exception as e:
            logger.error(f"Error retrieving detection details: {e}")
            return None
    
    def get_statistics(self):
        """
        Get database statistics.
        
        Returns:
            dict: Statistics about logged detections
        """
        try:
            # Create new cursor for these queries
            cursor = self.conn.cursor()
            
            # Total detections
            cursor.execute('SELECT COUNT(*) FROM detections')
            total_detections = cursor.fetchone()[0]
            
            # Total logos detected
            cursor.execute('SELECT COUNT(*) FROM logo_detections')
            total_logos = cursor.fetchone()[0]
            
            # Fake logos detected
            cursor.execute('SELECT COUNT(*) FROM logo_detections WHERE is_fake = 1')
            total_fakes = cursor.fetchone()[0]
            
            # Brand distribution
            cursor.execute('''
                SELECT brand, COUNT(*) as count
                FROM logo_detections
                GROUP BY brand
                ORDER BY count DESC
            ''')
            brand_distribution = dict(cursor.fetchall())
            
            cursor.close()
            
            return {
                'total_detections': total_detections,
                'total_logos': total_logos,
                'total_fakes': total_fakes,
                'brand_distribution': brand_distribution
            }
        
        except Exception as e:
            logger.error(f"Error retrieving statistics: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
