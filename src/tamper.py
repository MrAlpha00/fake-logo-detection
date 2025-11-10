"""
Tamper detection module using Error Level Analysis (ELA).
Detects image manipulation by analyzing JPEG compression artifacts.
"""
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import io
from datetime import datetime
from src.utils import get_logger

logger = get_logger(__name__)


def error_level_analysis(image, quality=90):
    """
    Perform Error Level Analysis to detect image tampering.
    
    ELA works by re-saving the image at a known JPEG quality and comparing
    the difference. Edited regions show different error levels than original regions.
    
    Args:
        image: Input image (BGR numpy array)
        quality: JPEG quality for re-compression (default 90)
    
    Returns:
        dict: {
            'ela_image': numpy array (BGR) showing error levels,
            'mean_brightness': float (average error level),
            'is_suspicious': bool (True if high error levels detected),
            'suspiciousness_score': float (0-1)
        }
    """
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Save to bytes buffer with specified JPEG quality
    buffer = io.BytesIO()
    pil_image.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    
    # Load the recompressed image
    recompressed = Image.open(buffer)
    recompressed_array = np.array(recompressed)
    
    # Compute absolute difference (error level)
    original_array = np.array(pil_image)
    
    # Ensure same dimensions
    if original_array.shape != recompressed_array.shape:
        recompressed_array = cv2.resize(recompressed_array, 
                                        (original_array.shape[1], original_array.shape[0]))
    
    # Calculate error level
    error = cv2.absdiff(original_array, recompressed_array)
    
    # Enhance error visibility by scaling
    # Multiply by a factor to make subtle differences more visible
    ela_enhanced = np.clip(error * 10, 0, 255).astype(np.uint8)
    
    # Compute mean brightness of error image (higher = more suspicious)
    mean_brightness = np.mean(error)
    
    # Suspiciousness threshold calibration
    # High error levels suggest tampering, but very low can also indicate multiple resaves
    suspiciousness_score = min(1.0, mean_brightness / 50.0)
    
    # Flag as suspicious if mean error is above threshold
    is_suspicious = mean_brightness > 15.0
    
    # Convert back to BGR for OpenCV compatibility
    ela_bgr = cv2.cvtColor(ela_enhanced, cv2.COLOR_RGB2BGR)
    
    logger.info(f"ELA analysis: mean_brightness={mean_brightness:.2f}, "
                f"suspicious={is_suspicious}, score={suspiciousness_score:.3f}")
    
    return {
        'ela_image': ela_bgr,
        'mean_brightness': float(mean_brightness),
        'is_suspicious': is_suspicious,
        'suspiciousness_score': float(suspiciousness_score)
    }


def detect_clone_regions(image, threshold=0.95):
    """
    Detect potential clone-stamp tampering by finding duplicate regions.
    
    This is a simplified clone detection using template matching.
    For production, consider more sophisticated methods like SIFT-based duplicate detection.
    
    Args:
        image: Input image (BGR)
        threshold: Similarity threshold for clone detection
    
    Returns:
        list: List of suspicious clone region pairs
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clones = []
    
    # Sample small patches and look for duplicates
    patch_size = 32
    step = 16
    
    patches = []
    locations = []
    
    for y in range(0, gray.shape[0] - patch_size, step):
        for x in range(0, gray.shape[1] - patch_size, step):
            patch = gray[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            locations.append((x, y))
    
    # Compare patches (simplified for demo - full implementation would use more efficient matching)
    # This is computationally expensive for large images
    for i in range(min(len(patches), 100)):  # Limit to first 100 patches for speed
        for j in range(i+1, min(len(patches), 100)):
            # Compute normalized cross-correlation
            corr = cv2.matchTemplate(patches[i], patches[j], cv2.TM_CCOEFF_NORMED)[0, 0]
            
            if corr > threshold:
                clones.append({
                    'location1': locations[i],
                    'location2': locations[j],
                    'similarity': float(corr)
                })
    
    logger.info(f"Clone detection found {len(clones)} potential duplicate regions")
    return clones


def analyze_noise_patterns(image):
    """
    Analyze noise patterns to detect inconsistencies suggesting manipulation.
    
    Authentic images should have consistent noise across the image.
    Tampered regions often show different noise characteristics.
    
    Args:
        image: Input image (BGR)
    
    Returns:
        dict: Noise analysis results
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur and compute difference (noise extraction)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = cv2.absdiff(gray, blurred)
    
    # Divide image into regions and compute noise variance
    h, w = gray.shape
    grid_size = 4
    block_h, block_w = h // grid_size, w // grid_size
    
    variances = []
    for i in range(grid_size):
        for j in range(grid_size):
            block = noise[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            variances.append(np.var(block))
    
    # Compute variance of variances (should be low for authentic images)
    variance_consistency = np.var(variances)
    
    # Flag inconsistent noise patterns
    is_inconsistent = variance_consistency > 20.0
    
    return {
        'noise_variance_consistency': float(variance_consistency),
        'is_inconsistent': is_inconsistent,
        'regional_variances': [float(v) for v in variances]
    }


def extract_exif_metadata(image_path_or_bytes):
    """
    Extract EXIF metadata from image for forensic analysis.
    
    Analyzes camera model, software edits, GPS location, timestamps, and other
    metadata that can reveal image authenticity and manipulation history.
    
    Args:
        image_path_or_bytes: File path or bytes buffer containing image
    
    Returns:
        dict: {
            'has_exif': bool,
            'camera_make': str or None,
            'camera_model': str or None,
            'software': str or None (editing software used),
            'datetime_original': str or None,
            'datetime_digitized': str or None,
            'gps_latitude': float or None,
            'gps_longitude': float or None,
            'gps_altitude': float or None,
            'orientation': int or None,
            'flash': str or None,
            'focal_length': str or None,
            'iso': int or None,
            'exposure_time': str or None,
            'f_number': str or None,
            'tampering_indicators': list (signs of manipulation),
            'metadata_warnings': list (suspicious metadata patterns)
        }
    """
    try:
        # Open image
        if isinstance(image_path_or_bytes, (str, bytes)):
            if isinstance(image_path_or_bytes, str):
                img = Image.open(image_path_or_bytes)
            else:
                img = Image.open(io.BytesIO(image_path_or_bytes))
        else:
            # Assume it's already a PIL Image
            img = image_path_or_bytes
        
        # Get EXIF data
        exif_data = img._getexif() if hasattr(img, '_getexif') else None
        
        if not exif_data:
            logger.info("No EXIF data found in image")
            return {
                'has_exif': False,
                'tampering_indicators': ['No EXIF data - may have been stripped'],
                'metadata_warnings': ['Missing metadata suggests possible editing']
            }
        
        # Parse EXIF tags
        exif = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            exif[tag] = value
        
        # Extract key fields
        result = {
            'has_exif': True,
            'camera_make': exif.get('Make'),
            'camera_model': exif.get('Model'),
            'software': exif.get('Software'),
            'datetime_original': exif.get('DateTimeOriginal'),
            'datetime_digitized': exif.get('DateTimeDigitized'),
            'orientation': exif.get('Orientation'),
            'flash': exif.get('Flash'),
            'focal_length': str(exif.get('FocalLength')) if exif.get('FocalLength') else None,
            'iso': exif.get('ISOSpeedRatings'),
            'exposure_time': str(exif.get('ExposureTime')) if exif.get('ExposureTime') else None,
            'f_number': str(exif.get('FNumber')) if exif.get('FNumber') else None,
            'gps_latitude': None,
            'gps_longitude': None,
            'gps_altitude': None,
            'tampering_indicators': [],
            'metadata_warnings': []
        }
        
        # Extract GPS data if available
        if 'GPSInfo' in exif:
            gps_info = {}
            for key in exif['GPSInfo'].keys():
                decode = GPSTAGS.get(key, key)
                gps_info[decode] = exif['GPSInfo'][key]
            
            # Parse GPS coordinates
            if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info:
                lat = gps_info['GPSLatitude']
                lat_ref = gps_info['GPSLatitudeRef']
                result['gps_latitude'] = convert_to_degrees(lat, lat_ref)
            
            if 'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:
                lon = gps_info['GPSLongitude']
                lon_ref = gps_info['GPSLongitudeRef']
                result['gps_longitude'] = convert_to_degrees(lon, lon_ref)
            
            if 'GPSAltitude' in gps_info:
                result['gps_altitude'] = float(gps_info['GPSAltitude'])
        
        # Analyze for tampering indicators
        if result['software']:
            editing_software = ['photoshop', 'gimp', 'paint.net', 'lightroom', 
                              'affinity', 'pixlr', 'canva', 'snapseed']
            if any(sw in result['software'].lower() for sw in editing_software):
                result['tampering_indicators'].append(
                    f"Edited with {result['software']} - may have modifications"
                )
        
        # Check for timestamp inconsistencies
        if result['datetime_original'] and result['datetime_digitized']:
            try:
                dt_orig = datetime.strptime(result['datetime_original'], '%Y:%m:%d %H:%M:%S')
                dt_digit = datetime.strptime(result['datetime_digitized'], '%Y:%m:%d %H:%M:%S')
                time_diff = abs((dt_orig - dt_digit).total_seconds())
                
                if time_diff > 60:  # More than 1 minute difference
                    result['metadata_warnings'].append(
                        f"Timestamp mismatch: {time_diff:.0f}s between capture and digitization"
                    )
            except:
                pass
        
        # Missing expected metadata for modern cameras
        if not result['camera_make'] and not result['camera_model']:
            result['metadata_warnings'].append(
                "No camera information - unusual for modern devices"
            )
        
        logger.info(f"EXIF extraction: camera={result.get('camera_model')}, "
                   f"software={result.get('software')}, "
                   f"warnings={len(result['metadata_warnings'])}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error extracting EXIF data: {e}")
        return {
            'has_exif': False,
            'tampering_indicators': [f'Error reading metadata: {str(e)}'],
            'metadata_warnings': []
        }


def convert_to_degrees(value, ref):
    """
    Convert GPS coordinates to degrees.
    
    EXIF GPS coordinates are stored as rational numbers (tuples of numerator/denominator).
    
    Args:
        value: GPS coordinate value (degrees, minutes, seconds) as rational tuples
        ref: Reference direction (N/S for latitude, E/W for longitude)
    
    Returns:
        float: Coordinate in decimal degrees
    """
    try:
        # Handle rational numbers from EXIF (numerator, denominator tuples)
        def to_decimal(rational):
            if isinstance(rational, tuple) and len(rational) == 2:
                return float(rational[0]) / float(rational[1]) if rational[1] != 0 else 0.0
            elif isinstance(rational, (int, float)):
                return float(rational)
            else:
                return 0.0
        
        d = to_decimal(value[0])  # degrees
        m = to_decimal(value[1])  # minutes
        s = to_decimal(value[2])  # seconds
        
        degrees = d + (m / 60.0) + (s / 3600.0)
        
        if ref in ['S', 'W']:
            degrees = -degrees
        
        return degrees
    except (IndexError, TypeError, ZeroDivisionError) as e:
        logger.error(f"Error converting GPS coordinates: {e}")
        return 0.0
