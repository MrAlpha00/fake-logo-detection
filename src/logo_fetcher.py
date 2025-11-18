"""
Online Logo Fetcher Module
Fetches brand logos from online sources using Brandfetch API.

NOTE: Brandfetch offers 500,000 free requests/month but requires registration.
To use this feature:
1. Sign up at https://brandfetch.com/developers
2. Get your API key from the dashboard
3. Set environment variable BRANDFETCH_API_KEY=your_key_here

Alternatively, Brandfetch allows unauthenticated requests for basic logo fetching.
"""
import requests
import cv2
import numpy as np
import os
from pathlib import Path
from src.utils import get_logger

logger = get_logger(__name__)


class OnlineLogoFetcher:
    """Fetch brand logos from online sources."""
    
    def __init__(self, api_key=None):
        """
        Initialize logo fetcher.
        
        Args:
            api_key: Optional API key for Brandfetch (will try env var if None)
        """
        # Try to get API key from environment if not provided
        self.api_key = api_key or os.environ.get('BRANDFETCH_API_KEY')
        self.base_url = "https://api.brandfetch.io/v2"
        self.cache_dir = Path('data/fetched_logos')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Warn if no API key (may limit functionality)
        if not self.api_key:
            logger.warning("No Brandfetch API key found. Using unauthenticated mode (limited features).")
    
    def fetch_logo_by_domain(self, domain):
        """
        Fetch logo by company domain name.
        
        Args:
            domain: Company domain (e.g., 'nike.com', 'apple.com')
        
        Returns:
            dict: Logo data with keys 'logo_url', 'logo_image', 'brand_colors', 'brand_name'
                  or None if not found
        """
        try:
            # Brandfetch API endpoint
            url = f"{self.base_url}/brands/{domain}"
            
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch logo for {domain}: {response.status_code}")
                return None
            
            data = response.json()
            
            # Extract logo URL (prefer SVG, fallback to PNG)
            logo_url = None
            if 'logos' in data and len(data['logos']) > 0:
                logo_data = data['logos'][0]
                if 'formats' in logo_data:
                    formats = logo_data['formats']
                    # Prefer PNG for compatibility
                    for fmt in formats:
                        if fmt.get('format') == 'png':
                            logo_url = fmt.get('src')
                            break
                    if not logo_url and formats:
                        logo_url = formats[0].get('src')
            
            if not logo_url:
                logger.warning(f"No logo URL found for {domain}")
                return None
            
            # Download logo image
            logo_image = self._download_image(logo_url, domain)
            
            # Extract brand colors
            brand_colors = []
            if 'colors' in data:
                for color in data['colors'][:5]:  # Top 5 colors
                    if 'hex' in color:
                        brand_colors.append(color['hex'])
            
            brand_name = data.get('name', domain.split('.')[0].capitalize())
            
            return {
                'logo_url': logo_url,
                'logo_image': logo_image,
                'brand_colors': brand_colors,
                'brand_name': brand_name,
                'domain': domain
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching logo for {domain}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching logo for {domain}: {e}")
            return None
    
    def fetch_logo_by_name(self, company_name):
        """
        Fetch logo by company name.
        
        Args:
            company_name: Company name (e.g., 'Nike', 'Apple')
        
        Returns:
            dict: Logo data or None if not found
        """
        try:
            # Search for brand first
            search_url = f"{self.base_url}/search/{company_name}"
            
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code != 200 or not response.json():
                logger.warning(f"Failed to search for {company_name}")
                return None
            
            results = response.json()
            if not results or len(results) == 0:
                return None
            
            # Get first result's domain
            first_result = results[0]
            domain = first_result.get('domain')
            
            if not domain:
                return None
            
            # Fetch logo using domain
            return self.fetch_logo_by_domain(domain)
            
        except Exception as e:
            logger.error(f"Error searching for {company_name}: {e}")
            return None
    
    def _download_image(self, url, identifier):
        """
        Download image from URL and convert to OpenCV format.
        
        Args:
            url: Image URL
            identifier: Identifier for caching (domain or name)
        
        Returns:
            numpy.ndarray: OpenCV image (BGR format) or None if failed
        """
        try:
            # Check cache first
            cache_path = self.cache_dir / f"{identifier.replace('.', '_').replace('/', '_')}.png"
            if cache_path.exists():
                logger.info(f"Loading cached logo for {identifier}")
                return cv2.imread(str(cache_path))
            
            # Download image
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return None
            
            # Convert to numpy array
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is not None:
                # Cache the image
                cv2.imwrite(str(cache_path), image)
                logger.info(f"Cached logo for {identifier}")
            
            return image
            
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            return None
    
    def fetch_multiple_logos(self, identifiers, by='domain'):
        """
        Fetch multiple logos in batch.
        
        Args:
            identifiers: List of domains or company names
            by: 'domain' or 'name'
        
        Returns:
            list: List of logo data dicts (None for failed fetches)
        """
        results = []
        for identifier in identifiers:
            if by == 'domain':
                logo_data = self.fetch_logo_by_domain(identifier)
            else:
                logo_data = self.fetch_logo_by_name(identifier)
            results.append(logo_data)
        
        return results
    
    def save_to_templates_db(self, logo_data, target_dir='data/logos_db'):
        """
        Save fetched logo to the templates database for detection.
        
        Args:
            logo_data: Logo data dict from fetch_logo_by_domain/name
            target_dir: Directory where templates are stored
        
        Returns:
            Path to saved template or None if failed
        """
        if not logo_data or logo_data.get('logo_image') is None:
            logger.warning("Cannot save empty logo data")
            return None
        
        try:
            target_path = Path(target_dir)
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename based on brand name
            brand_name = logo_data['brand_name'].lower().replace(' ', '_')
            filename = f"logo_fetched_{brand_name}.png"
            file_path = target_path / filename
            
            # Save image
            success = cv2.imwrite(str(file_path), logo_data['logo_image'])
            
            if success:
                logger.info(f"Saved fetched logo to templates database: {file_path}")
                return file_path
            else:
                logger.error(f"Failed to save logo to {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving logo to templates database: {e}")
            return None
