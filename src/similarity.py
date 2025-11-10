"""
Visual similarity search module using FAISS or Annoy for nearest neighbor retrieval.
Finds visually similar logos from a reference database.
"""
import cv2
import numpy as np
import pickle
from pathlib import Path
from src.utils import get_logger

logger = get_logger(__name__)

# Try importing FAISS, fallback to Annoy if not available
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS available for similarity search")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, will use Annoy fallback")

try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
    logger.info("Annoy available for similarity search")
except ImportError:
    ANNOY_AVAILABLE = False
    logger.warning("Annoy not available")


class SimilaritySearcher:
    """
    Visual similarity search using deep learning embeddings and nearest neighbor index.
    """
    
    def __init__(self, reference_dir='data/logos_db', index_path='models/similarity_index.pkl',
                 use_faiss=True):
        """
        Initialize similarity searcher.
        
        Args:
            reference_dir: Directory containing reference logo images
            index_path: Path to save/load the search index
            use_faiss: Prefer FAISS over Annoy if available
        """
        self.reference_dir = Path(reference_dir)
        self.index_path = Path(index_path)
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        
        # Embedding dimension (using simple feature extraction)
        self.embedding_dim = 128
        
        # Storage for reference data
        self.reference_embeddings = []
        self.reference_paths = []
        self.reference_names = []
        
        # Index for fast search
        self.index = None
        
        # Build or load index
        if self.index_path.exists():
            self.load_index()
        else:
            self.build_index()
    
    def embed_image(self, image):
        """
        Generate embedding vector for an image.
        
        Uses color histogram and ORB features for a compact representation.
        For production, consider using pre-trained CNN features (ResNet, etc.)
        
        Args:
            image: Input image (BGR numpy array)
        
        Returns:
            numpy.ndarray: Embedding vector (128-dim)
        """
        # Resize to standard size
        resized = cv2.resize(image, (64, 64))
        
        # Color histogram features (48 dims: 16 per channel)
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([resized], [i], None, [16], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
        
        # Texture features using Canny edges (64 dims)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Divide into 4x4 grid and compute edge density
        texture_features = []
        grid_size = 4
        block_h, block_w = edges.shape[0] // grid_size, edges.shape[1] // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                block = edges[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                texture_features.append(np.mean(block) / 255.0)
        
        # Shape features using moments (16 dims)
        moments = cv2.moments(gray)
        shape_features = [
            moments.get('m00', 0) / 1000.0,
            moments.get('m10', 0) / 1000.0,
            moments.get('m01', 0) / 1000.0,
            moments.get('m20', 0) / 1000.0,
            moments.get('m11', 0) / 1000.0,
            moments.get('m02', 0) / 1000.0,
            moments.get('m30', 0) / 1000.0,
            moments.get('m21', 0) / 1000.0,
            moments.get('m12', 0) / 1000.0,
            moments.get('m03', 0) / 1000.0,
        ]
        
        # Pad shape features to 16 dims
        shape_features.extend([0.0] * (16 - len(shape_features)))
        
        # Combine all features
        embedding = np.array(hist_features + texture_features + shape_features[:16], 
                            dtype=np.float32)
        
        # Ensure correct dimension (128)
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dim]
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def build_index(self):
        """Build similarity search index from reference logos."""
        logger.info(f"Building similarity index from {self.reference_dir}")
        
        if not self.reference_dir.exists():
            logger.warning(f"Reference directory not found: {self.reference_dir}")
            return
        
        # Load all reference images and compute embeddings
        for img_path in sorted(self.reference_dir.glob('*.png')) + sorted(self.reference_dir.glob('*.jpg')):
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                embedding = self.embed_image(image)
                
                self.reference_embeddings.append(embedding)
                self.reference_paths.append(str(img_path))
                
                # Extract name from filename
                name = img_path.stem.split('_')[-1].capitalize()
                self.reference_names.append(name)
                
                logger.debug(f"Added {img_path.name} to index")
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
        
        if len(self.reference_embeddings) == 0:
            logger.warning("No reference embeddings created")
            return
        
        # Convert to numpy array
        embeddings_array = np.array(self.reference_embeddings, dtype=np.float32)
        
        # Build index
        if self.use_faiss:
            self._build_faiss_index(embeddings_array)
        elif ANNOY_AVAILABLE:
            self._build_annoy_index(embeddings_array)
        else:
            logger.warning("No indexing library available, using linear search")
        
        # Save index
        self.save_index()
        
        logger.info(f"Built index with {len(self.reference_embeddings)} reference logos")
    
    def _build_faiss_index(self, embeddings):
        """Build FAISS index for fast similarity search."""
        # Use L2 distance (Euclidean)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings)
        logger.info("Built FAISS index")
    
    def _build_annoy_index(self, embeddings):
        """Build Annoy index for fast similarity search."""
        self.index = AnnoyIndex(self.embedding_dim, 'euclidean')
        
        for i, embedding in enumerate(embeddings):
            self.index.add_item(i, embedding)
        
        self.index.build(10)  # 10 trees
        logger.info("Built Annoy index")
    
    def search(self, query_image, top_k=5):
        """
        Search for top-k most similar logos from reference database.
        
        Args:
            query_image: Query image (BGR numpy array)
            top_k: Number of similar images to return
        
        Returns:
            list: List of dicts with keys 'path', 'name', 'distance', 'similarity'
        """
        # Generate query embedding
        query_embedding = self.embed_image(query_image)
        
        if self.index is None:
            # Fallback to linear search
            return self._linear_search(query_embedding, top_k)
        
        # Search using index
        if self.use_faiss:
            return self._search_faiss(query_embedding, top_k)
        elif ANNOY_AVAILABLE:
            return self._search_annoy(query_embedding, top_k)
        else:
            return self._linear_search(query_embedding, top_k)
    
    def _search_faiss(self, query_embedding, top_k):
        """Search using FAISS index."""
        query_array = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_array, min(top_k, len(self.reference_paths)))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.reference_paths):
                results.append({
                    'path': self.reference_paths[idx],
                    'name': self.reference_names[idx],
                    'distance': float(dist),
                    'similarity': float(1.0 / (1.0 + dist))  # Convert distance to similarity
                })
        
        return results
    
    def _search_annoy(self, query_embedding, top_k):
        """Search using Annoy index."""
        indices, distances = self.index.get_nns_by_vector(
            query_embedding, min(top_k, len(self.reference_paths)), include_distances=True
        )
        
        results = []
        for idx, dist in zip(indices, distances):
            if idx < len(self.reference_paths):
                results.append({
                    'path': self.reference_paths[idx],
                    'name': self.reference_names[idx],
                    'distance': float(dist),
                    'similarity': float(1.0 / (1.0 + dist))
                })
        
        return results
    
    def _linear_search(self, query_embedding, top_k):
        """Fallback linear search without index."""
        if len(self.reference_embeddings) == 0:
            return []
        
        # Compute distances to all references
        distances = []
        for ref_embedding in self.reference_embeddings:
            dist = np.linalg.norm(query_embedding - ref_embedding)
            distances.append(dist)
        
        # Get top-k indices
        top_indices = np.argsort(distances)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'path': self.reference_paths[idx],
                'name': self.reference_names[idx],
                'distance': float(distances[idx]),
                'similarity': float(1.0 / (1.0 + distances[idx]))
            })
        
        return results
    
    def save_index(self):
        """Save index and reference data to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'reference_paths': self.reference_paths,
            'reference_names': self.reference_names,
            'reference_embeddings': self.reference_embeddings,
            'use_faiss': self.use_faiss
        }
        
        with open(self.index_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved index to {self.index_path}")
    
    def load_index(self):
        """Load index and reference data from disk."""
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
            
            self.reference_paths = data['reference_paths']
            self.reference_names = data['reference_names']
            self.reference_embeddings = data['reference_embeddings']
            
            # Rebuild index structure
            embeddings_array = np.array(self.reference_embeddings, dtype=np.float32)
            
            if data.get('use_faiss') and FAISS_AVAILABLE:
                self._build_faiss_index(embeddings_array)
            elif ANNOY_AVAILABLE:
                self._build_annoy_index(embeddings_array)
            
            logger.info(f"Loaded index from {self.index_path}")
        except Exception as e:
            logger.error(f"Error loading index: {e}, rebuilding...")
            self.build_index()
