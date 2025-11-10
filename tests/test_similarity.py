"""
Unit tests for similarity search module.
Tests image embedding and nearest neighbor search.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import cv2
from src.similarity import SimilaritySearcher


def create_test_image(color=(100, 150, 200), size=(100, 100)):
    """Helper to create a test image."""
    return np.full((size[0], size[1], 3), color, dtype=np.uint8)


def test_embed_image_returns_correct_dimension():
    """Embedding should return vector of correct dimension."""
    searcher = SimilaritySearcher(reference_dir='data/logos_db')
    image = create_test_image()
    
    embedding = searcher.embed_image(image)
    
    assert embedding.shape == (searcher.embedding_dim,), \
        f"Embedding should have dimension {searcher.embedding_dim}"


def test_embed_image_normalized():
    """Embedding should be L2 normalized."""
    searcher = SimilaritySearcher(reference_dir='data/logos_db')
    image = create_test_image()
    
    embedding = searcher.embed_image(image)
    
    norm = np.linalg.norm(embedding)
    assert norm == pytest.approx(1.0, abs=0.01), \
        "Embedding should be L2 normalized"


def test_embed_image_deterministic():
    """Same image should produce same embedding."""
    searcher = SimilaritySearcher(reference_dir='data/logos_db')
    image = create_test_image()
    
    embedding1 = searcher.embed_image(image)
    embedding2 = searcher.embed_image(image)
    
    np.testing.assert_array_almost_equal(embedding1, embedding2,
        err_msg="Same image should produce identical embeddings")


def test_embed_image_different_for_different_images():
    """Different images should produce different embeddings."""
    searcher = SimilaritySearcher(reference_dir='data/logos_db')
    
    image1 = create_test_image(color=(255, 0, 0))
    image2 = create_test_image(color=(0, 255, 0))
    
    embedding1 = searcher.embed_image(image1)
    embedding2 = searcher.embed_image(image2)
    
    # Embeddings should be different
    assert not np.allclose(embedding1, embedding2), \
        "Different images should have different embeddings"


def test_similarity_searcher_initialization():
    """SimilaritySearcher should initialize correctly."""
    searcher = SimilaritySearcher(reference_dir='data/logos_db')
    
    assert searcher.embedding_dim == 128
    assert len(searcher.reference_embeddings) > 0, \
        "Should load reference embeddings from database"


def test_search_returns_results():
    """Search should return list of similar images."""
    searcher = SimilaritySearcher(reference_dir='data/logos_db')
    query_image = create_test_image()
    
    results = searcher.search(query_image, top_k=3)
    
    assert isinstance(results, list)
    assert len(results) <= 3, "Should return at most top_k results"


def test_search_result_format():
    """Search results should have correct format."""
    searcher = SimilaritySearcher(reference_dir='data/logos_db')
    query_image = create_test_image()
    
    results = searcher.search(query_image, top_k=5)
    
    if len(results) > 0:
        result = results[0]
        assert 'path' in result
        assert 'name' in result
        assert 'distance' in result
        assert 'similarity' in result


def test_search_similarity_range():
    """Similarity scores should be in valid range."""
    searcher = SimilaritySearcher(reference_dir='data/logos_db')
    query_image = create_test_image()
    
    results = searcher.search(query_image, top_k=5)
    
    for result in results:
        assert result['similarity'] >= 0, "Similarity should be non-negative"
        assert result['distance'] >= 0, "Distance should be non-negative"


def test_search_identical_image_highest_similarity():
    """Identical image should have highest similarity."""
    searcher = SimilaritySearcher(reference_dir='data/logos_db')
    
    # Load a reference image
    if len(searcher.reference_paths) > 0:
        ref_path = searcher.reference_paths[0]
        ref_image = cv2.imread(ref_path)
        
        results = searcher.search(ref_image, top_k=1)
        
        if len(results) > 0:
            # First result should be the same image with high similarity
            assert results[0]['similarity'] > 0.9, \
                "Identical image should have very high similarity"


def test_build_index_creates_embeddings():
    """Building index should create embeddings for all references."""
    # Create temporary searcher
    searcher = SimilaritySearcher(reference_dir='data/logos_db')
    
    # Should have embeddings for reference logos
    assert len(searcher.reference_embeddings) > 0
    assert len(searcher.reference_paths) > 0
    assert len(searcher.reference_names) > 0
    
    # Counts should match
    assert len(searcher.reference_embeddings) == len(searcher.reference_paths)
    assert len(searcher.reference_embeddings) == len(searcher.reference_names)


def test_embed_different_image_sizes():
    """Embedding should handle different image sizes."""
    searcher = SimilaritySearcher(reference_dir='data/logos_db')
    
    sizes = [(50, 50), (100, 100), (200, 150), (300, 400)]
    
    for size in sizes:
        image = create_test_image(size=size)
        embedding = searcher.embed_image(image)
        
        assert embedding.shape == (searcher.embedding_dim,), \
            f"Should produce consistent embedding dimension for size {size}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
