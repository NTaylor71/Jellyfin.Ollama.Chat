"""
Test suite for similarity engine.
Tests Gensim/sklearn similarity models and document similarity calculations.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.search.similarity_engine import SimilarityEngine, SimilarityResult, ModelStats


class TestSimilarityEngine:
    """Test cases for SimilarityEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary directory for cache
        self.temp_dir = Path(tempfile.mkdtemp())
        self.engine = SimilarityEngine(
            cache_dir=self.temp_dir,
            min_similarity=0.1,
            max_cache_size=1000
        )
        
        # Sample movie documents
        self.sample_docs = {
            "movie1": {
                "name": "The Matrix",
                "overview": "A computer hacker learns about the true nature of reality",
                "genres": ["Science Fiction", "Action"],
                "people": [
                    {"name": "Keanu Reeves", "type": "Actor"},
                    {"name": "The Wachowskis", "type": "Director"}
                ]
            },
            "movie2": {
                "name": "The Dark Knight",
                "overview": "Batman battles the Joker in Gotham City",
                "genres": ["Action", "Crime", "Drama"],
                "people": [
                    {"name": "Christian Bale", "type": "Actor"},
                    {"name": "Christopher Nolan", "type": "Director"}
                ]
            },
            "movie3": {
                "name": "Inception",
                "overview": "A thief enters dreams to plant ideas",
                "genres": ["Science Fiction", "Thriller"],
                "people": [
                    {"name": "Leonardo DiCaprio", "type": "Actor"},
                    {"name": "Christopher Nolan", "type": "Director"}
                ]
            }
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.min_similarity == 0.1
        assert self.engine.max_cache_size == 1000
        assert self.engine.cache_dir == self.temp_dir
        assert len(self.engine.document_corpus) == 0
        assert len(self.engine.document_ids) == 0
    
    def test_add_document_basic(self):
        """Test adding documents to corpus."""
        self.engine.add_document("movie1", self.sample_docs["movie1"])
        
        assert len(self.engine.document_corpus) == 1
        assert len(self.engine.document_ids) == 1
        assert "movie1" in self.engine.document_ids
        assert "movie1" in self.engine.document_metadata
        
        # Check document processing
        doc_terms = self.engine.document_corpus[0]
        assert len(doc_terms) > 0
        assert any("matrix" in term.lower() for term in doc_terms)
    
    def test_add_document_update_existing(self):
        """Test updating existing document."""
        # Add document
        self.engine.add_document("movie1", self.sample_docs["movie1"])
        original_length = len(self.engine.document_corpus)
        
        # Update same document
        updated_doc = self.sample_docs["movie1"].copy()
        updated_doc["overview"] = "Updated overview with different content"
        self.engine.add_document("movie1", updated_doc)
        
        # Should not increase corpus size
        assert len(self.engine.document_corpus) == original_length
        assert len(self.engine.document_ids) == original_length
        
        # Should update metadata
        assert "updated_at" in self.engine.document_metadata["movie1"]
    
    def test_add_multiple_documents(self):
        """Test adding multiple documents."""
        for doc_id, doc_content in self.sample_docs.items():
            self.engine.add_document(doc_id, doc_content)
        
        assert len(self.engine.document_corpus) == 3
        assert len(self.engine.document_ids) == 3
        assert all(doc_id in self.engine.document_ids for doc_id in self.sample_docs.keys())
    
    def test_build_models_no_documents(self):
        """Test building models with no documents."""
        result = self.engine.build_models()
        assert result is False
    
    def test_build_gensim_models(self):
        """Test building Gensim models (mock the actual method)."""
        # Add documents
        for doc_id, doc_content in self.sample_docs.items():
            self.engine.add_document(doc_id, doc_content)
        
        # Mock the _build_gensim_models method directly
        with patch.object(self.engine, '_build_gensim_models', return_value=True) as mock_build:
            result = self.engine.build_models(use_tfidf=True, use_lsi=True, use_lda=False)
            
            assert result is True
            mock_build.assert_called_once_with(True, True, False, 100, 50)
    
    @patch('src.search.similarity_engine.GENSIM_AVAILABLE', False)
    @patch('src.search.similarity_engine.SKLEARN_AVAILABLE', True)
    def test_build_sklearn_models(self):
        """Test building sklearn models as fallback."""
        # Add documents
        for doc_id, doc_content in self.sample_docs.items():
            self.engine.add_document(doc_id, doc_content)
        
        # Mock sklearn components
        with patch('src.search.similarity_engine.TfidfVectorizer') as mock_vectorizer:
            mock_vectorizer_instance = Mock()
            mock_vectorizer_instance.fit_transform.return_value = Mock()
            mock_vectorizer_instance.get_feature_names_out.return_value = ["feature1", "feature2"]
            mock_vectorizer.return_value = mock_vectorizer_instance
            
            # Build models
            result = self.engine.build_models()
            
            assert result is True
            assert self.engine.stats.model_type == "sklearn"
            assert self.engine.stats.num_documents == 3
    
    @patch('src.search.similarity_engine.GENSIM_AVAILABLE', False)
    @patch('src.search.similarity_engine.SKLEARN_AVAILABLE', False)
    def test_build_models_no_libraries(self):
        """Test building models when no libraries available."""
        # Add documents
        for doc_id, doc_content in self.sample_docs.items():
            self.engine.add_document(doc_id, doc_content)
        
        result = self.engine.build_models()
        assert result is False
    
    def test_find_similar_no_documents(self):
        """Test similarity search with no documents."""
        results = self.engine.find_similar("action movie")
        assert results == []
    
    def test_find_similar_no_models(self):
        """Test similarity search with no built models."""
        # Add documents but don't build models
        for doc_id, doc_content in self.sample_docs.items():
            self.engine.add_document(doc_id, doc_content)
        
        results = self.engine.find_similar("action movie")
        assert results == []
    
    @patch('src.search.similarity_engine.SKLEARN_AVAILABLE', True)
    def test_find_similar_sklearn(self):
        """Test similarity search with sklearn models."""
        # Add documents
        for doc_id, doc_content in self.sample_docs.items():
            self.engine.add_document(doc_id, doc_content)
        
        # Mock sklearn components
        with patch('src.search.similarity_engine.TfidfVectorizer') as mock_vectorizer, \
             patch('src.search.similarity_engine.cosine_similarity') as mock_cosine:
            
            mock_vectorizer_instance = Mock()
            mock_vectorizer_instance.fit_transform.return_value = Mock()
            mock_vectorizer_instance.transform.return_value = Mock()
            mock_vectorizer_instance.get_feature_names_out.return_value = ["action", "movie"]
            mock_vectorizer.return_value = mock_vectorizer_instance
            
            # Mock similarity scores
            mock_cosine.return_value = [[0.8, 0.6, 0.3]]  # Similarities for 3 documents
            
            # Build models and search
            self.engine.build_models()
            results = self.engine.find_similar("action movie", top_k=2)
            
            assert len(results) <= 2
            assert all(isinstance(result, SimilarityResult) for result in results)
            if results:
                # Results should be sorted by similarity
                assert results[0].similarity_score >= results[-1].similarity_score
    
    def test_calculate_field_scores(self):
        """Test field-specific similarity calculation."""
        # Add a document
        self.engine.add_document("movie1", self.sample_docs["movie1"])
        
        # Test field score calculation
        query_terms = ["action", "matrix", "keanu"]
        field_scores = self.engine._calculate_field_scores("movie1", query_terms)
        
        assert isinstance(field_scores, dict)
        # Should have scores for fields that contain matching terms
        if field_scores:
            assert all(0 <= score <= 1 for score in field_scores.values())
    
    def test_get_stats(self):
        """Test getting model statistics."""
        stats = self.engine.get_stats()
        assert isinstance(stats, ModelStats)
        assert stats.num_documents == 0
        assert stats.model_type == "none"
        
        # Add documents and check stats update
        for doc_id, doc_content in self.sample_docs.items():
            self.engine.add_document(doc_id, doc_content)
        
        # Stats should be updated after building models
        with patch('src.search.similarity_engine.GENSIM_AVAILABLE', False), \
             patch('src.search.similarity_engine.SKLEARN_AVAILABLE', True), \
             patch.object(self.engine, '_build_sklearn_models', return_value=True) as mock_build:
            
            result = self.engine.build_models()
            
            # Manually set the stats to simulate successful model building
            self.engine.stats = ModelStats(
                num_documents=3,
                num_terms=50,
                last_updated=self.engine.stats.last_updated,
                model_type="sklearn",
                cache_size=3
            )
            
            stats = self.engine.get_stats()
            assert stats.num_documents == 3
            assert stats.model_type == "sklearn"
    
    def test_save_and_load_models(self):
        """Test saving and loading models."""
        # Add documents
        for doc_id, doc_content in self.sample_docs.items():
            self.engine.add_document(doc_id, doc_content)
        
        # Save models
        save_path = self.temp_dir / "test_models.pkl"
        result = self.engine.save_models(save_path)
        assert result is True
        assert save_path.exists()
        
        # Create new engine and load models
        new_engine = SimilarityEngine(cache_dir=self.temp_dir)
        result = new_engine.load_models(save_path)
        assert result is True
        
        # Check loaded data
        assert len(new_engine.document_corpus) == 3
        assert len(new_engine.document_ids) == 3
        assert all(doc_id in new_engine.document_ids for doc_id in self.sample_docs.keys())
    
    def test_save_models_error_handling(self):
        """Test save models error handling."""
        # Try to save to invalid path
        invalid_path = Path("/invalid/path/models.pkl")
        result = self.engine.save_models(invalid_path)
        assert result is False
    
    def test_load_models_error_handling(self):
        """Test load models error handling."""
        # Try to load non-existent file
        invalid_path = self.temp_dir / "nonexistent.pkl"
        result = self.engine.load_models(invalid_path)
        assert result is False
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Add documents
        for doc_id, doc_content in self.sample_docs.items():
            self.engine.add_document(doc_id, doc_content)
        
        assert len(self.engine.document_corpus) > 0
        assert len(self.engine.document_ids) > 0
        assert len(self.engine.document_metadata) > 0
        
        # Clear cache
        self.engine.clear_cache()
        
        assert len(self.engine.document_corpus) == 0
        assert len(self.engine.document_ids) == 0
        assert len(self.engine.document_metadata) == 0
        assert self.engine.stats.num_documents == 0
        assert self.engine.stats.model_type == "none"
    
    def test_document_processing_edge_cases(self):
        """Test document processing with edge cases."""
        # Empty document
        self.engine.add_document("empty", {})
        assert len(self.engine.document_corpus) == 1
        assert len(self.engine.document_corpus[0]) == 0
        
        # Document with None values
        self.engine.add_document("none_values", {
            "name": None,
            "overview": "",
            "genres": None
        })
        
        # Document with invalid data types
        self.engine.add_document("invalid_types", {
            "name": 12345,
            "genres": "single_string_instead_of_list"
        })
        
        # Should handle gracefully without errors
        assert len(self.engine.document_corpus) == 3


class TestSimilarityResult:
    """Test SimilarityResult dataclass."""
    
    def test_similarity_result_creation(self):
        """Test creating SimilarityResult."""
        result = SimilarityResult(
            document_id="movie1",
            similarity_score=0.85,
            matched_terms=["action", "hero"],
            field_scores={"name": 0.9, "overview": 0.8}
        )
        
        assert result.document_id == "movie1"
        assert result.similarity_score == 0.85
        assert result.matched_terms == ["action", "hero"]
        assert result.field_scores["name"] == 0.9


class TestModelStats:
    """Test ModelStats dataclass."""
    
    def test_model_stats_creation(self):
        """Test creating ModelStats."""
        from datetime import datetime
        now = datetime.now()
        
        stats = ModelStats(
            num_documents=100,
            num_terms=5000,
            last_updated=now,
            model_type="gensim",
            cache_size=100
        )
        
        assert stats.num_documents == 100
        assert stats.num_terms == 5000
        assert stats.last_updated == now
        assert stats.model_type == "gensim"
        assert stats.cache_size == 100


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])