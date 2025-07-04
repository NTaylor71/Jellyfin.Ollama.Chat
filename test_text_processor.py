"""
Test suite for text processing pipeline.
Tests NLTK processing, normalization, and movie-specific field handling.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.search.text_processor import TextProcessor, ProcessedText


class TestTextProcessor:
    """Test cases for TextProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TextProcessor()
    
    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.language == "en"
        assert isinstance(self.processor.movie_terms, set)
        assert "action" in self.processor.movie_terms
        assert "director" in self.processor.movie_terms
    
    def test_normalize_text_basic(self):
        """Test basic text normalization."""
        # Basic cleanup
        result = self.processor.normalize_text("  Multiple   spaces  ")
        assert result == "Multiple spaces"
        
        # Diacritics
        result = self.processor.normalize_text("Café with Naïve Résumé")
        assert result == "Cafe with Naive Resume"
        
        # Special characters
        result = self.processor.normalize_text("It's a \"great\" movie – really!")
        assert result == "It's a \"great\" movie - really!"
    
    def test_normalize_text_movie_specific(self):
        """Test movie-specific text normalization."""
        # Movie with diacritics
        result = self.processor.normalize_text("Amélie")
        assert result == "Amelie"
        
        # Finnish movie title
        result = self.processor.normalize_text("Mika Rättö")
        assert result == "Mika Ratto"
        
        # German umlaut
        result = self.processor.normalize_text("Das Böse")
        assert result == "Das Bose"
    
    def test_tokenize_basic(self):
        """Test basic tokenization."""
        tokens = self.processor.tokenize("The Matrix is great!")
        expected = ["the", "matrix", "is", "great"]
        assert tokens == expected
    
    def test_tokenize_movie_title(self):
        """Test tokenization of movie titles."""
        tokens = self.processor.tokenize("The Lord of the Rings: Fellowship")
        assert "lord" in tokens
        assert "rings" in tokens
        assert "fellowship" in tokens
        # Punctuation should be filtered out
        assert ":" not in tokens
    
    def test_tokenize_empty_input(self):
        """Test tokenization with empty input."""
        tokens = self.processor.tokenize("")
        assert tokens == []
        
        tokens = self.processor.tokenize(None)
        assert tokens == []
    
    def test_remove_stopwords(self):
        """Test stop word removal."""
        tokens = ["the", "matrix", "is", "a", "great", "movie"]
        filtered = self.processor.remove_stopwords(tokens)
        
        # Common stop words should be removed
        assert "the" not in filtered
        assert "is" not in filtered
        assert "a" not in filtered
        
        # Content words should remain
        assert "matrix" in filtered
        assert "great" in filtered
        assert "movie" in filtered
    
    def test_remove_stopwords_preserves_movie_terms(self):
        """Test that movie-specific terms are preserved."""
        tokens = ["action", "drama", "director", "actor"]
        filtered = self.processor.remove_stopwords(tokens)
        
        # Movie terms should not be removed even if in stop words
        assert "action" in filtered
        assert "drama" in filtered
        assert "director" in filtered
        assert "actor" in filtered
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        tokens = ["matrix", "neo", "ai", "x", "a", "123"]
        keywords = self.processor.extract_keywords(tokens)
        
        # Valid keywords
        assert "matrix" in keywords
        assert "neo" in keywords
        assert "ai" in keywords
        
        # Single characters and numbers should be filtered
        assert "x" not in keywords
        assert "a" not in keywords
        assert "123" not in keywords
    
    def test_process_text_complete_pipeline(self):
        """Test complete text processing pipeline."""
        text = "The Matrix: A sci-fi masterpiece starring Keanu Reeves!"
        result = self.processor.process_text(text)
        
        assert isinstance(result, ProcessedText)
        assert result.original == text
        assert "matrix" in result.keywords
        assert "sci-fi" in result.keywords or "sci" in result.keywords
        assert "masterpiece" in result.keywords
        assert "keanu" in result.keywords
        assert "reeves" in result.keywords
        
        # Stop words should be removed
        assert "the" not in result.keywords
        assert "a" not in result.keywords
    
    def test_process_movie_field_name(self):
        """Test processing movie name field."""
        result = self.processor.process_movie_field("name", "The Dark Knight")
        assert "dark" in result
        assert "knight" in result
        assert "the" not in result  # Stop word removed
    
    def test_process_movie_field_people(self):
        """Test processing people field."""
        people = [
            {"name": "Christian Bale", "type": "Actor", "role": "Batman"},
            {"name": "Christopher Nolan", "type": "Director"}
        ]
        result = self.processor.process_movie_field("people", people)
        
        assert "christian" in result
        assert "bale" in result
        assert "christopher" in result
        assert "nolan" in result
    
    def test_process_movie_field_genres(self):
        """Test processing genres field."""
        genres = ["Action", "Crime", "Drama"]
        result = self.processor.process_movie_field("genres", genres)
        
        assert "action" in result
        assert "crime" in result
        assert "drama" in result
    
    def test_process_movie_field_enhanced_fields(self):
        """Test processing enhanced fields from LLM."""
        enhanced = {
            "enhanced_summary": "dark psychological thriller with complex characters",
            "enhanced_themes": "justice, morality, chaos"
        }
        result = self.processor.process_movie_field("enhanced_fields", enhanced)
        
        assert "dark" in result
        assert "psychological" in result
        assert "thriller" in result
        assert "justice" in result
        assert "morality" in result
        assert "chaos" in result
    
    def test_process_movie_document(self):
        """Test processing complete movie document."""
        movie_doc = {
            "name": "The Dark Knight",
            "overview": "Batman fights the Joker in Gotham City",
            "people": [
                {"name": "Christian Bale", "type": "Actor"},
                {"name": "Heath Ledger", "type": "Actor"},
                {"name": "Christopher Nolan", "type": "Director"}
            ],
            "genres": ["Action", "Crime", "Drama"],
            "enhanced_fields": {
                "enhanced_summary": "psychological crime thriller"
            }
        }
        
        result = self.processor.process_movie_document(movie_doc)
        
        # Check all fields are processed
        assert "name" in result
        assert "overview" in result
        assert "people" in result
        assert "genres" in result
        assert "enhanced_fields" in result
        
        # Check content extraction
        assert "dark" in result["name"]
        assert "knight" in result["name"]
        assert "batman" in result["overview"]
        assert "joker" in result["overview"]
        assert "christian" in result["people"]
        assert "nolan" in result["people"]
        assert "action" in result["genres"]
        assert "psychological" in result["enhanced_fields"]
    
    def test_detect_language_basic(self):
        """Test basic language detection."""
        # English (default)
        lang = self.processor.detect_language("The quick brown fox")
        assert lang == "en"
        
        # Empty text
        lang = self.processor.detect_language("")
        assert lang == "en"
        
        # Text with special characters (basic heuristic)
        lang = self.processor.detect_language("Niño está feliz")
        assert lang == "es"  # Should detect Spanish due to 'ñ'
    
    def test_char_mappings(self):
        """Test character mapping functionality."""
        # Test various diacritics
        test_cases = [
            ("café", "cafe"),
            ("naïve", "naive"),
            ("résumé", "resume"),
            ("piñata", "pinata"),
            ("björk", "bjork"),
            ("größe", "grosse")
        ]
        
        for input_text, expected in test_cases:
            result = self.processor.normalize_text(input_text)
            assert expected in result.lower()
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # None input
        result = self.processor.process_text(None)
        assert result.original == ""
        assert result.keywords == []
        
        # Empty string
        result = self.processor.process_text("")
        assert result.original == ""
        assert result.keywords == []
        
        # Only punctuation
        result = self.processor.process_text("!@#$%^&*()")
        assert len(result.keywords) == 0
        
        # Only numbers
        result = self.processor.process_text("123 456 789")
        assert len(result.keywords) == 0
        
        # Mixed valid and invalid
        result = self.processor.process_text("movie! 123 @#$ great")
        assert "movie" in result.keywords
        assert "great" in result.keywords
        assert len([k for k in result.keywords if k.isdigit()]) == 0


class TestTextProcessorWithoutNLTK:
    """Test TextProcessor fallback behavior when NLTK is not available."""
    
    def test_basic_tokenize_fallback(self):
        """Test basic tokenization fallback."""
        processor = TextProcessor()
        
        # Test the _basic_tokenize method directly
        tokens = processor._basic_tokenize("The Matrix is great!")
        assert "matrix" in tokens
        assert "is" in tokens
        assert "great" in tokens
        # Punctuation should be filtered
        assert "!" not in tokens
    
    def test_fallback_processing(self):
        """Test processing with fallback methods."""
        processor = TextProcessor()
        
        # Even without NLTK, basic processing should work
        text = "Action movie with great actors"
        result = processor.process_text(text)
        
        assert isinstance(result, ProcessedText)
        assert len(result.keywords) > 0
        assert any("action" in keyword.lower() for keyword in result.keywords)


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])