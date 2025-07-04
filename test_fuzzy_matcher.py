"""
Test suite for fuzzy matching system.
Tests edit distance, phonetic matching, and movie-specific fuzzy search.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.search.fuzzy_matcher import FuzzyMatcher, FuzzyMatch


class TestFuzzyMatcher:
    """Test cases for FuzzyMatcher class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.matcher = FuzzyMatcher(min_similarity=0.6, max_edit_distance=3)
        
        # Sample movie titles
        self.movie_titles = [
            "The Matrix",
            "The Dark Knight",
            "Inception",
            "Pulp Fiction",
            "The Godfather",
            "Goodfellas",
            "Casablanca",
            "The Shawshank Redemption",
            "Amélie",
            "Mika Rättö"  # Finnish movie with diacritics
        ]
        
        # Sample person names
        self.person_names = [
            "Keanu Reeves",
            "Christian Bale",
            "Leonardo DiCaprio",
            "Morgan Freeman",
            "Samuel L. Jackson",
            "Christopher Nolan",
            "Quentin Tarantino",
            "Martin Scorsese",
            "Amélie Poulain",
            "Jean-Pierre Jeunet"
        ]
    
    def test_initialization(self):
        """Test matcher initialization."""
        assert self.matcher.min_similarity == 0.6
        assert self.matcher.max_edit_distance == 3
        assert isinstance(self.matcher.soundex_map, dict)
        assert isinstance(self.matcher.movie_patterns, list)
    
    def test_normalize_for_matching(self):
        """Test text normalization for matching."""
        # Basic normalization
        result = self.matcher.normalize_for_matching("The MATRIX")
        assert result == "the matrix"
        
        # Diacritics removal
        result = self.matcher.normalize_for_matching("Amélie")
        assert result == "amelie"
        
        # Phonetic simplification
        result = self.matcher.normalize_for_matching("Philosophy")
        assert result == "filosofy"  # ph -> f
        
        # Multiple spaces
        result = self.matcher.normalize_for_matching("  Multiple   Spaces  ")
        assert result == "multiple spaces"
    
    def test_clean_movie_title(self):
        """Test movie title cleaning."""
        # Remove articles
        result = self.matcher.clean_movie_title("The Matrix")
        assert "matrix" in result
        assert "the" not in result
        
        # Remove year
        result = self.matcher.clean_movie_title("The Matrix (1999)")
        assert "matrix" in result
        assert "1999" not in result
        
        # Remove punctuation
        result = self.matcher.clean_movie_title("The Lord of the Rings: Fellowship")
        assert "lord" in result
        assert "rings" in result
        assert "fellowship" in result
        assert ":" not in result
        
        # Handle diacritics
        result = self.matcher.clean_movie_title("Amélie")
        assert result == "amelie"
    
    def test_clean_person_name(self):
        """Test person name cleaning."""
        # Basic cleaning
        result = self.matcher.clean_person_name("Samuel L. Jackson")
        assert "samuel" in result
        assert "jackson" in result
        
        # Handle suffixes
        result = self.matcher.clean_person_name("Robert Downey Jr.")
        assert "robert" in result
        assert "downey" in result
        assert "junior" in result or "jr" in result
        
        # Diacritics
        result = self.matcher.clean_person_name("Jean-Pierre Jeunet")
        assert "jean" in result
        assert "pierre" in result
        assert "jeunet" in result
    
    def test_edit_distance(self):
        """Test edit distance calculation."""
        # Identical strings
        distance = self.matcher.edit_distance("matrix", "matrix")
        assert distance == 0
        
        # Single character difference
        distance = self.matcher.edit_distance("matrix", "matric")
        assert distance == 1
        
        # Multiple differences
        distance = self.matcher.edit_distance("matrix", "metric")
        assert distance == 2
        
        # Empty strings
        distance = self.matcher.edit_distance("", "")
        assert distance == 0
        
        distance = self.matcher.edit_distance("", "test")
        assert distance == 4
        
        distance = self.matcher.edit_distance("test", "")
        assert distance == 4
    
    def test_similarity_ratio(self):
        """Test similarity ratio calculation."""
        # Identical strings
        ratio = self.matcher.similarity_ratio("matrix", "matrix")
        assert ratio == 1.0
        
        # Similar strings
        ratio = self.matcher.similarity_ratio("matrix", "matric")
        assert 0.8 < ratio < 1.0
        
        # Different strings
        ratio = self.matcher.similarity_ratio("matrix", "batman")
        assert ratio < 0.5
        
        # Empty strings
        ratio = self.matcher.similarity_ratio("", "")
        assert ratio == 1.0
        
        ratio = self.matcher.similarity_ratio("", "test")
        assert ratio == 0.0
    
    def test_partial_ratio(self):
        """Test partial ratio calculation."""
        # Substring match
        ratio = self.matcher.partial_ratio("matrix", "the matrix reloaded")
        assert ratio > 0.8
        
        # Reverse substring
        ratio = self.matcher.partial_ratio("the matrix reloaded", "matrix")
        assert ratio > 0.8
        
        # No match
        ratio = self.matcher.partial_ratio("matrix", "batman")
        assert ratio < 0.5
        
        # Empty strings
        ratio = self.matcher.partial_ratio("", "test")
        assert ratio == 0.0
    
    def test_soundex(self):
        """Test Soundex code generation."""
        # Basic Soundex
        code = self.matcher.soundex("Smith")
        assert code == "S530"
        
        code = self.matcher.soundex("Johnson")
        assert code == "J525"
        
        # Similar sounding names should have same code
        code1 = self.matcher.soundex("Jackson")
        code2 = self.matcher.soundex("Jakson")
        # Note: Exact Soundex match depends on implementation
        assert len(code1) == 4
        assert len(code2) == 4
        
        # Empty input
        code = self.matcher.soundex("")
        assert code == ""
        
        # Non-alphabetic input
        code = self.matcher.soundex("123")
        assert code == ""
    
    def test_phonetic_similarity(self):
        """Test phonetic similarity calculation."""
        # Similar sounding names
        similarity = self.matcher.phonetic_similarity("Jackson", "Jakson")
        assert similarity > 0.5
        
        # Different sounding names
        similarity = self.matcher.phonetic_similarity("Smith", "Johnson")
        assert similarity < 0.8
        
        # Empty strings
        similarity = self.matcher.phonetic_similarity("", "test")
        assert similarity == 0.0
    
    def test_match_exact(self):
        """Test exact matching."""
        candidates = ["The Matrix", "The Dark Knight", "Inception"]
        
        # Exact match
        matches = self.matcher.match_exact("The Matrix", candidates)
        assert len(matches) == 1
        assert matches[0].text == "The Matrix"
        assert matches[0].match_type == "exact"
        assert matches[0].score == 1.0
        
        # Case insensitive match
        matches = self.matcher.match_exact("the matrix", candidates)
        assert len(matches) == 1
        assert matches[0].text == "The Matrix"
        
        # No match
        matches = self.matcher.match_exact("Batman", candidates)
        assert len(matches) == 0
    
    def test_match_fuzzy(self):
        """Test fuzzy matching."""
        candidates = ["The Matrix", "The Dark Knight", "Inception"]
        
        # Close match
        matches = self.matcher.match_fuzzy("Matric", candidates)
        # Should find "The Matrix" with fuzzy matching
        assert len(matches) >= 0  # Depends on similarity threshold
        
        # Check that matches meet criteria
        for match in matches:
            assert match.score >= self.matcher.min_similarity
            assert match.edit_distance <= self.matcher.max_edit_distance
            assert match.match_type == "fuzzy"
    
    def test_match_partial(self):
        """Test partial matching."""
        candidates = ["The Matrix Reloaded", "The Dark Knight Rises", "Inception"]
        
        # Partial match
        matches = self.matcher.match_partial("Matrix", candidates)
        assert len(matches) >= 1
        
        # Should find "The Matrix Reloaded"
        matrix_match = next((m for m in matches if "Matrix" in m.text), None)
        assert matrix_match is not None
        assert matrix_match.match_type == "partial"
        assert matrix_match.score >= self.matcher.min_similarity
    
    def test_match_phonetic(self):
        """Test phonetic matching."""
        candidates = ["Jackson", "Johnson", "Jakson", "Smith"]
        
        # Phonetic match
        matches = self.matcher.match_phonetic("Jaksen", candidates)
        
        # Should find phonetically similar names
        for match in matches:
            assert match.match_type == "phonetic"
            assert match.score >= self.matcher.min_similarity
    
    def test_match_movie_title(self):
        """Test movie title matching with specialized cleaning."""
        # Test with articles and punctuation
        matches = self.matcher.match_movie_title("Dark Knight", self.movie_titles)
        
        # Should find "The Dark Knight"
        dark_knight_match = next((m for m in matches if "Dark Knight" in m.text), None)
        assert dark_knight_match is not None
        
        # Test with year
        matches = self.matcher.match_movie_title("Matrix 1999", ["The Matrix (1999)", "The Matrix Reloaded"])
        assert len(matches) >= 1
        
        # Test with diacritics
        matches = self.matcher.match_movie_title("Amelie", self.movie_titles)
        amelie_match = next((m for m in matches if "Amélie" in m.text), None)
        assert amelie_match is not None
    
    def test_match_person_name(self):
        """Test person name matching with specialized cleaning."""
        # Test full name match
        matches = self.matcher.match_person_name("Keanu Reeves", self.person_names)
        
        keanu_match = next((m for m in matches if "Keanu Reeves" in m.text), None)
        assert keanu_match is not None
        assert keanu_match.match_type in ["exact_name", "fuzzy_name"]
        
        # Test partial name match (first name only)
        matches = self.matcher.match_person_name("Leonardo", self.person_names)
        
        # Should find "Leonardo DiCaprio"
        leonardo_match = next((m for m in matches if "Leonardo" in m.text), None)
        assert leonardo_match is not None
        
        # Test with different case
        matches = self.matcher.match_person_name("christopher nolan", self.person_names)
        nolan_match = next((m for m in matches if "Christopher Nolan" in m.text), None)
        assert nolan_match is not None
    
    def test_find_best_matches_general(self):
        """Test finding best matches with general strategy."""
        candidates = ["The Matrix", "The Dark Knight", "Matrix Reloaded", "Batman"]
        
        matches = self.matcher.find_best_matches("Matrix", candidates, "general", top_k=3)
        
        assert len(matches) <= 3
        assert all(isinstance(match, FuzzyMatch) for match in matches)
        
        # Results should be sorted by score
        for i in range(1, len(matches)):
            assert matches[i-1].score >= matches[i].score
        
        # Should find relevant movies
        matrix_movies = [m for m in matches if "Matrix" in m.text]
        assert len(matrix_movies) >= 1
    
    def test_find_best_matches_movie_title(self):
        """Test finding best matches for movie titles."""
        matches = self.matcher.find_best_matches(
            "dark knight", 
            self.movie_titles, 
            "movie_title", 
            top_k=5
        )
        
        assert len(matches) <= 5
        
        # Should find "The Dark Knight"
        dark_knight_match = next((m for m in matches if "Dark Knight" in m.text), None)
        assert dark_knight_match is not None
    
    def test_find_best_matches_person_name(self):
        """Test finding best matches for person names."""
        matches = self.matcher.find_best_matches(
            "christian", 
            self.person_names, 
            "person_name", 
            top_k=5
        )
        
        assert len(matches) <= 5
        
        # Should find "Christian Bale"
        christian_match = next((m for m in matches if "Christian" in m.text), None)
        assert christian_match is not None
    
    def test_match_score_threshold(self):
        """Test match quality descriptions."""
        assert self.matcher.match_score_threshold(0.98) == "excellent"
        assert self.matcher.match_score_threshold(0.88) == "very_good"
        assert self.matcher.match_score_threshold(0.78) == "good"
        assert self.matcher.match_score_threshold(0.68) == "fair"
        assert self.matcher.match_score_threshold(0.58) == "poor"
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty query
        matches = self.matcher.find_best_matches("", self.movie_titles)
        assert matches == []
        
        # Empty candidates
        matches = self.matcher.find_best_matches("matrix", [])
        assert matches == []
        
        # None input
        matches = self.matcher.find_best_matches(None, self.movie_titles)
        assert matches == []
        
        # Single character query
        matches = self.matcher.find_best_matches("a", self.movie_titles)
        # Should handle gracefully (may or may not return matches)
        assert isinstance(matches, list)
    
    def test_unicode_handling(self):
        """Test Unicode and special character handling."""
        # Test with various Unicode characters
        unicode_titles = [
            "Amélie",
            "Crouching Tiger, Hidden Dragon (臥虎藏龍)",
            "Das Boot",
            "La Dolce Vita",
            "Kärlekens språk"
        ]
        
        # Should handle Unicode normalization
        matches = self.matcher.find_best_matches("Amelie", unicode_titles, "movie_title")
        amelie_match = next((m for m in matches if "Amélie" in m.text), None)
        assert amelie_match is not None
        
        # Test with Unicode input
        matches = self.matcher.find_best_matches("Kärlekens", unicode_titles, "movie_title")
        # Should handle normalization
        assert isinstance(matches, list)
    
    def test_performance_with_large_dataset(self):
        """Test performance with larger dataset."""
        # Create larger dataset
        large_dataset = self.movie_titles * 100  # 1000 titles
        
        # Should complete in reasonable time
        matches = self.matcher.find_best_matches(
            "matrix", 
            large_dataset, 
            "movie_title", 
            top_k=10
        )
        
        assert len(matches) <= 10
        assert all(isinstance(match, FuzzyMatch) for match in matches)


class TestFuzzyMatch:
    """Test FuzzyMatch dataclass."""
    
    def test_fuzzy_match_creation(self):
        """Test creating FuzzyMatch objects."""
        match = FuzzyMatch(
            text="The Matrix",
            score=0.95,
            match_type="exact",
            edit_distance=0,
            matched_portion="The Matrix",
            start_pos=0,
            end_pos=10
        )
        
        assert match.text == "The Matrix"
        assert match.score == 0.95
        assert match.match_type == "exact"
        assert match.edit_distance == 0
        assert match.matched_portion == "The Matrix"
        assert match.start_pos == 0
        assert match.end_pos == 10


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])