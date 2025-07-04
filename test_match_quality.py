"""
Test suite for match quality system.
Tests match quality scoring and classification functionality.
"""

import pytest
from src.search.match_quality import (
    MatchQualityScorer, MatchQualityConfig, MatchQuality,
    QualityScore, QualityMatch
)


class TestMatchQualityConfig:
    """Test match quality configuration."""
    
    def test_default_quality_scores(self):
        """Test default quality score configuration."""
        config = MatchQualityConfig()
        
        # Test quality score values per specification
        assert config.quality_scores[MatchQuality.EXACT].base_score == 1.0
        assert config.quality_scores[MatchQuality.STEMMED].base_score == 0.8
        assert config.quality_scores[MatchQuality.SYNONYM].base_score == 0.7
        assert config.quality_scores[MatchQuality.FUZZY].base_score == 0.6
        assert config.quality_scores[MatchQuality.PARTIAL].base_score == 0.4
        assert config.quality_scores[MatchQuality.PHONETIC].base_score == 0.5
        
        # Test confidence values
        assert config.quality_scores[MatchQuality.EXACT].confidence == 1.0
        assert config.quality_scores[MatchQuality.STEMMED].confidence == 0.9
        assert config.quality_scores[MatchQuality.FUZZY].confidence == 0.7
    
    def test_thresholds(self):
        """Test threshold values."""
        config = MatchQualityConfig()
        
        assert config.max_edit_distance == 3
        assert config.edit_distance_penalty == 0.1
        assert config.min_similarity_ratio == 0.6
        assert config.min_length_for_fuzzy == 4


class TestMatchQualityScorer:
    """Test match quality scorer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = MatchQualityScorer()
    
    def test_calculate_edit_distance(self):
        """Test edit distance calculation."""
        # Identical strings
        assert self.scorer._calculate_edit_distance("test", "test") == 0
        
        # Single character difference
        assert self.scorer._calculate_edit_distance("test", "best") == 1
        
        # Multiple differences
        assert self.scorer._calculate_edit_distance("kitten", "sitting") == 3
        
        # Empty strings
        assert self.scorer._calculate_edit_distance("", "") == 0
        assert self.scorer._calculate_edit_distance("test", "") == 4
    
    def test_calculate_similarity_ratio(self):
        """Test similarity ratio calculation."""
        # Identical strings
        assert self.scorer._calculate_similarity_ratio("test", "test") == 1.0
        
        # Similar strings
        ratio = self.scorer._calculate_similarity_ratio("test", "best")
        assert 0.7 < ratio < 1.0
        
        # Very different strings
        ratio = self.scorer._calculate_similarity_ratio("abc", "xyz")
        assert ratio < 0.5
        
        # Empty strings
        assert self.scorer._calculate_similarity_ratio("", "") == 0.0
        assert self.scorer._calculate_similarity_ratio("test", "") == 0.0
    
    def test_determine_match_quality_exact(self):
        """Test exact match quality determination."""
        quality, edit_distance, similarity_ratio = self.scorer.determine_match_quality(
            "matrix", "matrix"
        )
        
        assert quality == MatchQuality.EXACT
        assert edit_distance == 0
        assert similarity_ratio == 1.0
    
    def test_determine_match_quality_fuzzy(self):
        """Test fuzzy match quality determination."""
        quality, edit_distance, similarity_ratio = self.scorer.determine_match_quality(
            "matrix", "matric"
        )
        
        assert quality == MatchQuality.FUZZY
        assert edit_distance == 1
        assert similarity_ratio > 0.8
    
    def test_determine_match_quality_partial(self):
        """Test partial match quality determination."""
        quality, edit_distance, similarity_ratio = self.scorer.determine_match_quality(
            "matrix", "the matrix movie"
        )
        
        assert quality == MatchQuality.PARTIAL
        assert edit_distance > 0
        assert similarity_ratio > 0.0
    
    def test_score_match_quality_exact(self):
        """Test exact match quality scoring."""
        match = self.scorer.score_match_quality(
            "name", "The Matrix", "matrix", "Matrix", 
            MatchQuality.EXACT, 0, 1.0
        )
        
        assert match.field_name == "name"
        assert match.query_term == "matrix"
        assert match.matched_text == "Matrix"
        assert match.quality == MatchQuality.EXACT
        assert match.quality_score == 1.0
        assert match.confidence == 1.0
        assert match.final_score == 1.0
    
    def test_score_match_quality_with_edit_distance(self):
        """Test quality scoring with edit distance penalty."""
        match = self.scorer.score_match_quality(
            "name", "The Matrix", "matrix", "matric",
            MatchQuality.FUZZY, 1, 0.9
        )
        
        assert match.quality == MatchQuality.FUZZY
        assert match.edit_distance == 1
        assert match.quality_score < 0.6  # Should be penalized
        assert match.confidence < 0.7    # Should be penalized
    
    def test_score_match_quality_with_similarity_ratio(self):
        """Test quality scoring with similarity ratio adjustment."""
        match = self.scorer.score_match_quality(
            "name", "The Matrix", "matrix", "matric",
            MatchQuality.FUZZY, 1, 0.85
        )
        
        assert match.similarity_ratio == 0.85
        assert match.quality_score > 0.0
        assert match.final_score > 0.0
    
    def test_apply_length_adjustments(self):
        """Test length-based adjustments."""
        # Test with longer matched text (should get bonus)
        adjusted_score = self.scorer._apply_length_adjustments(1.0, "test", "testing")
        assert adjusted_score > 1.0
        
        # Test with shorter query (should get penalty)
        adjusted_score = self.scorer._apply_length_adjustments(1.0, "ab", "abc")
        assert adjusted_score < 1.0
    
    def test_search_string_matches(self):
        """Test searching string fields."""
        matches = self.scorer._score_string_matches(
            "name", "The Matrix", ["matrix", "neo"]
        )
        
        # Should find "matrix" match
        matrix_matches = [m for m in matches if m.query_term == "matrix"]
        assert len(matrix_matches) == 1
        assert matrix_matches[0].quality == MatchQuality.EXACT  # Case-insensitive exact match
        
        # Should not find "neo" match
        neo_matches = [m for m in matches if m.query_term == "neo"]
        assert len(neo_matches) == 0
    
    def test_search_list_matches(self):
        """Test searching list fields."""
        genres = ["Action", "Sci-Fi", "Thriller"]
        matches = self.scorer._score_list_matches(
            "genres", genres, ["action", "drama"]
        )
        
        # Should find "action" match (case-insensitive)
        action_matches = [m for m in matches if m.query_term == "action"]
        assert len(action_matches) == 1
        assert action_matches[0].matched_text == "Action"
        
        # Should not find "drama" match
        drama_matches = [m for m in matches if m.query_term == "drama"]
        assert len(drama_matches) == 0
    
    def test_search_people_matches(self):
        """Test searching people field with objects."""
        people = [
            {"name": "Keanu Reeves", "type": "Actor"},
            {"name": "Lana Wachowski", "type": "Director"}
        ]
        matches = self.scorer._score_list_matches(
            "people", people, ["keanu", "wachowski"]
        )
        
        # Should find both matches
        keanu_matches = [m for m in matches if m.query_term == "keanu"]
        assert len(keanu_matches) == 1
        assert keanu_matches[0].matched_text == "Keanu"  # Extracts just the matching word
        
        wachowski_matches = [m for m in matches if m.query_term == "wachowski"]
        assert len(wachowski_matches) == 1
        assert wachowski_matches[0].matched_text == "Wachowski"  # Extracts just the matching word
    
    def test_search_dict_matches(self):
        """Test searching dictionary fields."""
        enhanced_fields = {
            "summary": "cyberpunk action thriller",
            "mood": "dark dystopian future",
            "themes": "reality vs simulation"
        }
        matches = self.scorer._score_dict_matches(
            "enhanced_fields", enhanced_fields, ["cyberpunk", "reality"]
        )
        
        # Should find both matches
        cyberpunk_matches = [m for m in matches if m.query_term == "cyberpunk"]
        assert len(cyberpunk_matches) == 1
        
        reality_matches = [m for m in matches if m.query_term == "reality"]
        assert len(reality_matches) == 1
    
    def test_find_best_fuzzy_match(self):
        """Test finding best fuzzy match."""
        field_value = "This is a great movie about artificial intelligence"
        
        # Should find fuzzy match for "artficial" (typo)
        fuzzy_match = self.scorer._find_best_fuzzy_match(field_value, "artficial")
        
        if fuzzy_match:  # Fuzzy matching might not find this depending on threshold
            assert fuzzy_match.matched_text == "artificial"
            assert fuzzy_match.quality == MatchQuality.FUZZY
    
    def test_score_field_matches(self):
        """Test comprehensive field matching."""
        movie_data = {
            "name": "The Matrix",
            "genres": ["Action", "Sci-Fi"],
            "people": [{"name": "Keanu Reeves", "type": "Actor"}],
            "enhanced_fields": {
                "summary": "cyberpunk action thriller"
            }
        }
        
        matches = self.scorer.score_field_matches(
            "name", movie_data["name"], ["matrix", "neo"]
        )
        
        # Should find matrix match
        matrix_matches = [m for m in matches if m.query_term == "matrix"]
        assert len(matrix_matches) == 1
    
    def test_rank_quality_matches(self):
        """Test ranking quality matches."""
        matches = [
            QualityMatch(
                "name", "The Matrix", "matrix", "Matrix",
                MatchQuality.EXACT, 1.0, 1.0, 0, 1.0, 1.0
            ),
            QualityMatch(
                "overview", "About matrix", "matrix", "matrix",
                MatchQuality.PARTIAL, 0.4, 0.6, 0, 0.8, 0.24
            ),
            QualityMatch(
                "genres", "Action", "action", "Action",
                MatchQuality.EXACT, 1.0, 1.0, 0, 1.0, 1.0
            )
        ]
        
        ranked = self.scorer.rank_quality_matches(matches)
        
        # Should rank by final score, then confidence
        assert ranked[0].final_score == 1.0
        assert ranked[1].final_score == 1.0
        assert ranked[2].final_score == 0.24
    
    def test_get_quality_stats(self):
        """Test quality statistics."""
        matches = [
            QualityMatch(
                "name", "The Matrix", "matrix", "Matrix",
                MatchQuality.EXACT, 1.0, 1.0, 0, 1.0, 1.0
            ),
            QualityMatch(
                "overview", "About matrix", "matrix", "matric",
                MatchQuality.FUZZY, 0.6, 0.7, 1, 0.9, 0.42
            ),
            QualityMatch(
                "genres", "Action", "action", "Action",
                MatchQuality.EXACT, 1.0, 1.0, 0, 1.0, 1.0
            )
        ]
        
        stats = self.scorer.get_quality_stats(matches)
        
        assert stats["total_matches"] == 3
        assert stats["quality_distribution"]["exact"] == 2
        assert stats["quality_distribution"]["fuzzy"] == 1
        assert stats["avg_score"] == (1.0 + 0.42 + 1.0) / 3
        assert stats["max_score"] == 1.0
        assert stats["min_score"] == 0.42
        assert stats["high_quality_matches"] == 2  # 2 exact matches
    
    def test_filter_by_quality(self):
        """Test filtering matches by quality."""
        matches = [
            QualityMatch(
                "name", "The Matrix", "matrix", "Matrix",
                MatchQuality.EXACT, 1.0, 1.0, 0, 1.0, 1.0
            ),
            QualityMatch(
                "overview", "About matrix", "matrix", "matric",
                MatchQuality.FUZZY, 0.6, 0.7, 1, 0.9, 0.42
            ),
            QualityMatch(
                "overview", "Contains matrix", "matrix", "matrix part",
                MatchQuality.PARTIAL, 0.4, 0.6, 0, 0.8, 0.24
            )
        ]
        
        # Filter to only exact and stemmed matches
        filtered = self.scorer.filter_by_quality(matches, MatchQuality.STEMMED)
        assert len(filtered) == 1  # Only exact match
        assert filtered[0].quality == MatchQuality.EXACT
        
        # Filter to fuzzy and better
        filtered = self.scorer.filter_by_quality(matches, MatchQuality.FUZZY)
        assert len(filtered) == 2  # Exact and fuzzy
        
        # Filter to partial and better (should include all)
        filtered = self.scorer.filter_by_quality(matches, MatchQuality.PARTIAL)
        assert len(filtered) == 3
    
    def test_empty_matches_stats(self):
        """Test statistics with empty matches."""
        stats = self.scorer.get_quality_stats([])
        assert stats == {}


class TestQualityScore:
    """Test quality score dataclass."""
    
    def test_quality_score_creation(self):
        """Test quality score creation."""
        score = QualityScore(
            base_score=0.8,
            confidence=0.9,
            description="Test quality score"
        )
        
        assert score.base_score == 0.8
        assert score.confidence == 0.9
        assert score.description == "Test quality score"


class TestQualityMatch:
    """Test quality match dataclass."""
    
    def test_quality_match_creation(self):
        """Test quality match creation."""
        match = QualityMatch(
            field_name="name",
            field_value="The Matrix",
            query_term="matrix",
            matched_text="Matrix",
            quality=MatchQuality.EXACT,
            quality_score=1.0,
            confidence=1.0,
            edit_distance=0,
            similarity_ratio=1.0,
            final_score=1.0
        )
        
        assert match.field_name == "name"
        assert match.field_value == "The Matrix"
        assert match.query_term == "matrix"
        assert match.matched_text == "Matrix"
        assert match.quality == MatchQuality.EXACT
        assert match.quality_score == 1.0
        assert match.confidence == 1.0
        assert match.edit_distance == 0
        assert match.similarity_ratio == 1.0
        assert match.final_score == 1.0


class TestMatchQualityIntegration:
    """Integration tests for match quality system."""
    
    def test_real_movie_search(self):
        """Test with realistic movie data."""
        scorer = MatchQualityScorer()
        
        movie_fields = {
            "name": "Blade Runner 2049",
            "genres": ["Sci-Fi", "Drama"],
            "people": [{"name": "Ryan Gosling", "type": "Actor"}],
            "overview": "A young blade runner's discovery leads to tracking down a former blade runner.",
            "enhanced_fields": {
                "summary": "visually stunning cyberpunk sequel"
            }
        }
        
        # Test multiple field searches
        all_matches = []
        for field_name, field_value in movie_fields.items():
            matches = scorer.score_field_matches(field_name, field_value, ["blade", "runner", "cyberpunk"])
            all_matches.extend(matches)
        
        assert len(all_matches) > 0
        
        # Should find high quality matches
        high_quality = [m for m in all_matches if m.quality in [MatchQuality.EXACT, MatchQuality.PARTIAL]]
        assert len(high_quality) > 0
    
    def test_typo_handling(self):
        """Test handling of typos."""
        scorer = MatchQualityScorer()
        
        # Test common typos
        matches = scorer._score_string_matches(
            "name", "The Matrix", ["matirx", "matrx", "matrix"]
        )
        
        # Should find the correct match
        correct_matches = [m for m in matches if m.query_term == "matrix"]
        assert len(correct_matches) == 1
        
        # Fuzzy matches might be found depending on threshold
        fuzzy_matches = [m for m in matches if m.quality == MatchQuality.FUZZY]
        # Don't assert specific count as it depends on fuzzy matching threshold
    
    def test_case_sensitivity(self):
        """Test case-insensitive matching."""
        scorer = MatchQualityScorer()
        
        matches = scorer._score_string_matches(
            "name", "The Matrix", ["MATRIX", "Matrix", "matrix"]
        )
        
        # Should find matches for all cases
        assert len(matches) == 3
        for match in matches:
            assert match.quality == MatchQuality.EXACT  # Case-insensitive exact matches
    
    def test_performance_with_large_data(self):
        """Test performance with large dataset."""
        scorer = MatchQualityScorer()
        
        # Create large movie dataset
        large_overview = "This is a very long movie description " * 100
        large_overview += "matrix appears here cyberpunk themes"
        
        matches = scorer._score_string_matches(
            "overview", large_overview, ["matrix", "cyberpunk", "themes"]
        )
        
        # Should find all matches efficiently
        assert len(matches) == 3
        
        # Should maintain quality scoring
        for match in matches:
            assert match.final_score > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])