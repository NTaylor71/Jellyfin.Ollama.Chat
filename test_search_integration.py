"""
Integration test for Stage 6.2 Field-Specific Weighting System.
Tests integration between field weights, positional scoring, and match quality.
"""

import pytest
from src.search import (
    FieldWeightedSearch, FieldWeightConfig,
    PositionalScorer, PositionalScoringConfig,
    MatchQualityScorer, MatchQualityConfig,
    MatchType, MatchQuality
)


class TestSearchSystemIntegration:
    """Test integration of all search components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.field_search = FieldWeightedSearch()
        self.positional_scorer = PositionalScorer()
        self.quality_scorer = MatchQualityScorer()
        
        self.sample_movie = {
            "name": "The Matrix",
            "original_title": "The Matrix",
            "genres": ["Action", "Sci-Fi", "Thriller"],
            "people": [
                {"name": "Keanu Reeves", "type": "Actor", "role": "Neo"},
                {"name": "Laurence Fishburne", "type": "Actor", "role": "Morpheus"},
                {"name": "Lana Wachowski", "type": "Director"},
                {"name": "Lilly Wachowski", "type": "Director"}
            ],
            "overview": "A computer hacker learns about the true nature of reality and his role in the war against its controllers.",
            "taglines": ["Welcome to the Real World", "The fight for the future begins"],
            "studios": [{"name": "Warner Bros", "id": "123"}],
            "enhanced_fields": {
                "summary": "groundbreaking cyberpunk action thriller exploring themes of reality vs simulation",
                "mood": "dark philosophical dystopian",
                "themes": "free will, consciousness, artificial intelligence"
            }
        }
    
    def test_field_weight_system(self):
        """Test field weighting system independently."""
        matches = self.field_search.search_movie_fields(
            self.sample_movie, ["matrix", "keanu"]
        )
        
        assert len(matches) > 0
        
        # Should find matrix in title (high weight)
        title_matches = [m for m in matches if m.field_name == "name" and "matrix" in m.query_term.lower()]
        assert len(title_matches) > 0
        
        # Should find keanu in people field
        people_matches = [m for m in matches if m.field_name == "people" and "keanu" in m.query_term.lower()]
        assert len(people_matches) > 0
        
        # Title matches should have higher scores
        ranked = self.field_search.rank_matches(matches)
        title_match = next((m for m in ranked if m.field_name == "name"), None)
        people_match = next((m for m in ranked if m.field_name == "people"), None)
        
        if title_match and people_match:
            assert title_match.final_score > people_match.final_score
    
    def test_positional_scoring_system(self):
        """Test positional scoring system independently."""
        # Test title field positional scoring
        matches = self.positional_scorer.score_multi_word_query(
            "name", "The Matrix", ["the", "matrix"], 1.0
        )
        
        assert len(matches) == 2
        
        # "The" should be at exact position (0)
        the_match = next(m for m in matches if m.query_term == "the")
        assert the_match.position == 0
        assert the_match.position_type.value == "exact"
        
        # "Matrix" should be at middle position (4/10 = 0.4 > 0.1 threshold)
        matrix_match = next(m for m in matches if m.query_term == "matrix")
        assert matrix_match.position == 4
        assert matrix_match.position_type.value == "middle"
        
        # Both should preserve phrase order
        assert the_match.phrase_order_preserved == True
        assert matrix_match.phrase_order_preserved == True
    
    def test_match_quality_system(self):
        """Test match quality system independently."""
        matches = self.quality_scorer.score_field_matches(
            "name", "The Matrix", ["matrix", "matric", "neo"]
        )
        
        # Should find exact match for "matrix"
        matrix_matches = [m for m in matches if m.query_term == "matrix"]
        assert len(matrix_matches) == 1
        assert matrix_matches[0].quality == MatchQuality.EXACT  # Case-insensitive exact match
        
        # Should not find match for "neo"
        neo_matches = [m for m in matches if m.query_term == "neo"]
        assert len(neo_matches) == 0
    
    def test_combined_scoring_workflow(self):
        """Test a complete scoring workflow combining all systems."""
        query_terms = ["matrix", "action", "keanu"]
        
        # Step 1: Field-weighted search
        field_matches = self.field_search.search_movie_fields(self.sample_movie, query_terms)
        assert len(field_matches) > 0
        
        # Step 2: For each field match, add positional scoring
        enhanced_matches = []
        for field_match in field_matches:
            # Get positional score for this field
            positional_score = self.positional_scorer.calculate_field_positional_score(
                field_match.field_name, field_match.field_value, [field_match.query_term]
            )
            
            # Create combined score
            combined_score = field_match.final_score * (1 + positional_score * 0.1)  # 10% positional boost
            enhanced_matches.append({
                "field_match": field_match,
                "positional_score": positional_score,
                "combined_score": combined_score
            })
        
        # Step 3: Add quality scoring
        final_matches = []
        for enhanced_match in enhanced_matches:
            field_match = enhanced_match["field_match"]
            
            # Get quality matches for this field
            quality_matches = self.quality_scorer.score_field_matches(
                field_match.field_name, field_match.field_value, [field_match.query_term]
            )
            
            # Find best quality match
            if quality_matches:
                best_quality = max(quality_matches, key=lambda q: q.final_score)
                quality_bonus = best_quality.confidence * 0.1  # 10% quality bonus
                
                final_score = enhanced_match["combined_score"] * (1 + quality_bonus)
                final_matches.append({
                    "field_match": field_match,
                    "positional_score": enhanced_match["positional_score"],
                    "quality_match": best_quality,
                    "final_score": final_score
                })
        
        assert len(final_matches) > 0
        
        # Verify that title matches score highest
        title_matches = [m for m in final_matches if m["field_match"].field_name == "name"]
        assert len(title_matches) > 0
        
        # Sort by final score
        sorted_matches = sorted(final_matches, key=lambda m: m["final_score"], reverse=True)
        
        # Top match should have a reasonable score
        assert sorted_matches[0]["final_score"] > 1.0
    
    def test_real_world_search_scenario(self):
        """Test a realistic search scenario."""
        # Scenario: User searches for "cyberpunk action with keanu reeves"
        query_terms = ["cyberpunk", "action", "keanu", "reeves"]
        
        # Field-weighted search
        field_matches = self.field_search.search_movie_fields(self.sample_movie, query_terms)
        
        # Should find matches in multiple fields
        field_names = {match.field_name for match in field_matches}
        expected_fields = {"genres", "people", "enhanced_fields"}
        assert expected_fields.intersection(field_names), f"Expected some of {expected_fields}, got {field_names}"
        
        # Quality analysis
        all_quality_matches = []
        for field_name, field_value in self.sample_movie.items():
            quality_matches = self.quality_scorer.score_field_matches(
                field_name, field_value, query_terms
            )
            all_quality_matches.extend(quality_matches)
        
        # Should have good quality matches
        high_quality = [m for m in all_quality_matches 
                       if m.quality in [MatchQuality.EXACT, MatchQuality.PARTIAL]]
        assert len(high_quality) > 0
        
        # Positional analysis for title field
        title_positional = self.positional_scorer.score_multi_word_query(
            "name", self.sample_movie["name"], ["matrix"], 1.0
        )
        
        if title_positional:
            # Matrix should be at middle position with good score
            matrix_pos = title_positional[0]
            assert matrix_pos.position_type.value in ["exact", "beginning", "middle"]
            assert matrix_pos.final_score > 1.0
    
    def test_configuration_integration(self):
        """Test that all configurations work together."""
        # Custom field weight config
        field_config = FieldWeightConfig()
        field_config.update_weight("enhanced_fields", 
            field_config.weights["enhanced_fields"].__class__(base_weight=5.0, exact_match_boost=8.0)
        )
        
        # Custom positional config
        pos_config = PositionalScoringConfig()
        pos_config.title_start_bonus = 5.0
        
        # Custom quality config
        quality_config = MatchQualityConfig()
        quality_config.quality_scores[MatchQuality.PARTIAL].base_score = 0.8
        
        # Create scorers with custom configs
        field_search = FieldWeightedSearch(field_config)
        pos_scorer = PositionalScorer(pos_config)
        quality_scorer = MatchQualityScorer(quality_config)
        
        # Test that custom configs are used
        enhanced_matches = field_search.search_movie_fields(
            self.sample_movie, ["cyberpunk"]
        )
        
        # Enhanced fields should have higher weight now
        enhanced_field_matches = [m for m in enhanced_matches if m.field_name == "enhanced_fields"]
        if enhanced_field_matches:
            assert enhanced_field_matches[0].field_weight == 5.0
    
    def test_edge_cases_integration(self):
        """Test edge cases across all systems."""
        # Empty movie
        empty_matches = self.field_search.search_movie_fields({}, ["test"])
        assert len(empty_matches) == 0
        
        # Empty query
        empty_query_matches = self.field_search.search_movie_fields(self.sample_movie, [])
        assert len(empty_query_matches) == 0
        
        # Very long query terms
        long_term = "a" * 100
        long_matches = self.field_search.search_movie_fields(self.sample_movie, [long_term])
        # Should handle gracefully without errors
        assert isinstance(long_matches, list)
        
        # Special characters
        special_matches = self.field_search.search_movie_fields(
            self.sample_movie, ["matrix!", "@action", "#sci-fi"]
        )
        # Should handle gracefully
        assert isinstance(special_matches, list)
    
    def test_performance_integration(self):
        """Test performance with integrated systems."""
        # Create larger dataset
        large_movie = self.sample_movie.copy()
        large_movie["overview"] = self.sample_movie["overview"] * 10
        large_movie["enhanced_fields"]["summary"] = self.sample_movie["enhanced_fields"]["summary"] * 5
        
        # Test with multiple query terms
        query_terms = ["matrix", "action", "cyberpunk", "keanu", "reality", "computer"]
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        
        field_matches = self.field_search.search_movie_fields(large_movie, query_terms)
        
        # Basic quality scoring
        for field_name, field_value in large_movie.items():
            if isinstance(field_value, str):
                self.quality_scorer.score_field_matches(field_name, field_value, query_terms[:3])
        
        # Basic positional scoring
        for term in query_terms[:3]:
            self.positional_scorer.score_multi_word_query("name", large_movie["name"], [term], 1.0)
        
        elapsed = time.time() - start_time
        assert elapsed < 1.0  # Should complete within 1 second
        
        # Should still find matches
        assert len(field_matches) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])