"""
Test suite for positional scorer system.
Tests positional scoring and ranking functionality.
"""

import pytest
from src.search.positional_scorer import (
    PositionalScorer, PositionalScoringConfig, PositionalMatch,
    PositionType
)


class TestPositionalScoringConfig:
    """Test positional scoring configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PositionalScoringConfig()
        
        assert config.first_word_bonus == 2.0
        assert config.title_start_bonus == 3.0
        assert config.phrase_order_bonus == 1.5
        assert config.exact_position_bonus == 2.5
        
        assert config.beginning_threshold == 0.1
        assert config.end_threshold == 0.9
        assert config.position_decay_factor == 0.8
        
        assert "name" in config.high_priority_fields
        assert "original_title" in config.high_priority_fields
        
        assert config.word_position_bonuses[0] == 2.0
        assert config.word_position_bonuses[1] == 1.5


class TestPositionalScorer:
    """Test positional scorer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = PositionalScorer()
    
    def test_get_position_type(self):
        """Test position type determination."""
        # Test exact position (beginning)
        assert self.scorer._get_position_type(0, 100) == PositionType.EXACT
        
        # Test beginning
        assert self.scorer._get_position_type(5, 100) == PositionType.BEGINNING
        
        # Test middle
        assert self.scorer._get_position_type(50, 100) == PositionType.MIDDLE
        
        # Test end
        assert self.scorer._get_position_type(95, 100) == PositionType.END
    
    def test_get_word_position(self):
        """Test word position calculation."""
        text = "The Matrix is a great movie"
        
        # Position 0 (start of "The") = word position 0
        assert self.scorer._get_word_position(text, 0) == 0
        
        # Position 4 (start of "Matrix") = word position 1
        assert self.scorer._get_word_position(text, 4) == 1
        
        # Position 11 (start of "is") = word position 2
        assert self.scorer._get_word_position(text, 11) == 2
    
    def test_find_all_positions(self):
        """Test finding all positions of a term."""
        text = "the matrix and the matrix reloaded"
        positions = self.scorer._find_all_positions(text, "matrix")
        
        assert len(positions) == 2
        assert 4 in positions  # First "matrix"
        assert 19 in positions  # Second "matrix"
    
    def test_score_position_exact(self):
        """Test scoring exact position (beginning)."""
        match = self.scorer.score_position(
            "name", "The Matrix", "the", 0, 1.0
        )
        
        assert match.field_name == "name"
        assert match.query_term == "the"
        assert match.position == 0
        assert match.position_type == PositionType.EXACT
        assert match.word_position == 0
        assert match.final_score > 5.0  # Should get multiple bonuses
    
    def test_score_position_beginning(self):
        """Test scoring beginning position."""
        match = self.scorer.score_position(
            "overview", "A great movie about the matrix", "great", 2, 1.0
        )
        
        assert match.position == 2
        assert match.position_type == PositionType.BEGINNING
        assert match.word_position == 1
        assert match.final_score > 1.0
    
    def test_score_position_middle(self):
        """Test scoring middle position."""
        long_text = "This is a very long movie description that contains many words and the matrix appears in the middle"
        match = self.scorer.score_position(
            "overview", long_text, "matrix", 82, 1.0
        )
        
        assert match.position_type == PositionType.MIDDLE
        assert match.final_score > 0.5
        assert match.final_score < 2.0  # Should be less than beginning
    
    def test_score_position_title_field(self):
        """Test scoring in high-priority title field."""
        match = self.scorer.score_position(
            "name", "The Matrix", "matrix", 4, 1.0
        )
        
        assert match.field_name == "name"
        assert match.position_type == PositionType.MIDDLE  # Position 4/10 = 0.4 > 0.1 threshold
        # Should still get word position bonus for second word
        assert match.final_score > 1.0
    
    def test_score_multi_word_query(self):
        """Test multi-word query scoring."""
        matches = self.scorer.score_multi_word_query(
            "name", "The Matrix Reloaded", ["matrix", "reloaded"], 1.0
        )
        
        assert len(matches) == 2
        
        # Find matrix match
        matrix_match = next(m for m in matches if m.query_term == "matrix")
        assert matrix_match.position == 4
        
        # Find reloaded match
        reloaded_match = next(m for m in matches if m.query_term == "reloaded")
        assert reloaded_match.position == 11
    
    def test_check_multi_word_phrase_order(self):
        """Test phrase order checking."""
        # Test correct order
        result = self.scorer._check_multi_word_phrase_order(
            "the matrix reloaded", ["matrix", "reloaded"], 0, 4
        )
        assert result == True
        
        # Test incorrect order (reloaded before matrix)
        result = self.scorer._check_multi_word_phrase_order(
            "reloaded matrix", ["matrix", "reloaded"], 0, 9
        )
        assert result == False
    
    def test_phrase_order_bonus(self):
        """Test phrase order bonus application."""
        matches = self.scorer.score_multi_word_query(
            "name", "The Matrix Reloaded", ["matrix", "reloaded"], 1.0
        )
        
        # Both matches should have phrase order preserved
        for match in matches:
            assert match.phrase_order_preserved == True
            # Should get phrase order bonus
            assert match.positional_bonus > 1.0
    
    def test_rank_positional_matches(self):
        """Test ranking positional matches."""
        matches = [
            PositionalMatch(
                "name", "The Matrix", "matrix", 4, PositionType.BEGINNING, 1, True, 1.0, 3.0, 3.0
            ),
            PositionalMatch(
                "overview", "About the matrix", "matrix", 10, PositionType.MIDDLE, 2, True, 1.0, 1.5, 1.5
            ),
            PositionalMatch(
                "name", "The Matrix", "the", 0, PositionType.EXACT, 0, True, 1.0, 5.0, 5.0
            )
        ]
        
        ranked = self.scorer.rank_positional_matches(matches)
        
        assert ranked[0].final_score == 5.0  # "the" at exact position
        assert ranked[1].final_score == 3.0  # "matrix" at beginning
        assert ranked[2].final_score == 1.5  # "matrix" in middle
    
    def test_get_best_positional_match(self):
        """Test getting best positional match."""
        matches = [
            PositionalMatch(
                "name", "The Matrix", "matrix", 4, PositionType.BEGINNING, 1, True, 1.0, 3.0, 3.0
            ),
            PositionalMatch(
                "overview", "About the matrix", "matrix", 10, PositionType.MIDDLE, 2, True, 1.0, 1.5, 1.5
            )
        ]
        
        best = self.scorer.get_best_positional_match(matches)
        assert best.final_score == 3.0
        assert best.field_name == "name"
    
    def test_calculate_field_positional_score(self):
        """Test field positional score calculation."""
        score = self.scorer.calculate_field_positional_score(
            "name", "The Matrix Reloaded", ["matrix", "reloaded"]
        )
        
        assert score > 0.0
        assert isinstance(score, float)
    
    def test_get_position_stats(self):
        """Test position statistics."""
        matches = [
            PositionalMatch(
                "name", "The Matrix", "the", 0, PositionType.EXACT, 0, True, 1.0, 5.0, 5.0
            ),
            PositionalMatch(
                "name", "The Matrix", "matrix", 4, PositionType.BEGINNING, 1, True, 1.0, 3.0, 3.0
            ),
            PositionalMatch(
                "overview", "About the matrix", "matrix", 10, PositionType.MIDDLE, 2, False, 1.0, 1.5, 1.5
            )
        ]
        
        stats = self.scorer.get_position_stats(matches)
        
        assert stats["total_matches"] == 3
        assert stats["position_types"]["exact"] == 1
        assert stats["position_types"]["beginning"] == 1
        assert stats["position_types"]["middle"] == 1
        assert stats["avg_word_position"] == 1.0  # (0+1+2)/3
        assert stats["phrase_order_preserved"] == 2/3
        assert stats["best_score"] == 5.0
        assert stats["avg_score"] == (5.0 + 3.0 + 1.5) / 3
    
    def test_empty_matches_stats(self):
        """Test statistics with empty matches."""
        stats = self.scorer.get_position_stats([])
        assert stats == {}
    
    def test_position_decay(self):
        """Test position decay in middle positions."""
        # Test positions at different points in middle
        early_middle = self.scorer.score_position(
            "overview", "A" * 100, "test", 20, 1.0
        )
        
        late_middle = self.scorer.score_position(
            "overview", "A" * 100, "test", 80, 1.0
        )
        
        # Earlier middle position should score higher
        assert early_middle.final_score > late_middle.final_score
    
    def test_word_position_bonuses(self):
        """Test word position bonuses."""
        text = "First second third fourth fifth word"
        
        # Test different word positions
        first_word = self.scorer.score_position("test", text, "first", 0, 1.0)
        second_word = self.scorer.score_position("test", text, "second", 6, 1.0)
        fifth_word = self.scorer.score_position("test", text, "fifth", 25, 1.0)
        
        # First word should have highest bonus
        assert first_word.positional_bonus > second_word.positional_bonus
        assert second_word.positional_bonus > fifth_word.positional_bonus


class TestPositionalMatch:
    """Test positional match dataclass."""
    
    def test_positional_match_creation(self):
        """Test positional match creation."""
        match = PositionalMatch(
            field_name="name",
            field_value="The Matrix",
            query_term="matrix",
            position=4,
            position_type=PositionType.BEGINNING,
            word_position=1,
            phrase_order_preserved=True,
            base_score=1.0,
            positional_bonus=3.0,
            final_score=3.0
        )
        
        assert match.field_name == "name"
        assert match.field_value == "The Matrix"
        assert match.query_term == "matrix"
        assert match.position == 4
        assert match.position_type == PositionType.BEGINNING
        assert match.word_position == 1
        assert match.phrase_order_preserved == True
        assert match.base_score == 1.0
        assert match.positional_bonus == 3.0
        assert match.final_score == 3.0


class TestPositionalScorerIntegration:
    """Integration tests for positional scorer."""
    
    def test_real_movie_title_matching(self):
        """Test with real movie titles."""
        scorer = PositionalScorer()
        
        # Test exact title match
        matches = scorer.score_multi_word_query(
            "name", "The Lord of the Rings", ["lord", "rings"], 1.0
        )
        
        assert len(matches) == 2
        
        # Both should preserve phrase order
        for match in matches:
            assert match.phrase_order_preserved == True
        
        # "Lord" should be at beginning, "Rings" at end
        lord_match = next(m for m in matches if m.query_term == "lord")
        rings_match = next(m for m in matches if m.query_term == "rings")
        
        assert lord_match.position < rings_match.position
    
    def test_tagline_vs_title_scoring(self):
        """Test scoring differences between fields."""
        scorer = PositionalScorer()
        
        # Same term in title vs tagline
        title_match = scorer.score_position("name", "The Matrix", "matrix", 4, 1.0)
        tagline_match = scorer.score_position("taglines", "Welcome to the Matrix", "matrix", 15, 1.0)
        
        # Title should score higher due to high priority field bonus
        assert title_match.final_score > tagline_match.final_score
    
    def test_performance_with_long_text(self):
        """Test performance with long text."""
        scorer = PositionalScorer()
        
        # Create long text
        long_text = "This is a very long movie description " * 100
        long_text += "matrix appears here"
        
        match = scorer.score_position("overview", long_text, "matrix", len(long_text) - 18, 1.0)
        
        assert match.position_type == PositionType.END
        assert match.final_score > 0.0
    
    def test_edge_cases(self):
        """Test edge cases."""
        scorer = PositionalScorer()
        
        # Empty text
        score = scorer.calculate_field_positional_score("name", "", ["test"])
        assert score == 0.0
        
        # Empty query
        score = scorer.calculate_field_positional_score("name", "The Matrix", [])
        assert score == 0.0
        
        # Single character
        match = scorer.score_position("name", "A", "a", 0, 1.0)
        assert match.final_score > 0.0
        
        # No matches
        matches = scorer.score_multi_word_query("name", "The Matrix", ["xyz"], 1.0)
        assert len(matches) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])