"""
Positional scoring system for movie search.
Provides scoring based on position of matches within fields.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class PositionType(Enum):
    """Types of positions for scoring."""
    BEGINNING = "beginning"
    MIDDLE = "middle"
    END = "end"
    EXACT = "exact"


@dataclass
class PositionalMatch:
    """A positional match result."""
    field_name: str
    field_value: str
    query_term: str
    position: int
    position_type: PositionType
    word_position: int
    phrase_order_preserved: bool
    base_score: float
    positional_bonus: float
    final_score: float


class PositionalScoringConfig:
    """Configuration for positional scoring."""
    
    def __init__(self):
        """Initialize with default scoring values per specification."""
        # Position-based bonuses
        self.first_word_bonus = 2.0
        self.title_start_bonus = 3.0
        self.phrase_order_bonus = 1.5
        self.exact_position_bonus = 2.5
        
        # Position decay factors
        self.beginning_threshold = 0.1  # First 10% of field
        self.end_threshold = 0.9        # Last 10% of field
        self.position_decay_factor = 0.8
        
        # High-priority fields for positional scoring
        self.high_priority_fields = {
            "name", "original_title", "sort_name", "taglines"
        }
        
        # Word position bonuses (decreasing)
        self.word_position_bonuses = {
            0: 2.0,  # First word
            1: 1.5,  # Second word
            2: 1.2,  # Third word
            3: 1.0,  # Fourth word and beyond
        }


class PositionalScorer:
    """Positional scoring system for movie search."""
    
    def __init__(self, config: Optional[PositionalScoringConfig] = None):
        """Initialize with scoring configuration."""
        self.config = config or PositionalScoringConfig()
    
    def score_position(self, field_name: str, field_value: str, query_term: str, 
                      position: int, base_score: float = 1.0) -> PositionalMatch:
        """Score a match based on its position in the field."""
        field_length = len(field_value)
        
        # Determine position type
        position_type = self._get_position_type(position, field_length)
        
        # Calculate word position
        word_position = self._get_word_position(field_value, position)
        
        # Calculate positional bonus
        positional_bonus = self._calculate_positional_bonus(
            field_name, position, field_length, position_type, word_position
        )
        
        # Check phrase order preservation (for multi-word queries)
        phrase_order_preserved = self._check_phrase_order(field_value, [query_term], position)
        
        # Apply phrase order bonus
        if phrase_order_preserved:
            positional_bonus *= self.config.phrase_order_bonus
        
        final_score = base_score * positional_bonus
        
        return PositionalMatch(
            field_name=field_name,
            field_value=field_value,
            query_term=query_term,
            position=position,
            position_type=position_type,
            word_position=word_position,
            phrase_order_preserved=phrase_order_preserved,
            base_score=base_score,
            positional_bonus=positional_bonus,
            final_score=final_score
        )
    
    def score_multi_word_query(self, field_name: str, field_value: str, 
                             query_terms: List[str], base_score: float = 1.0) -> List[PositionalMatch]:
        """Score multiple query terms with positional information."""
        matches = []
        field_lower = field_value.lower()
        
        for i, query_term in enumerate(query_terms):
            query_lower = query_term.lower()
            
            # Find all positions of this term
            positions = self._find_all_positions(field_lower, query_lower)
            
            for position in positions:
                # Check if this preserves phrase order
                phrase_order_preserved = self._check_multi_word_phrase_order(
                    field_lower, query_terms, i, position
                )
                
                match = self.score_position(field_name, field_value, query_term, position, base_score)
                match.phrase_order_preserved = phrase_order_preserved
                
                # Apply additional phrase order bonus for multi-word queries
                if phrase_order_preserved and len(query_terms) > 1:
                    match.positional_bonus *= self.config.phrase_order_bonus
                    match.final_score = match.base_score * match.positional_bonus
                
                matches.append(match)
        
        return matches
    
    def _get_position_type(self, position: int, field_length: int) -> PositionType:
        """Determine the type of position within the field."""
        if position == 0:
            return PositionType.EXACT
        
        relative_position = position / field_length
        
        if relative_position <= self.config.beginning_threshold:
            return PositionType.BEGINNING
        elif relative_position >= self.config.end_threshold:
            return PositionType.END
        else:
            return PositionType.MIDDLE
    
    def _get_word_position(self, field_value: str, character_position: int) -> int:
        """Get the word position of a character position."""
        # Count words before the character position
        words_before = field_value[:character_position].split()
        return len(words_before)
    
    def _calculate_positional_bonus(self, field_name: str, position: int, field_length: int, 
                                  position_type: PositionType, word_position: int) -> float:
        """Calculate the positional bonus for a match."""
        bonus = 1.0
        
        # Base position bonus
        if position_type == PositionType.EXACT:
            bonus *= self.config.exact_position_bonus
        elif position_type == PositionType.BEGINNING:
            bonus *= self.config.first_word_bonus
        elif position_type == PositionType.MIDDLE:
            # Apply gradual decay from beginning
            relative_position = position / field_length
            decay = 1.0 - (relative_position * self.config.position_decay_factor)
            bonus *= max(decay, 0.2)  # Minimum bonus of 0.2
        
        # Word position bonus
        word_bonus = self.config.word_position_bonuses.get(word_position, 1.0)
        bonus *= word_bonus
        
        # Title field bonus
        if field_name in self.config.high_priority_fields and position_type == PositionType.BEGINNING:
            bonus *= self.config.title_start_bonus
        
        return bonus
    
    def _find_all_positions(self, text: str, query: str) -> List[int]:
        """Find all positions of a query term in text."""
        positions = []
        start = 0
        
        while True:
            pos = text.find(query, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        return positions
    
    def _check_phrase_order(self, field_value: str, query_terms: List[str], position: int) -> bool:
        """Check if phrase order is preserved."""
        if len(query_terms) <= 1:
            return True
        
        # For single term, always true
        return True
    
    def _check_multi_word_phrase_order(self, field_lower: str, query_terms: List[str], 
                                     current_term_index: int, current_position: int) -> bool:
        """Check if multi-word phrase order is preserved."""
        if len(query_terms) <= 1:
            return True
        
        # Check if previous terms appear before current position
        for i in range(current_term_index):
            prev_term = query_terms[i].lower()
            prev_positions = self._find_all_positions(field_lower, prev_term)
            
            # Find the closest previous position
            valid_prev_positions = [pos for pos in prev_positions if pos < current_position]
            if not valid_prev_positions:
                return False
        
        # Check if next terms appear after current position
        current_term_end = current_position + len(query_terms[current_term_index])
        for i in range(current_term_index + 1, len(query_terms)):
            next_term = query_terms[i].lower()
            next_positions = self._find_all_positions(field_lower, next_term)
            
            # Find the closest next position
            valid_next_positions = [pos for pos in next_positions if pos > current_term_end]
            if not valid_next_positions:
                return False
        
        return True
    
    def rank_positional_matches(self, matches: List[PositionalMatch]) -> List[PositionalMatch]:
        """Rank matches by positional score."""
        return sorted(matches, key=lambda m: m.final_score, reverse=True)
    
    def get_best_positional_match(self, matches: List[PositionalMatch]) -> Optional[PositionalMatch]:
        """Get the best positional match."""
        if not matches:
            return None
        
        ranked = self.rank_positional_matches(matches)
        return ranked[0]
    
    def calculate_field_positional_score(self, field_name: str, field_value: str, 
                                       query_terms: List[str]) -> float:
        """Calculate total positional score for a field."""
        if not query_terms:
            return 0.0
        
        matches = self.score_multi_word_query(field_name, field_value, query_terms)
        
        if not matches:
            return 0.0
        
        # Use the best match score for the field
        best_match = self.get_best_positional_match(matches)
        return best_match.final_score if best_match else 0.0
    
    def get_position_stats(self, matches: List[PositionalMatch]) -> Dict[str, Any]:
        """Get statistics about positional matches."""
        if not matches:
            return {}
        
        position_types = [m.position_type.value for m in matches]
        word_positions = [m.word_position for m in matches]
        phrase_orders = [m.phrase_order_preserved for m in matches]
        
        return {
            "total_matches": len(matches),
            "position_types": {
                "beginning": position_types.count("beginning"),
                "middle": position_types.count("middle"),
                "end": position_types.count("end"),
                "exact": position_types.count("exact"),
            },
            "avg_word_position": sum(word_positions) / len(word_positions),
            "phrase_order_preserved": sum(phrase_orders) / len(phrase_orders),
            "best_score": max(m.final_score for m in matches),
            "avg_score": sum(m.final_score for m in matches) / len(matches),
        }