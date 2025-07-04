"""
Field-specific weighting system for movie search.
Provides configurable weights for different movie fields and match types.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class MatchType(Enum):
    """Types of matches for scoring."""
    EXACT = "exact"
    PARTIAL = "partial"
    FUZZY = "fuzzy"
    STEMMED = "stemmed"
    SYNONYM = "synonym"


@dataclass
class FieldWeight:
    """Weight configuration for a movie field."""
    base_weight: float = 1.0
    exact_match_boost: float = 1.0
    partial_match_boost: float = 1.0
    fuzzy_match_boost: float = 1.0
    stemmed_match_boost: float = 1.0
    synonym_match_boost: float = 1.0


@dataclass
class WeightedMatch:
    """A weighted match result."""
    field_name: str
    field_value: str
    query_term: str
    match_type: MatchType
    base_score: float
    field_weight: float
    match_boost: float
    final_score: float
    position: int = 0


class FieldWeightConfig:
    """Configuration for field weights following the specification."""
    
    def __init__(self):
        """Initialize with default weights per specification."""
        self.weights = {
            # Title fields - highest priority
            "name": FieldWeight(base_weight=3.0, exact_match_boost=5.0, partial_match_boost=2.0),
            "original_title": FieldWeight(base_weight=2.8, exact_match_boost=4.5, partial_match_boost=1.8),
            "sort_name": FieldWeight(base_weight=2.5, exact_match_boost=4.0, partial_match_boost=1.5),
            
            # Genre fields - high priority
            "genres": FieldWeight(base_weight=2.5, exact_match_boost=4.0, partial_match_boost=1.5),
            
            # People fields - medium-high priority
            "people": FieldWeight(base_weight=2.0, exact_match_boost=3.5, partial_match_boost=1.2),
            
            # Content fields - medium priority
            "overview": FieldWeight(base_weight=1.5, exact_match_boost=2.5, partial_match_boost=1.0),
            "taglines": FieldWeight(base_weight=1.8, exact_match_boost=3.0, partial_match_boost=1.1),
            
            # Production fields - lower priority
            "studios": FieldWeight(base_weight=1.2, exact_match_boost=2.0, partial_match_boost=0.8),
            "production_locations": FieldWeight(base_weight=1.0, exact_match_boost=1.5, partial_match_boost=0.6),
            
            # Technical fields - lowest priority
            "official_rating": FieldWeight(base_weight=0.8, exact_match_boost=1.2, partial_match_boost=0.5),
            "container": FieldWeight(base_weight=0.5, exact_match_boost=1.0, partial_match_boost=0.3),
            
            # Enhanced fields (AI-generated) - high priority for search
            "enhanced_fields": FieldWeight(base_weight=2.2, exact_match_boost=3.8, partial_match_boost=1.4),
        }
    
    def get_field_weight(self, field_name: str) -> FieldWeight:
        """Get weight configuration for a field."""
        return self.weights.get(field_name, FieldWeight())
    
    def update_weight(self, field_name: str, weight: FieldWeight):
        """Update weight for a field."""
        self.weights[field_name] = weight
    
    def get_all_weights(self) -> Dict[str, FieldWeight]:
        """Get all field weights."""
        return self.weights.copy()


class FieldWeightedSearch:
    """Field-weighted search system for movies."""
    
    def __init__(self, config: Optional[FieldWeightConfig] = None):
        """Initialize with weight configuration."""
        self.config = config or FieldWeightConfig()
    
    def calculate_field_score(self, field_name: str, field_value: Any, query_term: str, 
                            match_type: MatchType, base_score: float, position: int = 0) -> WeightedMatch:
        """Calculate weighted score for a field match."""
        field_weight = self.config.get_field_weight(field_name)
        
        # Select appropriate boost based on match type
        match_boost = {
            MatchType.EXACT: field_weight.exact_match_boost,
            MatchType.PARTIAL: field_weight.partial_match_boost,
            MatchType.FUZZY: field_weight.fuzzy_match_boost,
            MatchType.STEMMED: field_weight.stemmed_match_boost,
            MatchType.SYNONYM: field_weight.synonym_match_boost,
        }.get(match_type, 1.0)
        
        # Calculate final score
        final_score = base_score * field_weight.base_weight * match_boost
        
        return WeightedMatch(
            field_name=field_name,
            field_value=str(field_value),
            query_term=query_term,
            match_type=match_type,
            base_score=base_score,
            field_weight=field_weight.base_weight,
            match_boost=match_boost,
            final_score=final_score,
            position=position
        )
    
    def search_movie_fields(self, movie_data: Dict[str, Any], query_terms: List[str]) -> List[WeightedMatch]:
        """Search movie fields with weighted scoring."""
        matches = []
        
        for query_term in query_terms:
            query_lower = query_term.lower()
            
            # Search each field
            for field_name, field_value in movie_data.items():
                if field_value is None:
                    continue
                
                field_matches = self._search_field(field_name, field_value, query_term, query_lower)
                matches.extend(field_matches)
        
        return matches
    
    def _search_field(self, field_name: str, field_value: Any, query_term: str, query_lower: str) -> List[WeightedMatch]:
        """Search a specific field for matches."""
        matches = []
        
        if isinstance(field_value, str):
            matches.extend(self._search_string_field(field_name, field_value, query_term, query_lower))
        elif isinstance(field_value, list):
            matches.extend(self._search_list_field(field_name, field_value, query_term, query_lower))
        elif isinstance(field_value, dict):
            matches.extend(self._search_dict_field(field_name, field_value, query_term, query_lower))
        
        return matches
    
    def _search_string_field(self, field_name: str, field_value: str, query_term: str, query_lower: str) -> List[WeightedMatch]:
        """Search string field for matches."""
        matches = []
        field_lower = field_value.lower()
        
        # Exact match
        if query_lower == field_lower:
            matches.append(self.calculate_field_score(
                field_name, field_value, query_term, MatchType.EXACT, 1.0, 0
            ))
        # Partial match
        elif query_lower in field_lower:
            # Higher score for matches at beginning
            position = field_lower.find(query_lower)
            base_score = 0.8 if position == 0 else 0.6
            matches.append(self.calculate_field_score(
                field_name, field_value, query_term, MatchType.PARTIAL, base_score, position
            ))
        
        return matches
    
    def _search_list_field(self, field_name: str, field_value: List[Any], query_term: str, query_lower: str) -> List[WeightedMatch]:
        """Search list field for matches."""
        matches = []
        
        for i, item in enumerate(field_value):
            if isinstance(item, str):
                item_lower = item.lower()
                if query_lower == item_lower:
                    matches.append(self.calculate_field_score(
                        field_name, item, query_term, MatchType.EXACT, 1.0, i
                    ))
                elif query_lower in item_lower:
                    position = item_lower.find(query_lower)
                    base_score = 0.8 if position == 0 else 0.6
                    matches.append(self.calculate_field_score(
                        field_name, item, query_term, MatchType.PARTIAL, base_score, position
                    ))
            elif isinstance(item, dict) and "name" in item:
                # Handle Person, Studio objects
                name_lower = item["name"].lower()
                if query_lower == name_lower:
                    matches.append(self.calculate_field_score(
                        field_name, item["name"], query_term, MatchType.EXACT, 1.0, i
                    ))
                elif query_lower in name_lower:
                    position = name_lower.find(query_lower)
                    base_score = 0.8 if position == 0 else 0.6
                    matches.append(self.calculate_field_score(
                        field_name, item["name"], query_term, MatchType.PARTIAL, base_score, position
                    ))
        
        return matches
    
    def _search_dict_field(self, field_name: str, field_value: Dict[str, Any], query_term: str, query_lower: str) -> List[WeightedMatch]:
        """Search dictionary field for matches."""
        matches = []
        
        # For enhanced_fields, search all values
        if field_name == "enhanced_fields":
            for key, value in field_value.items():
                if isinstance(value, str):
                    value_lower = value.lower()
                    if query_lower in value_lower:
                        position = value_lower.find(query_lower)
                        base_score = 0.8 if position == 0 else 0.6
                        matches.append(self.calculate_field_score(
                            field_name, f"{key}: {value}", query_term, MatchType.PARTIAL, base_score, position
                        ))
        
        return matches
    
    def rank_matches(self, matches: List[WeightedMatch]) -> List[WeightedMatch]:
        """Rank matches by weighted score."""
        return sorted(matches, key=lambda m: m.final_score, reverse=True)
    
    def get_top_matches(self, matches: List[WeightedMatch], limit: int = 10) -> List[WeightedMatch]:
        """Get top N matches."""
        ranked = self.rank_matches(matches)
        return ranked[:limit]
    
    def calculate_total_score(self, matches: List[WeightedMatch]) -> float:
        """Calculate total score for a set of matches."""
        return sum(match.final_score for match in matches)