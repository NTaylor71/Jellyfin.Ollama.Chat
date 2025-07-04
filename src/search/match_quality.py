"""
Match quality scoring system for movie search.
Provides scoring based on the quality and type of matches.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class MatchQuality(Enum):
    """Quality levels for matches."""
    EXACT = "exact"
    STEMMED = "stemmed"
    FUZZY = "fuzzy"
    SYNONYM = "synonym"
    PARTIAL = "partial"
    PHONETIC = "phonetic"


@dataclass
class QualityScore:
    """Quality score configuration."""
    base_score: float
    confidence: float
    description: str


@dataclass
class QualityMatch:
    """A quality-scored match result."""
    field_name: str
    field_value: str
    query_term: str
    matched_text: str
    quality: MatchQuality
    quality_score: float
    confidence: float
    edit_distance: int
    similarity_ratio: float
    final_score: float


class MatchQualityConfig:
    """Configuration for match quality scoring."""
    
    def __init__(self):
        """Initialize with default quality scores per specification."""
        self.quality_scores = {
            MatchQuality.EXACT: QualityScore(
                base_score=1.0,
                confidence=1.0,
                description="Perfect exact match"
            ),
            MatchQuality.STEMMED: QualityScore(
                base_score=0.8,
                confidence=0.9,
                description="Stemmed form match (run/running)"
            ),
            MatchQuality.SYNONYM: QualityScore(
                base_score=0.7,
                confidence=0.8,
                description="Synonym match (movie/film)"
            ),
            MatchQuality.FUZZY: QualityScore(
                base_score=0.6,
                confidence=0.7,
                description="Fuzzy match with edit distance"
            ),
            MatchQuality.PARTIAL: QualityScore(
                base_score=0.4,
                confidence=0.6,
                description="Partial substring match"
            ),
            MatchQuality.PHONETIC: QualityScore(
                base_score=0.5,
                confidence=0.65,
                description="Phonetically similar match"
            ),
        }
        
        # Edit distance thresholds
        self.max_edit_distance = 3
        self.edit_distance_penalty = 0.1
        
        # Similarity ratio thresholds
        self.min_similarity_ratio = 0.6
        self.similarity_bonus = 0.2
        
        # Length-based adjustments
        self.min_length_for_fuzzy = 4
        self.length_adjustment_factor = 0.05


class MatchQualityScorer:
    """Match quality scoring system."""
    
    def __init__(self, config: Optional[MatchQualityConfig] = None):
        """Initialize with quality configuration."""
        self.config = config or MatchQualityConfig()
    
    def score_match_quality(self, field_name: str, field_value: str, query_term: str, 
                          matched_text: str, quality: MatchQuality, 
                          edit_distance: int = 0, similarity_ratio: float = 1.0) -> QualityMatch:
        """Score a match based on its quality."""
        quality_config = self.config.quality_scores.get(quality, 
                                                        self.config.quality_scores[MatchQuality.PARTIAL])
        
        # Base quality score
        quality_score = quality_config.base_score
        confidence = quality_config.confidence
        
        # Adjust for edit distance
        if edit_distance > 0:
            edit_penalty = min(edit_distance * self.config.edit_distance_penalty, 0.5)
            quality_score *= (1.0 - edit_penalty)
            confidence *= (1.0 - edit_penalty)
        
        # Adjust for similarity ratio
        if similarity_ratio < 1.0:
            similarity_adjustment = (similarity_ratio - 0.5) * self.config.similarity_bonus
            quality_score *= (1.0 + similarity_adjustment)
        
        # Length-based adjustments
        quality_score = self._apply_length_adjustments(quality_score, query_term, matched_text)
        
        # Final score
        final_score = quality_score * confidence
        
        return QualityMatch(
            field_name=field_name,
            field_value=field_value,
            query_term=query_term,
            matched_text=matched_text,
            quality=quality,
            quality_score=quality_score,
            confidence=confidence,
            edit_distance=edit_distance,
            similarity_ratio=similarity_ratio,
            final_score=final_score
        )
    
    def determine_match_quality(self, query_term: str, matched_text: str) -> Tuple[MatchQuality, int, float]:
        """Determine the quality of a match."""
        query_lower = query_term.lower()
        matched_lower = matched_text.lower()
        
        # Exact match
        if query_lower == matched_lower:
            return MatchQuality.EXACT, 0, 1.0
        
        # Calculate edit distance and similarity
        edit_distance = self._calculate_edit_distance(query_lower, matched_lower)
        similarity_ratio = self._calculate_similarity_ratio(query_lower, matched_lower)
        
        # Determine quality based on metrics
        if edit_distance == 0:
            return MatchQuality.EXACT, 0, 1.0
        elif edit_distance <= 1 and len(query_term) > 3:
            return MatchQuality.FUZZY, edit_distance, similarity_ratio
        elif similarity_ratio >= 0.8:
            return MatchQuality.STEMMED, edit_distance, similarity_ratio
        elif similarity_ratio >= 0.6:
            return MatchQuality.FUZZY, edit_distance, similarity_ratio
        elif query_lower in matched_lower or matched_lower in query_lower:
            return MatchQuality.PARTIAL, edit_distance, similarity_ratio
        else:
            return MatchQuality.PARTIAL, edit_distance, similarity_ratio
    
    def _calculate_edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._calculate_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _calculate_similarity_ratio(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings."""
        if not s1 or not s2:
            return 0.0
        
        edit_distance = self._calculate_edit_distance(s1, s2)
        max_length = max(len(s1), len(s2))
        
        return 1.0 - (edit_distance / max_length)
    
    def _apply_length_adjustments(self, quality_score: float, query_term: str, matched_text: str) -> float:
        """Apply length-based adjustments to quality score."""
        query_length = len(query_term)
        matched_length = len(matched_text)
        
        # Bonus for longer matches
        if matched_length >= query_length:
            length_bonus = min((matched_length - query_length) * self.config.length_adjustment_factor, 0.2)
            quality_score *= (1.0 + length_bonus)
        
        # Penalty for very short matches
        if query_length < self.config.min_length_for_fuzzy:
            quality_score *= 0.8
        
        return quality_score
    
    def score_field_matches(self, field_name: str, field_value: Any, query_terms: List[str]) -> List[QualityMatch]:
        """Score all matches in a field."""
        matches = []
        
        if isinstance(field_value, str):
            matches.extend(self._score_string_matches(field_name, field_value, query_terms))
        elif isinstance(field_value, list):
            matches.extend(self._score_list_matches(field_name, field_value, query_terms))
        elif isinstance(field_value, dict):
            matches.extend(self._score_dict_matches(field_name, field_value, query_terms))
        
        return matches
    
    def _score_string_matches(self, field_name: str, field_value: str, query_terms: List[str]) -> List[QualityMatch]:
        """Score matches in a string field."""
        matches = []
        field_lower = field_value.lower()
        
        for query_term in query_terms:
            query_lower = query_term.lower()
            
            # Find exact matches
            if query_lower in field_lower:
                start_pos = field_lower.find(query_lower)
                matched_text = field_value[start_pos:start_pos + len(query_term)]
                
                quality, edit_distance, similarity_ratio = self.determine_match_quality(query_term, matched_text)
                
                match = self.score_match_quality(
                    field_name, field_value, query_term, matched_text,
                    quality, edit_distance, similarity_ratio
                )
                matches.append(match)
            
            # Find fuzzy matches if no exact match
            elif len(query_term) >= self.config.min_length_for_fuzzy:
                fuzzy_match = self._find_best_fuzzy_match(field_value, query_term)
                if fuzzy_match:
                    matches.append(fuzzy_match)
        
        return matches
    
    def _score_list_matches(self, field_name: str, field_value: List[Any], query_terms: List[str]) -> List[QualityMatch]:
        """Score matches in a list field."""
        matches = []
        
        for item in field_value:
            if isinstance(item, str):
                item_matches = self._score_string_matches(field_name, item, query_terms)
                matches.extend(item_matches)
            elif isinstance(item, dict) and "name" in item:
                item_matches = self._score_string_matches(field_name, item["name"], query_terms)
                matches.extend(item_matches)
        
        return matches
    
    def _score_dict_matches(self, field_name: str, field_value: Dict[str, Any], query_terms: List[str]) -> List[QualityMatch]:
        """Score matches in a dictionary field."""
        matches = []
        
        for key, value in field_value.items():
            if isinstance(value, str):
                value_matches = self._score_string_matches(f"{field_name}.{key}", value, query_terms)
                matches.extend(value_matches)
        
        return matches
    
    def _find_best_fuzzy_match(self, field_value: str, query_term: str) -> Optional[QualityMatch]:
        """Find the best fuzzy match in a field."""
        words = field_value.split()
        best_match = None
        best_score = 0.0
        
        for word in words:
            if len(word) >= self.config.min_length_for_fuzzy:
                similarity_ratio = self._calculate_similarity_ratio(query_term.lower(), word.lower())
                
                if similarity_ratio >= self.config.min_similarity_ratio:
                    edit_distance = self._calculate_edit_distance(query_term.lower(), word.lower())
                    
                    if edit_distance <= self.config.max_edit_distance:
                        match = self.score_match_quality(
                            "fuzzy_field", field_value, query_term, word,
                            MatchQuality.FUZZY, edit_distance, similarity_ratio
                        )
                        
                        if match.final_score > best_score:
                            best_match = match
                            best_score = match.final_score
        
        return best_match
    
    def rank_quality_matches(self, matches: List[QualityMatch]) -> List[QualityMatch]:
        """Rank matches by quality score."""
        return sorted(matches, key=lambda m: (m.final_score, m.confidence), reverse=True)
    
    def get_quality_stats(self, matches: List[QualityMatch]) -> Dict[str, Any]:
        """Get statistics about match quality."""
        if not matches:
            return {}
        
        qualities = [m.quality.value for m in matches]
        scores = [m.final_score for m in matches]
        confidences = [m.confidence for m in matches]
        
        return {
            "total_matches": len(matches),
            "quality_distribution": {
                "exact": qualities.count("exact"),
                "stemmed": qualities.count("stemmed"),
                "fuzzy": qualities.count("fuzzy"),
                "synonym": qualities.count("synonym"),
                "partial": qualities.count("partial"),
                "phonetic": qualities.count("phonetic"),
            },
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "avg_confidence": sum(confidences) / len(confidences),
            "high_quality_matches": sum(1 for m in matches if m.quality in [MatchQuality.EXACT, MatchQuality.STEMMED]),
        }
    
    def filter_by_quality(self, matches: List[QualityMatch], min_quality: MatchQuality = MatchQuality.PARTIAL) -> List[QualityMatch]:
        """Filter matches by minimum quality."""
        quality_order = [
            MatchQuality.EXACT,
            MatchQuality.STEMMED,
            MatchQuality.SYNONYM,
            MatchQuality.PHONETIC,
            MatchQuality.FUZZY,
            MatchQuality.PARTIAL,
        ]
        
        min_quality_index = quality_order.index(min_quality)
        
        return [match for match in matches 
                if quality_order.index(match.quality) <= min_quality_index]