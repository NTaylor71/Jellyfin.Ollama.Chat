"""
Search system for movie discovery and retrieval.
"""

from .text_processor import TextProcessor, ProcessedText
from .similarity_engine import SimilarityEngine, SimilarityResult, ModelStats
from .fuzzy_matcher import FuzzyMatcher, FuzzyMatch
from .field_weights import FieldWeightedSearch, FieldWeightConfig, FieldWeight, MatchType, WeightedMatch
from .positional_scorer import PositionalScorer, PositionalScoringConfig, PositionalMatch, PositionType
from .match_quality import MatchQualityScorer, MatchQualityConfig, MatchQuality, QualityMatch, QualityScore

__all__ = [
    "TextProcessor", "ProcessedText",
    "SimilarityEngine", "SimilarityResult", "ModelStats", 
    "FuzzyMatcher", "FuzzyMatch",
    "FieldWeightedSearch", "FieldWeightConfig", "FieldWeight", "MatchType", "WeightedMatch",
    "PositionalScorer", "PositionalScoringConfig", "PositionalMatch", "PositionType",
    "MatchQualityScorer", "MatchQualityConfig", "MatchQuality", "QualityMatch", "QualityScore"
]