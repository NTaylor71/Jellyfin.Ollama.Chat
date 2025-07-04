"""
Search system for movie discovery and retrieval.
"""

from .text_processor import TextProcessor, ProcessedText
from .similarity_engine import SimilarityEngine, SimilarityResult, ModelStats
from .fuzzy_matcher import FuzzyMatcher, FuzzyMatch

__all__ = [
    "TextProcessor", "ProcessedText",
    "SimilarityEngine", "SimilarityResult", "ModelStats", 
    "FuzzyMatcher", "FuzzyMatch"
]