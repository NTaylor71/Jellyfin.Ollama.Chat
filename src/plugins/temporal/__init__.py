"""
Temporal analysis plugins package.

Provides sophisticated temporal understanding for both content ingestion 
and user query processing.

Dual-Use Plugins:
- SpacyWithFallbackIngestionAndQueryPlugin: spaCy-based temporal analysis for both ingestion and queries

Ingestion-Specific Plugins:
- HeidelTimeIngestionPlugin: Rich cultural/historical temporal analysis

Query-Specific Plugins: 
- DucklingQueryPlugin: Fast user query temporal understanding
- SUTimeQueryPlugin: Complex temporal reasoning
"""

from .spacy_with_fallback_ingestion_and_query import SpacyWithFallbackIngestionAndQueryPlugin
from .heideltime_ingestion import HeidelTimeIngestionPlugin
from .duckling_query import DucklingQueryPlugin  
from .sutime_query import SUTimeQueryPlugin

# Backward compatibility aliases
SophisticatedTemporalPlugin = SpacyWithFallbackIngestionAndQueryPlugin
TemporalExpressionPlugin = SpacyWithFallbackIngestionAndQueryPlugin

__all__ = [
    'SpacyWithFallbackIngestionAndQueryPlugin',
    'HeidelTimeIngestionPlugin',
    'DucklingQueryPlugin', 
    'SUTimeQueryPlugin',
    # Backward compatibility
    'SophisticatedTemporalPlugin',
    'TemporalExpressionPlugin'
]