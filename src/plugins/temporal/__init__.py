"""
Temporal analysis plugins package.

Provides sophisticated temporal understanding for both content ingestion 
and user query processing.

Ingestion Plugins:
- HeidelTimeIngestionPlugin: Rich cultural/historical temporal analysis

Query Plugins: 
- DucklingQueryPlugin: Fast user query temporal understanding
- SUTimeQueryPlugin: Complex temporal reasoning
"""

from .heideltime_ingestion import HeidelTimeIngestionPlugin
from .duckling_query import DucklingQueryPlugin  
from .sutime_query import SUTimeQueryPlugin

__all__ = [
    'HeidelTimeIngestionPlugin',
    'DucklingQueryPlugin', 
    'SUTimeQueryPlugin'
]