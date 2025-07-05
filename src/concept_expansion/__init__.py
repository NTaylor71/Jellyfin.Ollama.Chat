"""
Concept Expansion Infrastructure for Stage 3: Procedural Concept Expansion.

Provides reusable concept expansion services for plugins across the intelligence pipeline.
Supports multiple backends (ConceptNet, LLM, Gensim) with unified caching and interface.
"""

from src.concept_expansion.providers.conceptnet_client import ConceptNetClient
from src.concept_expansion.concept_expander import ConceptExpander, ExpansionMethod
from src.concept_expansion.providers.base_provider import BaseProvider, ProviderMetadata, ExpansionRequest
from src.concept_expansion.providers.conceptnet_provider import ConceptNetProvider

__all__ = [
    "ConceptExpander",
    "ConceptNetClient", 
    "ExpansionMethod",
    "BaseProvider",
    "ProviderMetadata", 
    "ExpansionRequest",
    "ConceptNetProvider"
]