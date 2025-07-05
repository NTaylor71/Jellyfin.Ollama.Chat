"""
Concept Expansion Infrastructure for Stage 3: Procedural Concept Expansion.

Provides reusable concept expansion services for plugins across the intelligence pipeline.
Supports multiple backends (ConceptNet, LLM, Gensim) with unified caching and interface.
"""

from .providers.conceptnet_client import ConceptNetClient
from .concept_expander import ConceptExpander, ExpansionMethod
from .providers.base_provider import BaseProvider, ProviderMetadata, ExpansionRequest
from .providers.conceptnet_provider import ConceptNetProvider
from .providers.gensim_provider import GensimProvider
from .providers.spacy_temporal_provider import SpacyTemporalProvider
from .providers.heideltime_provider import HeidelTimeProvider
from .providers.sutime_provider import SUTimeProvider

__all__ = [
    "ConceptExpander",
    "ConceptNetClient", 
    "ExpansionMethod",
    "BaseProvider",
    "ProviderMetadata", 
    "ExpansionRequest",
    "ConceptNetProvider",
    "GensimProvider",
    "SpacyTemporalProvider", 
    "HeidelTimeProvider",
    "SUTimeProvider"
]