"""
Linguistic analysis plugins for advanced language deconstruction.
These plugins work symmetrically on both content (during ingestion) and queries (during search).
"""

from .base import LinguisticPlugin, DualUsePlugin
from .conceptnet import ConceptNetExpansionPlugin
from .semantic_roles import SemanticRoleLabelerPlugin

__all__ = [
    "LinguisticPlugin",
    "DualUsePlugin", 
    "ConceptNetExpansionPlugin",
    "SemanticRoleLabelerPlugin"
]