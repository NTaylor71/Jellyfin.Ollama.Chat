"""
ConceptExpander: Reusable concept expansion infrastructure for Stage 3+.

Provides unified interface for concept expansion with multiple backends:
- ConceptNet (Stage 3.1) 
- LLM (Stage 3.2)
- Multi-source fusion (Stage 3.3)

Used by plugins throughout the intelligence pipeline for consistent concept expansion.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime

from src.concept_expansion.providers.base_provider import BaseProvider, ExpansionRequest
from src.concept_expansion.providers.conceptnet_provider import ConceptNetProvider
from src.concept_expansion.providers.llm.llm_provider import LLMProvider
from src.data.cache_manager import get_cache_manager, CacheStrategy
from src.shared.plugin_contracts import (
    PluginResult, CacheKey, CacheType, ConfidenceScore, PluginMetadata, PluginType,
    create_field_expansion_result
)

logger = logging.getLogger(__name__)


class ExpansionMethod(Enum):
    """Available concept expansion methods with their capabilities."""
    CONCEPTNET = "conceptnet"      # Literal/linguistic relationships, context-unaware
    LLM = "llm"                    # Semantic understanding, context-aware
    GENSIM = "gensim"              # Statistical similarity, corpus-based
    DUCKLING = "duckling"          # Temporal parsing, time-aware
    HEIDELTIME = "heideltime"      # Temporal extraction, context-aware
    SUTIME = "sutime"              # Temporal understanding, rule-based
    MULTI_SOURCE = "multi_source"  # Combined semantic + literal + temporal
    AUTO = "auto"                  # Choose best method automatically


class ConceptExpander:
    """
    Reusable concept expansion service for the intelligence pipeline.
    
    Provides unified interface for expanding concepts using different backends
    with consistent caching and error handling. Used by:
    
    - EmbedDataEmbellisherPlugin instances during ingestion
    - Query processing plugins for search term expansion
    - Content analysis plugins for pattern discovery
    - Future plugins requiring concept understanding
    
    Example Usage:
        expander = ConceptExpander()
        concepts = await expander.expand_concept("action", "movie")
        # Returns: ["fight", "combat", "battle", "intense", "fast-paced"]
    """
    
    def __init__(self, cache_strategy: CacheStrategy = CacheStrategy.CACHE_FIRST):
        """
        Initialize ConceptExpander.
        
        Args:
            cache_strategy: How to handle caching (default: cache-first)
        """
        self.cache_strategy = cache_strategy
        self.cache_manager = get_cache_manager()
        
        # Initialize providers
        self.conceptnet_provider = ConceptNetProvider()
        self.llm_provider = LLMProvider()
        self.providers = {
            ExpansionMethod.CONCEPTNET: self.conceptnet_provider,
            ExpansionMethod.LLM: self.llm_provider
        }
        
        # Method capabilities for intelligent selection
        self.method_capabilities = {
            ExpansionMethod.CONCEPTNET: {
                "type": "literal",
                "context_aware": False,
                "strengths": ["linguistic relationships", "cross-language", "factual connections"],
                "weaknesses": ["context-blind", "generic relationships", "poor compound terms"],
                "best_for": ["single words", "factual lookup", "linguistic similarity"]
            },
            ExpansionMethod.LLM: {
                "type": "semantic", 
                "context_aware": True,
                "strengths": ["context understanding", "domain knowledge", "compound concepts"],
                "weaknesses": ["requires API calls", "potential hallucination", "slower"],
                "best_for": ["movie genres", "contextual concepts", "compound terms"]
            },
            ExpansionMethod.GENSIM: {
                "type": "statistical",
                "context_aware": False, 
                "strengths": ["corpus similarity", "statistical patterns", "fast lookup"],
                "weaknesses": ["limited to training data", "no semantic understanding"],
                "best_for": ["text similarity", "topic modeling", "document similarity"]
            },
            ExpansionMethod.DUCKLING: {
                "type": "temporal",
                "context_aware": True,
                "strengths": ["natural language time parsing", "multi-language support", "fuzzy time expressions"],
                "weaknesses": ["limited to temporal concepts", "requires API calls"],
                "best_for": ["release dates", "time expressions", "natural language dates"]
            },
            ExpansionMethod.HEIDELTIME: {
                "type": "temporal", 
                "context_aware": True,
                "strengths": ["document-aware temporal extraction", "context-sensitive parsing", "multilingual"],
                "weaknesses": ["complex setup", "document context required", "slower processing"],
                "best_for": ["document temporal analysis", "historical dates", "context-aware time extraction"]
            },
            ExpansionMethod.SUTIME: {
                "type": "temporal",
                "context_aware": False,
                "strengths": ["rule-based reliability", "Stanford NLP integration", "precise parsing"],
                "weaknesses": ["rigid rules", "limited flexibility", "English-focused"],
                "best_for": ["structured time expressions", "formal dates", "rule-based parsing"]
            }
        }
        
    async def expand_concept(
        self,
        concept: str,
        media_context: str = "movie",
        method: ExpansionMethod = ExpansionMethod.CONCEPTNET,
        field_name: str = "concept",
        max_concepts: int = 10
    ) -> Optional[PluginResult]:
        """
        Expand a concept using specified method with cache-first behavior.
        
        Args:
            concept: Term to expand (e.g., "action", "dark comedy")
            media_context: Media type context ("movie", "book", etc.)
            method: Expansion method to use
            field_name: Source field name for cache key
            max_concepts: Maximum concepts to return
            
        Returns:
            PluginResult with expanded concepts, or None if failed
            
        Example:
            result = await expander.expand_concept("action", "movie")
            if result and result.success:
                expanded = result.enhanced_data["expanded_concepts"]
                # ["fight", "combat", "battle", "intense", "fast-paced"]
        """
        start_time = datetime.now()
        
        try:
            # Generate cache key based on expansion method
            cache_type = self._method_to_cache_type(method)
            cache_key = self.cache_manager.generate_cache_key(
                cache_type=cache_type,
                field_name=field_name,
                input_value=concept,
                media_context=media_context
            )
            
            # Use cache manager's get_or_compute pattern
            result = await self.cache_manager.get_or_compute(
                cache_key=cache_key,
                compute_func=lambda: self._compute_expansion(
                    concept, media_context, method, field_name, max_concepts, start_time
                ),
                strategy=self.cache_strategy
            )
            
            if result:
                logger.debug(f"Concept expansion for '{concept}': {len(result.enhanced_data.get('expanded_concepts', []))} concepts")
            else:
                logger.warning(f"Concept expansion failed for '{concept}'")
            
            return result
            
        except Exception as e:
            logger.error(f"ConceptExpander.expand_concept failed: {e}")
            return None
    
    async def _compute_expansion(
        self,
        concept: str,
        media_context: str,
        method: ExpansionMethod,
        field_name: str,
        max_concepts: int,
        start_time: datetime
    ) -> Optional[PluginResult]:
        """
        Compute concept expansion using specified method.
        
        This is called by cache manager when cache miss occurs.
        """
        if method == ExpansionMethod.CONCEPTNET:
            return await self._expand_with_conceptnet(
                concept, media_context, field_name, max_concepts, start_time
            )
        elif method == ExpansionMethod.LLM:
            return await self._expand_with_llm(
                concept, media_context, field_name, max_concepts, start_time
            )
        elif method == ExpansionMethod.GENSIM:
            # Placeholder for Stage 3.3 - Gensim similarity expansion  
            logger.info(f"Gensim expansion not implemented yet (Stage 3.3)")
            return None
        elif method == ExpansionMethod.MULTI_SOURCE:
            # Placeholder for Stage 3.3 - Multi-source fusion
            logger.info(f"Multi-source expansion not implemented yet (Stage 3.3)")
            return None
        elif method == ExpansionMethod.AUTO:
            # For now, default to ConceptNet. In Stage 3.3, choose best method
            return await self._expand_with_conceptnet(
                concept, media_context, field_name, max_concepts, start_time
            )
        else:
            logger.error(f"Unknown expansion method: {method}")
            return None
    
    async def _expand_with_conceptnet(
        self,
        concept: str,
        media_context: str,
        field_name: str,
        max_concepts: int,
        start_time: datetime
    ) -> Optional[PluginResult]:
        """
        Expand concept using ConceptNet provider.
        
        Returns PluginResult in the standard format for caching.
        """
        try:
            # Ensure provider is initialized
            if not await self.conceptnet_provider._ensure_initialized():
                logger.error("ConceptNet provider not available")
                return None
            
            # Create expansion request
            request = ExpansionRequest(
                concept=concept,
                media_context=media_context,
                max_concepts=max_concepts,
                field_name=field_name
            )
            
            # Call ConceptNet provider
            result = await self.conceptnet_provider.expand_concept(request)
            
            if result is None:
                logger.warning(f"ConceptNet provider failed for concept: {concept}")
                return None
            
            return result
            
        except Exception as e:
            logger.error(f"ConceptNet expansion failed: {e}")
            return None
    
    async def _expand_with_llm(
        self,
        concept: str,
        media_context: str,
        field_name: str,
        max_concepts: int,
        start_time: datetime
    ) -> Optional[PluginResult]:
        """
        Expand concept using LLM provider.
        
        Returns PluginResult in the standard format for caching.
        """
        try:
            # Ensure provider is initialized
            if not await self.llm_provider._ensure_initialized():
                logger.error("LLM provider not available")
                return None
            
            # Create expansion request
            request = ExpansionRequest(
                concept=concept,
                media_context=media_context,
                max_concepts=max_concepts,
                field_name=field_name
            )
            
            # Call LLM provider
            result = await self.llm_provider.expand_concept(request)
            
            if result is None:
                logger.warning(f"LLM provider failed for concept: {concept}")
                return None
            
            return result
            
        except Exception as e:
            logger.error(f"LLM expansion failed: {e}")
            return None
    
    def _method_to_cache_type(self, method: ExpansionMethod) -> CacheType:
        """Convert expansion method to cache type."""
        mapping = {
            ExpansionMethod.CONCEPTNET: CacheType.CONCEPTNET,
            ExpansionMethod.LLM: CacheType.LLM_CONCEPT,
            ExpansionMethod.GENSIM: CacheType.GENSIM_SIMILARITY,
            ExpansionMethod.MULTI_SOURCE: CacheType.CUSTOM,
            ExpansionMethod.AUTO: CacheType.CONCEPTNET  # Default for now
        }
        return mapping.get(method, CacheType.CONCEPTNET)
    
    async def batch_expand_concepts(
        self,
        concepts: List[str],
        media_context: str = "movie",
        method: ExpansionMethod = ExpansionMethod.CONCEPTNET,
        field_name: str = "concept"
    ) -> Dict[str, Optional[PluginResult]]:
        """
        Expand multiple concepts efficiently.
        
        Useful for plugins that need to expand many terms at once.
        
        Args:
            concepts: List of concepts to expand
            media_context: Media type context
            method: Expansion method
            field_name: Source field name
            
        Returns:
            Dictionary mapping concept -> PluginResult (or None if failed)
        """
        results = {}
        
        # Process concepts sequentially to respect rate limits
        # In future, could batch cache checks then parallelize API calls
        for concept in concepts:
            result = await self.expand_concept(
                concept=concept,
                media_context=media_context,
                method=method,
                field_name=field_name
            )
            results[concept] = result
        
        logger.info(f"Batch expanded {len(concepts)} concepts, {sum(1 for r in results.values() if r)} successful")
        return results
    
    def get_method_capabilities(self, method: ExpansionMethod) -> Dict[str, Any]:
        """
        Get capability information for an expansion method.
        
        Args:
            method: Expansion method to query
            
        Returns:
            Dictionary with method capabilities, strengths, weaknesses
        """
        return self.method_capabilities.get(method, {
            "type": "unknown",
            "context_aware": False,
            "strengths": [],
            "weaknesses": ["not implemented"],
            "best_for": []
        })
    
    def get_recommended_method(self, concept: str, media_context: str) -> ExpansionMethod:
        """
        Recommend best expansion method based on concept and context.
        
        Args:
            concept: Concept to expand
            media_context: Media type context
            
        Returns:
            Recommended ExpansionMethod
        """
        # Simple heuristics for method selection
        if " " in concept or len(concept.split()) > 1:
            # Compound terms - LLM is better for context
            return ExpansionMethod.LLM
        elif media_context in ["movie", "book", "music"] and concept in ["action", "comedy", "drama", "horror", "thriller"]:
            # Genre terms need context - LLM is better
            return ExpansionMethod.LLM
        else:
            # Single words, factual lookup - ConceptNet is fine
            return ExpansionMethod.CONCEPTNET
    
    async def close(self) -> None:
        """Clean up resources."""
        # Close all providers
        for provider in self.providers.values():
            await provider.close()


# Global expander instance for reuse
_concept_expander: Optional[ConceptExpander] = None


def get_concept_expander() -> ConceptExpander:
    """Get singleton ConceptExpander instance."""
    global _concept_expander
    if _concept_expander is None:
        _concept_expander = ConceptExpander()
    return _concept_expander