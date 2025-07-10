"""
ConceptNet provider for literal/linguistic concept expansion.
Implements the BaseProvider interface using ConceptNet API.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.providers.nlp.base_provider import (
    BaseProvider, ProviderMetadata, ExpansionRequest, ProviderError, ProviderNotAvailableError
)
from src.providers.knowledge.conceptnet_client import get_conceptnet_client, ConceptNetResponse
from src.shared.plugin_contracts import (
    PluginResult, CacheKey, CacheType, ConfidenceScore, PluginMetadata, PluginType,
    create_field_expansion_result
)

logger = logging.getLogger(__name__)


class ConceptNetProvider(BaseProvider):
    """
    ConceptNet provider for literal/linguistic concept expansion.
    
    Provides factual relationships, cross-language connections, and linguistic similarities
    without semantic understanding or context awareness.
    """
    
    def __init__(self):
        super().__init__()
        self.client = None
        self._client_initialized = False
    
    @property
    def metadata(self) -> ProviderMetadata:
        """Get ConceptNet provider metadata."""
        return ProviderMetadata(
            name="ConceptNet",
            provider_type="literal",
            context_aware=False,
            strengths=[
                "linguistic relationships",
                "cross-language connections", 
                "factual associations",
                "fast lookup",
                "large knowledge base"
            ],
            weaknesses=[
                "context-blind",
                "generic relationships",
                "poor compound terms",
                "no semantic understanding",
                "literal associations only"
            ],
            best_for=[
                "single words",
                "factual lookup",
                "linguistic similarity",
                "cross-language synonyms",
                "basic concept relationships"
            ]
        )
    
    async def initialize(self) -> bool:
        """Initialize the ConceptNet provider."""
        try:
            if not self._client_initialized:
                self.client = get_conceptnet_client()
                self._client_initialized = True
                logger.info("ConceptNet provider initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ConceptNet provider: {e}")
            return False
    
    async def expand_concept(self, request: ExpansionRequest) -> Optional[PluginResult]:
        """
        Expand concept using ConceptNet API.
        
        Args:
            request: Expansion request
            
        Returns:
            PluginResult with ConceptNet expansions
        """
        start_time = datetime.now()
        
        try:

            if not await self._ensure_initialized():
                raise ProviderNotAvailableError("ConceptNet provider not available", "ConceptNet")
            

            response: ConceptNetResponse = await self.client.expand_concept(
                concept=request.concept,
                media_context=request.media_context,
                limit=request.max_concepts * 2  
            )
            
            if not response.success:
                logger.warning(f"ConceptNet API failed: {response.error_message}")
                return None
            

            sorted_concepts = sorted(
                response.concepts,
                key=lambda c: response.confidence_scores.get(c, 0.0),
                reverse=True
            )[:request.max_concepts]
            
            
            filtered_scores = {
                concept: response.confidence_scores.get(concept, 0.0)
                for concept in sorted_concepts
            }
            

            total_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            
            return create_field_expansion_result(
                field_name=request.field_name,
                input_value=request.concept,
                expansion_result={
                    "expanded_concepts": sorted_concepts,
                    "original_concept": request.concept,
                    "expansion_method": "conceptnet",
                    "api_call_time_ms": response.api_call_time_ms,
                    "provider_type": "literal"
                },
                confidence_scores=filtered_scores,
                plugin_name="ConceptNetProvider",
                plugin_version="1.0.0",
                cache_type=CacheType.CONCEPTNET,
                execution_time_ms=total_time_ms,
                media_context=request.media_context,
                plugin_type=PluginType.CONCEPT_EXPANSION,
                api_endpoint=self.client.BASE_URL,
                model_used="ConceptNet5"
            )
            
        except Exception as e:
            logger.error(f"ConceptNet expansion failed: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check ConceptNet provider health."""
        try:
            if not self._client_initialized:
                return {
                    "status": "unhealthy",
                    "provider": "ConceptNet", 
                    "error": "Not initialized"
                }
            

            test_response = await self.client.expand_concept("test", "test", limit=1)
            
            return {
                "status": "healthy" if test_response.success else "degraded",
                "provider": "ConceptNet",
                "api_accessible": test_response.success,
                "api_endpoint": self.client.BASE_URL,
                "response_time_ms": test_response.api_call_time_ms,
                "error": test_response.error_message if not test_response.success else None
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "ConceptNet",
                "error": str(e)
            }
    
    def supports_concept(self, concept: str, media_context: str) -> bool:
        """
        Check if ConceptNet can handle this concept.
        
        ConceptNet works best with:
        - Single words
        - Simple phrases
        - Common concepts
        
        Less effective with:
        - Complex compound terms
        - Domain-specific jargon
        - Context-dependent meanings
        """

        if len(concept.split()) > 3:

            return False
        

        return True
    
    def get_recommended_parameters(self, concept: str, media_context: str) -> Dict[str, Any]:
        """Get recommended parameters for ConceptNet expansion."""
        params = {
            "max_concepts": 10
        }
        

        if " " in concept:

            params["max_concepts"] = 15
        

        if len(concept.split()) == 1:
            params["max_concepts"] = 20
        
        return params
    
    async def close(self) -> None:
        """Clean up ConceptNet provider resources."""
        if self.client:
            await self.client.close()
            self._client_initialized = False