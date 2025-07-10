"""
Generic LLM provider for concept expansion.

Implements the BaseProvider interface using pluggable LLM backends,
providing context-aware concept expansion with automatic backend selection.
"""

import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.providers.nlp.base_provider import (
    BaseProvider, ProviderMetadata, ExpansionRequest, ProviderError, ProviderNotAvailableError
)
from src.providers.llm.base_llm_client import (
    BaseLLMClient, LLMRequest, LLMResponse,
    LLMClientError, LLMClientNotAvailableError
)
from src.shared.plugin_contracts import (
    PluginResult, CacheKey, CacheType, ConfidenceScore, PluginMetadata, PluginType,
    create_field_expansion_result
)
from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class LLMProvider(BaseProvider):
    """
    Generic LLM provider for context-aware concept expansion.
    
    Supports multiple LLM backends (Ollama, OpenAI, etc.) with automatic
    backend selection and intelligent prompt engineering for movie concepts.
    """
    
    def __init__(self, backend: Optional[str] = None):
        super().__init__()
        self.settings = get_settings()
        self.backend_name = backend or self._detect_backend()
        self.client: Optional[BaseLLMClient] = None
        self._client_initialized = False
        
        logger.info(f"Initialized LLM provider with backend: {self.backend_name}")
    
    @property
    def metadata(self) -> ProviderMetadata:
        """Get LLM provider metadata."""
        return ProviderMetadata(
            name="LLM",
            provider_type="semantic",
            context_aware=True,
            strengths=[
                "context understanding",
                "domain knowledge",
                "compound concepts",
                "semantic relationships",
                "movie-specific understanding"
            ],
            weaknesses=[
                "requires API calls",
                "potential hallucination",
                "slower than cached lookups",
                "model-dependent quality",
                "resource intensive"
            ],
            best_for=[
                "movie genres",
                "contextual concepts",
                "compound terms",
                "thematic elements",
                "audience preferences"
            ]
        )
    
    async def initialize(self) -> bool:
        """Initialize the LLM provider and backend client."""
        try:
            if not self._client_initialized:
                self.client = self._create_backend_client(self.backend_name)
                if self.client:
                    success = await self.client.initialize()
                    if success:
                        self._client_initialized = True
                        logger.info(f"LLM provider initialized with {self.backend_name} backend")
                        return True
                    else:
                        logger.error(f"Failed to initialize {self.backend_name} backend")
                        return False
                else:
                    logger.error(f"Could not create {self.backend_name} backend client")
                    return False
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            return False
    
    async def expand_concept(self, request: ExpansionRequest) -> Optional[PluginResult]:
        """
        Expand concept using LLM backend.
        
        Args:
            request: Expansion request
            
        Returns:
            PluginResult with LLM-generated concept expansions
        """
        start_time = datetime.now()
        
        try:

            if not await self._ensure_initialized():
                raise ProviderNotAvailableError("LLM provider not available", "LLM")
            
            
            prompt = self._build_concept_prompt(
                request.concept,
                request.media_context,
                request.max_concepts
            )
            
            
            llm_request = LLMRequest(
                prompt=prompt,
                concept=request.concept,
                media_context=request.media_context,
                max_tokens=self._calculate_max_tokens(request.max_concepts),
                temperature=self.client.get_recommended_temperature("concept"),
                system_prompt=self._build_system_prompt(request.media_context)
            )
            

            llm_response: LLMResponse = await self.client.generate_completion(llm_request)
            
            if not llm_response.success:
                logger.warning(f"LLM completion failed: {llm_response.error_message}")
                return None
            

            concepts = self._parse_concepts_from_response(
                llm_response.text,
                request.max_concepts
            )
            
            if not concepts:
                logger.warning(f"No concepts extracted from LLM response for: {request.concept}")
                return None
            

            confidence_scores = self._generate_confidence_scores(
                concepts,
                request.concept,
                llm_response
            )
            

            total_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            
            return create_field_expansion_result(
                field_name=request.field_name,
                input_value=request.concept,
                expansion_result={
                    "expanded_concepts": concepts,
                    "original_concept": request.concept,
                    "expansion_method": "llm",
                    "backend": self.backend_name,
                    "model": llm_response.model,
                    "llm_response_time_ms": llm_response.response_time_ms,
                    "provider_type": "semantic"
                },
                confidence_scores=confidence_scores,
                plugin_name="LLMProvider",
                plugin_version="1.0.0",
                cache_type=CacheType.LLM_CONCEPT,
                execution_time_ms=total_time_ms,
                media_context=request.media_context,
                plugin_type=PluginType.CONCEPT_EXPANSION,
                api_endpoint=getattr(self.client, 'base_url', 'unknown'),
                model_used=llm_response.model
            )
            
        except Exception as e:
            logger.error(f"LLM concept expansion failed: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check LLM provider health."""
        try:
            if not self._client_initialized:
                return {
                    "status": "unhealthy",
                    "provider": "LLM",
                    "backend": self.backend_name,
                    "error": "Not initialized"
                }
            
            if not self.client:
                return {
                    "status": "unhealthy",
                    "provider": "LLM",
                    "backend": self.backend_name,
                    "error": "No backend client"
                }
            
            
            backend_health = await self.client.health_check()
            
            return {
                "status": backend_health.get("status", "unknown"),
                "provider": "LLM",
                "backend": self.backend_name,
                "backend_health": backend_health,
                "model_info": self.client.get_model_info() if self.client else None
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "LLM",
                "backend": self.backend_name,
                "error": str(e)
            }
    
    def supports_concept(self, concept: str, media_context: str) -> bool:
        """
        Check if LLM can handle this concept.
        
        LLM providers are generally good with all concepts, especially:
        - Compound terms
        - Context-dependent concepts
        - Domain-specific terms
        """

        if len(concept.split()) > 1:

            return True
        
        if media_context in ["movie", "book", "music", "tv"] and concept.lower() in [
            "action", "comedy", "drama", "horror", "thriller", "romance", "sci-fi", "fantasy"
        ]:

            return True
        

        return True
    
    def get_recommended_parameters(self, concept: str, media_context: str) -> Dict[str, Any]:
        """Get recommended parameters for LLM expansion."""
        params = {
            "max_concepts": 10,
            "temperature": 0.3
        }
        

        if " " in concept:

            params["max_concepts"] = 12
            params["temperature"] = 0.4
        

        if concept.lower() in ["action", "comedy", "drama", "horror", "thriller", "romance"]:
            params["max_concepts"] = 15
            params["temperature"] = 0.2
        
        return params
    
    async def close(self) -> None:
        """Clean up LLM provider resources."""
        if self.client:
            await self.client.close()
            self._client_initialized = False
    
    def _detect_backend(self) -> str:
        """
        Auto-detect which LLM backend to use based on configuration.
        
        Returns:
            Backend name to use
        """


        return "ollama"
    
    def _create_backend_client(self, backend: str) -> Optional[BaseLLMClient]:
        """
        Create backend client for specified backend.
        
        Args:
            backend: Backend name to create
            
        Returns:
            Backend client instance or None if not supported
        """
        if backend == "ollama":
            try:
                from src.providers.llm.ollama_backend_client import OllamaBackendClient
                return OllamaBackendClient()
            except ImportError as e:
                logger.error(f"Could not import Ollama backend: {e}")
                return None
        




        
        else:
            logger.error(f"Unknown LLM backend: {backend}")
            return None
    
    def _build_system_prompt(self, media_context: str) -> str:
        """Build system prompt for concept expansion."""
        return f"""You are an expert in {media_context} content analysis and recommendation systems. Your task is to help expand search concepts to improve content discovery.

When given a concept, provide related terms that would help users find similar {media_context} content they would enjoy. Focus on:
- Thematic elements and mood
- Genre characteristics and subgenres  
- Common narrative patterns
- Audience preferences and expectations
- Synonyms and related terminology

Be specific to {media_context} content and avoid generic terms."""
    
    def _build_concept_prompt(
        self,
        concept: str,
        media_context: str,
        max_concepts: int
    ) -> str:
        """Build context-aware prompt for concept expansion."""
        if self.client:
            
            return self.client.build_concept_expansion_prompt(
                concept, media_context, max_concepts
            )
        

        return f"""For the {media_context} concept "{concept}", provide {max_concepts} related concepts that would help users discover similar content.

Focus on {media_context}-specific elements that share themes, mood, or appeal with "{concept}".

Return only a comma-separated list of concepts:

Concept: {concept}
Related concepts:"""
    
    def _calculate_max_tokens(self, max_concepts: int) -> int:
        """Calculate maximum tokens needed for response."""

        
        return max_concepts * 15 + 50
    
    def _parse_concepts_from_response(
        self,
        response_text: str,
        max_concepts: int
    ) -> List[str]:
        """
        Parse concepts from LLM response text.
        
        Args:
            response_text: Raw LLM response
            max_concepts: Maximum concepts to return
            
        Returns:
            List of cleaned concept strings
        """

        text = response_text.strip()
        
        
        prefixes_to_remove = [
            "Related concepts:", "Concepts:", "Related terms:", "Related movies:",
            "Similar concepts:", "Here are", "The related concepts are:"
        ]
        
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        

        concepts = []
        

        if "," in text:
            raw_concepts = text.split(",")
        else:

            raw_concepts = text.split("\n")
        
        for concept in raw_concepts:
            cleaned = self._clean_concept(concept)
            if cleaned and len(cleaned) > 1:
                concepts.append(cleaned)
                
                if len(concepts) >= max_concepts:
                    break
        
        return concepts[:max_concepts]
    
    def _clean_concept(self, concept: str) -> str:
        """
        Clean and normalize a single concept.
        
        Args:
            concept: Raw concept string
            
        Returns:
            Cleaned concept string
        """

        cleaned = concept.strip()
        
        
        cleaned = re.sub(r'^\d+\.?\s*', '', cleaned)
        
        
        cleaned = re.sub(r'^[-•*]\s*', '', cleaned)
        
        
        cleaned = cleaned.strip('"\'')
        
        
        cleaned = re.sub(r'\s+', ' ', cleaned)
        

        cleaned = cleaned.lower().strip()
        
        return cleaned
    
    def _generate_confidence_scores(
        self,
        concepts: List[str],
        original_concept: str,
        llm_response: LLMResponse
    ) -> Dict[str, float]:
        """
        Generate confidence scores for extracted concepts.
        
        Args:
            concepts: List of extracted concepts
            original_concept: Original concept being expanded
            llm_response: LLM response metadata
            
        Returns:
            Dictionary mapping concept to confidence score
        """
        confidence_scores = {}
        base_confidence = 0.8
        

        if llm_response.response_time_ms > 5000:
            base_confidence *= 0.9
        
        for i, concept in enumerate(concepts):

            position_factor = 1.0 - (i * 0.05)
            position_factor = max(position_factor, 0.5)
            

            length_factor = min(len(concept) / 20.0, 1.2)
            length_factor = max(length_factor, 0.8)
            

            similarity_factor = 1.0
            if original_concept.lower() in concept.lower() or concept.lower() in original_concept.lower():
                similarity_factor = 1.1
            
            final_confidence = base_confidence * position_factor * length_factor * similarity_factor
            final_confidence = min(final_confidence, 0.95)
            final_confidence = max(final_confidence, 0.3) 
            
            confidence_scores[concept] = round(final_confidence, 3)
        
        return confidence_scores