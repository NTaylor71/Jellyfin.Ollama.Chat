"""
Duckling provider for temporal concept expansion.
Implements the BaseProvider interface using Duckling for time-aware expansion.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import re

from concept_expansion.providers.base_provider import (
    BaseProvider, ProviderMetadata, ExpansionRequest, ProviderError, ProviderNotAvailableError
)
from shared.plugin_contracts import (
    PluginResult, CacheKey, CacheType, ConfidenceScore, PluginMetadata, PluginType,
    create_field_expansion_result
)

logger = logging.getLogger(__name__)

# Duckling imports with fallback
try:
    from duckling import DucklingWrapper, Language
    DUCKLING_AVAILABLE = True
except ImportError:
    DUCKLING_AVAILABLE = False
    logger.warning("Duckling not available. Install with: pip install duckling")


class DucklingProvider(BaseProvider):
    """
    Duckling provider for temporal concept expansion.
    
    Uses Duckling NLP library to parse natural language time expressions
    and expand them into standardized temporal concepts.
    """
    
    def __init__(self, language: str = "en"):
        super().__init__()
        self.language = language
        self.duckling: Optional[DucklingWrapper] = None
        self._duckling_initialized = False
        
        # Pure Duckling temporal parser - no hard-coded patterns
    
    @property
    def metadata(self) -> ProviderMetadata:
        """Get Duckling provider metadata."""
        return ProviderMetadata(
            name="Duckling",
            provider_type="temporal",
            context_aware=True,
            strengths=[
                "natural language time parsing",
                "multi-language support",
                "fuzzy time expressions",
                "context-aware dates",
                "relative time understanding"
            ],
            weaknesses=[
                "limited to temporal concepts",
                "requires API calls",
                "complex setup",
                "language-dependent",
                "may over-interpret text"
            ],
            best_for=[
                "release dates",
                "time expressions",
                "natural language dates",
                "temporal queries",
                "chronological concepts"
            ]
        )
    
    async def initialize(self) -> bool:
        """Initialize the Duckling provider."""
        if not DUCKLING_AVAILABLE:
            logger.error("Duckling not available - cannot initialize DucklingProvider")
            return False
        
        try:
            if not self._duckling_initialized:
                logger.info("Initializing Duckling temporal parser")
                
                # Initialize Duckling wrapper
                language_map = {
                    "en": Language.EN,
                    "es": Language.ES,
                    "fr": Language.FR,
                    "de": Language.DE,
                    "it": Language.IT,
                    "pt": Language.PT
                }
                
                duckling_lang = language_map.get(self.language, Language.EN)
                self.duckling = DucklingWrapper(language=duckling_lang)
                
                # Test the connection
                test_result = self.duckling.parse("next Friday")
                self._duckling_initialized = True
                
                logger.info(f"Duckling initialized successfully for language: {self.language}")
                
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Duckling provider: {e}")
            logger.error("Make sure Duckling server is running")
            return False
    
    async def expand_concept(self, request: ExpansionRequest) -> Optional[PluginResult]:
        """
        Expand concept using Duckling temporal parsing.
        
        Args:
            request: Expansion request
            
        Returns:
            PluginResult with temporal expansions
        """
        start_time = datetime.now()
        
        try:
            # Ensure provider is initialized
            if not await self._ensure_initialized():
                raise ProviderNotAvailableError("Duckling provider not available", "Duckling")
            
            if not self.duckling:
                logger.error("Duckling not initialized")
                return None
            
            concept = request.concept.strip()
            
            # Pure Duckling temporal parsing - no pattern expansion
            temporal_concepts = []
            confidence_scores = {}
            
            # Direct Duckling parsing only
            parsed_results = self._parse_temporal_concept(concept)
            if parsed_results:
                temporal_concepts.extend(parsed_results)
            
            # Remove duplicates while preserving order
            unique_concepts = []
            seen = set()
            for concept_item in temporal_concepts:
                if concept_item not in seen:
                    unique_concepts.append(concept_item)
                    seen.add(concept_item)
            
            # Limit to max_concepts
            final_concepts = unique_concepts[:request.max_concepts]
            
            if not final_concepts:
                logger.warning(f"No temporal concepts found for: {concept}")
                return None
            
            # Generate confidence scores
            for i, concept_item in enumerate(final_concepts):
                # Higher confidence for earlier results
                confidence = max(0.3, 0.9 - (i * 0.1))
                confidence_scores[concept_item] = confidence
            
            # Calculate total execution time
            total_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create PluginResult using helper function
            return create_field_expansion_result(
                field_name=request.field_name,
                input_value=request.concept,
                expansion_result={
                    "expanded_concepts": final_concepts,
                    "original_concept": request.concept,
                    "expansion_method": "duckling",
                    "language": self.language,
                    "temporal_parsing": True,
                    "provider_type": "temporal"
                },
                confidence_scores=confidence_scores,
                plugin_name="DucklingProvider",
                plugin_version="1.0.0",
                cache_type=CacheType.DUCKLING_TIME,
                execution_time_ms=total_time_ms,
                media_context=request.media_context,
                plugin_type=PluginType.CONCEPT_EXPANSION,
                api_endpoint="duckling:temporal",
                model_used=f"duckling-{self.language}"
            )
            
        except Exception as e:
            logger.error(f"Duckling expansion failed: {e}")
            return None
    
    def _parse_temporal_concept(self, concept: str) -> List[str]:
        """
        Parse temporal concept using Duckling.
        
        Args:
            concept: Concept to parse
            
        Returns:
            List of temporal concepts
        """
        if not self.duckling:
            return []
        
        try:
            # Parse the concept
            results = self.duckling.parse(concept)
            
            temporal_concepts = []
            
            for result in results:
                if result.get("dim") == "time":
                    # Extract temporal information
                    value = result.get("value", {})
                    
                    if "grain" in value:
                        grain = value["grain"]
                        temporal_concepts.append(f"time-{grain}")
                    
                    if "type" in value:
                        time_type = value["type"]
                        temporal_concepts.append(f"temporal-{time_type}")
                    
                    # Add specific temporal concepts based on parsed data
                    if "month" in value:
                        temporal_concepts.append("monthly")
                    if "year" in value:
                        temporal_concepts.append("yearly")
                    if "day" in value:
                        temporal_concepts.append("daily")
            
            return temporal_concepts
            
        except Exception as e:
            logger.error(f"Error parsing temporal concept '{concept}': {e}")
            return []
    
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Duckling provider health."""
        try:
            if not DUCKLING_AVAILABLE:
                return {
                    "status": "unhealthy",
                    "provider": "Duckling",
                    "error": "Duckling not available"
                }
            
            if not self._duckling_initialized:
                return {
                    "status": "unhealthy",
                    "provider": "Duckling",
                    "error": "Not initialized"
                }
            
            if not self.duckling:
                return {
                    "status": "unhealthy",
                    "provider": "Duckling",
                    "error": "Duckling wrapper is None"
                }
            
            # Try a simple test parse
            try:
                test_result = self.duckling.parse("next week")
                success = isinstance(test_result, list)
                
                return {
                    "status": "healthy" if success else "degraded",
                    "provider": "Duckling",
                    "language": self.language,
                    "test_successful": success
                }
                
            except Exception as e:
                return {
                    "status": "degraded",
                    "provider": "Duckling",
                    "language": self.language,
                    "error": f"Parse test failed: {e}"
                }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "Duckling",
                "error": str(e)
            }
    
    def supports_concept(self, concept: str, media_context: str) -> bool:
        """
        Check if Duckling can handle this concept.
        
        Duckling works best with:
        - Temporal expressions ("next Friday", "last year")
        - Date-related terms ("release date", "premiere")
        - Time periods ("90s", "recent")
        
        Less effective with:
        - Non-temporal concepts
        - Abstract time concepts
        - Very specific dates without context
        """
        concept_lower = concept.lower()
        
        # Temporal keywords
        temporal_keywords = [
            "time", "date", "year", "month", "day", "week",
            "recent", "old", "new", "classic", "modern",
            "release", "premiere", "debut", "launch",
            "decade", "century", "era", "age",
            "90s", "80s", "70s", "60s", "50s",
            "yesterday", "today", "tomorrow",
            "last", "next", "current", "past", "future",
            "season", "summer", "winter", "spring", "fall"
        ]
        
        # Check if concept contains temporal keywords
        return any(keyword in concept_lower for keyword in temporal_keywords)
    
    def get_recommended_parameters(self, concept: str, media_context: str) -> Dict[str, Any]:
        """Get recommended parameters for Duckling expansion."""
        params = {
            "max_concepts": 8
        }
        
        # Temporal concepts often have fewer but more specific results
        if self.supports_concept(concept, media_context):
            params["max_concepts"] = 12
        
        return params
    
    async def close(self) -> None:
        """Clean up Duckling provider resources."""
        if self.duckling:
            # Duckling doesn't need explicit cleanup
            self.duckling = None
            self._duckling_initialized = False
            logger.info("Duckling provider closed")