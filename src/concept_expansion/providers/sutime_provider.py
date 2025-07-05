"""
SUTime provider for rule-based temporal concept expansion.
Implements the BaseProvider interface using SUTime for precise temporal parsing.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import re
import json

from concept_expansion.providers.base_provider import (
    BaseProvider, ProviderMetadata, ExpansionRequest, ProviderError, ProviderNotAvailableError
)
from shared.plugin_contracts import (
    PluginResult, CacheKey, CacheType, ConfidenceScore, PluginMetadata, PluginType,
    create_field_expansion_result
)

logger = logging.getLogger(__name__)

# SUTime imports with fallback
try:
    from sutime import SUTime
    SUTIME_AVAILABLE = True
except ImportError:
    SUTIME_AVAILABLE = False
    logger.warning("SUTime not available. Install with: pip install sutime")


class SUTimeProvider(BaseProvider):
    """
    SUTime provider for rule-based temporal concept expansion.
    
    Uses Stanford SUTime for precise, rule-based temporal expression parsing
    with high accuracy for structured time expressions.
    """
    
    def __init__(self):
        super().__init__()
        self.sutime: Optional[SUTime] = None
        self._sutime_initialized = False
        
        # Pure SUTime temporal parser - no hard-coded patterns
    
    @property
    def metadata(self) -> ProviderMetadata:
        """Get SUTime provider metadata."""
        return ProviderMetadata(
            name="SUTime",
            provider_type="temporal",
            context_aware=False,
            strengths=[
                "rule-based reliability",
                "Stanford NLP integration",
                "precise parsing",
                "structured output",
                "high accuracy"
            ],
            weaknesses=[
                "rigid rules",
                "limited flexibility",
                "English-focused",
                "JVM dependency",
                "no context understanding"
            ],
            best_for=[
                "structured time expressions",
                "formal dates",
                "rule-based parsing",
                "precise temporal extraction",
                "standardized formats"
            ]
        )
    
    async def initialize(self) -> bool:
        """Initialize the SUTime provider."""
        if not SUTIME_AVAILABLE:
            logger.error("SUTime not available - cannot initialize SUTimeProvider")
            return False
        
        try:
            if not self._sutime_initialized:
                logger.info("Initializing SUTime temporal parser")
                
                # Initialize SUTime with JVM settings
                self.sutime = SUTime(jvm_flags=['-Xmx4g'])
                
                # Test the initialization
                test_text = "The movie was released on December 25th, 2020."
                test_result = self.sutime.parse(test_text)
                
                self._sutime_initialized = True
                logger.info("SUTime initialized successfully")
                
            return True
        except Exception as e:
            logger.error(f"Failed to initialize SUTime provider: {e}")
            logger.error("Make sure Java is installed and JAVA_HOME is set")
            return False
    
    async def expand_concept(self, request: ExpansionRequest) -> Optional[PluginResult]:
        """
        Expand concept using SUTime temporal parsing.
        
        Args:
            request: Expansion request
            
        Returns:
            PluginResult with SUTime temporal expansions
        """
        start_time = datetime.now()
        
        try:
            # Ensure provider is initialized
            if not await self._ensure_initialized():
                raise ProviderNotAvailableError("SUTime provider not available", "SUTime")
            
            if not self.sutime:
                logger.error("SUTime not initialized")
                return None
            
            concept = request.concept.strip()
            
            # Create enriched text for better temporal parsing
            enriched_text = self._create_temporal_context(concept, request.media_context)
            
            # Pure SUTime temporal parsing - no rule expansion
            temporal_concepts = []
            confidence_scores = {}
            
            # Direct SUTime parsing only
            sutime_results = self._parse_with_sutime(enriched_text, concept)
            if sutime_results:
                temporal_concepts.extend(sutime_results)
            
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
            
            # Generate confidence scores based on parsing method
            for i, concept_item in enumerate(final_concepts):
                # Higher confidence for SUTime-parsed results
                if i < len(sutime_results):
                    confidence = max(0.8, 0.95 - (i * 0.03))
                else:
                    confidence = max(0.5, 0.8 - (i * 0.05))
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
                    "expansion_method": "sutime",
                    "rule_based": True,
                    "provider_type": "temporal"
                },
                confidence_scores=confidence_scores,
                plugin_name="SUTimeProvider",
                plugin_version="1.0.0",
                cache_type=CacheType.SUTIME,
                execution_time_ms=total_time_ms,
                media_context=request.media_context,
                plugin_type=PluginType.CONCEPT_EXPANSION,
                api_endpoint="sutime:temporal",
                model_used="sutime-stanford"
            )
            
        except Exception as e:
            logger.error(f"SUTime expansion failed: {e}")
            return None
    
    def _create_temporal_context(self, concept: str, media_context: str) -> str:
        """
        Create temporal context for SUTime parsing.
        
        Args:
            concept: Original concept
            media_context: Media context
            
        Returns:
            Enriched text with temporal context
        """
        templates = {
            "movie": f"The {media_context} {concept} was released. Many {concept} films have premiered over the years.",
            "tv": f"The {media_context} show about {concept} aired recently. The {concept} series has been on television.",
            "book": f"The {media_context} about {concept} was published. The {concept} book has been in print.",
            "music": f"The {media_context} album {concept} was released. The {concept} music has been available."
        }
        
        return templates.get(media_context, f"The {concept} content was created recently.")
    
    def _parse_with_sutime(self, text: str, original_concept: str) -> List[str]:
        """
        Parse text with SUTime to extract temporal concepts.
        
        Args:
            text: Text to parse
            original_concept: Original concept
            
        Returns:
            List of temporal concepts
        """
        if not self.sutime:
            return []
        
        try:
            # Parse with SUTime
            parsed_results = self.sutime.parse(text)
            
            temporal_concepts = []
            
            for result in parsed_results:
                # Extract temporal type and value
                if 'type' in result:
                    temporal_type = result['type']
                    temporal_concepts.append(f"temporal-{temporal_type.lower()}")
                
                if 'value' in result:
                    value = result['value']
                    
                    # Extract year information
                    if isinstance(value, str):
                        year_match = re.search(r'(\d{4})', value)
                        if year_match:
                            year = year_match.group(1)
                            decade = f"{year[:3]}0s"
                            temporal_concepts.extend([f"year-{year}", decade])
                    
                    # Extract temporal patterns
                    if 'PRESENT_REF' in str(value):
                        temporal_concepts.extend(["current", "present", "now"])
                    elif 'PAST_REF' in str(value):
                        temporal_concepts.extend(["past", "historical", "previous"])
                    elif 'FUTURE_REF' in str(value):
                        temporal_concepts.extend(["future", "upcoming", "forthcoming"])
                
                # Extract text from result
                if 'text' in result:
                    text_value = result['text'].lower()
                    if text_value != original_concept.lower():
                        temporal_concepts.append(text_value)
            
            return temporal_concepts
            
        except Exception as e:
            logger.error(f"Error parsing with SUTime: {e}")
            return []
    
    
    async def health_check(self) -> Dict[str, Any]:
        """Check SUTime provider health."""
        try:
            if not SUTIME_AVAILABLE:
                return {
                    "status": "unhealthy",
                    "provider": "SUTime",
                    "error": "SUTime not available"
                }
            
            if not self._sutime_initialized:
                return {
                    "status": "unhealthy",
                    "provider": "SUTime",
                    "error": "Not initialized"
                }
            
            if not self.sutime:
                return {
                    "status": "unhealthy",
                    "provider": "SUTime",
                    "error": "SUTime instance is None"
                }
            
            # Try a simple test parse
            try:
                test_text = "The movie premiered on January 1st, 2020."
                test_result = self.sutime.parse(test_text)
                success = isinstance(test_result, list) and len(test_result) > 0
                
                return {
                    "status": "healthy" if success else "degraded",
                    "provider": "SUTime",
                    "test_successful": success,
                    "parsed_results": len(test_result) if test_result else 0
                }
                
            except Exception as e:
                return {
                    "status": "degraded",
                    "provider": "SUTime",
                    "error": f"Parse test failed: {e}"
                }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "SUTime",
                "error": str(e)
            }
    
    def supports_concept(self, concept: str, media_context: str) -> bool:
        """
        Check if SUTime can handle this concept.
        
        SUTime works best with:
        - Structured temporal expressions
        - Formal dates and times
        - Standard temporal formats
        """
        concept_lower = concept.lower()
        
        # Basic temporal keywords that SUTime can parse
        temporal_keywords = [
            "date", "time", "year", "month", "day", "week",
            "release", "premiere", "debut", "launch", 
            "schedule", "timeline", "chronology",
            "period", "era", "age", "epoch",
            "yesterday", "today", "tomorrow",
            "last", "next", "this", "recent"
        ]
        
        return any(keyword in concept_lower for keyword in temporal_keywords)
    
    def get_recommended_parameters(self, concept: str, media_context: str) -> Dict[str, Any]:
        """Get recommended parameters for SUTime expansion."""
        params = {
            "max_concepts": 12
        }
        
        # SUTime can provide precise temporal analysis
        if self.supports_concept(concept, media_context):
            params["max_concepts"] = 15
        
        return params
    
    async def close(self) -> None:
        """Clean up SUTime provider resources."""
        if self.sutime:
            # SUTime cleanup if needed
            self.sutime = None
            self._sutime_initialized = False
            logger.info("SUTime provider closed")