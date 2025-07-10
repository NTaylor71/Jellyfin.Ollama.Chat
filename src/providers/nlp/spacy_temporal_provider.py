"""
SpaCy Temporal provider for temporal concept expansion.
Implements the BaseProvider interface using SpaCy for time-aware expansion.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import re

from src.providers.nlp.base_provider import (
    BaseProvider, ProviderMetadata, ExpansionRequest, ProviderError, ProviderNotAvailableError
)
from src.shared.plugin_contracts import (
    PluginResult, CacheKey, CacheType, ConfidenceScore, PluginMetadata, PluginType,
    create_field_expansion_result
)

logger = logging.getLogger(__name__)


import spacy


class SpacyTemporalProvider(BaseProvider):
    """
    SpaCy Temporal provider for temporal concept expansion.
    
    Uses SpaCy NLP library to parse natural language time expressions
    and expand them into standardized temporal concepts.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        super().__init__()
        self.model_name = model_name
        self.nlp = None
        self._spacy_initialized = False
        
    @property
    def metadata(self) -> ProviderMetadata:
        """Get SpaCy temporal provider metadata."""
        return ProviderMetadata(
            name="SpacyTemporal",
            provider_type="temporal",
            context_aware=True,
            strengths=[
                "natural language time parsing",
                "reliable entity recognition",
                "Python 3.12 compatible",
                "context-aware dates",
                "relative time understanding"
            ],
            weaknesses=[
                "limited to temporal concepts",
                "requires model download",
                "English-focused",
                "may miss complex expressions"
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
        """Initialize the SpaCy temporal provider."""
        
        try:
            if not self._spacy_initialized:
                logger.info("Initializing SpaCy temporal parser")
                

                self.nlp = spacy.load(self.model_name)
                

                test_doc = self.nlp("next Friday")
                self._spacy_initialized = True
                
                logger.info(f"SpaCy initialized successfully with model: {self.model_name}")
                
            return True
        except Exception as e:
            logger.error(f"Failed to initialize SpaCy temporal provider: {e}")
            logger.error("Make sure SpaCy model is installed: python -m spacy download en_core_web_sm")
            return False
    
    async def expand_concept(self, request: ExpansionRequest) -> Optional[PluginResult]:
        """
        Expand concept using SpaCy temporal parsing.
        
        Args:
            request: Expansion request
            
        Returns:
            PluginResult with temporal expansions
        """
        start_time = datetime.now()
        
        try:

            if not await self._ensure_initialized():
                raise ProviderNotAvailableError("SpaCy temporal provider not available", "SpacyTemporal")
            
            if not self.nlp:
                logger.error("SpaCy model not loaded")
                return None
            
            concept = request.concept.strip()
            

            temporal_concepts = await self._parse_temporal_concept(concept)
            
            if not temporal_concepts:
                logger.warning(f"No temporal concepts found for: {concept}")
                return None
            
            
            unique_concepts = []
            seen = set()
            for concept_item in temporal_concepts:
                if concept_item not in seen:
                    unique_concepts.append(concept_item)
                    seen.add(concept_item)
            

            final_concepts = unique_concepts[:request.max_concepts]
            

            confidence_scores = {}
            for i, concept_item in enumerate(final_concepts):

                confidence = max(0.3, 0.9 - (i * 0.1))
                confidence_scores[concept_item] = confidence
            

            total_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            
            return create_field_expansion_result(
                field_name=request.field_name,
                input_value=request.concept,
                expansion_result={
                    "expanded_concepts": final_concepts,
                    "original_concept": request.concept,
                    "expansion_method": "spacy_temporal",
                    "model": self.model_name,
                    "temporal_parsing": True,
                    "provider_type": "temporal"
                },
                confidence_scores=confidence_scores,
                plugin_name="SpacyTemporalProvider",
                plugin_version="1.0.0",
                cache_type=CacheType.SPACY_TEMPORAL,
                execution_time_ms=total_time_ms,
                media_context=request.media_context,
                plugin_type=PluginType.CONCEPT_EXPANSION,
                api_endpoint="spacy:temporal",
                model_used=f"spacy-{self.model_name}"
            )
            
        except Exception as e:
            logger.error(f"SpaCy temporal expansion failed: {e}")
            return None
    
    async def _parse_temporal_concept(self, concept: str) -> List[str]:
        """
        Parse temporal concept using SpaCy NER.
        
        Args:
            concept: Concept to parse
            
        Returns:
            List of temporal concepts
        """
        if not self.nlp:
            return []
        
        try:

            doc = self.nlp(concept)
            
            temporal_concepts = []
            

            for ent in doc.ents:
                if ent.label_ in ['DATE', 'TIME', 'ORDINAL', 'CARDINAL']:

                    entity_text = ent.text.lower()
                    
                    
                    temporal_concepts.append(entity_text)
                    

                    if ent.label_ == 'DATE':
                        if any(word in entity_text for word in ['week', 'weekly']):
                            temporal_concepts.append("weekly")
                        if any(word in entity_text for word in ['month', 'monthly']):
                            temporal_concepts.append("monthly")
                        if any(word in entity_text for word in ['year', 'yearly', 'annual']):
                            temporal_concepts.append("yearly")
                        if any(word in entity_text for word in ['day', 'daily']):
                            temporal_concepts.append("daily")
                        if any(word in entity_text for word in ['recent', 'new', 'latest']):
                            temporal_concepts.append("recent")
                        if any(word in entity_text for word in ['old', 'classic', 'vintage']):
                            temporal_concepts.append("classic")
                    
                    
                    temporal_concepts.append(f"temporal-{ent.label_.lower()}")
            

            if not temporal_concepts:
                try:
                    from src.shared.temporal_concept_generator import TemporalConceptGenerator
                    generator = TemporalConceptGenerator()
                    result = await generator.classify_temporal_concept(concept)
                    if result.enhanced_data.get("is_temporal", False):
                        temporal_concepts.append(concept.lower())
                        temporal_concepts.append("temporal-general")
                except Exception as e:
                    logger.debug(f"TemporalConceptGenerator fallback failed: {e}")
            
            return temporal_concepts
            
        except Exception as e:
            logger.error(f"Error parsing temporal concept '{concept}': {e}")
            return []
    
    def _fallback_temporal_detection(self, concept: str) -> bool:
        """
        Fallback temporal detection using LLM-based procedural intelligence.
        
        Uses TemporalConceptGenerator to determine if concept is temporal.
        
        Args:
            concept: Concept to check
            
        Returns:
            True if appears temporal
        """
        try:
            from src.shared.temporal_concept_generator import TemporalConceptGenerator
            
            
            generator = TemporalConceptGenerator()
            

            temporal_result = generator.classify_temporal_concept(concept)
            

            return temporal_result.confidence_score.overall > 0.3
            
        except Exception as e:
            logger.warning(f"LLM temporal detection failed: {e}")
            

            import re
            return bool(re.search(r'\b\d{4}s?\b', concept))
    
    async def health_check(self) -> Dict[str, Any]:
        """Check SpaCy temporal provider health."""
        try:
            
            if not self._spacy_initialized:
                return {
                    "status": "unhealthy",
                    "provider": "SpacyTemporal",
                    "error": "SpaCy not initialized"
                }
            
            if not self.nlp:
                return {
                    "status": "unhealthy",
                    "provider": "SpacyTemporal",
                    "error": "SpaCy model not loaded"
                }
            

            try:
                test_doc = self.nlp("next week")
                return {
                    "status": "healthy",
                    "provider": "SpacyTemporal",
                    "model": self.model_name,
                    "test_result": "success"
                }
            except Exception as test_error:
                return {
                    "status": "unhealthy",
                    "provider": "SpacyTemporal",
                    "error": f"Test failed: {test_error}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "provider": "SpacyTemporal",
                "error": str(e)
            }
    
    def can_handle_concept(self, concept: str, media_context: str) -> bool:
        """
        Check if SpaCy can handle this concept.
        
        SpaCy works best with:
        - Time expressions ("next Friday", "last week")
        - Date references ("2024", "1990s") 
        - Temporal adjectives ("recent", "old")
        """
        if not concept or len(concept.strip()) == 0:
            return False
        
        
        return self._fallback_temporal_detection(concept)
    
    def get_recommended_parameters(self, concept: str, media_context: str) -> Dict[str, Any]:
        """Get recommended parameters for SpaCy expansion."""
        return {
            "max_concepts": 8,
            "model": self.model_name,
            "temporal_focus": True
        }
    
    async def close(self) -> None:
        """Clean up SpaCy provider resources."""
        if self.nlp:

            self.nlp = None
            self._spacy_initialized = False
            logger.info("SpaCy temporal provider closed")