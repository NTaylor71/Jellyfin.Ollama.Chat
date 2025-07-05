"""
TemporalConceptGenerator: Procedural temporal intelligence via LLM analysis.

Replaces all hard-coded temporal patterns with LLM-driven understanding of
temporal concepts in different media contexts. Learns what "recent", "classic",
"90s", etc. actually mean for movies vs books vs music.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from concept_expansion.providers.llm.llm_provider import LLMProvider
from data.cache_manager import get_cache_manager, CacheStrategy
from shared.plugin_contracts import (
    PluginResult, CacheKey, CacheType, ConfidenceScore, PluginMetadata, PluginType,
    create_field_expansion_result
)

logger = logging.getLogger(__name__)


@dataclass
class TemporalConceptRequest:
    """Request for temporal concept generation."""
    temporal_term: str           # e.g., "recent", "classic", "90s"
    media_context: str          # e.g., "movie", "book", "music"
    max_concepts: int = 10
    bootstrap_mode: bool = False  # True for initial learning, False for cached lookup


class TemporalConceptGenerator:
    """
    Procedural temporal intelligence generator using LLM analysis.
    
    Instead of hard-coded patterns, asks LLM questions like:
    - "What does 'recent' mean for movies in 2024?"
    - "What temporal concepts describe 'classic' cinema?"
    - "How do people refer to 1990s music?"
    
    Caches all intelligence for reuse across the system.
    """
    
    def __init__(self, cache_strategy: CacheStrategy = CacheStrategy.CACHE_FIRST):
        """
        Initialize TemporalConceptGenerator.
        
        Args:
            cache_strategy: How to handle caching
        """
        self.cache_strategy = cache_strategy
        self.cache_manager = get_cache_manager()
        self.llm_provider = LLMProvider()
        
        # No hardcoded bootstrap questions - generate dynamically via LLM
    
    def classify_temporal_concept(self, concept: str) -> PluginResult:
        """
        Classify if a concept is temporal using LLM-based procedural intelligence.
        
        Args:
            concept: Concept to classify
            
        Returns:
            PluginResult with temporal classification and confidence
        """
        try:
            # Build LLM prompt for temporal classification
            prompt = f"""Analyze if this concept is temporal (time-related):
            
Concept: "{concept}"
            
Consider:
- Years, dates, time periods (1990s, recent, classic)
- Time qualities (old, new, modern, vintage)
- Time references (current, past, future)
- Time contexts (contemporary, historical, traditional)
- Release/publication terms (debut, premiere, launch)
            
Respond with:
1. Classification: TEMPORAL or NON_TEMPORAL
2. Confidence: 0.0-1.0
3. Reason: brief explanation
            
Format: [TEMPORAL|NON_TEMPORAL] [0.0-1.0] [reason]"""
            
            from concept_expansion.providers.llm.base_llm_client import LLMRequest
            llm_request = LLMRequest(
                prompt=prompt,
                concept=concept,
                media_context="general",
                max_tokens=50,
                temperature=0.1
            )
            
            # Synchronous call for classification
            import asyncio
            try:
                llm_response = asyncio.run(self.llm_provider.client.generate_completion(llm_request))
                if llm_response.success:
                    response_text = llm_response.text.strip()
                    parts = response_text.split()
                    
                    if len(parts) >= 2:
                        classification = parts[0]
                        confidence = float(parts[1])
                        reason = " ".join(parts[2:]) if len(parts) > 2 else "LLM analysis"
                        
                        is_temporal = classification.upper() == "TEMPORAL"
                        
                        from shared.plugin_contracts import PluginResult
                        return PluginResult(
                            enhanced_data={
                                "is_temporal": is_temporal,
                                "classification": classification,
                                "reason": reason
                            },
                            confidence_score=ConfidenceScore(overall=confidence),
                            plugin_metadata=PluginMetadata(
                                plugin_name="TemporalConceptGenerator",
                                plugin_version="1.0.0",
                                plugin_type=PluginType.CONCEPT_EXPANSION,
                                execution_time_ms=50
                            ),
                            cache_key=None,
                            cache_ttl_seconds=3600
                        )
                    
            except Exception as e:
                logger.debug(f"LLM temporal classification failed: {e}")
                
        except Exception as e:
            logger.warning(f"Temporal classification failed: {e}")
            
        # Fallback: simple numeric year detection
        import re
        has_year = bool(re.search(r'\b\d{4}s?\b', concept))
        
        from shared.plugin_contracts import PluginResult
        return PluginResult(
            enhanced_data={
                "is_temporal": has_year,
                "classification": "TEMPORAL" if has_year else "NON_TEMPORAL",
                "reason": "Fallback year detection"
            },
            confidence_score=ConfidenceScore(overall=0.6 if has_year else 0.4),
            plugin_metadata=PluginMetadata(
                plugin_name="TemporalConceptGenerator",
                plugin_version="1.0.0",
                plugin_type=PluginType.CONCEPT_EXPANSION,
                execution_time_ms=5
            ),
            cache_key=None,
            cache_ttl_seconds=3600
        )
    
    async def generate_temporal_concepts(
        self, 
        request: TemporalConceptRequest
    ) -> Optional[PluginResult]:
        """
        Generate temporal concepts for a term in media context.
        
        Args:
            request: Temporal concept generation request
            
        Returns:
            PluginResult with procedurally generated temporal concepts
        """
        start_time = datetime.now()
        
        try:
            # Generate cache key for this temporal concept
            cache_key = self.cache_manager.generate_cache_key(
                cache_type=CacheType.CUSTOM,
                field_name="temporal_concept",
                input_value=f"{request.temporal_term}:{request.media_context}",
                media_context=request.media_context
            )
            
            # Use cache manager's get_or_compute pattern
            result = await self.cache_manager.get_or_compute(
                cache_key=cache_key,
                compute_func=lambda: self._compute_temporal_concepts(request, start_time),
                strategy=self.cache_strategy
            )
            
            if result:
                logger.debug(f"Generated {len(result.enhanced_data.get('temporal_concepts', []))} temporal concepts for '{request.temporal_term}' in {request.media_context}")
            else:
                logger.warning(f"Failed to generate temporal concepts for '{request.temporal_term}' in {request.media_context}")
            
            return result
            
        except Exception as e:
            logger.error(f"TemporalConceptGenerator failed: {e}")
            return None
    
    async def _compute_temporal_concepts(
        self,
        request: TemporalConceptRequest,
        start_time: datetime
    ) -> Optional[PluginResult]:
        """
        Compute temporal concepts using LLM analysis.
        
        This is called by cache manager when cache miss occurs.
        """
        try:
            # Ensure LLM provider is available
            if not await self.llm_provider._ensure_initialized():
                logger.error("LLM provider not available for temporal concept generation")
                return None
            
            # Build intelligent prompt for temporal understanding
            prompt = self._build_temporal_analysis_prompt(
                request.temporal_term,
                request.media_context,
                request.max_concepts
            )
            
            # Create LLM request
            from concept_expansion.providers.llm.base_llm_client import LLMRequest
            llm_request = LLMRequest(
                prompt=prompt,
                concept=request.temporal_term,
                media_context=request.media_context,
                max_tokens=200,
                temperature=0.3,  # Lower temperature for more consistent temporal concepts
                system_prompt=self._build_temporal_system_prompt(request.media_context)
            )
            
            # Call LLM
            llm_response = await self.llm_provider.client.generate_completion(llm_request)
            
            if not llm_response.success:
                logger.warning(f"LLM failed for temporal concept generation: {llm_response.error_message}")
                return None
            
            # Parse temporal concepts from LLM response
            temporal_concepts = self._parse_temporal_concepts(
                llm_response.text,
                request.max_concepts
            )
            
            if not temporal_concepts:
                logger.warning(f"No temporal concepts extracted from LLM response for: {request.temporal_term}")
                return None
            
            # Generate confidence scores
            confidence_scores = self._generate_confidence_scores(
                temporal_concepts,
                request.temporal_term,
                llm_response
            )
            
            # Calculate total execution time
            total_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create PluginResult
            return create_field_expansion_result(
                field_name="temporal_concept",
                input_value=f"{request.temporal_term}:{request.media_context}",
                expansion_result={
                    "temporal_concepts": temporal_concepts,
                    "original_term": request.temporal_term,
                    "media_context": request.media_context,
                    "generation_method": "llm_temporal_analysis",
                    "llm_model": llm_response.model,
                    "provider_type": "temporal_intelligence"
                },
                confidence_scores=confidence_scores,
                plugin_name="TemporalConceptGenerator",
                plugin_version="1.0.0",
                cache_type=CacheType.CUSTOM,
                execution_time_ms=total_time_ms,
                media_context=request.media_context,
                plugin_type=PluginType.CONCEPT_EXPANSION,
                api_endpoint="temporal_concept_generator",
                model_used=llm_response.model
            )
            
        except Exception as e:
            logger.error(f"Temporal concept computation failed: {e}")
            return None
    
    def _build_temporal_system_prompt(self, media_context: str) -> str:
        """Build system prompt for temporal concept understanding."""
        return f"""You are an expert in {media_context} temporal language and how people describe time-related concepts in {media_context} contexts.

Your task is to understand what temporal terms actually mean when people talk about {media_context} content. Focus on:

- How people naturally describe time periods for {media_context}
- What words indicate recency, age, or historical periods
- Cultural and contextual meanings of temporal terms
- Real-world usage patterns in {media_context} discussions

Be specific to {media_context} culture and avoid generic temporal terms that don't relate to how people actually talk about {media_context}."""
    
    def _build_temporal_analysis_prompt(
        self,
        temporal_term: str,
        media_context: str,
        max_concepts: int
    ) -> str:
        """Build prompt for temporal concept analysis."""
        current_year = datetime.now().year
        
        return f"""Analyze the temporal term "{temporal_term}" in the context of {media_context} content.

What specific words and phrases do people use when they mean "{temporal_term}" {media_context}? 

Consider:
- Current year is {current_year}
- How "{temporal_term}" is used in {media_context} discussions
- What time periods or characteristics this implies
- Cultural context and real usage patterns

Provide {max_concepts} specific temporal concepts that relate to "{temporal_term}" in {media_context} contexts.

Return only a comma-separated list of concepts:

Temporal term: {temporal_term}
Media context: {media_context}
Related temporal concepts:"""
    
    def _parse_temporal_concepts(
        self,
        response_text: str,
        max_concepts: int
    ) -> List[str]:
        """
        Parse temporal concepts from LLM response.
        
        Args:
            response_text: Raw LLM response
            max_concepts: Maximum concepts to return
            
        Returns:
            List of cleaned temporal concept strings
        """
        # Clean up the response
        text = response_text.strip()
        
        # Remove common prefixes that LLMs might add
        prefixes_to_remove = [
            "Related temporal concepts:", "Temporal concepts:", "Concepts:",
            "Related terms:", "Here are", "The concepts are:"
        ]
        
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        # Split by commas and clean each concept
        concepts = []
        
        if "," in text:
            raw_concepts = text.split(",")
        else:
            # Try newline separation as fallback
            raw_concepts = text.split("\n")
        
        for concept in raw_concepts:
            cleaned = self._clean_temporal_concept(concept)
            if cleaned and len(cleaned) > 1:
                concepts.append(cleaned)
                
                if len(concepts) >= max_concepts:
                    break
        
        return concepts[:max_concepts]
    
    def _clean_temporal_concept(self, concept: str) -> str:
        """
        Clean and normalize a temporal concept.
        
        Args:
            concept: Raw concept string
            
        Returns:
            Cleaned concept string
        """
        import re
        
        # Basic cleaning
        cleaned = concept.strip()
        
        # Remove numbering (1., 2., etc.)
        cleaned = re.sub(r'^\d+\.?\s*', '', cleaned)
        
        # Remove bullet points
        cleaned = re.sub(r'^[-•*]\s*', '', cleaned)
        
        # Remove quotes
        cleaned = cleaned.strip('"\'')
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Lowercase and strip
        cleaned = cleaned.lower().strip()
        
        return cleaned
    
    def _generate_confidence_scores(
        self,
        concepts: List[str],
        original_term: str,
        llm_response
    ) -> Dict[str, float]:
        """
        Generate confidence scores for temporal concepts.
        
        Args:
            concepts: List of generated concepts
            original_term: Original temporal term
            llm_response: LLM response metadata
            
        Returns:
            Dictionary mapping concept to confidence score
        """
        confidence_scores = {}
        base_confidence = 0.85  # Higher than general concept expansion due to focused temporal analysis
        
        for i, concept in enumerate(concepts):
            # Earlier concepts typically have higher confidence
            position_factor = 1.0 - (i * 0.05)  # Decrease by 5% per position
            position_factor = max(position_factor, 0.6)  # Minimum 60%
            
            # Longer, more specific concepts might be better
            specificity_factor = min(len(concept.split()) / 3.0, 1.2)  # Bonus for multi-word
            specificity_factor = max(specificity_factor, 0.8)  # Minimum 80%
            
            # Temporal relevance bonus - use pattern-based calculation
            temporal_relevance = self._calculate_temporal_relevance_patterns(concept, original_term)
            
            final_confidence = base_confidence * position_factor * specificity_factor * temporal_relevance
            final_confidence = min(final_confidence, 0.95)  # Cap at 95%
            final_confidence = max(final_confidence, 0.4)   # Floor at 40%
            
            confidence_scores[concept] = round(final_confidence, 3)
        
        return confidence_scores
    
    async def bootstrap_temporal_intelligence(
        self,
        media_contexts: List[str] = None,
        temporal_categories: List[str] = None
    ) -> Dict[str, Any]:
        """
        Bootstrap temporal intelligence by learning from LLM analysis.
        
        Asks systematic questions to build initial temporal concept cache.
        
        Args:
            media_contexts: Media types to bootstrap (default: ["movie", "book", "music"])
            temporal_categories: Temporal categories to learn (default: all)
            
        Returns:
            Bootstrap results summary
        """
        if media_contexts is None:
            media_contexts = ["movie", "book", "music", "tv"]
        
        if temporal_categories is None:
            temporal_categories = list(self.bootstrap_questions.keys())
        
        logger.info(f"Bootstrapping temporal intelligence for {len(media_contexts)} media types, {len(temporal_categories)} categories")
        
        bootstrap_results = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "concepts_generated": 0,
            "media_contexts": media_contexts,
            "temporal_categories": temporal_categories
        }
        
        # Generate temporal terms dynamically via LLM analysis
        temporal_terms = await self._generate_temporal_terms_via_llm(media_contexts, temporal_categories)
        
        for media_context in media_contexts:
            for category in temporal_categories:
                if category in temporal_terms:
                    for term in temporal_terms[category][:2]:  # Limit to first 2 terms per category
                        request = TemporalConceptRequest(
                            temporal_term=term,
                            media_context=media_context,
                            max_concepts=8,
                            bootstrap_mode=True
                        )
                        
                        bootstrap_results["total_requests"] += 1
                        
                        try:
                            result = await self.generate_temporal_concepts(request)
                            
                            if result and result.success:
                                bootstrap_results["successful"] += 1
                                concepts = result.enhanced_data.get("temporal_concepts", [])
                                bootstrap_results["concepts_generated"] += len(concepts)
                                logger.debug(f"Bootstrapped {len(concepts)} concepts for '{term}' in {media_context}")
                            else:
                                bootstrap_results["failed"] += 1
                                logger.warning(f"Failed to bootstrap '{term}' in {media_context}")
                                
                        except Exception as e:
                            bootstrap_results["failed"] += 1
                            logger.error(f"Bootstrap error for '{term}' in {media_context}: {e}")
        
        logger.info(f"Bootstrap complete: {bootstrap_results['successful']}/{bootstrap_results['total_requests']} successful, {bootstrap_results['concepts_generated']} total concepts")
        return bootstrap_results
    
    async def close(self) -> None:
        """Clean up resources."""
        if self.llm_provider:
            await self.llm_provider.close()


    def _calculate_temporal_relevance_patterns(self, concept: str, original_term: str) -> float:
        """
        Calculate temporal relevance using LLM-based procedural intelligence.
        
        Args:
            concept: Generated concept
            original_term: Original temporal term
            
        Returns:
            Temporal relevance multiplier
        """
        relevance = 1.0
        
        try:
            # Use LLM to assess temporal relevance instead of hard-coded patterns
            prompt = f"""Analyze the temporal relevance between these concepts:
            
Original term: "{original_term}"
Generated concept: "{concept}"
            
Rate temporal relevance on scale 1.0-1.5 where:
- 1.0 = no temporal connection
- 1.2 = some temporal connection
- 1.5 = strong temporal connection
            
Consider: temporal similarity, semantic overlap, specificity
Respond with just the number (e.g., 1.3)"""
            
            # Quick LLM call for relevance scoring
            from concept_expansion.providers.llm.base_llm_client import LLMRequest
            llm_request = LLMRequest(
                prompt=prompt,
                concept=f"{original_term}:{concept}",
                media_context="general",
                max_tokens=10,
                temperature=0.1
            )
            
            # Synchronous call for pattern analysis
            import asyncio
            try:
                llm_response = asyncio.run(self.llm_provider.client.generate_completion(llm_request))
                if llm_response.success:
                    relevance = float(llm_response.text.strip())
                    relevance = max(1.0, min(1.5, relevance))  # Clamp to valid range
                else:
                    relevance = 1.0
            except:
                relevance = 1.0
                
        except Exception as e:
            logger.debug(f"LLM relevance calculation failed: {e}")
            # Simple fallback: check for semantic similarity
            if original_term.lower() in concept.lower() or concept.lower() in original_term.lower():
                relevance = 1.2
            else:
                relevance = 1.0
        
        return relevance

    async def _generate_temporal_terms_via_llm(
        self,
        media_contexts: List[str], 
        temporal_categories: List[str]
    ) -> Dict[str, List[str]]:
        """
        Generate temporal terms for categories using LLM analysis.
    
    Args:
        media_contexts: List of media contexts
        temporal_categories: List of temporal categories
        
    Returns:
        Dictionary of generated temporal terms per category
        """
        temporal_terms = {}
        
        try:
            from concept_expansion.providers.llm.llm_provider import LLMProvider
            from concept_expansion.providers.llm.base_llm_client import LLMRequest
            
            if not await self.llm_provider._ensure_initialized():
                # Fallback for all categories
                for category in temporal_categories:
                    temporal_terms[category] = [category, f"related-{category}"]
                return temporal_terms
            
            for category in temporal_categories:
                # Generate terms for this category via LLM
                prompt = f'''Generate 5-8 temporal terms that relate to the category "{category}" across different media contexts.

Consider how people actually talk about "{category}" in discussions of movies, books, music, TV shows, etc.

Return only a comma-separated list of relevant temporal terms:

Category: {category}
Media contexts: {', '.join(media_contexts)}
Temporal terms:'''

                llm_request = LLMRequest(
                    prompt=prompt,
                    concept=category,
                    media_context="general",
                    max_tokens=100,
                    temperature=0.4,
                    system_prompt="Generate natural temporal terms that people actually use in media discussions."
                )
                
                response = await self.llm_provider.client.generate_completion(llm_request)
                
                if response.success:
                    # Parse comma-separated terms
                    terms = [term.strip() for term in response.text.split(',') if term.strip()]
                    temporal_terms[category] = terms[:8]  # Limit to 8 terms
                    logger.debug(f"Generated {len(terms)} terms for category '{category}'")
                else:
                    # Fallback for this category
                    temporal_terms[category] = [category, f"related-{category}"]
                    
        except Exception as e:
            logger.error(f"LLM temporal term generation failed: {e}")
            # Fallback for all categories
            for category in temporal_categories:
                temporal_terms[category] = [category, f"related-{category}"]
        
        return temporal_terms


# Global instance for reuse
_temporal_concept_generator: Optional[TemporalConceptGenerator] = None


def get_temporal_concept_generator() -> TemporalConceptGenerator:
    """Get singleton TemporalConceptGenerator instance."""
    global _temporal_concept_generator
    if _temporal_concept_generator is None:
        _temporal_concept_generator = TemporalConceptGenerator()
    return _temporal_concept_generator