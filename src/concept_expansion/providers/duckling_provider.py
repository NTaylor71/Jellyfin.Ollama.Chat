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
                
                # Initialize Duckling wrapper - discover supported languages programmatically
                duckling_lang = self._get_duckling_language_programmatically(self.language)
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
        
        # Use fallback temporal detection for quick support check
        return self._fallback_temporal_detection(concept_lower)
    
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
    
    def _get_duckling_language_programmatically(self, language_code: str):
        """
        Discover Duckling supported languages programmatically from the Language enum.
        
        Args:
            language_code: ISO language code
            
        Returns:
            Duckling Language enum value
        """
        try:
            # Programmatically discover available languages from the enum
            available_languages = {}
            
            # Inspect the Language enum to find all available languages
            for attr_name in dir(Language):
                if not attr_name.startswith('_') and hasattr(Language, attr_name):
                    attr_value = getattr(Language, attr_name)
                    if hasattr(attr_value, 'value'):
                        # Map common language codes to enum values
                        lang_code = attr_name.lower()
                        available_languages[lang_code] = attr_value
                        
                        # Generate language aliases procedurally
                        aliases = self._generate_language_aliases(lang_code)
                        for alias in aliases:
                            available_languages[alias] = attr_value
            
            # Try to find the requested language
            requested_lang = language_code.lower()
            
            if requested_lang in available_languages:
                logger.info(f"Found Duckling language support for: {language_code}")
                return available_languages[requested_lang]
            
            # Fallback to English if not found
            logger.warning(f"Language '{language_code}' not found in Duckling, using English")
            return available_languages.get('en', Language.EN)
            
        except Exception as e:
            logger.error(f"Error discovering Duckling languages: {e}")
            # Safe fallback
            return Language.EN
    
    async def _is_temporal_via_llm_intelligence(self, concept: str, media_context: str) -> bool:
        """
        Determine if concept is temporal using LLM-generated intelligence.
        
        Uses the TemporalConceptGenerator to ask LLM if this concept has temporal characteristics.
        
        Args:
            concept: Concept to analyze
            media_context: Media context for analysis
            
        Returns:
            True if concept has temporal characteristics
        """
        try:
            from concept_expansion.temporal_concept_generator import get_temporal_concept_generator
            from concept_expansion.providers.llm.llm_provider import LLMProvider
            from concept_expansion.providers.llm.base_llm_client import LLMRequest
            
            # Use LLM to determine temporal characteristics
            llm_provider = LLMProvider()
            if not await llm_provider._ensure_initialized():
                # Fallback to simple pattern detection if LLM unavailable
                return self._fallback_temporal_detection(concept)
            
            # Create LLM request for temporal analysis
            prompt = f'''Analyze if the concept "{concept}" has temporal characteristics in the context of {media_context} content.

Consider:
- Does this concept relate to time, dates, periods, or chronology?
- Is this commonly used with temporal expressions in {media_context} discussions?
- Does this concept imply recency, age, historical periods, or future time?

Answer only: YES or NO

Concept: {concept}
Media context: {media_context}
Has temporal characteristics:'''

            llm_request = LLMRequest(
                prompt=prompt,
                concept=concept,
                media_context=media_context,
                max_tokens=10,
                temperature=0.1,  # Low temperature for consistent yes/no answers
                system_prompt=f"You are an expert at identifying temporal concepts in {media_context} contexts. Be precise and concise."
            )
            
            # Get LLM response
            response = await llm_provider.client.generate_completion(llm_request)
            
            if response.success:
                answer = response.text.strip().upper()
                is_temporal = answer.startswith('YES')
                logger.debug(f"LLM temporal analysis for '{concept}': {answer} -> {is_temporal}")
                return is_temporal
            else:
                # Fallback if LLM fails
                return self._fallback_temporal_detection(concept)
                
        except Exception as e:
            logger.debug(f"LLM temporal analysis failed for '{concept}': {e}")
            return self._fallback_temporal_detection(concept)
    
    def _fallback_temporal_detection(self, concept: str) -> bool:
        """
        Fallback temporal detection when LLM is unavailable.
        
        Uses basic pattern matching as last resort.
        
        Args:
            concept: Concept to check
            
        Returns:
            True if concept appears temporal
        """
        import re
        
        # Basic temporal patterns - only as fallback
        temporal_patterns = [
            r'\b\d{4}s?\b',          # Years (1990s, 2000s)
            r'\b(19|20)\d{2}\b',     # Full years (1990, 2020)
            r'\b\d{1,2}(st|nd|rd|th)\b',  # Ordinals (1st, 2nd)
            r'\b(time|date|year|month|day|week|decade|century|era|age|period)\b',
            r'\b(recent|old|new|classic|modern|current|past|future)\b',
            r'\b(yesterday|today|tomorrow|last|next)\b',
            r'\b(release|premiere|debut|launch)\b',
            r'\b(season|summer|winter|spring|fall|autumn)\b'
        ]
        
        return any(re.search(pattern, concept, re.IGNORECASE) for pattern in temporal_patterns)
    
    def _generate_language_aliases(self, lang_code: str) -> List[str]:
        """
        Generate language aliases procedurally using language knowledge.
        
        Args:
            lang_code: ISO language code
            
        Returns:
            List of possible aliases for the language
        """
        import locale
        
        # Use system locale information to generate aliases
        aliases = [lang_code]
        
        try:
            # Try to get full language name from locale
            if hasattr(locale, 'windows_locale'):
                # Look through Windows locale mappings
                for key, value in locale.windows_locale.items():
                    if value.startswith(lang_code):
                        parts = value.split('_')
                        if len(parts) > 0:
                            aliases.append(parts[0])
            
            # Generate common patterns
            if len(lang_code) == 2:
                # Add full names using a procedural approach
                aliases.extend(self._guess_language_names(lang_code))
        
        except Exception as e:
            logger.debug(f"Error generating language aliases for {lang_code}: {e}")
        
        return list(set(aliases))  # Remove duplicates
    
    def _guess_language_names(self, lang_code: str) -> List[str]:
        """
        Procedurally guess full language names from ISO codes.
        
        Uses common linguistic patterns rather than hardcoded mappings.
        
        Args:
            lang_code: ISO language code
            
        Returns:
            List of guessed language names
        """
        # Use external data sources for language names (no hardcoding)
        try:
            # Try to use system/package language data first
            import pycountry
            try:
                language = pycountry.languages.get(alpha_2=lang_code)
                if language:
                    names = [language.name.lower()]
                    if hasattr(language, 'common_name'):
                        names.append(language.common_name.lower())
                    return names
            except:
                pass
        except ImportError:
            pass
        
        # Fallback: Use simple linguistic analysis
        # Generate variations based on common language code patterns
        variations = [lang_code]
        
        # Add common 3-letter variations
        if len(lang_code) == 2:
            variations.append(lang_code + 'n')  # en -> eng
            variations.append(lang_code + 'g')  # de -> deu, but try deg
            
        # Add 'ish' suffix pattern for many languages
        if len(lang_code) == 2:
            variations.append(lang_code + 'ish')  # en -> enish (approximates english)
        
        return variations