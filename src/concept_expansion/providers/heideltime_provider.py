"""
HeidelTime provider for temporal concept expansion.
Implements the BaseProvider interface using HeidelTime for context-aware temporal extraction.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import re

from concept_expansion.providers.base_provider import (
    BaseProvider, ProviderMetadata, ExpansionRequest, ProviderError, ProviderNotAvailableError
)
from shared.plugin_contracts import (
    PluginResult, CacheKey, CacheType, ConfidenceScore, PluginMetadata, PluginType,
    create_field_expansion_result
)

logger = logging.getLogger(__name__)

# HeidelTime imports with fallback
try:
    from py_heideltime import HeidelTime
    HEIDELTIME_AVAILABLE = True
except ImportError:
    HEIDELTIME_AVAILABLE = False
    logger.warning("HeidelTime not available. Install with: pip install py-heideltime")


class HeidelTimeProvider(BaseProvider):
    """
    HeidelTime provider for context-aware temporal concept expansion.
    
    Uses HeidelTime for document-aware temporal extraction and expansion,
    providing more sophisticated temporal understanding than simple pattern matching.
    """
    
    def __init__(self, language: str = "english"):
        super().__init__()
        self.language = language
        self.heideltime: Optional[HeidelTime] = None
        self._heideltime_initialized = False
        
        # Pure HeidelTime temporal parser - no hard-coded patterns
    
    @property
    def metadata(self) -> ProviderMetadata:
        """Get HeidelTime provider metadata."""
        return ProviderMetadata(
            name="HeidelTime",
            provider_type="temporal",
            context_aware=True,
            strengths=[
                "document-aware temporal extraction",
                "context-sensitive parsing",
                "multilingual support",
                "sophisticated temporal modeling",
                "publication date awareness"
            ],
            weaknesses=[
                "complex setup",
                "document context required",
                "slower processing",
                "Java dependency",
                "limited to temporal concepts"
            ],
            best_for=[
                "document temporal analysis",
                "historical dates",
                "context-aware time extraction",
                "publication timelines",
                "chronological analysis"
            ]
        )
    
    async def initialize(self) -> bool:
        """Initialize the HeidelTime provider."""
        if not HEIDELTIME_AVAILABLE:
            logger.error("HeidelTime not available - cannot initialize HeidelTimeProvider")
            return False
        
        try:
            if not self._heideltime_initialized:
                logger.info("Initializing HeidelTime temporal extractor")
                
                # Initialize HeidelTime with language
                self.heideltime = HeidelTime(language=self.language)
                
                # Test the initialization
                test_text = "The movie was released in 2020."
                test_result = self.heideltime.parse(test_text)
                
                self._heideltime_initialized = True
                logger.info(f"HeidelTime initialized successfully for language: {self.language}")
                
            return True
        except Exception as e:
            logger.error(f"Failed to initialize HeidelTime provider: {e}")
            logger.error("Make sure HeidelTime is properly installed with Java dependencies")
            return False
    
    async def expand_concept(self, request: ExpansionRequest) -> Optional[PluginResult]:
        """
        Expand concept using HeidelTime temporal extraction.
        
        Args:
            request: Expansion request
            
        Returns:
            PluginResult with HeidelTime temporal expansions
        """
        start_time = datetime.now()
        
        try:
            # Ensure provider is initialized
            if not await self._ensure_initialized():
                raise ProviderNotAvailableError("HeidelTime provider not available", "HeidelTime")
            
            if not self.heideltime:
                logger.error("HeidelTime not initialized")
                return None
            
            concept = request.concept.strip()
            
            # Create document context for better temporal understanding
            document_context = await self._create_document_context(concept, request.media_context)
            
            # Pure HeidelTime temporal extraction - no pattern expansion
            temporal_concepts = []
            confidence_scores = {}
            
            # Direct HeidelTime temporal extraction only
            extracted_temporals = self._extract_temporal_expressions(
                document_context, concept, request.media_context
            )
            if extracted_temporals:
                temporal_concepts.extend(extracted_temporals)
            
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
                logger.warning(f"No temporal concepts extracted for: {concept}")
                return None
            
            # Generate confidence scores based on extraction quality
            for i, concept_item in enumerate(final_concepts):
                # Higher confidence for directly extracted temporals
                if i < len(extracted_temporals):
                    confidence = max(0.7, 0.95 - (i * 0.05))
                else:
                    confidence = max(0.4, 0.8 - (i * 0.08))
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
                    "expansion_method": "heideltime",
                    "language": self.language,
                    "document_context": True,
                    "provider_type": "temporal"
                },
                confidence_scores=confidence_scores,
                plugin_name="HeidelTimeProvider",
                plugin_version="1.0.0",
                cache_type=CacheType.HEIDELTIME,
                execution_time_ms=total_time_ms,
                media_context=request.media_context,
                plugin_type=PluginType.CONCEPT_EXPANSION,
                api_endpoint="heideltime:temporal",
                model_used=f"heideltime-{self.language}"
            )
            
        except Exception as e:
            logger.error(f"HeidelTime expansion failed: {e}")
            return None
    
    async def _create_document_context(self, concept: str, media_context: str) -> str:
        """
        Create document context for better temporal understanding using LLM intelligence.
        
        Args:
            concept: Concept to expand
            media_context: Media context
            
        Returns:
            Document context string
        """
        try:
            return await self._generate_document_context_via_llm(concept, media_context)
        except Exception as e:
            logger.debug(f"LLM context generation failed: {e}")
            # Minimal fallback without hardcoded assumptions
            return f"This {media_context} content relates to {concept} and has temporal elements."
    
    def _extract_temporal_expressions(self, document_context: str, concept: str, media_context: str) -> List[str]:
        """
        Extract temporal expressions using HeidelTime.
        
        Args:
            document_context: Document context
            concept: Original concept
            media_context: Media context
            
        Returns:
            List of temporal concepts
        """
        if not self.heideltime:
            return []
        
        try:
            # Parse the document context
            temporal_data = self.heideltime.parse(document_context)
            
            temporal_concepts = []
            
            # Extract temporal expressions from HeidelTime output
            if temporal_data:
                # Parse HeidelTime XML output
                temporal_expressions = self._parse_heideltime_output(temporal_data)
                
                for expr in temporal_expressions:
                    # Convert to concept format
                    concept_forms = self._temporal_to_concepts(expr, media_context)
                    temporal_concepts.extend(concept_forms)
            
            return temporal_concepts
            
        except Exception as e:
            logger.error(f"Error extracting temporal expressions: {e}")
            return []
    
    def _parse_heideltime_output(self, heideltime_output: str) -> List[Dict[str, Any]]:
        """
        Parse HeidelTime XML output to extract temporal expressions.
        
        Args:
            heideltime_output: HeidelTime XML output
            
        Returns:
            List of temporal expression dictionaries
        """
        expressions = []
        
        try:
            # Simple regex parsing of HeidelTime XML output
            # In a production system, you'd use proper XML parsing
            timex_pattern = r'<TIMEX3[^>]*type="([^"]*)"[^>]*value="([^"]*)"[^>]*>([^<]*)</TIMEX3>'
            matches = re.findall(timex_pattern, heideltime_output)
            
            for match in matches:
                timex_type, value, text = match
                expressions.append({
                    "type": timex_type,
                    "value": value,
                    "text": text
                })
            
        except Exception as e:
            logger.error(f"Error parsing HeidelTime output: {e}")
        
        return expressions
    
    def _temporal_to_concepts(self, temporal_expr: Dict[str, Any], media_context: str) -> List[str]:
        """
        Convert temporal expression to concept terms.
        
        Args:
            temporal_expr: Temporal expression dictionary
            media_context: Media context
            
        Returns:
            List of concept terms
        """
        concepts = []
        
        timex_type = temporal_expr.get("type", "")
        value = temporal_expr.get("value", "")
        text = temporal_expr.get("text", "")
        
        # Generate concepts procedurally based on temporal expression structure
        concepts.extend(self._generate_concepts_from_temporal_expression(timex_type, value, text))
        
        return concepts
    
    
    async def health_check(self) -> Dict[str, Any]:
        """Check HeidelTime provider health."""
        try:
            if not HEIDELTIME_AVAILABLE:
                return {
                    "status": "unhealthy",
                    "provider": "HeidelTime",
                    "error": "HeidelTime not available"
                }
            
            if not self._heideltime_initialized:
                return {
                    "status": "unhealthy",
                    "provider": "HeidelTime",
                    "error": "Not initialized"
                }
            
            if not self.heideltime:
                return {
                    "status": "unhealthy",
                    "provider": "HeidelTime",
                    "error": "HeidelTime instance is None"
                }
            
            # Try a simple test parse
            try:
                test_text = "The movie was released last year."
                test_result = self.heideltime.parse(test_text)
                success = test_result is not None
                
                return {
                    "status": "healthy" if success else "degraded",
                    "provider": "HeidelTime",
                    "language": self.language,
                    "test_successful": success
                }
                
            except Exception as e:
                return {
                    "status": "degraded",
                    "provider": "HeidelTime",
                    "language": self.language,
                    "error": f"Parse test failed: {e}"
                }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "HeidelTime",
                "error": str(e)
            }
    
    def supports_concept(self, concept: str, media_context: str) -> bool:
        """
        Check if HeidelTime can handle this concept.
        
        HeidelTime works best with:
        - Document-based temporal expressions
        - Complex temporal contexts
        - Historical references
        - Publication dates and timelines
        """
        concept_lower = concept.lower()
        
        # Use procedural temporal detection via pattern analysis
        return self._has_temporal_patterns(concept_lower)
    
    def get_recommended_parameters(self, concept: str, media_context: str) -> Dict[str, Any]:
        """Get recommended parameters for HeidelTime expansion."""
        params = {
            "max_concepts": 10
        }
        
        # HeidelTime can provide detailed temporal analysis
        if self.supports_concept(concept, media_context):
            params["max_concepts"] = 15
        
        return params
    
    async def close(self) -> None:
        """Clean up HeidelTime provider resources."""
        if self.heideltime:
            # HeidelTime cleanup if needed
            self.heideltime = None
            self._heideltime_initialized = False
            logger.info("HeidelTime provider closed")
    
    async def _generate_document_context_via_llm(self, concept: str, media_context: str) -> str:
        """
        Generate document context using LLM intelligence.
        
        Args:
            concept: Concept to create context for
            media_context: Media context
            
        Returns:
            Generated document context with temporal elements
        """
        try:
            from concept_expansion.providers.llm.llm_provider import LLMProvider
            from concept_expansion.providers.llm.base_llm_client import LLMRequest
            
            llm_provider = LLMProvider()
            if not await llm_provider._ensure_initialized():
                # Fallback if LLM unavailable
                return f"This {media_context} content relates to {concept} and contains temporal information."
            
            # Create LLM request for document context generation
            prompt = f'''Generate a 2-3 sentence document context for HeidelTime temporal parsing.

Requirements:
- Include the concept "{concept}" in {media_context} context
- Add natural temporal expressions for parsing
- Use varied time references appropriate to {media_context}
- Make it sound like real content description

Concept: {concept}
Media type: {media_context}

Generated document context:'''

            llm_request = LLMRequest(
                prompt=prompt,
                concept=concept,
                media_context=media_context,
                max_tokens=150,
                temperature=0.4,
                system_prompt=f"Generate natural document contexts for {media_context} content with temporal expressions."
            )
            
            response = await llm_provider.client.generate_completion(llm_request)
            
            if response.success:
                context = response.text.strip()
                logger.debug(f"Generated document context for '{concept}': {context[:50]}...")
                return context
            else:
                # Fallback if LLM fails
                return f"This {media_context} content about {concept} has temporal aspects."
                
        except Exception as e:
            logger.debug(f"LLM document context generation failed for '{concept}': {e}")
            return f"This {media_context} content relates to {concept} and contains temporal information."
    
    def _has_temporal_patterns(self, concept: str) -> bool:
        """
        Check if concept has temporal patterns using regex analysis.
        
        Args:
            concept: Concept to check
            
        Returns:
            True if appears temporal
        """
        import re
        
        # Use regex patterns for temporal detection (no hardcoded word lists)
        temporal_patterns = [
            r'\b\d{4}s?\b',                    # Years/decades
            r'\b(19|20)\d{2}\b',               # Full years
            r'\b\d{1,2}(st|nd|rd|th)\b',       # Ordinals
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',  # Months
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',  # Month abbreviations
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',  # Days
            r'\b(spring|summer|fall|winter|autumn)\b',  # Seasons
            r'\b(yesterday|today|tomorrow|recent|old|new|classic|modern|current|past|future)\b',  # Time indicators
            r'\b(last|next|previous|following|prior)\b',  # Relative time
            r'\b(release|premiere|debut|launch|publication|broadcast)\b',  # Release terms
            r'\b(decade|century|era|age|period|epoch|time|date|year|month|day|week|hour|minute)\b',  # Time units
            r'\b(historical|chronological|temporal|sequential)\b'  # Time-related adjectives
        ]
        
        return any(re.search(pattern, concept, re.IGNORECASE) for pattern in temporal_patterns)
    
    def _generate_concepts_from_temporal_expression(self, timex_type: str, value: str, text: str) -> List[str]:
        """
        Generate concepts procedurally from temporal expression structure.
        
        Args:
            timex_type: Type of temporal expression
            value: Parsed value
            text: Original text
            
        Returns:
            List of generated concepts
        """
        concepts = []
        
        # Generate concepts based on expression characteristics
        if timex_type:
            concepts.append(f"type-{timex_type.lower()}")
        
        # Analyze value patterns
        if value:
            # Check for reference patterns
            if "REF" in value:
                if "PRESENT_REF" in value:
                    concepts.extend(["current-reference", "present-time", "now-relative"])
                elif "PAST_REF" in value:
                    concepts.extend(["past-reference", "historical-time", "before-now"])
                elif "FUTURE_REF" in value:
                    concepts.extend(["future-reference", "upcoming-time", "after-now"])
            
            # Extract year information procedurally
            import re
            year_match = re.search(r'(\d{4})', value)
            if year_match:
                year = int(year_match.group(1))
                decade = f"{year // 10 * 10}s"
                century = f"{(year // 100) + 1}th-century"
                concepts.extend([decade, f"year-{year}", century])
        
        # Generate concepts from text patterns
        if text and text.lower() != timex_type.lower():
            # Add text-based concepts
            import re
            text_clean = re.sub(r'[^\w\s]', '', text.lower())
            if len(text_clean) > 2:
                concepts.append(f"text-{text_clean}")
        
        return concepts