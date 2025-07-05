"""
Temporal Analysis Plugin - Orchestrates temporal understanding using multiple providers.

Combines SpaCy, HeidelTime, SUTime parsers with TemporalConceptGenerator for
intelligent temporal analysis with media-aware context.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime

from src.plugins.base_concept import BaseConceptPlugin, ProcessingStrategy
from src.plugins.base import (
    PluginMetadata, PluginResourceRequirements, PluginExecutionContext,
    PluginExecutionResult, PluginType, ExecutionPriority
)
from src.concept_expansion.providers.spacy_temporal_provider import SpacyTemporalProvider
from src.concept_expansion.providers.heideltime_provider import HeidelTimeProvider
from src.concept_expansion.providers.sutime_provider import SUTimeProvider
from src.concept_expansion.temporal_concept_generator import (
    TemporalConceptGenerator, TemporalConceptRequest
)
from src.concept_expansion.providers.base_provider import ExpansionRequest
from src.shared.plugin_contracts import PluginResult, CacheType
from src.data.cache_manager import CacheStrategy

logger = logging.getLogger(__name__)


class TemporalAnalysisPlugin(BaseConceptPlugin):
    """
    Plugin that orchestrates temporal analysis using multiple providers.
    
    Features:
    - Pure parsing: SpaCy, HeidelTime, SUTime
    - Intelligent expansion: TemporalConceptGenerator (LLM-based)
    - Media-aware temporal understanding
    - Combines parsing accuracy with semantic intelligence
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="TemporalAnalysisPlugin",
            version="2.0.0",
            description="Orchestrates multi-provider temporal analysis with intelligence",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["temporal", "time", "date", "parsing", "nlp"],
            execution_priority=ExecutionPriority.HIGH
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        return PluginResourceRequirements(
            min_cpu_cores=1.0,
            preferred_cpu_cores=2.0,
            min_memory_mb=300.0,
            preferred_memory_mb=800.0,
            requires_gpu=False,
            max_execution_time_seconds=20.0,
            can_use_distributed_resources=True
        )
    
    async def _initialize_providers(self) -> None:
        """Initialize temporal analysis providers."""
        # Initialize pure parsers
        self.providers["spacy"] = SpacyTemporalProvider()
        await self.providers["spacy"].initialize()
        
        self.providers["heideltime"] = HeidelTimeProvider()
        await self.providers["heideltime"].initialize()
        
        self.providers["sutime"] = SUTimeProvider()
        await self.providers["sutime"].initialize()
        
        # Initialize intelligent temporal generator
        self.temporal_generator = TemporalConceptGenerator(CacheStrategy.CACHE_FIRST)
        
        self._logger.info(f"Initialized {len(self.providers)} temporal providers + intelligence")
    
    async def embellish_embed_data(
        self, 
        data: Dict[str, Any], 
        context: PluginExecutionContext
    ) -> Dict[str, Any]:
        """
        Enhance data with temporal analysis using appropriate strategy.
        """
        start_time = datetime.now()
        
        try:
            # Select processing strategy
            strategy = self._select_processing_strategy(context)
            
            # Extract text for temporal analysis
            text_content = self._extract_temporal_text(data)
            if not text_content:
                self._logger.debug("No text content for temporal analysis")
                return data
            
            # Detect media context
            media_context = self._detect_media_context(data)
            
            self._logger.info(
                f"Analyzing temporal content using {strategy.value} strategy "
                f"for {media_context} context"
            )
            
            # Execute analysis based on strategy
            if strategy == ProcessingStrategy.HIGH_RESOURCE:
                temporal_data = await self._high_resource_temporal_analysis(
                    text_content, media_context, context
                )
            elif strategy == ProcessingStrategy.MEDIUM_RESOURCE:
                temporal_data = await self._medium_resource_temporal_analysis(
                    text_content, media_context, context
                )
            else:
                temporal_data = await self._low_resource_temporal_analysis(
                    text_content, media_context, context
                )
            
            # Add temporal analysis to data
            if "enhanced_fields" not in data:
                data["enhanced_fields"] = {}
            
            data["enhanced_fields"]["temporal_analysis"] = temporal_data
            data["enhanced_fields"]["temporal_metadata"] = {
                "strategy": strategy.value,
                "media_context": media_context,
                "expression_count": len(temporal_data.get("expressions", [])),
                "normalized_count": len(temporal_data.get("normalized", [])),
                "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000
            }
            
            # Create searchable temporal tags
            temporal_tags = self._create_temporal_search_tags(temporal_data)
            if temporal_tags:
                data["enhanced_fields"]["temporal_search_tags"] = temporal_tags
            
            self._logger.info(
                f"Temporal analysis complete: {len(temporal_data.get('expressions', []))} "
                f"expressions found"
            )
            
            return data
            
        except Exception as e:
            self._logger.error(f"Temporal analysis failed: {e}")
            return data
    
    def _extract_temporal_text(self, data: Dict[str, Any]) -> str:
        """Extract text that might contain temporal information."""
        text_parts = []
        
        # Text fields likely to contain temporal info
        temporal_fields = [
            "Overview", "description", "plot", "summary",
            "ProductionYear", "ReleaseDate", "PremiereDate",
            "content", "text", "taglines", "Taglines"
        ]
        
        for field in temporal_fields:
            if field in data:
                value = data[field]
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, (int, float)):
                    text_parts.append(str(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            text_parts.append(item)
        
        return " ".join(text_parts)
    
    def _detect_media_context(self, data: Dict[str, Any]) -> str:
        """Detect media type from data for context-aware analysis."""
        # Check explicit type fields
        if "Type" in data:
            media_type = data["Type"].lower()
            if "movie" in media_type:
                return "movie"
            elif "series" in media_type or "episode" in media_type:
                return "tv"
            elif "book" in media_type:
                return "book"
            elif "audio" in media_type or "music" in media_type:
                return "music"
        
        # Check for movie-specific fields
        movie_fields = ["ProductionYear", "OfficialRating", "CriticRating"]
        if any(field in data for field in movie_fields):
            return "movie"
        
        # Default
        return "movie"
    
    async def _high_resource_temporal_analysis(
        self,
        text: str,
        media_context: str,
        context: PluginExecutionContext
    ) -> Dict[str, Any]:
        """
        High resource: All parsers + intelligent expansion in parallel.
        """
        all_expressions = []
        all_normalized = []
        parser_results = {}
        
        # Parallel parsing with all providers
        parsing_tasks = []
        for parser_name in ["spacy", "heideltime", "sutime"]:
            if parser_name in self.providers:
                parsing_tasks.append(
                    self._parse_with_provider(parser_name, text, media_context)
                )
        
        # Execute parsers in parallel
        parser_outputs = await asyncio.gather(*parsing_tasks, return_exceptions=True)
        
        # Process parser results
        parser_names = ["spacy", "heideltime", "sutime"]
        for i, result in enumerate(parser_outputs):
            if isinstance(result, Exception):
                self._logger.warning(f"Parser {parser_names[i]} failed: {result}")
                continue
            
            if result and result.success:
                parser_data = result.enhanced_data
                expressions = parser_data.get("expanded_concepts", [])
                
                # Store raw parser results
                parser_results[parser_names[i]] = {
                    "expressions": expressions,
                    "count": len(expressions)
                }
                
                # Collect unique expressions
                for expr in expressions:
                    if expr not in all_expressions:
                        all_expressions.append(expr)
        
        # Queue intelligent temporal expansion for detected expressions
        if all_expressions and self.queue_manager:
            # Queue batch temporal intelligence
            task_data = {
                "expressions": all_expressions[:10],  # Limit
                "media_context": media_context,
                "generator": "temporal_concept"
            }
            task_id = await self._queue_task(
                "temporal_intelligence",
                task_data,
                ExecutionPriority.HIGH
            )
            self._logger.debug(f"Queued temporal intelligence generation: {task_id}")
        
        # Direct intelligent expansion for top expressions
        top_expressions = all_expressions[:5]
        for expr in top_expressions:
            try:
                temporal_request = TemporalConceptRequest(
                    temporal_term=expr,
                    media_context=media_context,
                    max_concepts=8
                )
                intel_result = await self.temporal_generator.generate_temporal_concepts(
                    temporal_request
                )
                if intel_result and intel_result.success:
                    intelligent_concepts = intel_result.enhanced_data.get(
                        "temporal_concepts", []
                    )
                    all_normalized.extend(intelligent_concepts)
            except Exception as e:
                self._logger.warning(f"Temporal intelligence failed for '{expr}': {e}")
        
        # Deduplicate normalized concepts
        unique_normalized = list(dict.fromkeys(all_normalized))
        
        return {
            "expressions": all_expressions,
            "normalized": unique_normalized,
            "parser_results": parser_results,
            "analysis_methods": list(parser_results.keys()),
            "temporal_scope": self._determine_temporal_scope(all_expressions),
            "confidence_level": "high" if len(parser_results) >= 2 else "medium"
        }
    
    async def _medium_resource_temporal_analysis(
        self,
        text: str,
        media_context: str,
        context: PluginExecutionContext
    ) -> Dict[str, Any]:
        """
        Medium resource: SpaCy + selective intelligence.
        """
        all_expressions = []
        all_normalized = []
        
        # Use SpaCy (most reliable single parser)
        if "spacy" in self.providers:
            try:
                result = await self._parse_with_provider("spacy", text, media_context)
                if result and result.success:
                    expressions = result.enhanced_data.get("expanded_concepts", [])
                    all_expressions.extend(expressions)
            except Exception as e:
                self._logger.warning(f"SpaCy temporal parsing failed: {e}")
        
        # Add intelligent expansion for top expressions
        for expr in all_expressions[:3]:
            try:
                temporal_request = TemporalConceptRequest(
                    temporal_term=expr,
                    media_context=media_context,
                    max_concepts=5
                )
                intel_result = await self.temporal_generator.generate_temporal_concepts(
                    temporal_request
                )
                if intel_result and intel_result.success:
                    concepts = intel_result.enhanced_data.get("temporal_concepts", [])
                    all_normalized.extend(concepts)
            except Exception as e:
                self._logger.warning(f"Temporal intelligence failed: {e}")
        
        return {
            "expressions": all_expressions,
            "normalized": list(dict.fromkeys(all_normalized))[:10],
            "analysis_methods": ["spacy", "temporal_intelligence"],
            "temporal_scope": self._determine_temporal_scope(all_expressions),
            "confidence_level": "medium"
        }
    
    async def _low_resource_temporal_analysis(
        self,
        text: str,
        media_context: str,
        context: PluginExecutionContext
    ) -> Dict[str, Any]:
        """
        Low resource: Single parser, no intelligence.
        """
        all_expressions = []
        
        # Try SpaCy first (most reliable)
        parser_used = None
        for parser_name in ["spacy", "sutime", "heideltime"]:
            if parser_name in self.providers:
                try:
                    result = await self._parse_with_provider(
                        parser_name, text, media_context
                    )
                    if result and result.success:
                        expressions = result.enhanced_data.get("expanded_concepts", [])
                        all_expressions.extend(expressions)
                        parser_used = parser_name
                        break
                except Exception as e:
                    self._logger.warning(f"{parser_name} failed: {e}")
        
        return {
            "expressions": all_expressions[:10],
            "normalized": [],  # No normalization in low resource
            "analysis_methods": [parser_used] if parser_used else [],
            "temporal_scope": self._determine_temporal_scope(all_expressions),
            "confidence_level": "low"
        }
    
    async def _parse_with_provider(
        self,
        provider_name: str,
        text: str,
        media_context: str
    ) -> Optional[PluginResult]:
        """Parse temporal expressions using a specific provider."""
        provider = self.providers.get(provider_name)
        if not provider:
            return None
        
        try:
            # Temporal providers expect the text as the concept
            request = ExpansionRequest(
                concept=text,
                media_context=media_context,
                max_concepts=20  # More for temporal parsing
            )
            
            # Use cache
            cache_key = self.cache_manager.generate_cache_key(
                cache_type=self._provider_to_cache_type(provider_name),
                field_name="temporal_text",
                input_value=text[:100],  # First 100 chars as key
                media_context=media_context
            )
            
            result = await self.cache_manager.get_or_compute(
                cache_key=cache_key,
                compute_func=lambda: provider.expand_concept(request),
                strategy=CacheStrategy.CACHE_FIRST
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Temporal parser '{provider_name}' failed: {e}")
            return None
    
    def _determine_temporal_scope(self, expressions: List[str]) -> str:
        """Determine the temporal scope from expressions."""
        if not expressions:
            return "unknown"
        
        # Check for different temporal patterns
        has_decade = any("decade" in expr.lower() or "0s" in expr for expr in expressions)
        has_year = any(expr.isdigit() and len(expr) == 4 for expr in expressions)
        has_relative = any(
            word in expr.lower() 
            for expr in expressions 
            for word in ["recent", "last", "ago", "current"]
        )
        
        if has_decade:
            return "decade"
        elif has_year:
            return "year"
        elif has_relative:
            return "relative"
        else:
            return "general"
    
    def _create_temporal_search_tags(self, temporal_data: Dict[str, Any]) -> List[str]:
        """Create searchable tags from temporal analysis."""
        tags = []
        
        # Add scope tag
        scope = temporal_data.get("temporal_scope", "unknown")
        tags.append(f"temporal_scope_{scope}")
        
        # Add expression-based tags
        for expr in temporal_data.get("expressions", [])[:10]:
            # Clean expression for tag
            clean_expr = expr.lower().replace(" ", "_").replace("-", "_")
            tags.append(f"temporal_{clean_expr}")
        
        # Add normalized concept tags
        for concept in temporal_data.get("normalized", [])[:5]:
            clean_concept = concept.lower().replace(" ", "_")
            tags.append(f"temporal_concept_{clean_concept}")
        
        # Add analysis method tags
        for method in temporal_data.get("analysis_methods", []):
            tags.append(f"temporal_method_{method}")
        
        return list(set(tags))  # Deduplicate
    
    def _provider_to_cache_type(self, provider_name: str) -> CacheType:
        """Map provider name to cache type."""
        mapping = {
            "spacy": CacheType.SPACY_TEMPORAL,
            "heideltime": CacheType.HEIDELTIME,
            "sutime": CacheType.SUTIME
        }
        return mapping.get(provider_name, CacheType.CUSTOM)