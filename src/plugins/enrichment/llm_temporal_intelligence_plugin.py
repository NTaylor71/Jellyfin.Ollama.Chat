"""
LLM Temporal Intelligence Plugin
HTTP-only plugin that calls LLM service for temporal concept understanding.
"""

import logging
from typing import Dict, Any, List

from src.plugins.http_base import HTTPBasePlugin
from src.plugins.base import PluginMetadata, PluginResourceRequirements, PluginType, ExecutionPriority

logger = logging.getLogger(__name__)


class LLMTemporalIntelligencePlugin(HTTPBasePlugin):
    """
    HTTP-only plugin that extracts temporal concepts using ONLY LLM.
    
    Features:
    - Calls LLM service endpoint for temporal reasoning
    - Understands implicit temporal relationships
    - Contextual temporal concept extraction
    - No provider management - just HTTP calls
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="LLMTemporalIntelligencePlugin",
            version="1.0.0",
            description="Extracts temporal concepts using LLM reasoning",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["temporal", "intelligence", "llm", "reasoning", "concepts"],
            execution_priority=ExecutionPriority.HIGH  # LLM is expensive, prioritize
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        """LLM temporal intelligence plugins have higher resource requirements due to model inference."""
        return PluginResourceRequirements(
            min_cpu_cores=1.0,
            preferred_cpu_cores=2.0,
            min_memory_mb=512.0,
            preferred_memory_mb=2048.0,
            requires_gpu=True,  # LLM inference benefits significantly from GPU
            min_gpu_memory_mb=2048.0,
            preferred_gpu_memory_mb=8192.0,
            max_execution_time_seconds=90.0  # Temporal reasoning can be complex
        )
    
    async def enrich_field(
        self, 
        field_name: str, 
        field_value: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich a field with LLM temporal intelligence.
        
        Args:
            field_name: Name of the field being enriched
            field_value: Value of the field
            config: Plugin configuration
            
        Returns:
            Dict containing LLM temporal intelligence results
        """
        try:
            # Convert field value to text
            if isinstance(field_value, str):
                text = field_value
            elif isinstance(field_value, list):
                text = " ".join(str(item) for item in field_value)
            else:
                text = str(field_value)
            
            if not text.strip():
                self._logger.debug(f"No text found in field {field_name}")
                result = {"llm_temporal": []}
                # Normalize all Unicode text in the result
                return self.normalize_text(result)
            
            self._logger.debug(f"Extracting temporal concepts from {len(text)} characters using LLM")
            
            # Call LLM service (using general expand endpoint for temporal analysis)
            # LLM expand endpoint expects LLMRequest format
            service_url = await self.get_plugin_service_url()
            request_data = {
                "concept": text,
                "media_context": "movie",
                "max_concepts": config.get("max_concepts", 15),
                "field_name": field_name,
                "options": {
                    "task_type": "temporal_intelligence",
                    "analysis_type": config.get("analysis_type", "comprehensive"),
                    "extract_periods": config.get("extract_periods", True),
                    "extract_sequences": config.get("extract_sequences", True),
                    "extract_relationships": config.get("extract_relationships", True),
                    "reasoning_depth": config.get("reasoning_depth", "medium"),
                    "temperature": config.get("temperature", 0.2),  # Lower for more consistent results
                    "context_window": config.get("context_window", "full")
                }
            }
            
            response = await self.http_post(service_url, request_data)
            
            # Process response from LLMRequest format
            if response.get("success", False):
                result_data = response.get("result", {})
                # LLM service returns temporal data in expanded_concepts
                temporal_concepts = result_data.get("expanded_concepts", 
                                  result_data.get("temporal_concepts", 
                                  result_data.get("concepts", [])))
                metadata = response.get("metadata", {})
            else:
                temporal_concepts = []
                metadata = {"error": response.get("error_message", "Unknown error")}
            
            self._logger.info(
                f"LLM extracted {len(temporal_concepts)} temporal concepts from field {field_name}"
            )
            
            result = {
                "llm_temporal": temporal_concepts,
                "original_text": text[:200] + "..." if len(text) > 200 else text,
                "field_name": field_name,
                "metadata": {
                    "provider": "llm_temporal",
                    "concept_count": len(temporal_concepts),
                    "text_length": len(text),
                    "analysis_type": request_data["options"]["analysis_type"],
                    "reasoning_depth": request_data["options"]["reasoning_depth"],
                    "service_metadata": metadata
                }
            }
            
            # Normalize all Unicode text in the result
            return self.normalize_text(result)
            
        except Exception as e:
            self._logger.error(f"LLM temporal intelligence failed for field {field_name}: {e}")
            # Return empty result on error
            result = {
                "llm_temporal": [],
                "original_text": "",
                "field_name": field_name,
                "error": str(e),
                "metadata": {
                    "provider": "llm_temporal",
                    "success": False
                }
            }
            
            # Normalize all Unicode text in the result
            return self.normalize_text(result)
    
    async def analyze_temporal_concepts(
        self, 
        text: str, 
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Direct temporal concept analysis method for backwards compatibility.
        
        Args:
            text: Text to analyze for temporal concepts
            config: Analysis configuration
            
        Returns:
            Temporal concept analysis results
        """
        if config is None:
            config = {}
            
        # Use enrich_field with dummy field info
        result = await self.enrich_field("text", text, config)
        response = {
            "temporal_concepts": result.get("llm_temporal", []),
            "metadata": result.get("metadata", {})
        }
        
        # Normalize all Unicode text in the result
        return self.normalize_text(response)
    
    async def extract_temporal_periods(
        self,
        text: str,
        period_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Extract specific types of temporal periods from text.
        
        Args:
            text: Text to analyze
            period_types: Types of periods to extract (historical, seasonal, lifecycle, etc.)
            
        Returns:
            Temporal period extraction results
        """
        if period_types is None:
            period_types = ["historical", "seasonal", "lifecycle", "narrative"]
        
        config = {
            "analysis_type": "periods",
            "extract_periods": True,
            "extract_sequences": False,
            "extract_relationships": False,
            "period_types": period_types,
            "reasoning_depth": "deep"
        }
        
        result = await self.enrich_field("periods", text, config)
        
        # Filter for period concepts
        temporal_concepts = result.get("llm_temporal", [])
        period_concepts = [
            concept for concept in temporal_concepts
            if concept.get("type") in ["period", "era", "timeframe"] or
               any(ptype in str(concept.get("description", "")).lower() for ptype in period_types)
        ]
        
        response = {
            "temporal_periods": period_concepts,
            "period_types": period_types,
            "total_concepts": len(temporal_concepts),
            "period_count": len(period_concepts),
            "metadata": result.get("metadata", {})
        }
        
        # Normalize all Unicode text in the result
        return self.normalize_text(response)
    
    async def analyze_temporal_sequences(
        self,
        text: str,
        sequence_type: str = "narrative"
    ) -> Dict[str, Any]:
        """
        Analyze temporal sequences and progressions in text.
        
        Args:
            text: Text to analyze for sequences
            sequence_type: Type of sequence (narrative, process, historical, causal)
            
        Returns:
            Temporal sequence analysis results
        """
        config = {
            "analysis_type": "sequences",
            "extract_periods": False,
            "extract_sequences": True,
            "extract_relationships": True,
            "sequence_type": sequence_type,
            "reasoning_depth": "deep"
        }
        
        result = await self.enrich_field("sequences", text, config)
        
        response = {
            "temporal_sequences": result.get("llm_temporal", []),
            "sequence_type": sequence_type,
            "metadata": result.get("metadata", {})
        }
        
        # Normalize all Unicode text in the result
        return self.normalize_text(response)
    
    async def extract_temporal_relationships(
        self,
        text: str,
        relationship_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Extract temporal relationships between events or concepts.
        
        Args:
            text: Text to analyze for relationships
            relationship_types: Types of relationships (before, after, during, caused_by, etc.)
            
        Returns:
            Temporal relationship extraction results
        """
        if relationship_types is None:
            relationship_types = ["before", "after", "during", "while", "caused_by", "leads_to"]
        
        config = {
            "analysis_type": "relationships",
            "extract_periods": False,
            "extract_sequences": False,
            "extract_relationships": True,
            "relationship_types": relationship_types,
            "reasoning_depth": "deep"
        }
        
        result = await self.enrich_field("relationships", text, config)
        
        response = {
            "temporal_relationships": result.get("llm_temporal", []),
            "relationship_types": relationship_types,
            "metadata": result.get("metadata", {})
        }
        
        # Normalize all Unicode text in the result
        return self.normalize_text(response)
    
    async def comprehensive_temporal_analysis(
        self,
        text: str,
        context: str = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive temporal analysis including all aspects.
        
        Args:
            text: Text to analyze comprehensively
            context: Additional context for better analysis
            
        Returns:
            Comprehensive temporal analysis results
        """
        config = {
            "analysis_type": "comprehensive",
            "extract_periods": True,
            "extract_sequences": True,
            "extract_relationships": True,
            "reasoning_depth": "deep",
            "context": context,
            "include_implicit": True,
            "include_inferred": True
        }
        
        result = await self.enrich_field("comprehensive", text, config)
        temporal_concepts = result.get("llm_temporal", [])
        
        # Categorize concepts by type
        periods = [c for c in temporal_concepts if c.get("category") == "period"]
        sequences = [c for c in temporal_concepts if c.get("category") == "sequence"]
        relationships = [c for c in temporal_concepts if c.get("category") == "relationship"]
        implicit = [c for c in temporal_concepts if c.get("implicit", False)]
        
        response = {
            "comprehensive_analysis": {
                "temporal_periods": periods,
                "temporal_sequences": sequences,
                "temporal_relationships": relationships,
                "implicit_temporal": implicit,
                "all_concepts": temporal_concepts
            },
            "analysis_summary": {
                "total_concepts": len(temporal_concepts),
                "periods_found": len(periods),
                "sequences_found": len(sequences),
                "relationships_found": len(relationships),
                "implicit_found": len(implicit)
            },
            "context": context,
            "metadata": result.get("metadata", {})
        }
        
        # Normalize all Unicode text in the result
        return self.normalize_text(response)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin and service health."""
        base_health = await super().health_check()
        
        try:
            # Test LLM temporal service connectivity
            service_url = self.get_service_url("llm", "health")
            health_response = await self.http_get(service_url)
            
            base_health["service_health"] = {
                "llm_temporal_service": "healthy",
                "service_response": health_response
            }
            
        except Exception as e:
            base_health["service_health"] = {
                "llm_temporal_service": "unhealthy",
                "error": str(e)
            }
        
        return base_health