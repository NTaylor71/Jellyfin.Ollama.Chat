"""
HeidelTime Temporal Plugin
HTTP-only plugin that calls HeidelTime service for temporal information extraction.
"""

import logging
from typing import Dict, Any, List

from src.plugins.http_base import HTTPBasePlugin
from src.plugins.base import PluginMetadata, PluginType, ExecutionPriority

logger = logging.getLogger(__name__)


class HeidelTimeTemporalPlugin(HTTPBasePlugin):
    """
    HTTP-only plugin that extracts temporal information using ONLY HeidelTime.
    
    Features:
    - Calls HeidelTime service endpoint for temporal tagging
    - Rule-based temporal expression recognition
    - High precision temporal normalization
    - No provider management - just HTTP calls
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="HeidelTimeTemporalPlugin",
            version="1.0.0",
            description="Extracts temporal information using HeidelTime rule-based system",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["temporal", "heideltime", "dates", "normalization", "rules"],
            execution_priority=ExecutionPriority.NORMAL
        )
    
    async def enrich_field(
        self, 
        field_name: str, 
        field_value: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich a field with HeidelTime temporal extraction.
        
        Args:
            field_name: Name of the field being enriched
            field_value: Value of the field
            config: Plugin configuration
            
        Returns:
            Dict containing HeidelTime temporal extraction results
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
                return {"heideltime_temporal": []}
            
            self._logger.debug(f"Extracting temporal info from {len(text)} characters using HeidelTime")
            
            # Call HeidelTime provider via NLP service
            # HeidelTime provider expects ProviderRequest format
            service_url = self.get_plugin_service_url()
            request_data = {
                "concept": text,
                "media_context": "movie",
                "max_concepts": config.get("max_concepts", 20),
                "field_name": field_name,
                "options": {
                    "document_type": config.get("document_type", "news"),  # news, narrative, colloquial, scientific
                    "document_creation_time": config.get("document_creation_time", None),
                    "language": config.get("language", "english"),
                    "output_format": config.get("output_format", "timeml"),
                    "normalize_timex": config.get("normalize_timex", True),
                    "extract_temporal_relations": config.get("extract_temporal_relations", False)
                }
            }
            
            response = await self.http_post(service_url, request_data)
            
            # Process response from ProviderResponse format
            if response.get("success", False):
                result_data = response.get("result", {})
                temporal_expressions = result_data.get("temporal_expressions", result_data.get("expressions", []))
                metadata = response.get("metadata", {})
            else:
                temporal_expressions = []
                metadata = {"error": response.get("error_message", "Unknown error")}
            
            self._logger.info(
                f"HeidelTime extracted {len(temporal_expressions)} temporal expressions from field {field_name}"
            )
            
            return {
                "heideltime_temporal": temporal_expressions,
                "original_text": text[:200] + "..." if len(text) > 200 else text,
                "field_name": field_name,
                "metadata": {
                    "provider": "heideltime",
                    "expression_count": len(temporal_expressions),
                    "text_length": len(text),
                    "document_type": request_data["options"]["document_type"],
                    "language": request_data["options"]["language"],
                    "service_metadata": metadata
                }
            }
            
        except Exception as e:
            self._logger.error(f"HeidelTime temporal extraction failed for field {field_name}: {e}")
            # Return empty result on error
            return {
                "heideltime_temporal": [],
                "original_text": "",
                "field_name": field_name,
                "error": str(e),
                "metadata": {
                    "provider": "heideltime",
                    "success": False
                }
            }
    
    async def extract_temporal_expressions(
        self, 
        text: str, 
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Direct temporal extraction method for backwards compatibility.
        
        Args:
            text: Text to extract temporal information from
            config: Extraction configuration
            
        Returns:
            Temporal extraction results
        """
        if config is None:
            config = {}
            
        # Use enrich_field with dummy field info
        result = await self.enrich_field("text", text, config)
        return {
            "temporal_expressions": result.get("heideltime_temporal", []),
            "metadata": result.get("metadata", {})
        }
    
    async def extract_with_document_time(
        self,
        text: str,
        document_creation_time: str,
        document_type: str = "news"
    ) -> Dict[str, Any]:
        """
        Extract temporal expressions with specific document creation time.
        
        Args:
            text: Text to extract temporal information from
            document_creation_time: When the document was created (ISO format)
            document_type: Type of document (news, narrative, colloquial, scientific)
            
        Returns:
            Temporal extraction results
        """
        config = {
            "document_creation_time": document_creation_time,
            "document_type": document_type,
            "normalize_timex": True
        }
        
        result = await self.enrich_field("text", text, config)
        
        return {
            "temporal_expressions": result.get("heideltime_temporal", []),
            "document_creation_time": document_creation_time,
            "document_type": document_type,
            "metadata": result.get("metadata", {})
        }
    
    async def extract_for_narrative(
        self,
        text: str,
        reference_time: str = None
    ) -> Dict[str, Any]:
        """
        Extract temporal expressions optimized for narrative text.
        
        Args:
            text: Narrative text to process
            reference_time: Reference time for relative expressions
            
        Returns:
            Temporal extraction results for narrative
        """
        config = {
            "document_type": "narrative",
            "document_creation_time": reference_time,
            "extract_temporal_relations": True,
            "normalize_timex": True
        }
        
        result = await self.enrich_field("narrative", text, config)
        
        return {
            "narrative_temporal": result.get("heideltime_temporal", []),
            "reference_time": reference_time,
            "metadata": result.get("metadata", {})
        }
    
    async def extract_scientific_temporal(
        self,
        text: str,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """
        Extract temporal expressions from scientific text.
        
        Args:
            text: Scientific text to process
            normalize: Whether to normalize temporal expressions
            
        Returns:
            Scientific temporal extraction results
        """
        config = {
            "document_type": "scientific",
            "normalize_timex": normalize,
            "language": "english",
            "output_format": "timeml"
        }
        
        result = await self.enrich_field("scientific", text, config)
        
        return {
            "scientific_temporal": result.get("heideltime_temporal", []),
            "normalized": normalize,
            "metadata": result.get("metadata", {})
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin and service health."""
        base_health = await super().health_check()
        
        try:
            # Test HeidelTime temporal service connectivity
            service_url = self.get_service_url("temporal", "health")
            health_response = await self.http_get(service_url)
            
            base_health["service_health"] = {
                "heideltime_temporal_service": "healthy",
                "service_response": health_response
            }
            
        except Exception as e:
            base_health["service_health"] = {
                "heideltime_temporal_service": "unhealthy",
                "error": str(e)
            }
        
        return base_health