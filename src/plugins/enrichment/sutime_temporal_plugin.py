"""
SUTime Temporal Plugin
HTTP-only plugin that calls SUTime service for temporal information extraction.
"""

import logging
from typing import Dict, Any, List

from src.plugins.http_base import HTTPBasePlugin
from src.plugins.base import PluginMetadata, PluginType, ExecutionPriority

logger = logging.getLogger(__name__)


class SUTimeTemporalPlugin(HTTPBasePlugin):
    """
    HTTP-only plugin that extracts temporal information using ONLY SUTime.
    
    Features:
    - Calls SUTime service endpoint for temporal recognition
    - Stanford NLP-based temporal expression recognition
    - Robust temporal normalization and resolution
    - No provider management - just HTTP calls
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="SUTimeTemporalPlugin",
            version="1.0.0",
            description="Extracts temporal information using Stanford SUTime",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["temporal", "sutime", "stanford", "dates", "normalization"],
            execution_priority=ExecutionPriority.NORMAL
        )
    
    async def enrich_field(
        self, 
        field_name: str, 
        field_value: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich a field with SUTime temporal extraction.
        
        Args:
            field_name: Name of the field being enriched
            field_value: Value of the field
            config: Plugin configuration
            
        Returns:
            Dict containing SUTime temporal extraction results
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
                return {"sutime_temporal": []}
            
            self._logger.debug(f"Extracting temporal info from {len(text)} characters using SUTime")
            
            # Call SUTime provider via dedicated SUTime service  
            # SUTime provider expects ProviderRequest format
            service_url = self.get_plugin_service_url()
            request_data = {
                "concept": text,
                "media_context": "movie",
                "max_concepts": config.get("max_concepts", 20),
                "field_name": field_name,
                "options": {
                    "reference_date": config.get("reference_date", None),
                    "include_nested": config.get("include_nested", True),
                    "include_range": config.get("include_range", True),
                    "resolve_relative": config.get("resolve_relative", True),
                    "timezone": config.get("timezone", "UTC"),
                    "output_format": config.get("output_format", "json")
                }
            }
            
            response = await self.http_post(service_url, request_data)
            
            # Process response from ProviderResponse format
            if response.get("success", False):
                result_data = response.get("result", {})
                # SUTime provider returns expanded_concepts containing temporal annotations
                temporal_annotations = result_data.get("expanded_concepts", [])
                metadata = response.get("metadata", {})
                metadata.update(result_data.get("analysis", {}))
            else:
                temporal_annotations = []
                metadata = {"error": response.get("error_message", "Unknown error")}
            
            self._logger.info(
                f"SUTime extracted {len(temporal_annotations)} temporal annotations from field {field_name}"
            )
            
            return {
                "sutime_temporal": temporal_annotations,
                "original_text": text[:200] + "..." if len(text) > 200 else text,
                "field_name": field_name,
                "metadata": {
                    "provider": "sutime",
                    "annotation_count": len(temporal_annotations),
                    "text_length": len(text),
                    "reference_date": request_data["options"]["reference_date"],
                    "timezone": request_data["options"]["timezone"],
                    "service_metadata": metadata
                }
            }
            
        except Exception as e:
            self._logger.error(f"SUTime temporal extraction failed for field {field_name}: {e}")
            # Return empty result on error
            return {
                "sutime_temporal": [],
                "original_text": "",
                "field_name": field_name,
                "error": str(e),
                "metadata": {
                    "provider": "sutime",
                    "success": False
                }
            }
    
    async def extract_temporal_annotations(
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
            "temporal_annotations": result.get("sutime_temporal", []),
            "metadata": result.get("metadata", {})
        }
    
    async def extract_with_reference_date(
        self,
        text: str,
        reference_date: str,
        timezone: str = "UTC"
    ) -> Dict[str, Any]:
        """
        Extract temporal expressions with specific reference date.
        
        Args:
            text: Text to extract temporal information from
            reference_date: Reference date for relative expressions (ISO format)
            timezone: Timezone for interpretation
            
        Returns:
            Temporal extraction results
        """
        config = {
            "reference_date": reference_date,
            "timezone": timezone,
            "resolve_relative": True,
            "include_range": True
        }
        
        result = await self.enrich_field("text", text, config)
        
        return {
            "temporal_annotations": result.get("sutime_temporal", []),
            "reference_date": reference_date,
            "timezone": timezone,
            "metadata": result.get("metadata", {})
        }
    
    async def extract_time_ranges(
        self,
        text: str,
        include_nested: bool = True
    ) -> Dict[str, Any]:
        """
        Extract temporal ranges and durations from text.
        
        Args:
            text: Text to process for time ranges
            include_nested: Whether to include nested temporal expressions
            
        Returns:
            Time range extraction results
        """
        config = {
            "include_nested": include_nested,
            "include_range": True,
            "resolve_relative": True,
            "output_format": "json"
        }
        
        result = await self.enrich_field("ranges", text, config)
        
        # Filter for range/duration expressions
        temporal_annotations = result.get("sutime_temporal", [])
        range_annotations = [
            annotation for annotation in temporal_annotations
            if annotation.get("type") in ["DURATION", "SET", "TIME"] or 
               "range" in str(annotation.get("value", "")).lower()
        ]
        
        return {
            "time_ranges": range_annotations,
            "total_annotations": len(temporal_annotations),
            "range_count": len(range_annotations),
            "metadata": result.get("metadata", {})
        }
    
    async def normalize_temporal_expressions(
        self,
        text: str,
        reference_date: str = None,
        output_format: str = "iso"
    ) -> Dict[str, Any]:
        """
        Extract and normalize temporal expressions to standard format.
        
        Args:
            text: Text containing temporal expressions
            reference_date: Reference date for normalization
            output_format: Output format (iso, timex3, json)
            
        Returns:
            Normalized temporal information
        """
        config = {
            "reference_date": reference_date,
            "resolve_relative": True,
            "output_format": output_format,
            "include_nested": True,
            "include_range": True
        }
        
        result = await self.enrich_field("text", text, config)
        
        return {
            "normalized_temporal": result.get("sutime_temporal", []),
            "output_format": output_format,
            "reference_date": reference_date,
            "metadata": result.get("metadata", {})
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin and service health."""
        base_health = await super().health_check()
        
        try:
            # Test SUTime temporal service connectivity
            service_url = self.get_service_url("temporal", "health")
            health_response = await self.http_get(service_url)
            
            base_health["service_health"] = {
                "sutime_temporal_service": "healthy",
                "service_response": health_response
            }
            
        except Exception as e:
            base_health["service_health"] = {
                "sutime_temporal_service": "unhealthy",
                "error": str(e)
            }
        
        return base_health