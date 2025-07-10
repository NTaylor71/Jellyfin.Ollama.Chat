"""
SpaCy Temporal Plugin
HTTP-only plugin that calls SpaCy service for temporal information extraction.
"""

import logging
from typing import Dict, Any, List

from src.plugins.http_base import HTTPBasePlugin
from src.plugins.base import PluginMetadata, PluginType, ExecutionPriority

logger = logging.getLogger(__name__)


class SpacyTemporalPlugin(HTTPBasePlugin):
    """
    HTTP-only plugin that extracts temporal information using ONLY SpaCy.
    
    Features:
    - Calls SpaCy service endpoint for temporal entity recognition
    - Extracts dates, times, durations from text
    - Named entity recognition for temporal expressions
    - No provider management - just HTTP calls
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="SpacyTemporalPlugin",
            version="1.0.0",
            description="Extracts temporal information using SpaCy NLP",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["temporal", "dates", "spacy", "nlp", "entities"],
            execution_priority=ExecutionPriority.NORMAL
        )
    
    async def enrich_field(
        self, 
        field_name: str, 
        field_value: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich a field with SpaCy temporal extraction.
        
        Args:
            field_name: Name of the field being enriched
            field_value: Value of the field
            config: Plugin configuration
            
        Returns:
            Dict containing SpaCy temporal extraction results
        """
        try:

            if isinstance(field_value, str):
                text = field_value
            elif isinstance(field_value, list):
                text = " ".join(str(item) for item in field_value)
            else:
                text = str(field_value)
            
            if not text.strip():
                self._logger.debug(f"No text found in field {field_name}")
                result = {"spacy_temporal": []}
                
                return self.normalize_text(result)
            
            self._logger.debug(f"Extracting temporal info from {len(text)} characters using SpaCy")
            
            

            service_url = await self.get_plugin_service_url()
            request_data = {
                "concept": text,
                "media_context": "movie",
                "max_concepts": config.get("max_concepts", 20),
                "field_name": field_name,
                "options": {
                    "extract_dates": config.get("extract_dates", True),
                    "extract_times": config.get("extract_times", True),
                    "extract_durations": config.get("extract_durations", True),
                    "entity_types": config.get("entity_types", ["DATE", "TIME", "DURATION"]),
                    "normalize_dates": config.get("normalize_dates", True),
                    "reference_date": config.get("reference_date", None)
                }
            }
            
            response = await self.http_post(service_url, request_data)
            
            
            if response.get("success", False):
                result_data = response.get("result", {})

                temporal_entities = result_data.get("expanded_concepts", [])
                metadata = response.get("metadata", {})
                metadata.update(result_data.get("analysis", {}))
            else:
                temporal_entities = []
                metadata = {"error": response.get("error_message", "Unknown error")}
            
            self._logger.info(
                f"SpaCy extracted {len(temporal_entities)} temporal entities from field {field_name}"
            )
            
            result = {
                "spacy_temporal": temporal_entities,
                "original_text": text[:200] + "..." if len(text) > 200 else text,
                "field_name": field_name,
                "metadata": {
                    "provider": "spacy",
                    "entity_count": len(temporal_entities),
                    "text_length": len(text),
                    "service_metadata": metadata,
                    "extraction_types": request_data["options"]["entity_types"]
                }
            }
            
            
            return self.normalize_text(result)
            
        except Exception as e:
            self._logger.error(f"SpaCy temporal extraction failed for field {field_name}: {e}")
            
            result = {
                "spacy_temporal": [],
                "original_text": "",
                "field_name": field_name,
                "error": str(e),
                "metadata": {
                    "provider": "spacy",
                    "success": False
                }
            }
            
            
            return self.normalize_text(result)
    
    async def extract_temporal_entities(
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
            
        
        result = await self.enrich_field("text", text, config)
        response = {
            "temporal_entities": result.get("spacy_temporal", []),
            "metadata": result.get("metadata", {})
        }
        
        
        return self.normalize_text(response)
    
    async def extract_dates_only(
        self,
        text: str,
        reference_date: str = None
    ) -> Dict[str, Any]:
        """
        Extract only date entities from text.
        
        Args:
            text: Text to extract dates from
            reference_date: Reference date for relative dates
            
        Returns:
            Date extraction results
        """
        config = {
            "extract_dates": True,
            "extract_times": False,
            "extract_durations": False,
            "entity_types": ["DATE"],
            "reference_date": reference_date
        }
        
        result = await self.enrich_field("text", text, config)
        

        temporal_entities = result.get("spacy_temporal", [])
        date_entities = [
            entity for entity in temporal_entities 
            if entity.get("label") == "DATE" or entity.get("type") == "date"
        ]
        
        response = {
            "dates": date_entities,
            "metadata": result.get("metadata", {})
        }
        
        
        return self.normalize_text(response)
    
    async def normalize_temporal_expressions(
        self,
        text: str,
        reference_date: str = None
    ) -> Dict[str, Any]:
        """
        Extract and normalize temporal expressions.
        
        Args:
            text: Text containing temporal expressions
            reference_date: Reference date for normalization
            
        Returns:
            Normalized temporal information
        """
        config = {
            "normalize_dates": True,
            "reference_date": reference_date,
            "extract_dates": True,
            "extract_times": True,
            "extract_durations": True
        }
        
        result = await self.enrich_field("text", text, config)
        
        response = {
            "normalized_temporal": result.get("spacy_temporal", []),
            "reference_date": reference_date,
            "metadata": result.get("metadata", {})
        }
        
        
        return self.normalize_text(response)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin and service health."""
        base_health = await super().health_check()
        
        try:
            
            service_url = self.get_service_url("temporal", "health")
            health_response = await self.http_get(service_url)
            
            base_health["service_health"] = {
                "spacy_temporal_service": "healthy",
                "service_response": health_response
            }
            
        except Exception as e:
            base_health["service_health"] = {
                "spacy_temporal_service": "unhealthy",
                "error": str(e)
            }
        
        return base_health