"""
SpaCy Named Entity Recognition Plugin
HTTP-only plugin that calls SpaCy service for comprehensive entity extraction.
"""

import logging
from typing import Dict, Any, List

from src.plugins.http_base import HTTPBasePlugin
from src.plugins.base import PluginMetadata, PluginType, ExecutionPriority

logger = logging.getLogger(__name__)


class SpacyNERPlugin(HTTPBasePlugin):
    """
    HTTP-only plugin that extracts named entities using SpaCy.
    
    Features:
    - Comprehensive NER for all entity types (PERSON, ORG, GPE, WORK_OF_ART, etc.)
    - Structured entity organization by category
    - Confidence scoring for entities
    - Support for multiple spaCy models
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="SpacyNERPlugin",
            version="1.0.0",
            description="Comprehensive named entity recognition using SpaCy NLP",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["ner", "entities", "spacy", "nlp", "extraction"],
            execution_priority=ExecutionPriority.HIGH
        )
    
    async def enrich_field(
        self, 
        field_name: str, 
        field_value: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich a field with comprehensive SpaCy NER.
        
        Args:
            field_name: Name of the field being enriched
            field_value: Value of the field
            config: Plugin configuration
            
        Returns:
            Dict containing structured entity extraction results
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
                return self._create_empty_result(field_name)
            
            self._logger.debug(f"Extracting entities from {len(text)} characters using SpaCy NER")
            
            service_url = await self.get_plugin_service_url()
            request_data = {
                "text": text,
                "field_name": field_name,
                "options": {
                    "model": config.get("model", "en_core_web_sm"),
                    "entity_types": config.get("entity_types", "ALL"),
                    "confidence_threshold": config.get("confidence_threshold", 0.7),
                    "include_lemmas": config.get("include_lemmas", True),
                    "group_by_type": config.get("group_by_type", True),
                    "extract_noun_chunks": config.get("extract_noun_chunks", True)
                }
            }
            
            response = await self.http_post(service_url, request_data)
            
            if response.get("success", False):
                result_data = response.get("result", {})
                entities = result_data.get("entities", {})
                metadata = response.get("metadata", {})
                
                # Structure entities by type
                structured_entities = self._structure_entities(entities)
                
                result = {
                    "spacy_entities": structured_entities,
                    "entity_summary": self._create_entity_summary(structured_entities),
                    "original_text": text[:200] + "..." if len(text) > 200 else text,
                    "field_name": field_name,
                    "metadata": {
                        "provider": "spacy_ner",
                        "model_used": request_data["options"]["model"],
                        "total_entities": sum(len(ents) for ents in structured_entities.values()),
                        "text_length": len(text),
                        "service_metadata": metadata,
                        "extraction_config": request_data["options"]
                    }
                }
            else:
                result = self._create_error_result(
                    field_name, 
                    response.get("error_message", "Unknown error"),
                    text
                )
            
            self._logger.info(
                f"SpaCy NER extracted entities for field {field_name}: "
                f"{result['metadata']['total_entities']} total entities"
            )
            
            return self.normalize_text(result)
            
        except Exception as e:
            self._logger.error(f"SpaCy NER extraction failed for field {field_name}: {e}")
            return self.normalize_text(self._create_error_result(field_name, str(e)))
    
    def _structure_entities(self, entities: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Structure entities by type for better organization."""
        structured = {
            "people": [],
            "organizations": [],
            "locations": [],
            "works_of_art": [],
            "events": [],
            "products": [],
            "dates": [],
            "money": [],
            "quantities": [],
            "languages": [],
            "other": []
        }
        
        # Entity type mapping
        type_mapping = {
            "PERSON": "people",
            "ORG": "organizations", 
            "GPE": "locations",
            "LOC": "locations",
            "FAC": "locations",
            "WORK_OF_ART": "works_of_art",
            "EVENT": "events",
            "PRODUCT": "products",
            "DATE": "dates",
            "TIME": "dates",
            "MONEY": "money",
            "QUANTITY": "quantities",
            "PERCENT": "quantities",
            "LANGUAGE": "languages"
        }
        
        for entity_type, entity_list in entities.items():
            category = type_mapping.get(entity_type, "other")
            for entity in entity_list:
                structured[category].append({
                    "text": entity.get("text", ""),
                    "label": entity.get("label", entity_type),
                    "confidence": entity.get("confidence", 0.0),
                    "start": entity.get("start", 0),
                    "end": entity.get("end", 0),
                    "lemma": entity.get("lemma", entity.get("text", ""))
                })
        
        # Remove empty categories
        return {k: v for k, v in structured.items() if v}
    
    def _create_entity_summary(self, structured_entities: Dict[str, List]) -> Dict[str, Any]:
        """Create a summary of extracted entities."""
        summary = {}
        for category, entities in structured_entities.items():
            if entities:
                summary[category] = {
                    "count": len(entities),
                    "top_entities": [e["text"] for e in entities[:3]],
                    "confidence_avg": sum(e["confidence"] for e in entities) / len(entities)
                }
        return summary
    
    def _create_empty_result(self, field_name: str) -> Dict[str, Any]:
        """Create empty result structure."""
        return {
            "spacy_entities": {},
            "entity_summary": {},
            "original_text": "",
            "field_name": field_name,
            "metadata": {
                "provider": "spacy_ner",
                "total_entities": 0,
                "success": True
            }
        }
    
    def _create_error_result(self, field_name: str, error: str, text: str = "") -> Dict[str, Any]:
        """Create error result structure."""
        return {
            "spacy_entities": {},
            "entity_summary": {},
            "original_text": text[:200] + "..." if len(text) > 200 else text,
            "field_name": field_name,
            "error": error,
            "metadata": {
                "provider": "spacy_ner",
                "success": False,
                "total_entities": 0
            }
        }
    
    async def extract_people(
        self, 
        text: str, 
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract only people entities from text.
        
        Args:
            text: Text to extract people from
            config: Extraction configuration
            
        Returns:
            People extraction results
        """
        if config is None:
            config = {}
        
        config["entity_types"] = ["PERSON"]
        result = await self.enrich_field("text", text, config)
        
        people = result.get("spacy_entities", {}).get("people", [])
        
        return self.normalize_text({
            "people": people,
            "count": len(people),
            "metadata": result.get("metadata", {})
        })
    
    async def extract_organizations(
        self, 
        text: str, 
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract only organization entities from text.
        
        Args:
            text: Text to extract organizations from
            config: Extraction configuration
            
        Returns:
            Organization extraction results
        """
        if config is None:
            config = {}
        
        config["entity_types"] = ["ORG"]
        result = await self.enrich_field("text", text, config)
        
        orgs = result.get("spacy_entities", {}).get("organizations", [])
        
        return self.normalize_text({
            "organizations": orgs,
            "count": len(orgs),
            "metadata": result.get("metadata", {})
        })
    
    async def extract_locations(
        self, 
        text: str, 
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract location entities from text.
        
        Args:
            text: Text to extract locations from
            config: Extraction configuration
            
        Returns:
            Location extraction results
        """
        if config is None:
            config = {}
        
        config["entity_types"] = ["GPE", "LOC", "FAC"]
        result = await self.enrich_field("text", text, config)
        
        locations = result.get("spacy_entities", {}).get("locations", [])
        
        return self.normalize_text({
            "locations": locations,
            "count": len(locations),
            "metadata": result.get("metadata", {})
        })
    
    async def extract_works_of_art(
        self, 
        text: str, 
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract works of art references from text.
        
        Args:
            text: Text to extract works from
            config: Extraction configuration
            
        Returns:
            Works of art extraction results
        """
        if config is None:
            config = {}
        
        config["entity_types"] = ["WORK_OF_ART"]
        result = await self.enrich_field("text", text, config)
        
        works = result.get("spacy_entities", {}).get("works_of_art", [])
        
        return self.normalize_text({
            "works_of_art": works,
            "count": len(works),
            "metadata": result.get("metadata", {})
        })
    
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin and service health."""
        base_health = await super().health_check()
        
        try:
            service_url = self.get_service_url("ner", "health")
            health_response = await self.http_get(service_url)
            
            base_health["service_health"] = {
                "spacy_ner_service": "healthy",
                "service_response": health_response
            }
            
        except Exception as e:
            base_health["service_health"] = {
                "spacy_ner_service": "unhealthy",
                "error": str(e)
            }
        
        return base_health