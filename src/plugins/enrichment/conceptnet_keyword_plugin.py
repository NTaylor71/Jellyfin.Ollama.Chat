"""
ConceptNet Keyword Plugin
HTTP-only plugin that calls ConceptNet service for keyword expansion.
"""

import logging
from typing import Dict, Any, List

from src.plugins.http_base import HTTPBasePlugin
from src.plugins.base import PluginMetadata, PluginType, ExecutionPriority
from src.shared.text_utils import extract_key_concepts

logger = logging.getLogger(__name__)


class ConceptNetKeywordPlugin(HTTPBasePlugin):
    """
    HTTP-only plugin that expands keywords using ONLY ConceptNet.
    
    Features:
    - Calls ConceptNet service endpoint 
    - Fast, lightweight keyword expansion
    - Linguistic relationships and associations
    - No provider management - just HTTP calls
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="ConceptNetKeywordPlugin",
            version="1.0.0",
            description="Expands keywords using ConceptNet linguistic relationships",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["keyword", "expansion", "conceptnet", "linguistic"],
            execution_priority=ExecutionPriority.NORMAL
        )
    
    async def enrich_field(
        self, 
        field_name: str, 
        field_value: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich a field with ConceptNet keyword expansion.
        
        Args:
            field_name: Name of the field being enriched
            field_value: Value of the field
            config: Plugin configuration
            
        Returns:
            Dict containing ConceptNet expansion results
        """
        try:
            # Extract keywords from field value
            if isinstance(field_value, str):
                keywords = extract_key_concepts(field_value)
            elif isinstance(field_value, list):
                # Handle list fields (e.g., genres, tags)
                keywords = [str(item).strip() for item in field_value if str(item).strip()]
            else:
                # Convert other types to string
                keywords = extract_key_concepts(str(field_value))
            
            if not keywords:
                self._logger.debug(f"No keywords found in field {field_name}")
                return {"conceptnet_keywords": []}
            
            # Limit keywords for API efficiency
            max_keywords = config.get("max_keywords", 5)
            keywords = keywords[:max_keywords]
            
            self._logger.debug(f"Expanding {len(keywords)} keywords using ConceptNet")
            
            # Call ConceptNet service using direct HTTPBasePlugin routing
            service_url = await self.get_plugin_service_url()
            
            # ConceptNet provider expects single concept string, not keywords array
            # Convert keywords list to a single concept string
            concept_text = ", ".join(keywords)
            
            request_data = {
                "concept": concept_text,
                "media_context": "movie",
                "max_concepts": config.get("max_concepts", 10),
                "field_name": field_name,
                "options": {
                    "relation_types": config.get("relation_types", ["RelatedTo", "IsA", "PartOf"]),
                    "language": config.get("language", "en")
                }
            }
            
            response = await self.http_post(service_url, request_data)
            
            # Process response from ProviderResponse format
            if response.get("success", False):
                result_data = response.get("result", {})
                expanded_keywords = result_data.get("concepts", [])
                metadata = response.get("metadata", {})
            else:
                expanded_keywords = []
                metadata = {"error": response.get("error_message", "Unknown error")}
            
            self._logger.info(
                f"ConceptNet expanded {len(keywords)} keywords to {len(expanded_keywords)} concepts"
            )
            print(f'ConceptNet expanded {len(keywords)} keywords to {len(expanded_keywords)} concepts')
            
            result = {
                "conceptnet_keywords": expanded_keywords,
                "original_keywords": keywords,
                "field_name": field_name,
                "metadata": {
                    "provider": "conceptnet",
                    "original_count": len(keywords),
                    "expanded_count": len(expanded_keywords),
                    "service_metadata": metadata
                }
            }
            
            # Normalize all Unicode text in the result
            return self.normalize_text(result)
            
        except Exception as e:
            self._logger.error(f"ConceptNet keyword expansion failed for field {field_name}: {e}")
            # Return empty result on error
            return {
                "conceptnet_keywords": [],
                "original_keywords": [],
                "field_name": field_name,
                "error": str(e),
                "metadata": {
                    "provider": "conceptnet",
                    "success": False
                }
            }
    
    async def expand_keywords(
        self, 
        keywords: List[str], 
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Direct keyword expansion method for backwards compatibility.
        
        Args:
            keywords: List of keywords to expand
            config: Expansion configuration
            
        Returns:
            Expansion results
        """
        if config is None:
            config = {}
            
        # Use enrich_field with dummy field info
        result = await self.enrich_field("keywords", keywords, config)
        return {
            "expanded_keywords": result.get("conceptnet_keywords", []),
            "metadata": result.get("metadata", {})
        }
    
    def _extract_keywords_from_text(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords from text using text utilities."""
        try:
            keywords = extract_key_concepts(text)
            return keywords[:max_keywords]
        except Exception as e:
            self._logger.warning(f"Failed to extract keywords: {e}")
            # Fallback to simple word extraction
            words = text.lower().split()
            # Filter out common stop words and short words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            keywords = [word for word in words if len(word) > 2 and word not in stop_words]
            return keywords[:max_keywords]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin and service health."""
        base_health = await super().health_check()
        
        try:
            # Test ConceptNet service connectivity
            service_url = self.get_service_url("keyword", "health")
            health_response = await self.http_get(service_url)
            
            base_health["service_health"] = {
                "conceptnet_service": "healthy",
                "service_response": health_response
            }
            
        except Exception as e:
            base_health["service_health"] = {
                "conceptnet_service": "unhealthy",
                "error": str(e)
            }
        
        return base_health