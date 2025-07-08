"""
Gensim Similarity Plugin
HTTP-only plugin that calls Gensim service for statistical keyword similarity.
"""

import logging
from typing import Dict, Any, List

from src.plugins.http_base import HTTPBasePlugin
from src.plugins.base import PluginMetadata, PluginType, ExecutionPriority
from src.shared.text_utils import extract_key_concepts

logger = logging.getLogger(__name__)


class GensimSimilarityPlugin(HTTPBasePlugin):
    """
    HTTP-only plugin that finds similar keywords using ONLY Gensim.
    
    Features:
    - Calls Gensim service endpoint for statistical similarity
    - Word vector-based similarity matching
    - Fast statistical analysis
    - No provider management - just HTTP calls
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="GensimSimilarityPlugin",
            version="1.0.0",
            description="Finds similar keywords using Gensim word vectors",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["keyword", "similarity", "gensim", "vectors", "statistical"],
            execution_priority=ExecutionPriority.NORMAL
        )
    
    async def enrich_field(
        self, 
        field_name: str, 
        field_value: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich a field with Gensim similarity expansion.
        
        Args:
            field_name: Name of the field being enriched
            field_value: Value of the field
            config: Plugin configuration
            
        Returns:
            Dict containing Gensim similarity results
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
                return {"gensim_similar": []}
            
            # Limit keywords for processing efficiency
            max_keywords = config.get("max_keywords", 5)
            keywords = keywords[:max_keywords]
            
            self._logger.debug(f"Finding similar terms for {len(keywords)} keywords using Gensim")
            
            # Call Gensim provider via generic NLP service endpoint
            # Gensim provider expects ProviderRequest format
            service_url = self.get_plugin_service_url()
            
            # Convert keywords list to a single concept string for provider request
            concept_text = ", ".join(keywords)
            
            request_data = {
                "concept": concept_text,
                "media_context": "movie",
                "max_concepts": config.get("max_similar", 8),
                "field_name": field_name,
                "options": {
                    "task_type": "similarity_search",
                    "similarity_threshold": config.get("threshold", 0.7),
                    "model_type": config.get("model_type", "word2vec"),  # or "fasttext"
                    "include_scores": config.get("include_scores", True),
                    "filter_duplicates": config.get("filter_duplicates", True)
                }
            }
            
            response = await self.http_post(service_url, request_data)
            
            # Process response from ProviderResponse format
            if response.get("success", False):
                result_data = response.get("result", {})
                # Gensim provider returns expanded_concepts, use as similar terms
                similar_keywords = result_data.get("expanded_concepts", result_data.get("similar_terms", []))
                metadata = response.get("metadata", {})
            else:
                similar_keywords = []
                metadata = {"error": response.get("error_message", "Unknown error")}
            
            # Extract just the terms if scores are included
            if similar_keywords and isinstance(similar_keywords[0], dict):
                if config.get("include_scores", True):
                    # Keep the structure with scores
                    pass  # similar_keywords is already in the right format
                else:
                    # Extract just the terms
                    similar_keywords = [item.get("term", str(item)) for item in similar_keywords]
            
            self._logger.info(
                f"Gensim found {len(similar_keywords)} similar terms for {len(keywords)} keywords"
            )
            
            return {
                "gensim_similar": similar_keywords,
                "original_keywords": keywords,
                "field_name": field_name,
                "metadata": {
                    "provider": "gensim",
                    "original_count": len(keywords),
                    "similar_count": len(similar_keywords),
                    "threshold": config.get("threshold", 0.7),
                    "model_type": config.get("model_type", "word2vec"),
                    "service_metadata": metadata
                }
            }
            
        except Exception as e:
            self._logger.error(f"Gensim similarity search failed for field {field_name}: {e}")
            # Return empty result on error
            return {
                "gensim_similar": [],
                "original_keywords": [],
                "field_name": field_name,
                "error": str(e),
                "metadata": {
                    "provider": "gensim",
                    "success": False
                }
            }
    
    async def find_similar_terms(
        self, 
        terms: List[str], 
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Direct similarity search method for backwards compatibility.
        
        Args:
            terms: List of terms to find similarities for
            config: Search configuration
            
        Returns:
            Similarity results
        """
        if config is None:
            config = {}
            
        # Use enrich_field with dummy field info
        result = await self.enrich_field("terms", terms, config)
        return {
            "similar_terms": result.get("gensim_similar", []),
            "metadata": result.get("metadata", {})
        }
    
    async def find_most_similar(
        self,
        term: str,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Find most similar terms to a single term.
        
        Args:
            term: Single term to find similarities for
            config: Search configuration
            
        Returns:
            Similarity results for the term
        """
        if config is None:
            config = {}
            
        result = await self.enrich_field("term", [term], config)
        similar = result.get("gensim_similar", [])
        
        return {
            "term": term,
            "similar_terms": similar,
            "metadata": result.get("metadata", {})
        }
    
    async def compare_similarity(
        self,
        term1: str,
        term2: str,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Compare similarity between two specific terms.
        
        Args:
            term1: First term
            term2: Second term
            config: Comparison configuration
            
        Returns:
            Similarity score and metadata
        """
        try:
            service_url = self.get_service_url("nlp", "gensim/compare")
            request_data = {
                "term1": term1,
                "term2": term2,
                "model_type": config.get("model_type", "word2vec") if config else "word2vec"
            }
            
            response = await self.http_post(service_url, request_data)
            
            return {
                "term1": term1,
                "term2": term2,
                "similarity_score": response.get("similarity", 0.0),
                "metadata": response.get("metadata", {})
            }
            
        except Exception as e:
            self._logger.error(f"Gensim similarity comparison failed: {e}")
            return {
                "term1": term1,
                "term2": term2,
                "similarity_score": 0.0,
                "error": str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin and service health."""
        base_health = await super().health_check()
        
        try:
            # Test Gensim service connectivity
            service_url = self.get_service_url("nlp", "health")
            health_response = await self.http_get(service_url)
            
            base_health["service_health"] = {
                "gensim_service": "healthy",
                "service_response": health_response
            }
            
        except Exception as e:
            base_health["service_health"] = {
                "gensim_service": "unhealthy",
                "error": str(e)
            }
        
        return base_health