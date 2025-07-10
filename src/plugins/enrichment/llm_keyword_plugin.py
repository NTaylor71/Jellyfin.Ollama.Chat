"""
LLM Keyword Plugin
HTTP-only plugin that calls LLM service for semantic keyword expansion.
"""

import logging
from typing import Dict, Any, List

from src.plugins.http_base import HTTPBasePlugin
from src.plugins.base import PluginMetadata, PluginResourceRequirements, PluginType, ExecutionPriority
from src.shared.text_utils import extract_key_concepts

logger = logging.getLogger(__name__)


class LLMKeywordPlugin(HTTPBasePlugin):
    """
    HTTP-only plugin that expands keywords using ONLY LLM.
    
    Features:
    - Calls LLM service endpoint for semantic understanding
    - Context-aware keyword expansion
    - Understands nuance and relationships
    - No provider management - just HTTP calls
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="LLMKeywordPlugin",
            version="1.0.0",
            description="Expands keywords using LLM semantic understanding",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["keyword", "expansion", "llm", "semantic", "ai"],
            execution_priority=ExecutionPriority.HIGH
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        return PluginResourceRequirements(
            min_cpu_cores=1.0,
            preferred_cpu_cores=2.0,
            min_memory_mb=512.0,
            preferred_memory_mb=2048.0,
            requires_gpu=True,
            min_gpu_memory_mb=2048.0,
            preferred_gpu_memory_mb=8192.0,
            max_execution_time_seconds=60.0
        )
    
    async def enrich_field(
        self, 
        field_name: str, 
        field_value: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            if isinstance(field_value, str):
                keywords = extract_key_concepts(field_value)
                context_text = field_value
            elif isinstance(field_value, list):
                keywords = [str(item).strip() for item in field_value if str(item).strip()]
                context_text = ", ".join(keywords)
            else:
                text_value = str(field_value)
                keywords = extract_key_concepts(text_value)
                context_text = text_value
            
            if not keywords:
                self._logger.debug(f"No keywords found in field {field_name}")
                return {"llm_keywords": []}
            
            max_keywords = config.get("max_keywords", 3)
            keywords = keywords[:max_keywords]
            
            self._logger.debug(f"Expanding {len(keywords)} keywords using LLM")
            
            service_url = await self.get_plugin_service_url()
            
            keywords_text = ", ".join(keywords)
            concept_text = f"Keywords: {keywords_text}. Context: {context_text}"
            
            request_data = {
                "concept": concept_text,
                "media_context": "movie",
                "max_concepts": config.get("max_concepts", 15),
                "field_name": field_name,
                "options": {
                    "task_type": "keyword_expansion",
                    "expansion_style": config.get("expansion_style", "semantic_related"),
                    "prompt_template": config.get("prompt_template", self._get_default_prompt(field_name)),
                    "temperature": config.get("temperature", 0.3),
                    "smart_retry_until": config.get("smart_retry_until", "list")
                }
            }
            
            response = await self.http_post(service_url, request_data)
            
            if response.get("success", False):
                result_data = response.get("result", {})
                expanded_keywords = result_data.get("expanded_concepts", result_data.get("concepts", []))
                metadata = response.get("metadata", {})
            else:
                expanded_keywords = []
                metadata = {"error": response.get("error_message", "Unknown error")}
            
            if expanded_keywords and isinstance(expanded_keywords[0], dict):
                expanded_keywords = [item.get("concept", str(item)) for item in expanded_keywords]
            
            self._logger.info(
                f"LLM expanded {len(keywords)} keywords to {len(expanded_keywords)} concepts"
            )
            
            result = {
                "llm_keywords": expanded_keywords,
                "original_keywords": keywords,
                "field_name": field_name,
                "context": context_text[:200] + "..." if len(context_text) > 200 else context_text,
                "metadata": {
                    "provider": "llm",
                    "original_count": len(keywords),
                    "expanded_count": len(expanded_keywords),
                    "service_metadata": metadata,
                    "expansion_style": config.get("expansion_style", "semantic_related")
                }
            }
            
            return self.normalize_text(result)
            
        except Exception as e:
            self._logger.error(f"LLM keyword expansion failed for field {field_name}: {e}")
            
            return {
                "llm_keywords": [],
                "original_keywords": [],
                "field_name": field_name,
                "error": str(e),
                "metadata": {
                    "provider": "llm",
                    "success": False
                }
            }
    
    def _get_default_prompt(self, field_name: str) -> str:
        """Get field-specific prompt template for LLM expansion."""
        prompts = {
            "name": "Given this title/name: '{value}', provide related concepts, themes, and associated terms that would help find similar content.",
            "title": "Given this title: '{value}', provide related concepts, themes, and associated terms that would help find similar content.",
            "overview": "Given this description: '{value}', extract and expand key concepts, themes, genres, and related terms.",
            "description": "Given this description: '{value}', extract and expand key concepts, themes, and related terms.",
            "genres": "Given these genres: '{value}', provide related genres, subgenres, and thematic categories.",
            "tags": "Given these tags: '{value}', provide related tags, categories, and associated terms.",
            "summary": "Given this summary: '{value}', extract and expand key concepts and themes."
        }
        
        
        field_lower = field_name.lower()
        for key, prompt in prompts.items():
            if key in field_lower:
                return prompt
        

        return "Given this content: '{value}', provide related concepts, themes, and terms that would help categorize and find similar content."
    
    async def expand_keywords(
        self, 
        keywords: List[str], 
        config: Dict[str, Any] = None,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Direct keyword expansion method for backwards compatibility.
        
        Args:
            keywords: List of keywords to expand
            config: Expansion configuration
            context: Additional context for better expansion
            
        Returns:
            Expansion results
        """
        if config is None:
            config = {}
        
        
        if context:
            config["context"] = context
            
        
        result = await self.enrich_field("keywords", keywords, config)
        return {
            "expanded_keywords": result.get("llm_keywords", []),
            "metadata": result.get("metadata", {})
        }
    
    async def expand_with_custom_prompt(
        self,
        field_value: Any,
        custom_prompt: str,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Expand keywords using a custom prompt template.
        
        Args:
            field_value: Value to expand
            custom_prompt: Custom prompt template (use {value} as placeholder)
            config: Additional configuration
            
        Returns:
            Expansion results
        """
        if config is None:
            config = {}
            
        config["prompt_template"] = custom_prompt
        return await self.enrich_field("custom", field_value, config)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin and service health."""
        base_health = await super().health_check()
        
        try:
            
            service_url = self.get_service_url("llm", "health")
            health_response = await self.http_get(service_url)
            
            base_health["service_health"] = {
                "llm_service": "healthy",
                "service_response": health_response
            }
            
        except Exception as e:
            base_health["service_health"] = {
                "llm_service": "unhealthy", 
                "error": str(e)
            }
        
        return base_health