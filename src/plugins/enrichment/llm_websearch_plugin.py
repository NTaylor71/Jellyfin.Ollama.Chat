"""
LLM Web Search Plugin
HTTP-only plugin that combines web search with LLM processing for real-time data enrichment.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional

from src.plugins.http_base import HTTPBasePlugin
from src.plugins.base import (
    PluginMetadata, PluginResourceRequirements, PluginType, ExecutionPriority,
    PluginExecutionContext, PluginExecutionResult
)

logger = logging.getLogger(__name__)


class LLMWebSearchPlugin(HTTPBasePlugin):
    """
    HTTP-only plugin that performs web search followed by LLM analysis.
    
    Features:
    - Template-based search query generation
    - Multi-domain search with filtering
    - LLM processing of search results
    - Configurable rate limiting and caching
    - Real-time web data enrichment
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="LLMWebSearchPlugin",
            version="1.0.0",
            description="Performs web search followed by LLM processing for real-time enrichment",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["websearch", "llm", "realtime", "web", "search", "enrichment"],
            execution_priority=ExecutionPriority.HIGH  # Web search + LLM is resource intensive
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        """Web search + LLM requires significant resources and network access."""
        return PluginResourceRequirements(
            min_cpu_cores=1.0,
            preferred_cpu_cores=2.0,
            min_memory_mb=1024.0,
            preferred_memory_mb=3072.0,
            requires_gpu=True,  # LLM processing benefits from GPU
            min_gpu_memory_mb=2048.0,
            preferred_gpu_memory_mb=8192.0,
            max_execution_time_seconds=120.0,  # Web search + LLM can take time
            requires_network=True  # Web search requires internet access
        )
    
    async def execute(self, data: Any, context: PluginExecutionContext) -> PluginExecutionResult:
        """
        Execute the LLMWebSearchPlugin with direct data structure.
        
        Expected data structure:
        {
            'field_name': str,
            'field_value': Any,
            'config': Dict[str, Any]
        }
        """
        logger.info(f"🔍 LLMWebSearchPlugin.execute: Called with data type={type(data)}")
        logger.info(f"🔍 LLMWebSearchPlugin.execute: data keys={list(data.keys()) if isinstance(data, dict) else 'not dict'}")
        
        start_time = time.time()
        
        try:
            if isinstance(data, dict) and 'field_name' in data and 'field_value' in data and 'config' in data:
                # Direct field enrichment call
                field_name = data['field_name']
                field_value = data['field_value']
                config = data['config']
                
                logger.info(f"🔍 LLMWebSearchPlugin.execute: About to call enrich_field({field_name}, {field_value})")
                enrichment_result = await self.enrich_field(field_name, field_value, config)
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                return PluginExecutionResult(
                    success=True,
                    data=enrichment_result,
                    execution_time_ms=execution_time_ms,
                    metadata={
                        "plugin": self.metadata.name,
                        "field_name": field_name,
                        "enrichment_type": "websearch_llm"
                    }
                )
            else:
                # Fall back to base class behavior for standard enrichment
                return await super().execute(data, context)
                
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"🔍 LLMWebSearchPlugin.execute: Error - {e}")
            
            return PluginExecutionResult(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time_ms,
                metadata={
                    "plugin": self.metadata.name,
                    "error_type": type(e).__name__
                }
            )
    
    async def enrich_field(
        self, 
        field_name: str, 
        field_value: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich a field using web search + LLM processing.
        
        Args:
            field_name: Name of the field being enriched
            field_value: Value of the field (used for template variables)
            config: Plugin configuration including search templates and LLM prompts
            
        Returns:
            Dict containing web search results and LLM analysis
        """
        logger.info(f"🔍 LLMWebSearchPlugin.enrich_field called for field '{field_name}' with value '{field_value}'")
        try:
            # Extract configuration
            search_templates = config.get("search_templates", [])
            llm_processing = config.get("llm_processing", {})
            rate_limiting = config.get("rate_limiting", {})
            
            if not search_templates:
                self._logger.warning(f"No search templates provided for field {field_name}")
                return self._empty_result(field_name, "no_search_templates")
            
            if not llm_processing or not llm_processing.get("prompt"):
                self._logger.warning(f"No LLM processing prompt provided for field {field_name}")
                return self._empty_result(field_name, "no_llm_prompt")
            
            # Prepare template variables from field context
            template_variables = self._prepare_template_variables(field_name, field_value, config)
            
            # Get web search service URL
            service_url = await self.get_plugin_service_url()
            
            # Prepare search templates list (extract from config structure)
            search_template_list = []
            max_results_per_template = config.get("max_results_per_template", 10)
            domains = config.get("domains", None)
            
            for template_config in search_templates:
                if isinstance(template_config, dict):
                    template = template_config.get("template", "")
                    # Override per-template settings if provided
                    template_max_results = template_config.get("max_results", max_results_per_template)
                    template_domains = template_config.get("domains", domains)
                    
                    search_template_list.append(template)
                    
                    # Store per-template settings (we'll use global settings for simplicity)
                    if template_domains and not domains:
                        domains = template_domains
                elif isinstance(template_config, str):
                    search_template_list.append(template_config)
            
            # Prepare request data for combined web search + LLM processing
            request_data = {
                "search_templates": search_template_list,
                "template_variables": template_variables,
                "max_results_per_template": max_results_per_template,
                "domains": domains,
                "processing_prompt": llm_processing["prompt"],
                "expected_format": llm_processing.get("expected_format", "dict"),
                "fields": llm_processing.get("fields", None),
                "rate_limit_delay": rate_limiting.get("delay_between_searches", 2.0)
            }
            
            self._logger.debug(f"Executing web search + LLM for field {field_name} with {len(search_template_list)} templates")
            
            # Call LLM service web search endpoint (service_url already includes the endpoint)
            response = await self.http_post(service_url, request_data)
            
            # Process response - handle both full success and partial success (search works, LLM fails)
            search_results = response.get("search_results", [])
            processed_content = response.get("processed_content", {})
            metadata = response.get("metadata", {})
            
            if response.get("success", False):
                self._logger.info(
                    f"Web search + LLM completed for field {field_name}: "
                    f"{len(search_results)} results, {metadata.get('execution_time_ms', 0):.1f}ms"
                )
            elif search_results:
                # Partial success: search worked but LLM processing failed
                error_msg = response.get("error_message", "Unknown error")
                self._logger.warning(
                    f"Web search succeeded but LLM processing failed for field {field_name}: {error_msg}. "
                    f"Returning {len(search_results)} search results without LLM analysis."
                )
                processed_content = {"error": f"LLM processing failed: {error_msg}"}
            else:
                # Complete failure: no search results
                error_msg = response.get("error_message", "Unknown error")
                self._logger.error(f"Web search failed for field {field_name}: {error_msg}")
                result = self._empty_result(field_name, f"websearch_failed: {error_msg}")
                return self.normalize_text(result)
            
            # Build result for success or partial success cases
            result = {
                "websearch_results": search_results,
                "llm_analysis": processed_content,
                "search_metadata": {
                    "templates_used": search_template_list,
                    "total_results": len(search_results),
                    "domains_filtered": domains,
                    "field_name": field_name,
                    "processing_status": "complete" if response.get("success", False) else "partial",
                    **metadata
                }
            }
            
            # Normalize all Unicode text in the result
            return self.normalize_text(result)
            
        except Exception as e:
            self._logger.error(f"LLM web search failed for field {field_name}: {e}")
            return self.normalize_text(self._empty_result(field_name, str(e)))
    
    def _prepare_template_variables(self, field_name: str, field_value: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare template variables for search query substitution."""
        template_variables = config.get("template_variables", {})
        
        # Add field-specific variables
        template_variables[field_name] = str(field_value) if field_value else ""
        
        # Add common field mappings that might be useful for templates
        # These would typically come from the media item being enriched
        common_fields = ["Name", "Title", "OriginalTitle", "ProductionYear", "Year", 
                        "Director", "Genres", "Tags", "Overview", "Plot", "Studio", "Studios"]
        
        for common_field in common_fields:
            if common_field not in template_variables:
                template_variables[common_field] = template_variables.get(common_field.lower(), "")
        
        return template_variables
    
    def _empty_result(self, field_name: str, error_reason: str) -> Dict[str, Any]:
        """Return empty result structure on error."""
        return {
            "websearch_results": [],
            "llm_analysis": {},
            "search_metadata": {
                "field_name": field_name,
                "success": False,
                "error": error_reason,
                "templates_used": [],
                "total_results": 0
            }
        }
    
    async def search_only(
        self, 
        search_templates: List[str], 
        template_variables: Dict[str, Any],
        max_results_per_template: int = 10,
        domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform web search only without LLM processing.
        
        Args:
            search_templates: List of search query templates
            template_variables: Variables for template substitution
            max_results_per_template: Maximum results per search template
            domains: Optional list of domains to filter
            
        Returns:
            Search results only
        """
        try:
            service_url = await self.get_plugin_service_url()
            
            request_data = {
                "search_templates": search_templates,
                "template_variables": template_variables,
                "max_results_per_template": max_results_per_template,
                "domains": domains
            }
            
            search_url = f"{service_url}/websearch/search"
            response = await self.http_post(search_url, request_data)
            
            if response.get("success", False):
                return {
                    "success": True,
                    "search_results": response.get("search_results", []),
                    "metadata": response.get("metadata", {})
                }
            else:
                return {
                    "success": False,
                    "search_results": [],
                    "error": response.get("error_message", "Search failed")
                }
                
        except Exception as e:
            self._logger.error(f"Web search only failed: {e}")
            return {
                "success": False,
                "search_results": [],
                "error": str(e)
            }
    
    async def process_results_only(
        self, 
        search_results: List[Dict[str, Any]], 
        processing_prompt: str,
        template_variables: Dict[str, Any] = None,
        expected_format: str = "dict",
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process search results with LLM only (no web search).
        
        Args:
            search_results: Raw search results to process
            processing_prompt: LLM prompt for processing
            template_variables: Variables for prompt substitution
            expected_format: Expected output format
            fields: Expected output fields
            
        Returns:
            LLM processed results
        """
        try:
            service_url = await self.get_plugin_service_url()
            
            request_data = {
                "search_results": search_results,
                "processing_prompt": processing_prompt,
                "template_variables": template_variables or {},
                "expected_format": expected_format,
                "fields": fields
            }
            
            process_url = f"{service_url}/websearch/process"
            response = await self.http_post(process_url, request_data)
            
            if response.get("success", False):
                return {
                    "success": True,
                    "processed_content": response.get("processed_content", {}),
                    "metadata": response.get("metadata", {})
                }
            else:
                return {
                    "success": False,
                    "processed_content": {},
                    "error": response.get("error_message", "Processing failed")
                }
                
        except Exception as e:
            self._logger.error(f"Result processing failed: {e}")
            return {
                "success": False,
                "processed_content": {},
                "error": str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin and service health."""
        base_health = await super().health_check()
        
        try:
            # Test web search service connectivity
            service_url = self.get_service_url("llm", "health")
            health_response = await self.http_get(service_url)
            
            # Test a simple web search (without LLM processing)
            test_search_url = f"{self.get_plugin_service_url()}/websearch/search"
            test_data = {
                "search_templates": ["test search"],
                "template_variables": {},
                "max_results_per_template": 1
            }
            
            search_test = await self.http_post(test_search_url, test_data)
            
            base_health["service_health"] = {
                "llm_service": "healthy",
                "websearch_endpoint": "healthy" if search_test.get("success") else "degraded",
                "service_response": health_response,
                "test_search": search_test.get("success", False)
            }
            
        except Exception as e:
            base_health["service_health"] = {
                "llm_service": "unknown",
                "websearch_endpoint": "unhealthy",
                "error": str(e)
            }
        
        return base_health