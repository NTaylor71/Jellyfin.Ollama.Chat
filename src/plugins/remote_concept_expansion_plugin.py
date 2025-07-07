"""
Remote Concept Expansion Plugin
Extends ConceptExpansionPlugin to use HTTP services instead of direct provider calls.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.plugins.base_concept import BaseConceptPlugin, ProcessingStrategy
from src.plugins.http_provider_plugin import HTTPProviderPlugin, ServiceEndpoint, HTTPRequest
from src.plugins.base import (
    PluginMetadata, PluginResourceRequirements, PluginExecutionContext,
    PluginExecutionResult, PluginType, ExecutionPriority
)
from src.shared.text_utils import extract_key_concepts
from src.shared.config import get_settings
from src.shared.media_field_config import get_media_field_config, detect_media_type_from_data

logger = logging.getLogger(__name__)


class RemoteConceptExpansionPlugin(HTTPProviderPlugin, BaseConceptPlugin):
    """
    Plugin that orchestrates concept expansion using HTTP services.
    
    This plugin calls NLP and LLM services over HTTP instead of using providers directly.
    Benefits:
    - Reduced resource usage (no local NLP models)
    - Better scalability (services can be scaled independently)
    - Service-oriented architecture
    - Circuit breaker protection for service failures
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="RemoteConceptExpansionPlugin",
            version="1.0.0",
            description="Remote concept expansion using HTTP services",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["concept", "expansion", "http", "remote", "service"],
            execution_priority=ExecutionPriority.HIGH
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        return PluginResourceRequirements(
            min_cpu_cores=0.5,
            preferred_cpu_cores=1.0,
            min_memory_mb=100.0,
            preferred_memory_mb=300.0,
            requires_gpu=False,  # GPU is handled by services
            max_execution_time_seconds=60.0,
            can_use_distributed_resources=True
        )
    
    def get_service_endpoints(self) -> List[ServiceEndpoint]:
        """Get service endpoints for NLP and LLM services."""
        settings = get_settings()
        
        endpoints = []
        
        # NLP Service endpoint
        nlp_url = settings.nlp_service_url
        if nlp_url:
            endpoints.append(ServiceEndpoint(
                name="nlp_service",
                url=nlp_url,
                timeout_seconds=30.0,
                retry_attempts=3,
                retry_delay_seconds=1.0,
                health_check_path="/health",
                circuit_breaker_enabled=True,
                circuit_breaker_failure_threshold=5,
                circuit_breaker_timeout_seconds=60.0
            ))
        
        # LLM Service endpoint
        llm_url = settings.llm_service_url
        if llm_url:
            endpoints.append(ServiceEndpoint(
                name="llm_service",
                url=llm_url,
                timeout_seconds=45.0,
                retry_attempts=2,
                retry_delay_seconds=2.0,
                health_check_path="/health",
                circuit_breaker_enabled=True,
                circuit_breaker_failure_threshold=3,
                circuit_breaker_timeout_seconds=120.0
            ))
        
        return endpoints
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize both HTTP provider and concept plugin."""
        # Initialize HTTP provider
        http_success = await HTTPProviderPlugin.initialize(self, config)
        if not http_success:
            return False
        
        # Initialize concept plugin base (cache manager, etc.)
        concept_success = await BaseConceptPlugin.initialize(self, config)
        if not concept_success:
            return False
        
        # Check if we have at least one service available
        healthy_services = await self._check_all_services_health()
        if not healthy_services:
            self._logger.warning("No services are healthy - plugin will have limited functionality")
        
        self._logger.info(f"Remote concept expansion plugin initialized with {len(healthy_services)} healthy services")
        return True
    
    async def embellish_embed_data(
        self, 
        data: Dict[str, Any], 
        context: PluginExecutionContext
    ) -> Dict[str, Any]:
        """
        Enhance data with expanded concepts using remote services.
        """
        start_time = datetime.now()
        
        try:
            # Extract concepts from data
            concepts = await self._extract_concepts_from_data(data)
            if not concepts:
                self._logger.debug("No concepts found to expand")
                return data
            
            # Select processing strategy based on service availability
            strategy = await self._select_remote_processing_strategy(context)
            
            self._logger.info(
                f"Expanding {len(concepts)} concepts using {strategy.value} strategy with remote services"
            )
            
            # Execute expansion based on strategy
            if strategy == ProcessingStrategy.HIGH_RESOURCE:
                expanded_concepts = await self._high_resource_remote_expansion(concepts, context)
            elif strategy == ProcessingStrategy.MEDIUM_RESOURCE:
                expanded_concepts = await self._medium_resource_remote_expansion(concepts, context)
            elif strategy == ProcessingStrategy.LOW_RESOURCE:
                expanded_concepts = await self._low_resource_remote_expansion(concepts, context)
            else:  # QUEUE_ONLY - fallback to low resource
                expanded_concepts = await self._low_resource_remote_expansion(concepts, context)
            
            # Add expanded concepts to data
            if "enhanced_fields" not in data:
                data["enhanced_fields"] = {}
            
            data["enhanced_fields"]["expanded_concepts"] = expanded_concepts
            data["enhanced_fields"]["expansion_metadata"] = {
                "strategy": strategy.value,
                "concept_count": len(concepts),
                "expanded_count": sum(len(v) for v in expanded_concepts.values()),
                "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "service_based": True
            }
            
            self._logger.info(
                f"Remote concept expansion complete: {len(concepts)} → "
                f"{sum(len(v) for v in expanded_concepts.values())} concepts"
            )
            
            return data
            
        except Exception as e:
            self._logger.error(f"Remote concept expansion failed: {e}")
            return data  # Return original data on error
    
    async def _extract_concepts_from_data(self, data: Dict[str, Any]) -> List[str]:
        """Extract key concepts from data using media-type-aware configuration."""
        try:
            # Detect media type from data
            media_type = detect_media_type_from_data(data)
            
            # Get field configuration manager
            config_manager = await get_media_field_config()
            
            # Extract concepts using configured rules
            concepts = config_manager.extract_concepts_from_data(data, media_type)
            
            self._logger.debug(f"Extracted {len(concepts)} concepts for {media_type.value}")
            return concepts
            
        except Exception as e:
            self._logger.error(f"Failed to extract concepts using media config: {e}")
            # Fallback to basic extraction
            return self._fallback_concept_extraction(data)
    
    def _fallback_concept_extraction(self, data: Dict[str, Any]) -> List[str]:
        """Fallback concept extraction when config system fails."""
        concepts = set()
        
        # Generic text fields that might exist in any media type
        for key, value in data.items():
            if isinstance(value, str) and len(value.strip()) > 0:
                if key.lower() in ['name', 'title', 'description', 'overview', 'summary']:
                    field_concepts = extract_key_concepts(value)
                    concepts.update(field_concepts[:3])
            elif isinstance(value, list):
                for item in value[:5]:
                    if isinstance(item, str):
                        concepts.add(item.lower().strip())
                    elif isinstance(item, dict):
                        # Try common name fields in dictionaries
                        for name_field in ['Name', 'name', 'title', 'Title', 'label', 'Label']:
                            if name_field in item and isinstance(item[name_field], str):
                                concepts.add(item[name_field].lower().strip())
                                break
        
        return list(concepts)[:10]
    
    async def _select_remote_processing_strategy(
        self, 
        context: PluginExecutionContext
    ) -> ProcessingStrategy:
        """Select processing strategy based on service availability."""
        # Check service health
        nlp_healthy = await self._is_service_healthy("nlp_service")
        llm_healthy = await self._is_service_healthy("llm_service")
        
        if nlp_healthy and llm_healthy:
            return ProcessingStrategy.HIGH_RESOURCE
        elif nlp_healthy:
            return ProcessingStrategy.MEDIUM_RESOURCE
        else:
            return ProcessingStrategy.LOW_RESOURCE
    
    async def _is_service_healthy(self, service_name: str) -> bool:
        """Check if a service is healthy."""
        health = await self.check_service_health(service_name)
        return health.get("healthy", False)
    
    async def _high_resource_remote_expansion(
        self, 
        concepts: List[str], 
        context: PluginExecutionContext
    ) -> Dict[str, List[str]]:
        """
        High resource strategy: Use both NLP and LLM services with parallelism.
        """
        expanded_concepts = {}
        
        # Parallel processing of concepts
        max_parallel = min(len(concepts), 5)  # Limit to avoid overwhelming services
        
        # Split concepts for different services
        nlp_concepts = concepts[:10]    # NLP service (faster)
        llm_concepts = concepts[:5]     # LLM service (slower, more expensive)
        
        # Create tasks for parallel execution
        tasks = []
        
        # NLP service tasks
        for concept in nlp_concepts:
            task = self._expand_concept_with_nlp_service(concept, "movie")
            tasks.append(("nlp", concept, task))
        
        # LLM service tasks
        for concept in llm_concepts:
            task = self._expand_concept_with_llm_service(concept, "movie")
            tasks.append(("llm", concept, task))
        
        # Execute tasks in batches
        for i in range(0, len(tasks), max_parallel):
            batch = tasks[i:i + max_parallel]
            batch_results = await asyncio.gather(
                *[task for _, _, task in batch],
                return_exceptions=True
            )
            
            # Process batch results
            for j, (service_type, concept, _) in enumerate(batch):
                result = batch_results[j]
                
                if isinstance(result, Exception):
                    self._logger.warning(f"{service_type} service failed for '{concept}': {result}")
                    continue
                
                if result and result.get("success", False):
                    concept_expansions = result.get("data", {}).get("expanded_concepts", [])
                    
                    # Merge with existing expansions
                    if concept in expanded_concepts:
                        existing = expanded_concepts[concept]
                        merged = list(dict.fromkeys(existing + concept_expansions))
                        expanded_concepts[concept] = merged[:15]  # Limit total
                    else:
                        expanded_concepts[concept] = concept_expansions[:10]
        
        return expanded_concepts
    
    async def _medium_resource_remote_expansion(
        self, 
        concepts: List[str], 
        context: PluginExecutionContext
    ) -> Dict[str, List[str]]:
        """
        Medium resource strategy: Use NLP service primarily, selective LLM.
        """
        expanded_concepts = {}
        
        # Process with NLP service
        for concept in concepts[:10]:
            try:
                result = await self._expand_concept_with_nlp_service(concept, "movie")
                if result and result.get("success", False):
                    concept_expansions = result.get("data", {}).get("expanded_concepts", [])
                    expanded_concepts[concept] = concept_expansions[:8]
            except Exception as e:
                self._logger.warning(f"NLP service failed for '{concept}': {e}")
        
        # Selective LLM for top concepts
        if await self._is_service_healthy("llm_service"):
            for concept in concepts[:3]:
                try:
                    result = await self._expand_concept_with_llm_service(concept, "movie")
                    if result and result.get("success", False):
                        llm_expansions = result.get("data", {}).get("expanded_concepts", [])
                        
                        # Merge with existing
                        if concept in expanded_concepts:
                            existing = expanded_concepts[concept]
                            merged = list(dict.fromkeys(existing + llm_expansions))
                            expanded_concepts[concept] = merged[:12]
                        else:
                            expanded_concepts[concept] = llm_expansions[:8]
                except Exception as e:
                    self._logger.warning(f"LLM service failed for '{concept}': {e}")
        
        return expanded_concepts
    
    async def _low_resource_remote_expansion(
        self, 
        concepts: List[str], 
        context: PluginExecutionContext
    ) -> Dict[str, List[str]]:
        """
        Low resource strategy: Use only NLP service, sequential processing.
        """
        expanded_concepts = {}
        
        # Sequential processing with NLP service only
        for concept in concepts[:8]:
            try:
                result = await self._expand_concept_with_nlp_service(concept, "movie")
                if result and result.get("success", False):
                    concept_expansions = result.get("data", {}).get("expanded_concepts", [])
                    expanded_concepts[concept] = concept_expansions[:5]
            except Exception as e:
                self._logger.warning(f"NLP service failed for '{concept}': {e}")
        
        return expanded_concepts
    
    async def _expand_concept_with_nlp_service(
        self, 
        concept: str, 
        media_context: str
    ) -> Optional[Dict[str, Any]]:
        """Expand concept using NLP service."""
        request = HTTPRequest(
            endpoint="providers/conceptnet/expand",
            method="POST",
            data={
                "concept": concept,
                "media_context": media_context,
                "max_concepts": 10
            },
            timeout=30.0
        )
        
        response = await self.call_service("nlp_service", request)
        
        if response.success:
            return {
                "success": True,
                "data": response.data,
                "execution_time_ms": response.execution_time_ms
            }
        else:
            return {
                "success": False,
                "error": response.error_message
            }
    
    async def _expand_concept_with_llm_service(
        self, 
        concept: str, 
        media_context: str
    ) -> Optional[Dict[str, Any]]:
        """Expand concept using LLM service."""
        request = HTTPRequest(
            endpoint="providers/ollama/expand",
            method="POST",
            data={
                "concept": concept,
                "media_context": media_context,
                "max_concepts": 10
            },
            timeout=45.0
        )
        
        response = await self.call_service("llm_service", request)
        
        if response.success:
            return {
                "success": True,
                "data": response.data,
                "execution_time_ms": response.execution_time_ms
            }
        else:
            return {
                "success": False,
                "error": response.error_message
            }
    
    async def _initialize_providers(self) -> None:
        """Override base method - we don't initialize providers directly."""
        self._logger.info("Remote concept expansion plugin uses HTTP services, not direct providers")
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check including service connectivity."""
        base_health = await HTTPProviderPlugin.health_check(self)
        
        # Add service-specific health information
        service_health = {}
        for service_name in self.services:
            service_health[service_name] = await self.check_service_health(service_name)
        
        base_health.update({
            "service_health": service_health,
            "services_available": len([s for s in service_health.values() if s.get("healthy", False)])
        })
        
        return base_health