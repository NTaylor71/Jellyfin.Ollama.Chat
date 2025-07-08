"""
Remote Temporal Analysis Plugin
Extends TemporalAnalysisPlugin to use HTTP services instead of direct provider calls.
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

logger = logging.getLogger(__name__)


class RemoteTemporalAnalysisPlugin(HTTPProviderPlugin, BaseConceptPlugin):
    """
    Plugin that orchestrates temporal analysis using HTTP services.
    
    This plugin calls NLP services for temporal analysis instead of using providers directly.
    Benefits:
    - Reduced resource usage (no local temporal processing models)
    - Better scalability (services can be scaled independently)
    - Service-oriented architecture
    - Circuit breaker protection for service failures
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="RemoteTemporalAnalysisPlugin",
            version="1.0.0",
            description="Remote temporal analysis using HTTP services",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["temporal", "time", "date", "http", "remote", "service"],
            execution_priority=ExecutionPriority.HIGH
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        return PluginResourceRequirements(
            min_cpu_cores=0.5,
            preferred_cpu_cores=1.0,
            min_memory_mb=100.0,
            preferred_memory_mb=300.0,
            requires_gpu=False,  # GPU handled by services
            max_execution_time_seconds=45.0,
            can_use_distributed_resources=True
        )
    
    def get_service_endpoints(self) -> List[ServiceEndpoint]:
        """Get service endpoints for temporal analysis."""
        settings = get_settings()
        
        endpoints = []
        
        # NLP Service endpoint (handles temporal providers)
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
        
        # LLM Service endpoint (for intelligent temporal understanding)
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
        
        # Initialize concept plugin base
        concept_success = await BaseConceptPlugin.initialize(self, config)
        if not concept_success:
            return False
        
        # Check service availability
        healthy_services = await self._check_all_services_health()
        if not healthy_services:
            self._logger.warning("No services are healthy - plugin will have limited functionality")
        
        self._logger.info(f"Remote temporal analysis plugin initialized with {len(healthy_services)} healthy services")
        return True
    
    async def embellish_embed_data(
        self, 
        data: Dict[str, Any], 
        context: PluginExecutionContext
    ) -> Dict[str, Any]:
        """
        Enhance data with temporal analysis using remote services.
        """
        start_time = datetime.now()
        
        try:
            # Extract text for temporal analysis
            text_content = self._extract_temporal_text(data)
            if not text_content:
                self._logger.debug("No text content for temporal analysis")
                return data
            
            # Select processing strategy based on service availability
            strategy = await self._select_remote_processing_strategy(context)
            
            self._logger.info(
                f"Analyzing temporal content using {strategy.value} strategy with remote services"
            )
            
            # Execute temporal analysis based on strategy
            if strategy == ProcessingStrategy.HIGH_RESOURCE:
                temporal_results = await self._high_resource_temporal_analysis(text_content, data, context)
            elif strategy == ProcessingStrategy.MEDIUM_RESOURCE:
                temporal_results = await self._medium_resource_temporal_analysis(text_content, data, context)
            else:  # LOW_RESOURCE
                temporal_results = await self._low_resource_temporal_analysis(text_content, data, context)
            
            # Add temporal analysis results to data
            if "enhanced_fields" not in data:
                data["enhanced_fields"] = {}
            
            data["enhanced_fields"]["temporal_analysis"] = temporal_results
            data["enhanced_fields"]["temporal_metadata"] = {
                "strategy": strategy.value,
                "text_length": len(text_content),
                "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "service_based": True
            }
            
            self._logger.info(f"Remote temporal analysis complete")
            
            return data
            
        except Exception as e:
            self._logger.error(f"Remote temporal analysis failed: {e}")
            return data  # Return original data on error
    
    def _extract_temporal_text(self, data: Dict[str, Any]) -> str:
        """Extract text fields that might contain temporal information using pattern matching."""
        try:
            import re
            import yaml
            from pathlib import Path
            
            # Load temporal extraction rules
            detection_config_path = Path("config/media_types/media_detection.yaml")
            if detection_config_path.exists():
                with open(detection_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                temporal_rules = config.get("temporal_extraction_rules", {})
            else:
                temporal_rules = {}
            
            text_parts = []
            
            # Extract text fields using patterns
            text_patterns = temporal_rules.get("text_field_patterns", [])
            for field_name, field_value in data.items():
                if isinstance(field_value, str) and field_value.strip():
                    # Check if field name matches any text pattern
                    field_weight = self._get_field_weight(field_name, text_patterns)
                    if field_weight > 0:
                        text_parts.append(field_value)
            
            # Extract date fields using patterns  
            date_patterns = temporal_rules.get("date_field_patterns", [])
            for field_name, field_value in data.items():
                if isinstance(field_value, str) and field_value.strip():
                    # Check if field name matches any date pattern
                    field_weight = self._get_field_weight(field_name, date_patterns)
                    if field_weight > 0:
                        text_parts.append(f"Date: {field_value}")
            
            return " ".join(text_parts)
            
        except Exception as e:
            logger.warning(f"Pattern-based temporal text extraction failed: {e}")
            return self._fallback_temporal_text_extraction(data)
    
    def _get_field_weight(self, field_name: str, patterns: List[Dict]) -> float:
        """Get weight for a field based on pattern matching."""
        import re
        
        max_weight = 0.0
        field_lower = field_name.lower()
        
        for pattern_config in patterns:
            pattern = pattern_config.get("pattern", "")
            case_insensitive = pattern_config.get("case_insensitive", True)
            weight = pattern_config.get("weight", 1.0)
            
            flags = re.IGNORECASE if case_insensitive else 0
            
            try:
                if re.search(pattern, field_lower, flags):
                    max_weight = max(max_weight, weight)
            except re.error:
                logger.warning(f"Invalid regex pattern: {pattern}")
                continue
        
        return max_weight
    
    def _fallback_temporal_text_extraction(self, data: Dict[str, Any]) -> str:
        """Fallback temporal text extraction using simple heuristics."""
        import re
        
        text_parts = []
        
        for field_name, field_value in data.items():
            if isinstance(field_value, str) and field_value.strip():
                field_lower = field_name.lower()
                
                # Text fields that might contain temporal info
                if re.search(r'(name|title|overview|description|summary|plot|content|synopsis)', field_lower):
                    text_parts.append(field_value)
                
                # Date fields
                elif re.search(r'(date|year|time|created|modified)', field_lower):
                    text_parts.append(f"Date: {field_value}")
        
        return " ".join(text_parts)
    
    async def _select_remote_processing_strategy(
        self, 
        context: PluginExecutionContext
    ) -> ProcessingStrategy:
        """Select processing strategy based on service availability."""
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
    
    async def _high_resource_temporal_analysis(
        self, 
        text_content: str, 
        data: Dict[str, Any],
        context: PluginExecutionContext
    ) -> Dict[str, Any]:
        """
        High resource strategy: Use all available temporal providers + LLM intelligence.
        """
        results = {
            "temporal_entities": [],
            "temporal_concepts": [],
            "temporal_relationships": [],
            "media_context": self._detect_media_context(data)
        }
        
        # Parallel temporal parsing with multiple providers
        parse_tasks = []
        
        # SpaCy temporal parsing
        parse_tasks.append(
            self._parse_temporal_with_spacy(text_content)
        )
        
        # HeidelTime parsing
        parse_tasks.append(
            self._parse_temporal_with_heideltime(text_content)
        )
        
        # SUTime parsing
        parse_tasks.append(
            self._parse_temporal_with_sutime(text_content)
        )
        
        # Execute parsing tasks in parallel
        parse_results = await asyncio.gather(*parse_tasks, return_exceptions=True)
        
        # Collect temporal entities from all parsers
        for i, result in enumerate(parse_results):
            if isinstance(result, Exception):
                self._logger.warning(f"Temporal parser {i} failed: {result}")
                continue
            
            if result and result.get("success", False):
                entities = result.get("data", {}).get("temporal_entities", [])
                results["temporal_entities"].extend(entities)
        
        # Deduplicate temporal entities
        seen_entities = set()
        unique_entities = []
        for entity in results["temporal_entities"]:
            entity_key = f"{entity.get('text', '')}-{entity.get('type', '')}"
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                unique_entities.append(entity)
        
        results["temporal_entities"] = unique_entities
        
        # Intelligent temporal concept generation using LLM
        if await self._is_service_healthy("llm_service"):
            try:
                llm_result = await self._generate_temporal_concepts_with_llm(
                    text_content, 
                    results["temporal_entities"], 
                    results["media_context"]
                )
                
                if llm_result and llm_result.get("success", False):
                    llm_data = llm_result.get("data", {})
                    results["temporal_concepts"] = llm_data.get("temporal_concepts", [])
                    results["temporal_relationships"] = llm_data.get("temporal_relationships", [])
                    
            except Exception as e:
                self._logger.warning(f"LLM temporal concept generation failed: {e}")
        
        return results
    
    async def _medium_resource_temporal_analysis(
        self, 
        text_content: str, 
        data: Dict[str, Any],
        context: PluginExecutionContext
    ) -> Dict[str, Any]:
        """
        Medium resource strategy: Use primary temporal parser + selective LLM.
        """
        results = {
            "temporal_entities": [],
            "temporal_concepts": [],
            "media_context": self._detect_media_context(data)
        }
        
        # Use SpaCy as primary parser (most reliable)
        try:
            spacy_result = await self._parse_temporal_with_spacy(text_content)
            if spacy_result and spacy_result.get("success", False):
                results["temporal_entities"] = spacy_result.get("data", {}).get("temporal_entities", [])
        except Exception as e:
            self._logger.warning(f"SpaCy temporal parsing failed: {e}")
        
        # Selective LLM concept generation if we have temporal entities
        if results["temporal_entities"] and await self._is_service_healthy("llm_service"):
            try:
                llm_result = await self._generate_temporal_concepts_with_llm(
                    text_content, 
                    results["temporal_entities"][:5],  # Limit for medium resource
                    results["media_context"]
                )
                
                if llm_result and llm_result.get("success", False):
                    llm_data = llm_result.get("data", {})
                    results["temporal_concepts"] = llm_data.get("temporal_concepts", [])
                    
            except Exception as e:
                self._logger.warning(f"LLM temporal concept generation failed: {e}")
        
        return results
    
    async def _low_resource_temporal_analysis(
        self, 
        text_content: str, 
        data: Dict[str, Any],
        context: PluginExecutionContext
    ) -> Dict[str, Any]:
        """
        Low resource strategy: Basic temporal parsing only.
        """
        results = {
            "temporal_entities": [],
            "media_context": self._detect_media_context(data)
        }
        
        # Use SpaCy only for basic temporal parsing
        try:
            spacy_result = await self._parse_temporal_with_spacy(text_content)
            if spacy_result and spacy_result.get("success", False):
                results["temporal_entities"] = spacy_result.get("data", {}).get("temporal_entities", [])
        except Exception as e:
            self._logger.warning(f"SpaCy temporal parsing failed: {e}")
        
        return results
    
    def _detect_media_context(self, data: Dict[str, Any]) -> str:
        """Detect media context from data using media type detection."""
        from src.shared.media_field_config import detect_media_type_from_data
        
        try:
            media_type = detect_media_type_from_data(data)
            return media_type.value
        except Exception as e:
            logger.warning(f"Media context detection failed: {e}")
            return "movie"  # Default fallback
    
    async def _parse_temporal_with_spacy(self, text_content: str) -> Optional[Dict[str, Any]]:
        """Parse temporal entities using SpaCy via NLP service."""
        request = HTTPRequest(
            endpoint="providers/spacy/temporal",
            method="POST",
            data={
                "text": text_content,
                "extract_entities": True,
                "entity_types": ["DATE", "TIME", "DURATION"]
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
    
    async def _parse_temporal_with_heideltime(self, text_content: str) -> Optional[Dict[str, Any]]:
        """Parse temporal entities using HeidelTime via NLP service."""
        request = HTTPRequest(
            endpoint="providers/heideltime/parse",
            method="POST",
            data={
                "text": text_content,
                "document_type": "news",
                "language": "english"
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
    
    async def _parse_temporal_with_sutime(self, text_content: str) -> Optional[Dict[str, Any]]:
        """Parse temporal entities using SUTime via NLP service."""
        request = HTTPRequest(
            endpoint="providers/sutime/parse",
            method="POST",
            data={
                "text": text_content,
                "reference_date": datetime.now().isoformat()
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
    
    async def _generate_temporal_concepts_with_llm(
        self, 
        text_content: str, 
        temporal_entities: List[Dict[str, Any]], 
        media_context: str
    ) -> Optional[Dict[str, Any]]:
        """Generate temporal concepts using LLM service."""
        request = HTTPRequest(
            endpoint="providers/ollama/temporal-concepts",
            method="POST",
            data={
                "text": text_content,
                "temporal_entities": temporal_entities,
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
        self._logger.info("Remote temporal analysis plugin uses HTTP services, not direct providers")
    
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