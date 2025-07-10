"""
Dynamic Service Discovery Component

Replaces multiple hard-coded service discovery implementations with a single
dynamic component that queries service /health endpoints for capabilities.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime
import httpx
from urllib.parse import urlparse

from src.shared.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ServiceCapabilityInfo:
    """Service capability information discovered from /health endpoint."""
    service_name: str
    base_url: str
    status: str = "unknown"
    providers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    service_type: str = "unknown"
    capabilities: List[str] = field(default_factory=list)
    last_discovery: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.status == "healthy"

    def has_capability(self, capability: str) -> bool:
        """Check if service has specific capability."""
        return capability in self.capabilities

    def get_provider_names(self) -> List[str]:
        """Get list of provider names."""
        return list(self.providers.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "service_name": self.service_name,
            "base_url": self.base_url,
            "status": self.status,
            "providers": self.providers,
            "service_type": self.service_type,
            "capabilities": self.capabilities,
            "last_discovery": self.last_discovery.isoformat() if self.last_discovery else None,
            "metadata": self.metadata
        }


class DynamicServiceDiscovery:
    """
    Single dynamic service discovery component.
    
    Replaces all hard-coded service registries with dynamic discovery
    by querying service /health endpoints for capabilities.
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceCapabilityInfo] = {}
        self.http_client: Optional[httpx.AsyncClient] = None
        self.discovery_timeout = 5.0
        self._discovery_lock = asyncio.Lock()
        
    async def initialize(self) -> bool:
        """Initialize the discovery service."""
        try:
            self.http_client = httpx.AsyncClient(timeout=self.discovery_timeout)
            await self.discover_all_services()
            logger.info(f"Dynamic service discovery initialized with {len(self.services)} services")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize dynamic service discovery: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.http_client:
            await self.http_client.aclose()
    
    async def discover_all_services(self) -> Dict[str, ServiceCapabilityInfo]:
        """Discover all services dynamically by scanning known ports."""
        async with self._discovery_lock:
            settings = get_settings()
            
            # Discover services by scanning for active services
            # No hard-coded ports - scan port range for responding services
            service_ports = {}
            
            # Scan configurable microservice port range from settings
            for port in range(settings.SERVICE_SCAN_START_PORT, settings.SERVICE_SCAN_END_PORT):
                # Try to connect and see if service responds to /health
                try:
                    base_url = self._get_service_base_url(f"unknown-service-{port}", port)
                    health_response = await self.http_client.get(f"{base_url}/health", timeout=2.0)
                    
                    if health_response.status_code == 200:
                        # Service found! Extract name from health response only
                        health_data = health_response.json()
                        
                        # Extract service name from health data dynamically
                        service_name = f"service-{port}"  # Default fallback
                        
                        # Try to derive name from providers in health response
                        if "providers" in health_data:
                            provider_names = list(health_data["providers"].keys())
                            if provider_names:
                                service_name = f"{provider_names[0]}-service"
                        elif "llm_provider" in health_data:
                            service_name = "llm-service"
                        
                        service_ports[service_name] = port
                        logger.info(f"Discovered active service: {service_name} on port {port}")
                        
                except Exception:
                    # Service not responding on this port - skip
                    continue
            
            discovered_services = {}
            
            # Discover services in parallel
            tasks = []
            for service_name, port in service_ports.items():
                base_url = self._get_service_base_url(service_name, port)
                tasks.append(self._discover_single_service(service_name, base_url))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, ServiceCapabilityInfo):
                    discovered_services[result.service_name] = result
                    logger.info(f"Discovered service: {result.service_name} with capabilities: {result.capabilities}")
                elif isinstance(result, Exception):
                    logger.warning(f"Service discovery failed: {result}")
            
            self.services = discovered_services
            return discovered_services
    
    def _get_service_base_url(self, service_name: str, port: int) -> str:
        """Get base URL for service based on environment."""
        settings = get_settings()
        
        # Environment-aware URL generation
        if settings.ENV == "localhost":
            return f"http://localhost:{port}"
        else:  # docker environment
            return f"http://{service_name}:{port}"
    
    async def _discover_single_service(self, service_name: str, base_url: str) -> ServiceCapabilityInfo:
        """Discover capabilities of a single service."""
        try:
            # Get health info
            health_url = f"{base_url}/health"
            health_response = await self.http_client.get(health_url)
            
            if health_response.status_code != 200:
                raise Exception(f"Health endpoint returned {health_response.status_code}")
            
            health_data = health_response.json()
            
            # Get available endpoints from OpenAPI
            openapi_url = f"{base_url}/openapi.json"
            openapi_response = await self.http_client.get(openapi_url)
            
            available_endpoints = []
            if openapi_response.status_code == 200:
                openapi_data = openapi_response.json()
                available_endpoints = list(openapi_data.get("paths", {}).keys())
            
            # Parse service info from health response
            service_info = ServiceCapabilityInfo(
                service_name=service_name,
                base_url=base_url,
                status=health_data.get("status", "unknown"),
                last_discovery=datetime.now(),
                metadata={
                    "health": health_data,
                    "available_endpoints": available_endpoints
                }
            )
            
            # Extract providers and capabilities dynamically
            self._extract_providers_and_capabilities(service_info, health_data, available_endpoints)
            
            return service_info
            
        except Exception as e:
            logger.warning(f"Failed to discover service {service_name} at {base_url}: {e}")
            # Return minimal info for unreachable services
            return ServiceCapabilityInfo(
                service_name=service_name,
                base_url=base_url,
                status="unreachable",
                last_discovery=datetime.now()
            )
    
    def _extract_providers_and_capabilities(self, service_info: ServiceCapabilityInfo, health_data: Dict[str, Any], available_endpoints: List[str]) -> None:
        """Extract providers and capabilities from health data and available endpoints."""
        
        # Handle different health response formats
        if "providers" in health_data:
            # ConceptNet, Gensim, Spacy, Heideltime format
            service_info.providers = health_data["providers"]
            
            # Derive service type and capabilities from providers
            provider_types = [p.get("type", "unknown") for p in service_info.providers.values()]
            service_info.service_type = provider_types[0] if provider_types else "unknown"
            service_info.capabilities = list(service_info.providers.keys())
            
        elif "llm_provider" in health_data:
            # LLM service format
            llm_provider = health_data["llm_provider"]
            service_info.providers = {"llm_provider": llm_provider}
            service_info.service_type = llm_provider.get("type", "unknown")
            
            # Extract capabilities from available endpoints instead of hard-coding
            llm_capabilities = []
            for endpoint in available_endpoints:
                if "/providers/llm/" in endpoint:
                    # Extract capability from endpoint path dynamically
                    parts = endpoint.split("/")
                    if len(parts) >= 4:  # /providers/llm/capability
                        capability = parts[3]
                        if capability not in llm_capabilities:
                            llm_capabilities.append(capability)
            
            service_info.capabilities = ["llm_provider"] + llm_capabilities
            
        else:
            # Unknown format - derive capabilities from endpoints
            service_info.service_type = "unknown"
            capabilities = []
            
            # Extract capabilities from provider endpoints
            for endpoint in available_endpoints:
                if "/providers/" in endpoint:
                    parts = endpoint.split("/")
                    if len(parts) >= 3:  # /providers/capability
                        capability = parts[2]
                        if capability not in capabilities and capability != "":
                            capabilities.append(capability)
            
            service_info.capabilities = capabilities
    
    # Public API Methods
    
    async def get_service_by_name(self, service_name: str, force_refresh: bool = False) -> Optional[ServiceCapabilityInfo]:
        """Get service information by name."""
        if force_refresh or service_name not in self.services:
            await self.discover_all_services()
        
        return self.services.get(service_name)
    
    async def get_services_by_capability(self, capability: str, healthy_only: bool = True) -> List[ServiceCapabilityInfo]:
        """Get all services that have a specific capability."""
        if not self.services:
            await self.discover_all_services()
        
        matching_services = []
        for service in self.services.values():
            if service.has_capability(capability):
                if not healthy_only or service.is_healthy():
                    matching_services.append(service)
        
        return matching_services
    
    async def get_services_by_type(self, service_type: str, healthy_only: bool = True) -> List[ServiceCapabilityInfo]:
        """Get all services of a specific type."""
        if not self.services:
            await self.discover_all_services()
        
        matching_services = []
        for service in self.services.values():
            if service.service_type == service_type:
                if not healthy_only or service.is_healthy():
                    matching_services.append(service)
        
        return matching_services
    
    async def get_healthy_services(self) -> List[ServiceCapabilityInfo]:
        """Get all healthy services."""
        if not self.services:
            await self.discover_all_services()
        
        return [service for service in self.services.values() if service.is_healthy()]
    
    async def get_service_for_plugin(self, plugin_name: str) -> Optional[ServiceCapabilityInfo]:
        """Get best service for a plugin based on name pattern matching."""
        if not self.services:
            await self.discover_all_services()
        
        plugin_name_lower = plugin_name.lower()
        
        # Pattern-based routing (same logic as endpoint_config.py but dynamic)
        if "conceptnet" in plugin_name_lower:
            services = await self.get_services_by_capability("conceptnet")
            return services[0] if services else None
        elif "gensim" in plugin_name_lower:
            services = await self.get_services_by_capability("gensim")
            return services[0] if services else None
        elif "spacy" in plugin_name_lower:
            services = await self.get_services_by_capability("spacy_temporal")
            return services[0] if services else None
        elif "heideltime" in plugin_name_lower:
            services = await self.get_services_by_capability("heideltime")
            return services[0] if services else None
        elif "llm" in plugin_name_lower:
            services = await self.get_services_by_capability("llm_provider")
            return services[0] if services else None
        
        # Default fallback
        healthy_services = await self.get_healthy_services()
        return healthy_services[0] if healthy_services else None
    
    async def get_service_url(self, service_name: str, endpoint: str = "") -> Optional[str]:
        """Get complete service URL with endpoint."""
        service = await self.get_service_by_name(service_name)
        if not service:
            return None
        
        base_url = service.base_url.rstrip('/')
        if endpoint:
            return f"{base_url}/{endpoint.lstrip('/')}"
        else:
            return base_url
    
    async def get_endpoint_for_plugin(self, plugin_name: str) -> Optional[str]:
        """Get the correct endpoint path for a plugin by discovering available endpoints."""
        service_info = await self.get_service_for_plugin(plugin_name)
        if not service_info:
            return None
        
        available_endpoints = service_info.metadata.get("available_endpoints", [])
        plugin_lower = plugin_name.lower()
        
        # Find matching endpoint based on plugin name
        best_match = None
        for endpoint in available_endpoints:
            if "/providers/" in endpoint:
                # Check if endpoint matches plugin type
                if "conceptnet" in plugin_lower and "conceptnet" in endpoint:
                    best_match = endpoint
                    break
                elif "gensim" in plugin_lower and "gensim" in endpoint:
                    best_match = endpoint
                    break
                elif "spacy" in plugin_lower and "spacy" in endpoint:
                    best_match = endpoint
                    break
                elif "heideltime" in plugin_lower and "heideltime" in endpoint:
                    best_match = endpoint
                    break
                elif "llm" in plugin_lower and "llm" in endpoint:
                    if "keyword" in plugin_lower and "keywords" in endpoint:
                        best_match = endpoint
                        break
                    elif "websearch" in plugin_lower and "websearch" in endpoint:
                        best_match = endpoint
                        break
                    elif "llm/" in endpoint and "/expand" in endpoint:
                        best_match = endpoint  # General LLM endpoint
        
        return best_match.lstrip('/') if best_match else None
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get summary of discovered services."""
        healthy_count = len([s for s in self.services.values() if s.is_healthy()])
        
        return {
            "total_services": len(self.services),
            "healthy_services": healthy_count,
            "unhealthy_services": len(self.services) - healthy_count,
            "last_discovery": max(
                (s.last_discovery for s in self.services.values() if s.last_discovery),
                default=None
            ),
            "services": {name: service.to_dict() for name, service in self.services.items()}
        }


# Global singleton instance
_service_discovery: Optional[DynamicServiceDiscovery] = None


async def get_service_discovery() -> DynamicServiceDiscovery:
    """Get the global service discovery instance."""
    global _service_discovery
    if _service_discovery is None:
        _service_discovery = DynamicServiceDiscovery()
        await _service_discovery.initialize()
    return _service_discovery


async def cleanup_service_discovery() -> None:
    """Cleanup the global service discovery instance."""
    global _service_discovery
    if _service_discovery is not None:
        await _service_discovery.cleanup()
        _service_discovery = None