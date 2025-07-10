"""
Service Endpoint Configuration
Configuration-driven service endpoint mapping for plugins.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml

from src.plugins.config import BasePluginConfig
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RoutingPattern(BaseModel):
    """A routing pattern for plugin-to-service mapping."""
    pattern: str = Field(..., description="Regex pattern to match plugin names")
    service: str = Field(..., description="Target service name")
    endpoint: str = Field(..., description="Endpoint key for the service")


class DefaultRouting(BaseModel):
    """Default routing when no patterns match."""
    service: str = Field(..., description="Default service name")
    endpoint: str = Field(..., description="Default endpoint key")


class ServiceEndpointConfig(BasePluginConfig):
    """Configuration for service endpoint mapping."""
    
    nlp_endpoints: Dict[str, str] = Field(default_factory=dict, description="Combined NLP service endpoint mappings")
    llm_endpoints: Dict[str, str] = Field(default_factory=dict, description="LLM service endpoint mappings")
    routing_patterns: List[RoutingPattern] = Field(default_factory=list, description="Pattern-based routing rules")
    default_routing: Optional[DefaultRouting] = Field(default=None, description="Default routing fallback")


class ServiceEndpointMapper:
    """Maps plugin names to service endpoints using configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/plugins/service_endpoints.yml")
        self.config: Optional[ServiceEndpointConfig] = None
        self._compiled_patterns: List[Tuple[re.Pattern, str, str]] = []
        
    def load_config(self) -> bool:
        """Load endpoint configuration from file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Service endpoint config not found: {self.config_path}")
                self._load_default_config()
                return False
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Parse routing patterns
            patterns = []
            for pattern_data in config_data.get('plugin_routing', {}).get('patterns', []):
                patterns.append(RoutingPattern(**pattern_data))
            
            # Parse default routing
            default_data = config_data.get('plugin_routing', {}).get('default', {})
            default_routing = DefaultRouting(**default_data) if default_data else None
            
            # Merge all NLP endpoints from individual service sections
            nlp_endpoints = {}
            nlp_endpoints.update(config_data.get('conceptnet_endpoints', {}))
            nlp_endpoints.update(config_data.get('gensim_endpoints', {}))
            nlp_endpoints.update(config_data.get('spacy_endpoints', {}))
            nlp_endpoints.update(config_data.get('heideltime_endpoints', {}))
            nlp_endpoints.update(config_data.get('sutime_endpoints', {}))
            
            self.config = ServiceEndpointConfig(
                nlp_endpoints=nlp_endpoints,
                llm_endpoints=config_data.get('llm_endpoints', {}),
                routing_patterns=patterns,
                default_routing=default_routing
            )
            
            # Compile regex patterns for efficiency
            self._compile_patterns()
            
            logger.info(f"Loaded service endpoint configuration from {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load service endpoint config: {e}")
            self._load_default_config()
            return False
    
    def _load_default_config(self):
        """Load minimal fallback configuration - no hard-coded endpoints."""
        logger.warning("Loading minimal fallback configuration - endpoints will be discovered dynamically")
        
        self.config = ServiceEndpointConfig(
            nlp_endpoints={},  # Empty - will be discovered
            llm_endpoints={},  # Empty - will be discovered
            routing_patterns=[
                # Pattern matching only - no hard-coded endpoints
                RoutingPattern(pattern="conceptnet", service="conceptnet", endpoint=""),
                RoutingPattern(pattern="gensim", service="gensim", endpoint=""),
                RoutingPattern(pattern="spacy", service="spacy", endpoint=""),
                RoutingPattern(pattern="heideltime", service="heideltime", endpoint=""),
                RoutingPattern(pattern="llm", service="llm", endpoint=""),
            ],
            default_routing=DefaultRouting(service="", endpoint="")  # Empty - will be discovered
        )
        
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self._compiled_patterns = []
        
        if not self.config or not self.config.routing_patterns:
            return
            
        for pattern in self.config.routing_patterns:
            try:
                compiled_pattern = re.compile(pattern.pattern, re.IGNORECASE)
                self._compiled_patterns.append((compiled_pattern, pattern.service, pattern.endpoint))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern.pattern}': {e}")
    
    async def get_service_and_endpoint(self, plugin_name: str) -> Tuple[str, str]:
        """
        Get service name and endpoint path for a plugin using dynamic discovery.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Tuple of (service_name, endpoint_path)
        """
        # Try dynamic service discovery first
        try:
            from src.shared.dynamic_service_discovery import get_service_discovery
            discovery = await get_service_discovery()
            service_info = await discovery.get_service_for_plugin(plugin_name)
            
            if service_info and service_info.is_healthy():
                # Derive endpoint from plugin name and service capabilities
                endpoint_path = self._derive_endpoint_from_plugin(plugin_name, service_info)
                logger.debug(f"Dynamic routing: plugin '{plugin_name}' -> {service_info.service_name}/{endpoint_path}")
                return service_info.service_name, endpoint_path
        
        except Exception as e:
            logger.warning(f"Dynamic routing failed for '{plugin_name}': {e}")
        
        # Fallback to pattern matching
        if not self.config:
            self.load_config()
        
        # Try pattern matching
        for compiled_pattern, service, endpoint_key in self._compiled_patterns:
            if compiled_pattern.search(plugin_name):
                endpoint_path = self._get_endpoint_path(service, endpoint_key)
                logger.debug(f"Pattern matched plugin '{plugin_name}' to {service}/{endpoint_path}")
                return service, endpoint_path
        
        # Fall back to default
        if self.config.default_routing:
            service = self.config.default_routing.service
            endpoint_key = self.config.default_routing.endpoint
            endpoint_path = self._get_endpoint_path(service, endpoint_key)
            logger.debug(f"Using default routing for plugin '{plugin_name}': {service}/{endpoint_path}")
            return service, endpoint_path
        
        # Ultimate fallback - use dynamic discovery
        logger.warning(f"No routing found for plugin '{plugin_name}', attempting dynamic discovery")
        try:
            from src.shared.dynamic_service_discovery import get_service_discovery
            discovery = await get_service_discovery()
            service_info = await discovery.get_service_for_plugin(plugin_name)
            if service_info:
                endpoint = await discovery.get_endpoint_for_plugin(plugin_name)
                return service_info.service_name, endpoint or ""
        except Exception as e:
            logger.error(f"Dynamic fallback failed for '{plugin_name}': {e}")
        
        # Final fallback - return empty to let service handle
        logger.error(f"All routing methods failed for plugin '{plugin_name}'")
        return "", ""
    
    async def _derive_endpoint_from_plugin(self, plugin_name: str, service_info) -> str:
        """Derive endpoint path from plugin name by discovering available endpoints."""
        try:
            from src.shared.dynamic_service_discovery import get_service_discovery
            discovery = await get_service_discovery()
            
            # Get the actual endpoint from service discovery
            endpoint = await discovery.get_endpoint_for_plugin(plugin_name)
            if endpoint:
                return endpoint
                
        except Exception as e:
            logger.warning(f"Failed to discover endpoint for {plugin_name}: {e}")
        
        # Fallback: return empty string to let service handle routing
        logger.warning(f"No endpoint discovered for plugin '{plugin_name}', using service default")
        return ""
    
    def _get_endpoint_path(self, service: str, endpoint_key: str) -> str:
        """Get the actual endpoint path for a service and endpoint key."""
        if not self.config:
            return endpoint_key
            
        # Handle split services
        if service in ["conceptnet", "gensim", "spacy", "heideltime", "sutime"] and endpoint_key in self.config.nlp_endpoints:
            return self.config.nlp_endpoints[endpoint_key]
        elif service == "llm" and endpoint_key in self.config.llm_endpoints:
            return self.config.llm_endpoints[endpoint_key]
        else:
            logger.warning(f"Endpoint key '{endpoint_key}' not found for service '{service}'")
            return endpoint_key
    
    def get_service_endpoints(self, service: str) -> Dict[str, str]:
        """Get all available endpoints for a service."""
        if not self.config:
            self.load_config()
            
        # Handle split services
        if service in ["conceptnet", "gensim", "spacy", "heideltime", "sutime"]:
            return self.config.nlp_endpoints.copy()
        elif service == "llm":
            return self.config.llm_endpoints.copy()
        else:
            return {}


# Global instance
_endpoint_mapper: Optional[ServiceEndpointMapper] = None


def get_endpoint_mapper() -> ServiceEndpointMapper:
    """Get the global service endpoint mapper."""
    global _endpoint_mapper
    if _endpoint_mapper is None:
        _endpoint_mapper = ServiceEndpointMapper()
        _endpoint_mapper.load_config()
    return _endpoint_mapper


def get_plugin_service_and_endpoint(plugin_name: str) -> Tuple[str, str]:
    """Get service and endpoint for a plugin name."""
    return get_endpoint_mapper().get_service_and_endpoint(plugin_name)