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
    
    nlp_endpoints: Dict[str, str] = Field(default_factory=dict, description="NLP service endpoint mappings")
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
            
            self.config = ServiceEndpointConfig(
                nlp_endpoints=config_data.get('nlp_endpoints', {}),
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
        """Load fallback configuration if file loading fails."""
        logger.info("Loading default service endpoint configuration")
        
        self.config = ServiceEndpointConfig(
            nlp_endpoints={
                "conceptnet": "providers/conceptnet/expand",
                "gensim": "providers/gensim/similarity", 
                "spacy_temporal": "providers/spacy_temporal/expand",
                "heideltime": "providers/heideltime/expand",
                "sutime": "providers/sutime/expand"
            },
            llm_endpoints={
                "keywords": "providers/llm/keywords/expand",
                "general": "providers/llm/expand"
            },
            routing_patterns=[
                RoutingPattern(pattern="conceptnet", service="nlp", endpoint="conceptnet"),
                RoutingPattern(pattern="gensim", service="nlp", endpoint="gensim"),
                RoutingPattern(pattern="spacy", service="nlp", endpoint="spacy_temporal"),
                RoutingPattern(pattern="heideltime", service="nlp", endpoint="heideltime"),
                RoutingPattern(pattern="sutime", service="nlp", endpoint="sutime"),
                RoutingPattern(pattern="llm.*keyword", service="llm", endpoint="keywords"),
                RoutingPattern(pattern="llm", service="llm", endpoint="general"),
            ],
            default_routing=DefaultRouting(service="nlp", endpoint="gensim")
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
    
    def get_service_and_endpoint(self, plugin_name: str) -> Tuple[str, str]:
        """
        Get service name and endpoint path for a plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Tuple of (service_name, endpoint_path)
        """
        if not self.config:
            self.load_config()
        
        # Try pattern matching
        for compiled_pattern, service, endpoint_key in self._compiled_patterns:
            if compiled_pattern.search(plugin_name):
                endpoint_path = self._get_endpoint_path(service, endpoint_key)
                logger.debug(f"Matched plugin '{plugin_name}' to {service}/{endpoint_path}")
                return service, endpoint_path
        
        # Fall back to default
        if self.config.default_routing:
            service = self.config.default_routing.service
            endpoint_key = self.config.default_routing.endpoint
            endpoint_path = self._get_endpoint_path(service, endpoint_key)
            logger.debug(f"Using default routing for plugin '{plugin_name}': {service}/{endpoint_path}")
            return service, endpoint_path
        
        # Ultimate fallback
        logger.warning(f"No routing found for plugin '{plugin_name}', using hardcoded fallback")
        return "nlp", "providers/gensim/similarity"
    
    def _get_endpoint_path(self, service: str, endpoint_key: str) -> str:
        """Get the actual endpoint path for a service and endpoint key."""
        if not self.config:
            return endpoint_key
            
        if service == "nlp" and endpoint_key in self.config.nlp_endpoints:
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
            
        if service == "nlp":
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