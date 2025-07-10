"""
Base provider interface for concept expansion services.
All expansion providers (ConceptNet, LLM, Gensim, Temporal) implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from src.shared.plugin_contracts import PluginResult


@dataclass
class ProviderMetadata:
    """Metadata about the expansion provider."""
    name: str
    provider_type: str
    context_aware: bool
    strengths: List[str]
    weaknesses: List[str]
    best_for: List[str]
    version: str = "1.0.0"


@dataclass
class ExpansionRequest:
    """Request for concept expansion."""
    concept: str
    media_context: str = "movie"
    max_concepts: int = 10
    field_name: str = "concept"
    

    options: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.options is None:
            self.options = {}


class BaseProvider(ABC):
    """
    Abstract base class for all concept expansion providers.
    
    Each provider (ConceptNet, LLM, Gensim, etc.) implements this interface
    to provide a consistent API for concept expansion.
    """
    
    def __init__(self):
        self._initialized = False
        self._metadata: Optional[ProviderMetadata] = None
    
    @property
    @abstractmethod
    def metadata(self) -> ProviderMetadata:
        """Get provider metadata and capabilities."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the provider.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def expand_concept(self, request: ExpansionRequest) -> Optional[PluginResult]:
        """
        Expand a concept using this provider.
        
        Args:
            request: Expansion request with concept and parameters
            
        Returns:
            PluginResult with expanded concepts, or None if failed
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check provider health and availability.
        
        Returns:
            Dictionary with health status information
        """
        pass
    
    async def close(self) -> None:
        """Clean up provider resources (optional override)."""
        pass
    
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized
    
    def supports_concept(self, concept: str, media_context: str) -> bool:
        """
        Check if this provider can handle the given concept.
        
        Override in subclasses for provider-specific logic.
        
        Args:
            concept: Concept to expand
            media_context: Media context
            
        Returns:
            True if provider can handle this concept
        """
        return True
    
    def get_recommended_parameters(self, concept: str, media_context: str) -> Dict[str, Any]:
        """
        Get recommended parameters for this concept/context.
        
        Override in subclasses for provider-specific recommendations.
        
        Args:
            concept: Concept to expand
            media_context: Media context
            
        Returns:
            Dictionary of recommended parameters
        """
        return {}
    
    async def _ensure_initialized(self) -> bool:
        """Ensure provider is initialized before use."""
        if not self._initialized:
            self._initialized = await self.initialize()
        return self._initialized


class ProviderError(Exception):
    """Base exception for provider errors."""
    
    def __init__(self, message: str, provider_name: str = "unknown"):
        self.provider_name = provider_name
        super().__init__(f"[{provider_name}] {message}")


class ProviderNotAvailableError(ProviderError):
    """Raised when a provider is not available (e.g., API down, not configured)."""
    pass


class ProviderTimeoutError(ProviderError):
    """Raised when a provider operation times out."""
    pass


class ProviderConfigurationError(ProviderError):
    """Raised when a provider has configuration issues."""
    pass