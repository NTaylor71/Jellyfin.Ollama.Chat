"""
Abstract base class for LLM backend clients.

Provides a standard interface for all LLM backends to ensure consistent
behavior across different LLM providers (Ollama, OpenAI, Anthropic, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from datetime import datetime


@dataclass
class LLMResponse:
    """
    Standard response format for all LLM backends.
    
    Provides consistent data structure regardless of underlying LLM service.
    """
    text: str
    model: str
    success: bool = True
    tokens_used: Optional[int] = None
    response_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate response data."""
        if not self.success and not self.error_message:
            self.error_message = "Unknown error occurred"
        
        # Ensure metadata is not None
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LLMRequest:
    """
    Standard request format for LLM operations.
    
    Provides consistent input structure for concept expansion requests.
    """
    prompt: str
    concept: str
    media_context: str = "movie"
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and set defaults for request data."""
        if not self.prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if not self.concept.strip():
            raise ValueError("Concept cannot be empty")
        
        # Ensure extra_params is not None
        if self.extra_params is None:
            self.extra_params = {}


class BaseLLMClient(ABC):
    """
    Abstract interface for LLM backend clients.
    
    All LLM implementations (Ollama, OpenAI, Anthropic, etc.) must implement
    this interface to provide consistent concept expansion capabilities.
    """
    
    def __init__(self):
        """Initialize the LLM client."""
        self._initialized = False
        self._model_info: Optional[Dict[str, Any]] = None
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the LLM client.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """
        Generate text completion from prompt.
        
        Args:
            request: LLM request with prompt and parameters
            
        Returns:
            LLM response with generated text and metadata
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if LLM service is available and healthy.
        
        Returns:
            Dictionary with health status information
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model being used.
        
        Returns:
            Dictionary with model information (name, version, capabilities, etc.)
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Clean up LLM client resources."""
        pass
    
    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        return self._initialized
    
    async def _ensure_initialized(self) -> bool:
        """Ensure client is initialized before use."""
        if not self._initialized:
            self._initialized = await self.initialize()
        return self._initialized
    
    def supports_streaming(self) -> bool:
        """
        Check if this backend supports streaming responses.
        
        Override in subclasses if streaming is supported.
        
        Returns:
            True if streaming is supported
        """
        return False
    
    def get_max_tokens(self) -> Optional[int]:
        """
        Get maximum token limit for this model.
        
        Override in subclasses with specific model limits.
        
        Returns:
            Maximum tokens, or None if unlimited/unknown
        """
        return None
    
    def get_recommended_temperature(self, concept_type: str) -> float:
        """
        Get recommended temperature for concept expansion.
        
        Override in subclasses for model-specific recommendations.
        
        Args:
            concept_type: Type of concept being expanded
            
        Returns:
            Recommended temperature value (0.0-1.0)
        """
        # Default: Lower temperature for more focused concept expansion
        return 0.3
    
    def build_concept_expansion_prompt(
        self,
        concept: str,
        media_context: str,
        max_concepts: int = 10
    ) -> str:
        """
        Build a standard prompt for concept expansion.
        
        Override in subclasses for backend-specific prompt engineering.
        
        Args:
            concept: The concept to expand
            media_context: Media type context (movie, book, etc.)
            max_concepts: Maximum number of concepts to return
            
        Returns:
            Formatted prompt string
        """
        return f"""For the {media_context} concept "{concept}", provide {max_concepts} related concepts that would help in search and discovery.

Focus on:
- Synonyms and related terms
- Common themes and elements
- Genre characteristics
- Audience expectations

Return only the concepts as a simple comma-separated list, no explanations.

Example format: concept1, concept2, concept3

Concept: {concept}
Related concepts:"""


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    
    def __init__(self, message: str, backend_name: str = "unknown"):
        self.backend_name = backend_name
        super().__init__(f"[{backend_name}] {message}")


class LLMClientNotAvailableError(LLMClientError):
    """Raised when LLM service is not available."""
    pass


class LLMClientTimeoutError(LLMClientError):
    """Raised when LLM request times out."""
    pass


class LLMClientConfigurationError(LLMClientError):
    """Raised when LLM client has configuration issues."""
    pass


class LLMClientRateLimitError(LLMClientError):
    """Raised when LLM service rate limit is exceeded."""
    pass