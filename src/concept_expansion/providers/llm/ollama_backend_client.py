"""
Ollama backend client for LLM concept expansion.

Implements the BaseLLMClient interface for Ollama API integration,
providing hardware-aware concept expansion using local LLM models.
"""

import logging
import asyncio
import httpx
from typing import Dict, Optional, List, Any
from datetime import datetime

from src.concept_expansion.providers.llm.base_llm_client import (
    BaseLLMClient, LLMRequest, LLMResponse,
    LLMClientError, LLMClientNotAvailableError, LLMClientTimeoutError,
    LLMClientConfigurationError
)
from src.shared.config import get_settings
from src.shared.hardware_config import get_resource_limits

logger = logging.getLogger(__name__)


class OllamaBackendClient(BaseLLMClient):
    """
    Ollama-specific implementation of LLM client.
    
    Provides concept expansion using Ollama's local LLM models with
    hardware-aware configuration and performance optimization.
    """
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.base_url = self.settings.ollama_chat_url
        self.model = self.settings.OLLAMA_CHAT_MODEL
        self.timeout = self.settings.OLLAMA_CHAT_TIMEOUT
        
        # HTTP client for API calls
        self._client: Optional[httpx.AsyncClient] = None
        
        # Hardware awareness
        self._hardware_limits: Optional[Dict[str, Any]] = None
        self._model_info_cache: Optional[Dict[str, Any]] = None
        
        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 0.1  # 10 requests per second max
        
        logger.info(f"Initialized Ollama client: {self.base_url}, model: {self.model}")
    
    async def initialize(self) -> bool:
        """Initialize the Ollama client and check availability."""
        try:
            # Create HTTP client
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
            
            # Load hardware limits for optimization
            self._hardware_limits = await get_resource_limits()
            
            # Test connection and load model info
            health_status = await self.health_check()
            if health_status.get("status") != "healthy":
                logger.error(f"Ollama health check failed: {health_status}")
                return False
            
            # Cache model information
            self._model_info_cache = await self._fetch_model_info()
            
            logger.info("Ollama client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            if self._client:
                await self._client.aclose()
                self._client = None
            return False
    
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """
        Generate text completion using Ollama API.
        
        Args:
            request: LLM request with prompt and parameters
            
        Returns:
            LLM response with generated concepts
        """
        start_time = datetime.now()
        
        try:
            # Ensure client is initialized
            if not await self._ensure_initialized():
                raise LLMClientNotAvailableError("Ollama client not available", "Ollama")
            
            # Rate limiting
            await self._enforce_rate_limit()
            
            # Build request payload
            payload = self._build_request_payload(request)
            
            # Make API call
            response_data = await self._make_api_call(payload)
            
            # Parse response
            completion_text = self._extract_completion_text(response_data)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return LLMResponse(
                text=completion_text,
                model=self.model,
                success=True,
                response_time_ms=execution_time,
                metadata={
                    "backend": "ollama",
                    "base_url": self.base_url,
                    "request_params": {
                        "temperature": request.temperature,
                        "max_tokens": request.max_tokens
                    },
                    "response_data": response_data
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            error_msg = f"Ollama completion failed: {str(e)}"
            logger.error(error_msg)
            
            return LLMResponse(
                text="",
                model=self.model,
                success=False,
                response_time_ms=execution_time,
                error_message=error_msg,
                metadata={"backend": "ollama", "error": str(e)}
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama service health and model availability."""
        try:
            if not self._client:
                return {
                    "status": "unhealthy",
                    "backend": "ollama",
                    "error": "Client not initialized"
                }
            
            # Check if Ollama is running
            response = await self._client.get(f"{self.base_url}/api/tags")
            
            if response.status_code != 200:
                return {
                    "status": "unhealthy",
                    "backend": "ollama",
                    "error": f"Ollama API returned status {response.status_code}"
                }
            
            # Check if our model is available
            models_data = response.json()
            available_models = [model.get("name", "") for model in models_data.get("models", [])]
            model_available = any(self.model in model_name for model_name in available_models)
            
            if not model_available:
                return {
                    "status": "degraded",
                    "backend": "ollama",
                    "error": f"Model {self.model} not found",
                    "available_models": available_models
                }
            
            # Test a simple completion
            test_response = await self._test_completion()
            
            return {
                "status": "healthy" if test_response else "degraded",
                "backend": "ollama",
                "base_url": self.base_url,
                "model": self.model,
                "model_available": model_available,
                "available_models": available_models,
                "test_completion": test_response
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": "ollama",
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Ollama model."""
        if self._model_info_cache:
            return self._model_info_cache
        
        return {
            "name": self.model,
            "backend": "ollama",
            "base_url": self.base_url,
            "timeout": self.timeout,
            "capabilities": {
                "streaming": True,
                "chat": True,
                "completion": True
            },
            "hardware_optimized": bool(self._hardware_limits and self._hardware_limits.get("gpu_available"))
        }
    
    async def close(self) -> None:
        """Clean up Ollama client resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("Ollama client closed")
    
    def supports_streaming(self) -> bool:
        """Ollama supports streaming responses."""
        return True
    
    def get_max_tokens(self) -> Optional[int]:
        """Get maximum token limit for the model."""
        # Most Ollama models have context limits around 2048-4096 tokens
        # Return a conservative estimate
        return 2048
    
    def get_recommended_temperature(self, concept_type: str) -> float:
        """Get recommended temperature for concept expansion."""
        # Lower temperature for more focused concept expansion
        if concept_type in ["genre", "tag", "category"]:
            return 0.2  # Very focused for categorical concepts
        elif concept_type in ["description", "overview"]:
            return 0.4  # Slightly more creative for descriptive concepts
        else:
            return 0.3  # Default focused temperature
    
    def build_concept_expansion_prompt(
        self,
        concept: str,
        media_context: str,
        max_concepts: int = 10
    ) -> str:
        """Build Ollama-optimized prompt for concept expansion."""
        # Ollama works well with clear, direct prompts
        return f"""You are a {media_context} expert. For the concept "{concept}", list {max_concepts} related concepts that would help users find similar {media_context}s.

Guidelines:
- Focus on {media_context}-specific themes and elements
- Include synonyms, related genres, and common characteristics
- Consider what audiences who like "{concept}" would also enjoy
- Be specific to {media_context} content

Format: Return only a comma-separated list of concepts, no explanations.

Concept: {concept}
Related {media_context} concepts:"""
    
    async def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = asyncio.get_event_loop().time()
    
    def _build_request_payload(self, request: LLMRequest) -> Dict[str, Any]:
        """Build Ollama API request payload."""
        payload = {
            "model": self.model,
            "prompt": request.prompt,
            "stream": False,  # Use non-streaming for concept expansion
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens or 200,  # Reasonable default for concept lists
                "stop": ["\n\n", "Explanation:", "Note:"]  # Stop at explanations
            }
        }
        
        # Add system prompt if provided
        if request.system_prompt:
            payload["system"] = request.system_prompt
        
        # Add any extra parameters
        if request.extra_params:
            payload["options"].update(request.extra_params)
        
        return payload
    
    async def _make_api_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make API call to Ollama."""
        try:
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.TimeoutException:
            raise LLMClientTimeoutError(f"Request timed out after {self.timeout}s", "Ollama")
        except httpx.HTTPStatusError as e:
            raise LLMClientError(f"HTTP error {e.response.status_code}: {e.response.text}", "Ollama")
        except Exception as e:
            raise LLMClientError(f"Request failed: {str(e)}", "Ollama")
    
    def _extract_completion_text(self, response_data: Dict[str, Any]) -> str:
        """Extract completion text from Ollama response."""
        if not response_data.get("done", False):
            logger.warning("Ollama response indicates incomplete generation")
        
        response_text = response_data.get("response", "").strip()
        
        if not response_text:
            raise LLMClientError("Empty response from Ollama", "Ollama")
        
        return response_text
    
    async def _fetch_model_info(self) -> Dict[str, Any]:
        """Fetch detailed model information from Ollama."""
        try:
            response = await self._client.post(
                f"{self.base_url}/api/show",
                json={"name": self.model}
            )
            
            if response.status_code == 200:
                model_data = response.json()
                return {
                    "name": self.model,
                    "backend": "ollama",
                    "details": model_data.get("details", {}),
                    "modelfile": model_data.get("modelfile", ""),
                    "parameters": model_data.get("parameters", {}),
                    "template": model_data.get("template", "")
                }
            else:
                logger.warning(f"Could not fetch model info: {response.status_code}")
                return self.get_model_info()
                
        except Exception as e:
            logger.warning(f"Failed to fetch model info: {e}")
            return self.get_model_info()
    
    async def _test_completion(self) -> bool:
        """Test a simple completion to verify model functionality."""
        try:
            # Direct API call to avoid recursion with generate_completion
            payload = {
                "model": self.model,
                "prompt": "Complete this: The sky is",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 5
                }
            }
            
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "").strip()
                return len(response_text) > 0
            else:
                return False
                
        except Exception as e:
            logger.debug(f"Test completion failed: {e}")
            return False