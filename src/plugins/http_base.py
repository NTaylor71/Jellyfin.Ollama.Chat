"""
HTTP Base Plugin Class
Simplified base class for all HTTP-only plugins in the reorganized architecture.
"""

import asyncio
import logging
import time
from abc import abstractmethod
from typing import Dict, Any, Optional, List
import aiohttp

from src.plugins.base import (
    BasePlugin, PluginMetadata, PluginResourceRequirements, 
    PluginExecutionContext, PluginExecutionResult, PluginType
)
from src.plugins.http_provider_plugin import (
    ServiceEndpoint, HTTPRequest, HTTPResponse, CircuitBreakerInfo
)
from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class HTTPBasePlugin(BasePlugin):
    """
    Base class for all HTTP-only plugins in the simplified architecture.
    
    All plugins inherit from this class and make HTTP calls to services.
    They do NOT manage providers directly - that's the services' job.
    
    Features:
    - Simplified HTTP client with circuit breaker
    - Standard error handling and retries  
    - Service health monitoring
    - Consistent plugin interface
    - Minimal resource requirements
    """
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.session: Optional[aiohttp.ClientSession] = None
        self.circuit_breakers: Dict[str, CircuitBreakerInfo] = {}
        self._request_count = 0
        self._failed_requests = 0
        self._total_response_time = 0.0
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        """HTTP plugins have minimal resource requirements."""
        return PluginResourceRequirements(
            min_cpu_cores=0.1,
            preferred_cpu_cores=0.5,
            min_memory_mb=25.0,
            preferred_memory_mb=100.0,
            requires_gpu=False,
            max_execution_time_seconds=30.0
        )
    
    @abstractmethod
    async def enrich_field(
        self, 
        field_name: str, 
        field_value: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich a field with HTTP service calls.
        
        Args:
            field_name: Name of the field being enriched
            field_value: Value of the field
            config: Plugin-specific configuration
            
        Returns:
            Dict containing enrichment results
        """
        pass
    
    async def execute(self, data: Any, context: PluginExecutionContext) -> PluginExecutionResult:
        """
        Execute the plugin with given data and context.
        
        This is the standard plugin interface. For HTTP plugins, we delegate
        to the enrich_field method for each field in the data.
        """
        start_time = time.time()
        
        try:
            # Handle different data types
            if isinstance(data, dict):
                # Process each field in the dictionary
                enriched_data = data.copy()
                
                # Add enriched_fields section if not present
                if "enriched_fields" not in enriched_data:
                    enriched_data["enriched_fields"] = {}
                
                # Process fields that need enrichment
                fields_to_enrich = getattr(context, 'fields_to_enrich', ['name', 'overview', 'description'])
                
                for field_name in fields_to_enrich:
                    if field_name in data:
                        field_config = getattr(context, 'field_configs', {}).get(field_name, {})
                        enrichment = await self.enrich_field(field_name, data[field_name], field_config)
                        enriched_data["enriched_fields"][f"{field_name}_{self.metadata.name.lower()}"] = enrichment
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                return PluginExecutionResult(
                    success=True,
                    data=enriched_data,
                    execution_time_ms=execution_time_ms,
                    metadata={
                        "plugin": self.metadata.name,
                        "enriched_fields": list(enriched_data["enriched_fields"].keys())
                    }
                )
            
            else:
                # For non-dict data, try to enrich directly
                field_config = getattr(context, 'config', {})
                enrichment = await self.enrich_field("data", data, field_config)
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                return PluginExecutionResult(
                    success=True,
                    data=enrichment,
                    execution_time_ms=execution_time_ms,
                    metadata={
                        "plugin": self.metadata.name,
                        "data_type": type(data).__name__
                    }
                )
                
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self._logger.error(f"Plugin execution failed: {e}")
            
            return PluginExecutionResult(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time_ms,
                metadata={
                    "plugin": self.metadata.name,
                    "error_type": type(e).__name__
                }
            )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize HTTP client."""
        try:
            # Initialize HTTP session with reasonable defaults for plugins
            connector = aiohttp.TCPConnector(
                limit=50,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            timeout = aiohttp.ClientTimeout(total=30.0)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": f"JellyfinOllamaChat/{self.metadata.version}",
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
            )
            
            # Initialize circuit breakers for known services
            self._init_circuit_breakers()
            
            self._logger.info(f"Initialized HTTP plugin: {self.metadata.name}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize HTTP plugin: {e}")
            return False
    
    def _init_circuit_breakers(self):
        """Initialize circuit breakers for common services."""
        services = [
            "keyword_service",
            "temporal_service", 
            "nlp_service",
            "llm_service"
        ]
        
        for service_name in services:
            self.circuit_breakers[service_name] = CircuitBreakerInfo(
                service_name=service_name,
                failure_threshold=5,
                timeout_seconds=60.0
            )
    
    async def cleanup(self) -> None:
        """Clean up HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def http_post(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP POST with standard error handling.
        
        Args:
            url: Complete service URL
            data: Request payload
            
        Returns:
            Response data
            
        Raises:
            RuntimeError: If request fails after retries
        """
        service_name = self._extract_service_name(url)
        response = await self._call_service_with_retries(url, "POST", data, service_name)
        
        if not response.success:
            raise RuntimeError(f"HTTP POST to {url} failed: {response.error_message}")
        
        return response.data or {}
    
    async def http_get(self, url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make HTTP GET with standard error handling.
        
        Args:
            url: Complete service URL
            params: Query parameters
            
        Returns:
            Response data
            
        Raises:
            RuntimeError: If request fails after retries
        """
        service_name = self._extract_service_name(url)
        response = await self._call_service_with_retries(url, "GET", params, service_name)
        
        if not response.success:
            raise RuntimeError(f"HTTP GET to {url} failed: {response.error_message}")
        
        return response.data or {}
    
    def _extract_service_name(self, url: str) -> str:
        """Extract service name from URL for circuit breaker tracking."""
        if "keyword" in url or ":8001" in url:
            return "keyword_service"
        elif "temporal" in url or ":8004" in url:
            return "temporal_service"
        elif "nlp" in url or ":8002" in url:
            return "nlp_service"
        elif "llm" in url or ":8003" in url:
            return "llm_service"
        else:
            return "unknown_service"
    
    async def _call_service_with_retries(
        self, 
        url: str, 
        method: str,
        data: Optional[Dict[str, Any]],
        service_name: str
    ) -> HTTPResponse:
        """Call service with circuit breaker and retry logic."""
        circuit_breaker = self.circuit_breakers.get(service_name)
        
        # Check circuit breaker
        if circuit_breaker and not circuit_breaker.can_execute():
            return HTTPResponse(
                success=False,
                status_code=0,
                error_message=f"Circuit breaker open for {service_name}",
                execution_time_ms=0.0,
                service_name=service_name
            )
        
        # Execute request with retries
        start_time = time.time()
        retry_attempts = 3
        retry_delay = 1.0
        last_error = None
        
        for attempt in range(retry_attempts):
            try:
                response = await self._execute_http_request(url, method, data)
                
                # Record success
                if circuit_breaker:
                    circuit_breaker.record_success()
                
                self._record_request_success(time.time() - start_time)
                return response
                
            except Exception as e:
                last_error = e
                self._logger.warning(f"Attempt {attempt + 1} failed for {service_name}: {e}")
                
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
        
        # All attempts failed
        if circuit_breaker:
            circuit_breaker.record_failure()
        
        execution_time_ms = (time.time() - start_time) * 1000
        self._record_request_failure(execution_time_ms / 1000)
        
        return HTTPResponse(
            success=False,
            status_code=0,
            error_message=f"All {retry_attempts} attempts failed. Last error: {last_error}",
            execution_time_ms=execution_time_ms,
            service_name=service_name
        )
    
    async def _execute_http_request(
        self, 
        url: str, 
        method: str, 
        data: Optional[Dict[str, Any]]
    ) -> HTTPResponse:
        """Execute a single HTTP request."""
        if not self.session:
            raise RuntimeError("HTTP session not initialized")
        
        start_time = time.time()
        timeout = aiohttp.ClientTimeout(total=30.0)
        
        try:
            if method == "GET":
                async with self.session.get(
                    url=url,
                    params=data,
                    timeout=timeout
                ) as response:
                    return await self._process_response(response, start_time, url)
            else:  # POST
                async with self.session.post(
                    url=url,
                    json=data,
                    timeout=timeout
                ) as response:
                    return await self._process_response(response, start_time, url)
                    
        except asyncio.TimeoutError:
            execution_time_ms = (time.time() - start_time) * 1000
            return HTTPResponse(
                success=False,
                status_code=0,
                error_message="Request timeout",
                execution_time_ms=execution_time_ms,
                service_name=self._extract_service_name(url)
            )
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return HTTPResponse(
                success=False,
                status_code=0,
                error_message=f"Request failed: {str(e)}",
                execution_time_ms=execution_time_ms,
                service_name=self._extract_service_name(url)
            )
    
    async def _process_response(self, response, start_time: float, url: str) -> HTTPResponse:
        """Process HTTP response and return standardized result."""
        execution_time_ms = (time.time() - start_time) * 1000
        service_name = self._extract_service_name(url)
        
        if response.status == 200:
            try:
                response_data = await response.json()
                return HTTPResponse(
                    success=True,
                    status_code=response.status,
                    data=response_data,
                    execution_time_ms=execution_time_ms,
                    service_name=service_name
                )
            except Exception as e:
                return HTTPResponse(
                    success=False,
                    status_code=response.status,
                    error_message=f"Failed to parse JSON response: {e}",
                    execution_time_ms=execution_time_ms,
                    service_name=service_name
                )
        else:
            try:
                error_text = await response.text()
            except:
                error_text = f"HTTP {response.status}"
                
            return HTTPResponse(
                success=False,
                status_code=response.status,
                error_message=error_text,
                execution_time_ms=execution_time_ms,
                service_name=service_name
            )
    
    def _record_request_success(self, execution_time_seconds: float):
        """Record successful request metrics."""
        self._request_count += 1
        self._total_response_time += execution_time_seconds
    
    def _record_request_failure(self, execution_time_seconds: float):
        """Record failed request metrics."""
        self._request_count += 1
        self._failed_requests += 1
        self._total_response_time += execution_time_seconds
    
    async def get_plugin_metrics(self) -> Dict[str, Any]:
        """Get plugin performance metrics."""
        avg_response_time = (
            self._total_response_time / self._request_count if self._request_count > 0 else 0.0
        )
        
        return {
            "total_requests": self._request_count,
            "failed_requests": self._failed_requests,
            "success_rate": (
                (self._request_count - self._failed_requests) / self._request_count 
                if self._request_count > 0 else 0.0
            ),
            "average_response_time_seconds": avg_response_time,
            "circuit_breakers": {
                name: cb.get_status() for name, cb in self.circuit_breakers.items()
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check including HTTP metrics."""
        base_health = {
            "plugin_name": self.metadata.name,
            "plugin_type": "http_based",
            "is_initialized": self._is_initialized,
            "session_active": self.session is not None
        }
        
        if self._is_initialized:
            plugin_metrics = await self.get_plugin_metrics()
            base_health["metrics"] = plugin_metrics
        
        return base_health
    
    def get_service_url(self, service_name: str, endpoint: str = "") -> str:
        """
        Get service URL from environment configuration.
        
        Args:
            service_name: Name of the service (e.g., "keyword", "temporal")
            endpoint: Optional endpoint path
            
        Returns:
            Complete service URL
        """
        # Get URLs from environment/config
        service_urls = {
            "keyword": getattr(self.settings, "KEYWORD_SERVICE_URL", "http://localhost:8001"),
            "temporal": getattr(self.settings, "TEMPORAL_SERVICE_URL", "http://localhost:8004"),
            "nlp": getattr(self.settings, "NLP_SERVICE_URL", "http://localhost:8002"),
            "llm": getattr(self.settings, "LLM_SERVICE_URL", "http://localhost:8003"),
        }
        
        base_url = service_urls.get(service_name, f"http://localhost:8000")
        
        if endpoint:
            return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        else:
            return base_url