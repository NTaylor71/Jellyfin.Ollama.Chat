"""
HTTP Provider Plugin Base Class
Base class for plugins that call HTTP services with circuit breaker and retry logic.
"""

import asyncio
import logging
import time
from abc import abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import aiohttp
from pydantic import BaseModel, Field

from src.plugins.base import (
    BasePlugin, PluginMetadata, PluginResourceRequirements, 
    PluginExecutionContext, PluginExecutionResult, PluginType
)
from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Service is down, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back up


@dataclass
class ServiceEndpoint:
    """HTTP service endpoint configuration."""
    name: str
    url: str
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    health_check_path: str = "/health"
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: float = 60.0


class HTTPRequest(BaseModel):
    """HTTP request model for service calls."""
    endpoint: str
    method: str = "POST"
    data: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    timeout: Optional[float] = None


class HTTPResponse(BaseModel):
    """HTTP response model from service calls."""
    success: bool
    status_code: int
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: float
    service_name: str


class CircuitBreakerInfo:
    """Circuit breaker state management."""
    
    def __init__(self, service_name: str, failure_threshold: int = 5, timeout_seconds: float = 60.0):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.last_success_time = time.time()
    
    def record_success(self):
        """Record successful call."""
        self.failure_count = 0
        self.last_success_time = time.time()
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker for {self.service_name} closed - service recovered")
    
    def record_failure(self):
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED and self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker for {self.service_name} opened - service failing")
    
    def can_execute(self) -> bool:
        """Check if we can execute a request."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            # Check if timeout has elapsed
            if time.time() - self.last_failure_time > self.timeout_seconds:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker for {self.service_name} half-open - testing service")
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "can_execute": self.can_execute()
        }


class HTTPProviderPlugin(BasePlugin):
    """
    Base class for plugins that call HTTP services.
    
    Features:
    - Circuit breaker pattern for service failures
    - Automatic retry with exponential backoff
    - Service health monitoring
    - Request/response logging and metrics
    - Environment-aware URL configuration
    """
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.services: Dict[str, ServiceEndpoint] = {}
        self.circuit_breakers: Dict[str, CircuitBreakerInfo] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self._request_count = 0
        self._failed_requests = 0
        self._total_response_time = 0.0
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.__class__.__name__,
            version="1.0.0",
            description="HTTP service provider plugin",
            author="System",
            plugin_type=PluginType.GENERAL
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        return PluginResourceRequirements(
            min_cpu_cores=0.5,
            preferred_cpu_cores=1.0,
            min_memory_mb=50.0,
            preferred_memory_mb=200.0,
            requires_gpu=False,
            max_execution_time_seconds=60.0
        )
    
    @abstractmethod
    def get_service_endpoints(self) -> List[ServiceEndpoint]:
        """Get list of service endpoints this plugin uses."""
        pass
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize HTTP client and service endpoints."""
        try:
            # Initialize HTTP session
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            timeout = aiohttp.ClientTimeout(total=60.0)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": f"UniversalMediaIngestion/{self.metadata.version}",
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
            )
            
            # Initialize service endpoints
            endpoints = self.get_service_endpoints()
            for endpoint in endpoints:
                self.services[endpoint.name] = endpoint
                if endpoint.circuit_breaker_enabled:
                    self.circuit_breakers[endpoint.name] = CircuitBreakerInfo(
                        service_name=endpoint.name,
                        failure_threshold=endpoint.circuit_breaker_failure_threshold,
                        timeout_seconds=endpoint.circuit_breaker_timeout_seconds
                    )
            
            self._logger.info(f"Initialized HTTP provider with {len(self.services)} services")
            
            # Test service connectivity
            healthy_services = await self._check_all_services_health()
            if not healthy_services:
                self._logger.warning("No services are healthy - plugin may have limited functionality")
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize HTTP provider: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def call_service(
        self, 
        service_name: str, 
        request: HTTPRequest,
        context: Optional[PluginExecutionContext] = None
    ) -> HTTPResponse:
        """
        Call a service with circuit breaker and retry logic.
        """
        if service_name not in self.services:
            return HTTPResponse(
                success=False,
                status_code=0,
                error_message=f"Service '{service_name}' not configured",
                execution_time_ms=0.0,
                service_name=service_name
            )
        
        service = self.services[service_name]
        circuit_breaker = self.circuit_breakers.get(service_name)
        
        # Check circuit breaker
        if circuit_breaker and not circuit_breaker.can_execute():
            return HTTPResponse(
                success=False,
                status_code=0,
                error_message=f"Circuit breaker open for service '{service_name}'",
                execution_time_ms=0.0,
                service_name=service_name
            )
        
        # Execute request with retries
        start_time = time.time()
        last_error = None
        
        for attempt in range(service.retry_attempts):
            try:
                response = await self._execute_http_request(service, request)
                
                # Record success
                if circuit_breaker:
                    circuit_breaker.record_success()
                
                self._record_request_success(time.time() - start_time)
                return response
                
            except Exception as e:
                last_error = e
                self._logger.warning(f"Attempt {attempt + 1} failed for {service_name}: {e}")
                
                if attempt < service.retry_attempts - 1:
                    await asyncio.sleep(service.retry_delay_seconds * (2 ** attempt))
        
        # All attempts failed
        if circuit_breaker:
            circuit_breaker.record_failure()
        
        execution_time_ms = (time.time() - start_time) * 1000
        self._record_request_failure(execution_time_ms / 1000)
        
        return HTTPResponse(
            success=False,
            status_code=0,
            error_message=f"All {service.retry_attempts} attempts failed. Last error: {last_error}",
            execution_time_ms=execution_time_ms,
            service_name=service_name
        )
    
    async def _execute_http_request(
        self, 
        service: ServiceEndpoint, 
        request: HTTPRequest
    ) -> HTTPResponse:
        """Execute a single HTTP request."""
        if not self.session:
            raise RuntimeError("HTTP session not initialized")
        
        url = f"{service.url.rstrip('/')}/{request.endpoint.lstrip('/')}"
        timeout = request.timeout or service.timeout_seconds
        headers = request.headers or {}
        
        start_time = time.time()
        
        async with self.session.request(
            method=request.method,
            url=url,
            json=request.data,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            execution_time_ms = (time.time() - start_time) * 1000
            
            if response.status == 200:
                try:
                    response_data = await response.json()
                    return HTTPResponse(
                        success=True,
                        status_code=response.status,
                        data=response_data,
                        execution_time_ms=execution_time_ms,
                        service_name=service.name
                    )
                except Exception as e:
                    return HTTPResponse(
                        success=False,
                        status_code=response.status,
                        error_message=f"Failed to parse JSON response: {e}",
                        execution_time_ms=execution_time_ms,
                        service_name=service.name
                    )
            else:
                error_text = await response.text()
                return HTTPResponse(
                    success=False,
                    status_code=response.status,
                    error_message=f"HTTP {response.status}: {error_text}",
                    execution_time_ms=execution_time_ms,
                    service_name=service.name
                )
    
    async def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service."""
        if service_name not in self.services:
            return {"healthy": False, "error": f"Service '{service_name}' not configured"}
        
        service = self.services[service_name]
        circuit_breaker = self.circuit_breakers.get(service_name)
        
        try:
            health_request = HTTPRequest(
                endpoint=service.health_check_path,
                method="GET",
                timeout=5.0
            )
            
            response = await self._execute_http_request(service, health_request)
            
            health_info = {
                "healthy": response.success,
                "status_code": response.status_code,
                "response_time_ms": response.execution_time_ms,
                "url": f"{service.url}{service.health_check_path}"
            }
            
            if circuit_breaker:
                health_info["circuit_breaker"] = circuit_breaker.get_status()
            
            if not response.success:
                health_info["error"] = response.error_message
            
            return health_info
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "url": f"{service.url}{service.health_check_path}",
                "circuit_breaker": circuit_breaker.get_status() if circuit_breaker else None
            }
    
    async def _check_all_services_health(self) -> List[str]:
        """Check health of all services and return healthy ones."""
        healthy_services = []
        
        for service_name in self.services:
            health = await self.check_service_health(service_name)
            if health.get("healthy", False):
                healthy_services.append(service_name)
        
        return healthy_services
    
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
        
        service_health = {}
        for service_name in self.services:
            service_health[service_name] = await self.check_service_health(service_name)
        
        return {
            "total_requests": self._request_count,
            "failed_requests": self._failed_requests,
            "success_rate": (
                (self._request_count - self._failed_requests) / self._request_count 
                if self._request_count > 0 else 0.0
            ),
            "average_response_time_seconds": avg_response_time,
            "services": service_health
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check including service status."""
        base_health = await super().health_check()
        plugin_metrics = await self.get_plugin_metrics()
        
        base_health.update({
            "http_metrics": plugin_metrics,
            "session_active": self.session is not None
        })
        
        return base_health