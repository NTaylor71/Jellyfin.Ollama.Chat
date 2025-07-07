"""
Service Health Monitor Plugin
Monitors availability and health of all HTTP services in the system.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from src.plugins.http_provider_plugin import HTTPProviderPlugin, ServiceEndpoint, HTTPRequest
from src.plugins.base import (
    PluginMetadata, PluginResourceRequirements, PluginExecutionContext,
    PluginExecutionResult, PluginType, ExecutionPriority
)
from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Service health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealthHistory:
    """Track health history for a service."""
    service_name: str
    current_status: ServiceStatus = ServiceStatus.UNKNOWN
    last_check_time: datetime = field(default_factory=datetime.now)
    last_healthy_time: Optional[datetime] = None
    consecutive_failures: int = 0
    total_checks: int = 0
    total_failures: int = 0
    average_response_time_ms: float = 0.0
    recent_response_times: List[float] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    
    def record_check(self, healthy: bool, response_time_ms: float, error_message: Optional[str] = None):
        """Record a health check result."""
        self.total_checks += 1
        self.last_check_time = datetime.now()
        
        # Update response time tracking
        self.recent_response_times.append(response_time_ms)
        if len(self.recent_response_times) > 20:  # Keep last 20 measurements
            self.recent_response_times.pop(0)
        
        self.average_response_time_ms = sum(self.recent_response_times) / len(self.recent_response_times)
        
        if healthy:
            self.consecutive_failures = 0
            self.last_healthy_time = datetime.now()
            
            # Determine status based on response time
            if response_time_ms < 1000:  # < 1 second
                self.current_status = ServiceStatus.HEALTHY
            elif response_time_ms < 5000:  # < 5 seconds
                self.current_status = ServiceStatus.DEGRADED
            else:
                self.current_status = ServiceStatus.DEGRADED
        else:
            self.consecutive_failures += 1
            self.total_failures += 1
            self.current_status = ServiceStatus.UNHEALTHY
            
            if error_message:
                self.error_messages.append(f"{datetime.now().isoformat()}: {error_message}")
                if len(self.error_messages) > 10:  # Keep last 10 errors
                    self.error_messages.pop(0)
    
    def get_uptime_percentage(self, window_hours: int = 24) -> float:
        """Calculate uptime percentage over a time window."""
        if self.total_checks == 0:
            return 0.0
        
        # Simple calculation based on total success rate
        success_rate = (self.total_checks - self.total_failures) / self.total_checks
        return success_rate * 100.0
    
    def is_service_down(self, failure_threshold: int = 3) -> bool:
        """Check if service should be considered down."""
        return self.consecutive_failures >= failure_threshold
    
    def time_since_last_healthy(self) -> Optional[timedelta]:
        """Get time since last healthy check."""
        if self.last_healthy_time:
            return datetime.now() - self.last_healthy_time
        return None


class ServiceHealthMonitorPlugin(HTTPProviderPlugin):
    """
    Plugin that monitors health of all HTTP services.
    
    Features:
    - Continuous health monitoring
    - Health history tracking
    - Service availability metrics
    - Alerting for service failures
    - Performance monitoring
    - Circuit breaker integration
    """
    
    def __init__(self):
        super().__init__()
        self.service_history: Dict[str, ServiceHealthHistory] = {}
        self.monitoring_enabled = True
        self.monitoring_interval = 30.0  # seconds
        self.monitoring_task: Optional[asyncio.Task] = None
        self.alert_threshold_failures = 3
        self.performance_degradation_threshold = 2000.0  # ms
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="ServiceHealthMonitorPlugin",
            version="1.0.0",
            description="Monitors health and availability of all HTTP services",
            author="System",
            plugin_type=PluginType.GENERAL,
            tags=["monitoring", "health", "services", "availability"],
            execution_priority=ExecutionPriority.NORMAL
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        return PluginResourceRequirements(
            min_cpu_cores=0.25,
            preferred_cpu_cores=0.5,
            min_memory_mb=50.0,
            preferred_memory_mb=100.0,
            requires_gpu=False,
            max_execution_time_seconds=30.0
        )
    
    def get_service_endpoints(self) -> List[ServiceEndpoint]:
        """Get all service endpoints to monitor."""
        settings = get_settings()
        
        endpoints = []
        
        # Core application services
        if settings.nlp_service_url:
            endpoints.append(ServiceEndpoint(
                name="nlp_service",
                url=settings.nlp_service_url,
                timeout_seconds=10.0,
                retry_attempts=1,
                health_check_path="/health",
                circuit_breaker_enabled=False  # Monitor only, don't circuit break
            ))
        
        if settings.llm_service_url:
            endpoints.append(ServiceEndpoint(
                name="llm_service",
                url=settings.llm_service_url,
                timeout_seconds=15.0,
                retry_attempts=1,
                health_check_path="/health",
                circuit_breaker_enabled=False
            ))
        
        if settings.router_service_url:
            endpoints.append(ServiceEndpoint(
                name="router_service",
                url=settings.router_service_url,
                timeout_seconds=5.0,
                retry_attempts=1,
                health_check_path="/health",
                circuit_breaker_enabled=False
            ))
        
        # Infrastructure services
        endpoints.append(ServiceEndpoint(
            name="mongodb",
            url=f"http://{settings.mongodb_host}:{settings.mongodb_port}",
            timeout_seconds=5.0,
            retry_attempts=1,
            health_check_path="/",
            circuit_breaker_enabled=False
        ))
        
        # Redis (if available)
        if hasattr(settings, 'redis_url'):
            endpoints.append(ServiceEndpoint(
                name="redis",
                url=settings.redis_url,
                timeout_seconds=5.0,
                retry_attempts=1,
                health_check_path="/ping",
                circuit_breaker_enabled=False
            ))
        
        # FAISS service
        if settings.vectordb_url:
            endpoints.append(ServiceEndpoint(
                name="faiss_service",
                url=settings.vectordb_url,
                timeout_seconds=10.0,
                retry_attempts=1,
                health_check_path="/health",
                circuit_breaker_enabled=False
            ))
        
        # Ollama service
        if settings.ollama_chat_url:
            endpoints.append(ServiceEndpoint(
                name="ollama",
                url=settings.ollama_chat_url,
                timeout_seconds=10.0,
                retry_attempts=1,
                health_check_path="/api/tags",  # Ollama-specific health check
                circuit_breaker_enabled=False
            ))
        
        return endpoints
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize health monitoring."""
        if not await super().initialize(config):
            return False
        
        # Initialize health history for all services
        for service_name in self.services:
            self.service_history[service_name] = ServiceHealthHistory(service_name)
        
        # Configure monitoring parameters from config
        self.monitoring_interval = config.get("monitoring_interval", 30.0)
        self.alert_threshold_failures = config.get("alert_threshold_failures", 3)
        self.performance_degradation_threshold = config.get("performance_degradation_threshold", 2000.0)
        
        # Start continuous monitoring
        if self.monitoring_enabled:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._logger.info(f"Started health monitoring for {len(self.services)} services")
        
        return True
    
    async def cleanup(self) -> None:
        """Stop monitoring and cleanup."""
        self.monitoring_enabled = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        await super().cleanup()
    
    async def execute(self, data: Any, context: PluginExecutionContext) -> PluginExecutionResult:
        """Execute a health check on demand."""
        try:
            health_results = await self.check_all_services_health()
            
            return PluginExecutionResult(
                success=True,
                data={
                    "service_health": health_results,
                    "monitoring_status": {
                        "enabled": self.monitoring_enabled,
                        "interval_seconds": self.monitoring_interval,
                        "services_monitored": len(self.services)
                    }
                },
                metadata={
                    "check_time": datetime.now().isoformat(),
                    "total_services": len(self.services),
                    "healthy_services": len([s for s in health_results.values() if s["status"] == ServiceStatus.HEALTHY])
                }
            )
            
        except Exception as e:
            return PluginExecutionResult(
                success=False,
                error_message=f"Health check failed: {str(e)}"
            )
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        self._logger.info("Health monitoring loop started")
        
        while self.monitoring_enabled:
            try:
                await self.check_all_services_health()
                await self._check_for_alerts()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5.0)  # Brief pause before retrying
        
        self._logger.info("Health monitoring loop stopped")
    
    async def check_all_services_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all services and update history."""
        health_results = {}
        
        # Check all services in parallel
        tasks = []
        for service_name in self.services:
            tasks.append(self._check_single_service_health(service_name))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (service_name, result) in enumerate(zip(self.services.keys(), results)):
            if isinstance(result, Exception):
                self._logger.error(f"Health check failed for {service_name}: {result}")
                health_results[service_name] = {
                    "status": ServiceStatus.UNHEALTHY,
                    "error": str(result),
                    "response_time_ms": 0.0
                }
            else:
                health_results[service_name] = result
        
        return health_results
    
    async def _check_single_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a single service."""
        if service_name not in self.services:
            return {"status": ServiceStatus.UNKNOWN, "error": "Service not configured"}
        
        service = self.services[service_name]
        history = self.service_history[service_name]
        
        start_time = time.time()
        
        try:
            # Special handling for different service types
            if service_name == "ollama":
                health_result = await self._check_ollama_health(service)
            elif service_name == "mongodb":
                health_result = await self._check_mongodb_health(service)
            else:
                health_result = await self.check_service_health(service_name)
            
            response_time_ms = (time.time() - start_time) * 1000
            healthy = health_result.get("healthy", False)
            error_message = health_result.get("error")
            
            # Record in history
            history.record_check(healthy, response_time_ms, error_message)
            
            # Determine overall status
            if healthy:
                if response_time_ms > self.performance_degradation_threshold:
                    status = ServiceStatus.DEGRADED
                else:
                    status = ServiceStatus.HEALTHY
            else:
                status = ServiceStatus.UNHEALTHY
            
            return {
                "status": status,
                "healthy": healthy,
                "response_time_ms": response_time_ms,
                "consecutive_failures": history.consecutive_failures,
                "uptime_percentage": history.get_uptime_percentage(),
                "last_healthy": history.last_healthy_time.isoformat() if history.last_healthy_time else None,
                "error": error_message
            }
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            history.record_check(False, response_time_ms, str(e))
            
            return {
                "status": ServiceStatus.UNHEALTHY,
                "healthy": False,
                "response_time_ms": response_time_ms,
                "consecutive_failures": history.consecutive_failures,
                "uptime_percentage": history.get_uptime_percentage(),
                "error": str(e)
            }
    
    async def _check_ollama_health(self, service: ServiceEndpoint) -> Dict[str, Any]:
        """Special health check for Ollama service."""
        try:
            # Check if Ollama is responding and has models
            request = HTTPRequest(
                endpoint="api/tags",
                method="GET",
                timeout=5.0
            )
            
            response = await self._execute_http_request(service, request)
            
            if response.success and response.data:
                models = response.data.get("models", [])
                return {
                    "healthy": True,
                    "models_available": len(models),
                    "response_time_ms": response.execution_time_ms
                }
            else:
                return {
                    "healthy": False,
                    "error": response.error_message or "No models available"
                }
                
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def _check_mongodb_health(self, service: ServiceEndpoint) -> Dict[str, Any]:
        """Special health check for MongoDB service."""
        try:
            # Simple TCP connection check for MongoDB
            import socket
            host, port = service.url.replace("http://", "").split(":")
            port = int(port)
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            result = sock.connect_ex((host, port))
            sock.close()
            
            return {
                "healthy": result == 0,
                "error": "Connection failed" if result != 0 else None
            }
            
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def _check_for_alerts(self):
        """Check if any services need alerts."""
        for service_name, history in self.service_history.items():
            # Check for consecutive failures
            if history.consecutive_failures >= self.alert_threshold_failures:
                self._log_service_alert(
                    service_name, 
                    "UNHEALTHY", 
                    f"Service has failed {history.consecutive_failures} consecutive health checks"
                )
            
            # Check for performance degradation
            elif (history.current_status == ServiceStatus.DEGRADED and 
                  history.average_response_time_ms > self.performance_degradation_threshold):
                self._log_service_alert(
                    service_name,
                    "DEGRADED",
                    f"Service response time degraded to {history.average_response_time_ms:.1f}ms"
                )
    
    def _log_service_alert(self, service_name: str, alert_type: str, message: str):
        """Log service alert."""
        alert_message = f"SERVICE ALERT [{alert_type}] {service_name}: {message}"
        
        if alert_type == "UNHEALTHY":
            self._logger.error(alert_message)
        elif alert_type == "DEGRADED":
            self._logger.warning(alert_message)
        else:
            self._logger.info(alert_message)
    
    def get_service_summary(self) -> Dict[str, Any]:
        """Get summary of all service health."""
        summary = {
            "total_services": len(self.services),
            "healthy_services": 0,
            "degraded_services": 0,
            "unhealthy_services": 0,
            "unknown_services": 0,
            "overall_status": ServiceStatus.HEALTHY,
            "services": {}
        }
        
        for service_name, history in self.service_history.items():
            status = history.current_status
            summary["services"][service_name] = {
                "status": status,
                "uptime_percentage": history.get_uptime_percentage(),
                "average_response_time_ms": history.average_response_time_ms,
                "consecutive_failures": history.consecutive_failures
            }
            
            if status == ServiceStatus.HEALTHY:
                summary["healthy_services"] += 1
            elif status == ServiceStatus.DEGRADED:
                summary["degraded_services"] += 1
            elif status == ServiceStatus.UNHEALTHY:
                summary["unhealthy_services"] += 1
            else:
                summary["unknown_services"] += 1
        
        # Determine overall status
        if summary["unhealthy_services"] > 0:
            summary["overall_status"] = ServiceStatus.UNHEALTHY
        elif summary["degraded_services"] > 0:
            summary["overall_status"] = ServiceStatus.DEGRADED
        elif summary["healthy_services"] > 0:
            summary["overall_status"] = ServiceStatus.HEALTHY
        else:
            summary["overall_status"] = ServiceStatus.UNKNOWN
        
        return summary
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check including monitoring status."""
        base_health = await super().health_check()
        service_summary = self.get_service_summary()
        
        base_health.update({
            "monitoring_enabled": self.monitoring_enabled,
            "monitoring_interval": self.monitoring_interval,
            "service_summary": service_summary
        })
        
        return base_health