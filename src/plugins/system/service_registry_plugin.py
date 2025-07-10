"""
Service Registry Plugin - Manages service discovery and health monitoring.

Plugin that tracks available services, their health status, and capabilities
for dynamic service routing and failover.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
import httpx

from src.plugins.base import (
    BasePlugin, PluginMetadata, PluginResourceRequirements, 
    PluginExecutionContext, PluginExecutionResult, PluginType, ExecutionPriority
)
from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class ServiceInfo:
    """Information about a registered service."""
    
    def __init__(
        self, 
        name: str, 
        url: str, 
        service_type: str,
        capabilities: List[str] = None
    ):
        self.name = name
        self.url = url
        self.service_type = service_type
        self.capabilities = capabilities or []
        self.status = "unknown"
        self.last_health_check = None
        self.consecutive_failures = 0
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "url": self.url,
            "service_type": self.service_type,
            "capabilities": self.capabilities,
            "status": self.status,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "consecutive_failures": self.consecutive_failures,
            "metadata": self.metadata
        }


class ServiceRegistryPlugin(BasePlugin):
    """
    Plugin for service discovery and health monitoring.
    
    Features:
    - Automatic service discovery
    - Health monitoring with failover
    - Capability-based service selection
    - Service load balancing hints
    """
    
    def __init__(self):
        super().__init__()
        self.services: Dict[str, ServiceInfo] = {}
        self.http_client: Optional[httpx.AsyncClient] = None
        self.health_check_interval = 30
        self.health_check_task: Optional[asyncio.Task] = None
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="ServiceRegistryPlugin",
            version="1.0.0",
            description="Service discovery and health monitoring plugin",
            author="System",
            plugin_type=PluginType.GENERAL,
            tags=["service-discovery", "health-monitoring", "microservices"],
            execution_priority=ExecutionPriority.HIGH
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        return PluginResourceRequirements(
            min_cpu_cores=0.1,
            preferred_cpu_cores=0.2,
            min_memory_mb=50.0,
            preferred_memory_mb=100.0,
            max_execution_time_seconds=10.0
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the service registry."""
        try:
            logger.info("Initializing Service Registry Plugin...")
            
            
            self.http_client = httpx.AsyncClient(timeout=10.0)
            

            await self._register_default_services()
            

            self.health_check_task = asyncio.create_task(self._health_monitor_loop())
            
            logger.info(f"✅ Service Registry initialized with {len(self.services)} services")
            return True
            
        except Exception as e:
            logger.error(f"❌ Service Registry initialization failed: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup service registry resources."""
        logger.info("Cleaning up Service Registry...")
        

        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        

        if self.http_client:
            await self.http_client.aclose()
        
        logger.info("✅ Service Registry cleanup complete")
    
    async def execute(self, data: Any, context: PluginExecutionContext) -> PluginExecutionResult:
        """Execute service registry operations."""
        try:
            operation = data.get("operation") if isinstance(data, dict) else "status"
            
            if operation == "status":
                result = await self._get_registry_status()
            elif operation == "register":
                result = await self._register_service(data)
            elif operation == "unregister":
                result = await self._unregister_service(data)
            elif operation == "health_check":
                result = await self._perform_health_checks()
            elif operation == "discover":
                result = await self._discover_services(data)
            else:
                return PluginExecutionResult(
                    success=False,
                    error_message=f"Unknown operation: {operation}"
                )
            
            return PluginExecutionResult(
                success=True,
                data=result,
                metadata={"operation": operation}
            )
            
        except Exception as e:
            logger.error(f"Service registry operation failed: {e}")
            return PluginExecutionResult(
                success=False,
                error_message=str(e)
            )
    
    async def _register_default_services(self):
        """Register services discovered dynamically from health endpoints."""
        from src.shared.dynamic_service_discovery import get_service_discovery
        
        try:
            
            discovery = await get_service_discovery()
            discovered_services = await discovery.discover_all_services()
            

            for service_name, capability_info in discovered_services.items():
                self.services[service_name] = ServiceInfo(
                    name=service_name,
                    url=capability_info.base_url,
                    service_type=capability_info.service_type,
                    capabilities=capability_info.capabilities
                )
                
                
                self.services[service_name].status = capability_info.status
                self.services[service_name].last_health_check = capability_info.last_discovery
                self.services[service_name].metadata = capability_info.metadata
            
            logger.info(f"Dynamically registered {len(self.services)} services via discovery")
            
        except Exception as e:
            logger.error(f"Dynamic service registration failed: {e}")

            logger.warning("Operating with empty service registry - services will be discovered on demand")
    
    async def _health_monitor_loop(self):
        """Background task for continuous health monitoring."""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_checks(self) -> Dict[str, Any]:
        """Perform health checks on all registered services."""
        results = {}
        
        for service_name, service in self.services.items():
            try:

                health_url = f"{service.url}/health"
                response = await self.http_client.get(health_url, timeout=5.0)
                
                if response.status_code == 200:
                    service.status = "healthy"
                    service.consecutive_failures = 0
                    service.last_health_check = datetime.utcnow()
                    
                    
                    try:
                        health_data = response.json()
                        service.metadata = health_data
                    except:
                        pass
                    
                    results[service_name] = "healthy"
                else:
                    service.status = f"unhealthy_http_{response.status_code}"
                    service.consecutive_failures += 1
                    results[service_name] = f"unhealthy_{response.status_code}"
                
            except Exception as e:
                service.status = f"error_{str(e)[:50]}"
                service.consecutive_failures += 1
                results[service_name] = f"error"
                logger.warning(f"Health check failed for {service_name}: {e}")
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "results": results,
            "healthy_services": len([s for s in self.services.values() if s.status == "healthy"]),
            "total_services": len(self.services)
        }
    
    async def _get_registry_status(self) -> Dict[str, Any]:
        """Get current registry status."""
        return {
            "services": {name: service.to_dict() for name, service in self.services.items()},
            "healthy_count": len([s for s in self.services.values() if s.status == "healthy"]),
            "total_count": len(self.services),
            "last_health_check": max(
                (s.last_health_check for s in self.services.values() if s.last_health_check),
                default=None
            )
        }
    
    async def _register_service(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new service."""
        required_fields = ["name", "url", "service_type"]
        if not all(field in data for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")
        
        service = ServiceInfo(
            name=data["name"],
            url=data["url"],
            service_type=data["service_type"],
            capabilities=data.get("capabilities", [])
        )
        
        self.services[data["name"]] = service
        

        await self._perform_health_checks()
        
        return {
            "message": f"Service {data['name']} registered successfully",
            "service": service.to_dict()
        }
    
    async def _unregister_service(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Unregister a service."""
        service_name = data.get("name")
        if not service_name:
            raise ValueError("Service name required")
        
        if service_name in self.services:
            del self.services[service_name]
            return {"message": f"Service {service_name} unregistered successfully"}
        else:
            raise ValueError(f"Service {service_name} not found")
    
    async def _discover_services(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Discover services based on criteria."""
        service_type = data.get("service_type")
        capability = data.get("capability")
        status = data.get("status", "healthy")
        
        matching_services = []
        
        for service in self.services.values():

            if service_type and service.service_type != service_type:
                continue
            

            if capability and capability not in service.capabilities:
                continue
            

            if status and service.status != status:
                continue
            
            matching_services.append(service.to_dict())
        
        return {
            "matching_services": matching_services,
            "count": len(matching_services),
            "filters": {
                "service_type": service_type,
                "capability": capability,
                "status": status
            }
        }
    

    
    def get_healthy_services(self, service_type: Optional[str] = None) -> List[ServiceInfo]:
        """Get list of healthy services, optionally filtered by type."""
        services = [
            service for service in self.services.values() 
            if service.status == "healthy"
        ]
        
        if service_type:
            services = [s for s in services if s.service_type == service_type]
        
        return services
    
    def get_service_by_capability(self, capability: str) -> Optional[ServiceInfo]:
        """Get a healthy service that has the specified capability."""
        for service in self.services.values():
            if (service.status == "healthy" and 
                capability in service.capabilities):
                return service
        return None
    
    def get_best_service_for_plugin(self, plugin_name: str, plugin_type: str) -> Optional[ServiceInfo]:
        """Get the best service for executing a specific plugin."""

        if plugin_type in ["concept_expansion", "temporal_analysis"]:
            return self.get_service_by_capability("concept_expansion")
        elif plugin_type in ["query_processing", "llm_concept"]:
            return self.get_service_by_capability("semantic_understanding")
        

        healthy_services = self.get_healthy_services()
        return healthy_services[0] if healthy_services else None