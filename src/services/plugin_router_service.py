"""
Plugin Router Service - Routes plugin requests to appropriate services.

Central orchestration service that routes plugin execution requests to
the correct provider services based on plugin type and requirements.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
import httpx

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

from src.shared.config import get_settings
from src.plugins.endpoint_config import get_endpoint_mapper

logger = logging.getLogger(__name__)


class PluginExecutionRequest(BaseModel):
    """Request for plugin execution."""
    plugin_name: str = Field(..., min_length=1)
    plugin_type: str = Field(..., min_length=1)
    data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class PluginExecutionResponse(BaseModel):
    """Response from plugin execution."""
    success: bool
    execution_time_ms: float
    service_used: str
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ServiceRoute(BaseModel):
    """Service route configuration."""
    url: str
    service_type: str
    health_endpoint: str
    status: str = "unknown"
    last_check: Optional[float] = None


class RouterHealth(BaseModel):
    """Router service health."""
    status: str
    uptime_seconds: float
    services: Dict[str, ServiceRoute]
    total_requests: int
    failed_requests: int


class PluginRouter:
    """Routes plugin requests to appropriate services."""
    
    def __init__(self):
        self.services: Dict[str, ServiceRoute] = {}
        self.plugin_service_mapping: Dict[str, str] = {}
        self.request_count = 0
        self.error_count = 0
        self.start_time = asyncio.get_event_loop().time()
        self.http_client: Optional[httpx.AsyncClient] = None
        self.endpoint_mapper = get_endpoint_mapper()
    
    async def initialize_router(self):
        """Initialize the router with service discovery."""
        logger.info("Initializing Plugin Router...")
        
        settings = get_settings()
        
        # Configure service routes using config
        self.services = {
            "nlp_provider": ServiceRoute(
                url=settings.nlp_service_url,
                service_type="nlp",
                health_endpoint="/health"
            ),
            "llm_provider": ServiceRoute(
                url=settings.llm_service_url, 
                service_type="llm",
                health_endpoint="/health"
            )
        }
        
        # Plugin-to-service mapping (populated dynamically)
        self.plugin_service_mapping = {}
        
        # Discover plugins dynamically
        await self._discover_available_plugins()
        
        # Initialize HTTP client
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Perform initial health checks
        await self.check_all_service_health()
        
        logger.info(f"Plugin Router initialized. {len(self.services)} services, {len(self.plugin_service_mapping)} plugins configured.")
    
    async def _discover_available_plugins(self):
        """Dynamically discover available HTTP-only plugins."""
        try:
            from pathlib import Path
            import importlib.util
            
            field_enrichment_dir = Path("src/plugins/field_enrichment")
            if not field_enrichment_dir.exists():
                logger.warning("Field enrichment plugins directory not found")
                return
            
            for plugin_file in field_enrichment_dir.glob("*_plugin.py"):
                try:
                    # Extract plugin name from filename and convert to proper class name
                    module_name = plugin_file.stem
                    # Handle special cases like LLM, SUTime, HeidelTime
                    words = module_name.split('_')
                    class_name_parts = []
                    for word in words:
                        if word.lower() == 'llm':
                            class_name_parts.append('LLM')
                        elif word.lower() == 'sutime':
                            class_name_parts.append('SUTime')
                        elif word.lower() == 'heideltime':
                            class_name_parts.append('HeidelTime')
                        elif word.lower() == 'conceptnet':
                            class_name_parts.append('ConceptNet')
                        else:
                            class_name_parts.append(word.capitalize())
                    plugin_class_name = ''.join(class_name_parts)
                    
                    # Use configuration-driven service mapping
                    service_name, _ = self.endpoint_mapper.get_service_and_endpoint(plugin_class_name)
                    # Map to service provider names for consistency  
                    service_provider = f"{service_name}_provider"
                    
                    self.plugin_service_mapping[plugin_class_name] = service_provider
                    logger.debug(f"Discovered plugin: {plugin_class_name} -> {service_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process plugin file {plugin_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to discover plugins: {e}")
    
    async def cleanup_router(self):
        """Cleanup router resources."""
        logger.info("Cleaning up Plugin Router...")
        if self.http_client:
            await self.http_client.aclose()
    
    async def check_service_health(self, service_name: str) -> bool:
        """Check health of a specific service."""
        if service_name not in self.services:
            return False
        
        service = self.services[service_name]
        
        try:
            response = await self.http_client.get(
                f"{service.url}{service.health_endpoint}",
                timeout=5.0
            )
            
            if response.status_code == 200:
                service.status = "healthy"
                service.last_check = asyncio.get_event_loop().time()
                return True
            else:
                service.status = f"unhealthy_http_{response.status_code}"
                return False
                
        except Exception as e:
            service.status = f"error_{str(e)[:50]}"
            logger.warning(f"Health check failed for {service_name}: {e}")
            return False
    
    async def check_all_service_health(self):
        """Check health of all services."""
        tasks = [
            self.check_service_health(service_name) 
            for service_name in self.services.keys()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_service_for_plugin(self, plugin_name: str, plugin_type: str) -> str:
        """Determine which service should handle a plugin."""
        # Direct plugin name mapping
        if plugin_name in self.plugin_service_mapping:
            return self.plugin_service_mapping[plugin_name]
        
        # Plugin type-based routing
        if plugin_type in ["concept_expansion", "temporal_analysis"]:
            return "nlp_provider"
        elif plugin_type in ["query_processing", "llm_concept"]:
            return "llm_provider"
        
        # Default to NLP provider for unknown plugins
        logger.warning(f"Unknown plugin {plugin_name} ({plugin_type}), routing to nlp_provider")
        return "nlp_provider"
    
    async def route_plugin_execution(
        self, 
        plugin_name: str, 
        plugin_type: str, 
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> PluginExecutionResponse:
        """Route plugin execution to appropriate service."""
        start_time = asyncio.get_event_loop().time()
        self.request_count += 1
        
        try:
            # Determine target service
            service_name = self.get_service_for_plugin(plugin_name, plugin_type)
            
            if service_name not in self.services:
                raise HTTPException(
                    status_code=404,
                    detail=f"Service {service_name} not configured"
                )
            
            service = self.services[service_name]
            
            # Check service health
            if service.status != "healthy":
                await self.check_service_health(service_name)
                if service.status != "healthy":
                    raise HTTPException(
                        status_code=503,
                        detail=f"Service {service_name} is unhealthy: {service.status}"
                    )
            
            # Route to appropriate service endpoint
            if service.service_type == "nlp":
                # For NLP service, determine the specific provider
                provider_name = self._map_plugin_to_provider(plugin_name)
                endpoint = f"{service.url}/providers/{provider_name}/expand"
                
                # Convert data to provider request format
                request_data = {
                    "concept": data.get("concept", ""),
                    "media_context": data.get("media_context", "movie"),
                    "max_concepts": data.get("max_concepts", 10),
                    "field_name": data.get("field_name", "concept"),
                    "options": data.get("options", {})
                }
                
            elif service.service_type == "llm":
                endpoint = f"{service.url}/expand"
                
                # Convert data to LLM request format
                request_data = {
                    "concept": data.get("concept", ""),
                    "media_context": data.get("media_context", "movie"),
                    "max_concepts": data.get("max_concepts", 10),
                    "field_name": data.get("field_name", "concept"),
                    "options": data.get("options", {})
                }
            
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unknown service type: {service.service_type}"
                )
            
            # Execute request
            response = await self.http_client.post(
                endpoint,
                json=request_data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result_data = response.json()
                execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                
                return PluginExecutionResponse(
                    success=result_data.get("success", False),
                    execution_time_ms=execution_time_ms,
                    service_used=service_name,
                    result=result_data.get("result"),
                    error_message=result_data.get("error_message"),
                    metadata={
                        "service_execution_time_ms": result_data.get("execution_time_ms"),
                        "endpoint": endpoint,
                        **result_data.get("metadata", {})
                    }
                )
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Service request failed: {response.text}"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            self.error_count += 1
            execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            logger.error(f"Error routing plugin {plugin_name}: {e}")
            
            return PluginExecutionResponse(
                success=False,
                execution_time_ms=execution_time_ms,
                service_used="error",
                error_message=str(e)
            )
    
    def _map_plugin_to_provider(self, plugin_name: str) -> str:
        """Map plugin name to service endpoint using configuration."""
        # Use configuration-driven endpoint mapping
        service_name, endpoint_path = self.endpoint_mapper.get_service_and_endpoint(plugin_name)
        return endpoint_path
    
    def get_health_status(self) -> RouterHealth:
        """Get router health status."""
        current_time = asyncio.get_event_loop().time()
        uptime = current_time - self.start_time
        
        return RouterHealth(
            status="healthy" if any(s.status == "healthy" for s in self.services.values()) else "degraded",
            uptime_seconds=uptime,
            services=self.services,
            total_requests=self.request_count,
            failed_requests=self.error_count
        )


# Global router
router = PluginRouter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await router.initialize_router()
    yield
    # Shutdown
    await router.cleanup_router()


# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title="Plugin Router Service",
    description="Routes plugin requests to appropriate services",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=settings.CORS_METHODS,
        allow_headers=settings.CORS_HEADERS,
    )

# Prometheus metrics (creates /metrics endpoint)
if settings.ENABLE_METRICS:
    Instrumentator().instrument(app).expose(app)


@app.get("/health", response_model=RouterHealth)
async def health_check():
    """Get router health status."""
    return router.get_health_status()


@app.post("/services/health-check")
async def refresh_service_health():
    """Refresh health status of all services."""
    await router.check_all_service_health()
    return {"message": "Health check completed", "services": router.services}


@app.get("/services")
async def list_services():
    """List all configured services."""
    return {
        "services": router.services,
        "plugin_mapping": router.plugin_service_mapping
    }


@app.post("/execute", response_model=PluginExecutionResponse)
async def execute_plugin(
    request: PluginExecutionRequest,
    background_tasks: BackgroundTasks
):
    """Execute a plugin via the appropriate service."""
    return await router.route_plugin_execution(
        plugin_name=request.plugin_name,
        plugin_type=request.plugin_type,
        data=request.data,
        context=request.context
    )


@app.get("/plugins/{plugin_name}/service")
async def get_plugin_service(plugin_name: str, plugin_type: str = "unknown"):
    """Get which service handles a specific plugin."""
    service_name = router.get_service_for_plugin(plugin_name, plugin_type)
    service = router.services.get(service_name)
    
    return {
        "plugin_name": plugin_name,
        "plugin_type": plugin_type,
        "service_name": service_name,
        "service_url": service.url if service else None,
        "service_status": service.status if service else "not_found"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the service
    uvicorn.run(
        "src.services.plugin_router_service:app",
        host="0.0.0.0",
        port=8003,
        reload=False,  # Disable reload for testing
        log_level=settings.LOG_LEVEL.lower()
    )