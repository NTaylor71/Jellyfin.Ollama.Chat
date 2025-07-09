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
        self.initialization_state = "starting"  # starting -> discovering_services -> checking_dependencies -> ready
        self.initialization_progress = {}
        self.use_split_architecture = True  # Always use split architecture
    
    def _check_split_architecture(self) -> bool:
        """Always use split architecture mode."""
        return True
    
    def _configure_service_routes(self, settings) -> Dict[str, ServiceRoute]:
        """Configure service routes for split architecture."""
        import os
        
        services = {}
        
        logger.info("🔄 Configuring split architecture service routes")
        
        # Split NLP services with defaults
        split_services = {
            "conceptnet_provider": ("CONCEPTNET_SERVICE_URL", "http://conceptnet-service:8001"),
            "gensim_provider": ("GENSIM_SERVICE_URL", "http://gensim-service:8006"),
            "spacy_provider": ("SPACY_SERVICE_URL", "http://spacy-service:8007"),
            "heideltime_provider": ("HEIDELTIME_SERVICE_URL", "http://heideltime-service:8008")
        }
        
        for service_name, (env_var, default_url) in split_services.items():
            service_url = os.getenv(env_var, default_url)
            service_type = service_name.replace("_provider", "")
            services[service_name] = ServiceRoute(
                url=service_url,
                service_type=service_type,
                health_endpoint="/health/ready"
            )
            logger.info(f"  ✅ {service_name}: {service_url}")
        
        # Always include LLM service
        services["llm_provider"] = ServiceRoute(
            url=settings.llm_service_url,
            service_type="llm",
            health_endpoint="/health/ready"
        )
        logger.info(f"  ✅ llm_provider: {settings.llm_service_url}")
        
        return services
    
    def _map_plugin_to_split_service(self, plugin_class_name: str, service_name: str) -> str:
        """Map plugin to split service provider."""
        # Plugin-to-service mapping for split architecture
        plugin_service_mapping = {
            "ConceptNetKeywordPlugin": "conceptnet_provider",
            "GensimSimilarityPlugin": "gensim_provider",
            "SpacyTemporalPlugin": "spacy_provider",
            "HeidelTimeTemporalPlugin": "heideltime_provider",
            "LLMKeywordPlugin": "llm_provider",
            "LLMQAPlugin": "llm_provider",
            "LLMKeywordExtractionPlugin": "llm_provider"
        }
        
        # Check if we have a specific mapping for this plugin
        if plugin_class_name in plugin_service_mapping:
            mapped_service = plugin_service_mapping[plugin_class_name]
            logger.debug(f"Split architecture: {plugin_class_name} -> {mapped_service}")
            return mapped_service
        
        # Fallback to service name mapping
        if service_name == "nlp":
            logger.warning(f"Plugin {plugin_class_name} mapped to nlp service but using split architecture - routing to conceptnet_provider")
            return "conceptnet_provider"  # Default fallback
        
        return f"{service_name}_provider"
    
    async def initialize_router(self):
        """Initialize the router with service discovery."""
        logger.info("Initializing Plugin Router...")
        self.initialization_state = "discovering_services"
        self.initialization_progress = {"phase": "discovering_services", "current_task": "configuring service routes"}
        
        settings = get_settings()
        
        # Configure service routes based on architecture mode
        self.services = self._configure_service_routes(settings)
        
        # Plugin-to-service mapping (populated dynamically)
        self.plugin_service_mapping = {}
        
        # Discover plugins dynamically
        self.initialization_progress["current_task"] = "discovering available plugins"
        await self._discover_available_plugins()
        
        # Initialize HTTP client
        self.initialization_progress["current_task"] = "initializing HTTP client"
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Check dependencies (wait for services to be ready)
        self.initialization_state = "checking_dependencies"
        self.initialization_progress = {"phase": "checking_dependencies", "current_task": "waiting for service dependencies"}
        
        # Perform initial health checks
        await self.check_all_service_health()
        
        # Mark as ready when dependencies are healthy
        healthy_services = [s for s in self.services.values() if s.status == "healthy"]
        if len(healthy_services) == len(self.services):
            self.initialization_state = "ready"
            self.initialization_progress = {
                "phase": "completed",
                "services_ready": len(healthy_services),
                "total_services": len(self.services),
                "plugins_configured": len(self.plugin_service_mapping)
            }
            logger.info(f"✅ Plugin Router ready. {len(self.services)} services, {len(self.plugin_service_mapping)} plugins configured.")
        else:
            self.initialization_progress["error"] = "Some services are not healthy"
            logger.warning(f"⚠️ Plugin Router initialized but some services unhealthy: {len(healthy_services)}/{len(self.services)}")
    
    async def _discover_available_plugins(self):
        """Dynamically discover available HTTP-only plugins."""
        try:
            from pathlib import Path
            import importlib.util
            
            enrichment_dir = Path("src/plugins/enrichment")
            if not enrichment_dir.exists():
                logger.warning("Field enrichment plugins directory not found")
                return
            
            for plugin_file in enrichment_dir.glob("*_plugin.py"):
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
                    
                    # Map to service provider names using split architecture
                    service_provider = self._map_plugin_to_split_service(plugin_class_name, service_name)
                    
                    self.plugin_service_mapping[plugin_class_name] = service_provider
                    logger.debug(f"Discovered plugin: {plugin_class_name} -> {service_provider}")
                    
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
            return "conceptnet_provider"
        elif plugin_type in ["query_processing", "llm_concept"]:
            return "llm_provider"
        
        # Default to ConceptNet provider for unknown plugins
        logger.warning(f"Unknown plugin {plugin_name} ({plugin_type}), routing to conceptnet_provider")
        return "conceptnet_provider"
    
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
                # For split NLP services, use direct provider endpoint
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
                endpoint = f"{service.url}/providers/llm/expand"
                
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
        """Map plugin name to provider name using configuration."""
        # Use configuration-driven endpoint mapping
        service_name, endpoint_path = self.endpoint_mapper.get_service_and_endpoint(plugin_name)
        
        # Extract provider name from endpoint path (e.g., "providers/gensim/expand" -> "gensim")
        if endpoint_path.startswith("providers/") and "/" in endpoint_path[10:]:
            provider_name = endpoint_path.split("/")[1]
            logger.debug(f"Mapped {plugin_name} to provider: {provider_name}")
            return provider_name
        else:
            # Fallback for malformed paths
            logger.warning(f"Unexpected endpoint path format: {endpoint_path}, using as-is")
            return endpoint_path
    
    def get_health_status(self) -> RouterHealth:
        """Get router health status."""
        current_time = asyncio.get_event_loop().time()
        uptime = current_time - self.start_time
        
        # Determine overall service status based on initialization state
        service_status = "starting"
        if self.initialization_state == "ready":
            service_status = "healthy" if any(s.status == "healthy" for s in self.services.values()) else "degraded"
        elif self.initialization_state in ["discovering_services", "checking_dependencies"]:
            service_status = "initializing"
        else:
            service_status = "starting"
        
        return RouterHealth(
            status=service_status,
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


@app.get("/health/detailed")
async def detailed_health_check():
    """Get detailed health status with initialization progress."""
    health = router.get_health_status()
    
    return {
        "status": health.status,
        "initialization_state": router.initialization_state,
        "initialization_progress": router.initialization_progress,
        "uptime_seconds": health.uptime_seconds,
        "services": health.services,
        "total_requests": health.total_requests,
        "failed_requests": health.failed_requests,
        "ready": router.initialization_state == "ready"
    }


@app.get("/health/ready")
async def readiness_check():
    """Simple readiness check for Docker health checks."""
    if router.initialization_state == "ready":
        return {"ready": True, "status": "healthy"}
    else:
        return {"ready": False, "status": router.initialization_state}, 503


@app.post("/services/health")
async def refresh_service_health():
    """Refresh health status of all services."""
    await router.check_all_service_health()
    return {"message": "Health check completed", "services": router.services}


@app.get("/services")
async def list_services():
    """List all configured services."""
    return {
        "architecture": "split",
        "services": router.services,
        "plugin_mapping": router.plugin_service_mapping,
        "total_services": len(router.services)
    }


@app.post("/plugins/execute", response_model=PluginExecutionResponse)
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
        "src.services.orchestration.plugin_router_service:app",
        host="0.0.0.0",
        port=8003,
        reload=False,  # Disable reload for testing
        log_level=settings.LOG_LEVEL.lower()
    )