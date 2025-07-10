"""
ConceptNet Provider Service - Lightweight API-only service for ConceptNet concept expansion.

This service provides ConceptNet functionality without any model dependencies,
making it the lightest service in the NLP provider ecosystem.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

from src.shared.config import get_settings
from src.services.base_service import BaseService

logger = logging.getLogger(__name__)


class ProviderRequest(BaseModel):
    """Request to ConceptNet provider."""
    concept: str = Field(..., min_length=1, max_length=500)
    media_context: str = Field(default="movie")
    max_concepts: int = Field(default=10, ge=1, le=50)
    field_name: str = Field(default="concept")
    options: Dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    """Response from ConceptNet provider."""
    success: bool
    provider_name: str
    execution_time_ms: float
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ServiceHealth(BaseModel):
    """Service health status."""
    status: str
    uptime_seconds: float
    providers: Dict[str, Dict[str, Any]]
    total_requests: int
    failed_requests: int


class ConceptNetProviderManager(BaseService):
    """Manages ConceptNet provider instance."""
    
    def __init__(self):
        self.provider = None
        self.initialization_error = None
        self.request_count = 0
        self.error_count = 0
        self.start_time = None
        self.initialization_state = "starting"
    
    async def initialize_provider(self):
        """Initialize ConceptNet provider."""
        logger.info("Initializing ConceptNet provider...")
        self.start_time = asyncio.get_event_loop().time()
        self.initialization_state = "initializing"
        
        try:
            from src.providers.knowledge.conceptnet_provider import ConceptNetProvider
            
            self.provider = ConceptNetProvider()
            if await self.provider.initialize():
                self.initialization_state = "ready"
                logger.info("✅ ConceptNet provider initialized successfully")
            else:
                self.initialization_error = "Provider initialization failed"
                self.initialization_state = "failed"
                logger.error("❌ ConceptNet provider initialization failed")
                
        except ImportError as e:
            self.initialization_error = f"Import error: {e}"
            self.initialization_state = "failed"
            logger.error(f"❌ ConceptNet provider import failed: {e}")
        except Exception as e:
            self.initialization_error = str(e)
            self.initialization_state = "failed"
            logger.error(f"❌ ConceptNet provider error: {e}")
    
    async def cleanup_provider(self):
        """Cleanup ConceptNet provider."""
        logger.info("Cleaning up ConceptNet provider...")
        if self.provider and hasattr(self.provider, 'cleanup'):
            try:
                await self.provider.cleanup()
                logger.info("✅ ConceptNet provider cleaned up")
            except Exception as e:
                logger.error(f"❌ Error cleaning up ConceptNet provider: {e}")
    
    def get_provider(self):
        """Get the ConceptNet provider instance."""
        if self.provider is None:
            raise HTTPException(
                status_code=503,
                detail="ConceptNet provider not available. Service may be initializing or failed to start."
            )
        return self.provider
    
    def get_health_status(self) -> ServiceHealth:
        """Get service health status."""
        current_time = asyncio.get_event_loop().time()
        uptime = current_time - self.start_time if self.start_time else 0
        
        providers = {}
        if self.provider:
            try:
                metadata = self.provider.metadata
                providers["conceptnet"] = {
                    "status": "healthy",
                    "type": metadata.provider_type,
                    "version": metadata.version,
                    "initialized": True
                }
            except Exception as e:
                providers["conceptnet"] = {
                    "status": "error",
                    "error": str(e)
                }
        elif self.initialization_error:
            providers["conceptnet"] = {
                "status": "failed_to_initialize",
                "error": self.initialization_error
            }
        
        # Service status based on provider availability
        if self.initialization_state == "ready" and self.provider:
            service_status = "healthy"
        elif self.initialization_state == "initializing":
            service_status = "initializing"
        else:
            service_status = "unhealthy"
        
        return ServiceHealth(
            status=service_status,
            uptime_seconds=uptime,
            providers=providers,
            total_requests=self.request_count,
            failed_requests=self.error_count
        )


# Global provider manager
provider_manager = ConceptNetProviderManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await provider_manager.initialize_provider()
    yield
    # Shutdown
    await provider_manager.cleanup_provider()


# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title="ConceptNet Provider Service",
    description="Lightweight API-only service for ConceptNet concept expansion",
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

# Prometheus metrics
if settings.ENABLE_METRICS:
    Instrumentator().instrument(app).expose(app)


@app.get("/health", response_model=ServiceHealth)
async def health_check():
    """Get service health status."""
    return provider_manager.get_health_status()


@app.get("/health/ready")
async def readiness_check():
    """Simple readiness check for Docker health checks."""
    if provider_manager.initialization_state == "ready":
        return {"ready": True, "status": "healthy"}
    else:
        return {"ready": False, "status": provider_manager.initialization_state}, 503


@app.get("/providers")
async def list_providers():
    """List available providers."""
    provider_info = {}
    if provider_manager.provider:
        try:
            metadata = provider_manager.provider.metadata
            provider_info["conceptnet"] = {
                "name": metadata.name,
                "type": metadata.provider_type,
                "version": metadata.version,
                "strengths": metadata.strengths,
                "best_for": metadata.best_for,
                "context_aware": metadata.context_aware
            }
        except Exception as e:
            logger.error(f"Error getting provider metadata: {e}")
    
    return {
        "available_providers": ["conceptnet"] if provider_manager.provider else [],
        "providers": provider_info,
        "initialization_error": provider_manager.initialization_error
    }


@app.post("/providers/conceptnet/expand", response_model=ProviderResponse)
async def expand_concept(request: ProviderRequest):
    """Expand a concept using ConceptNet."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        provider = provider_manager.get_provider()
        provider_manager.request_count += 1
        
        # Import ExpansionRequest
        from src.providers.nlp.base_provider import ExpansionRequest
        
        # Create expansion request
        expansion_request = ExpansionRequest(
            concept=request.concept,
            media_context=request.media_context,
            max_concepts=request.max_concepts,
            field_name=request.field_name,
            options=request.options
        )
        
        # Execute expansion
        result = await provider.expand_concept(expansion_request)
        
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return ProviderResponse(
            success=result.success,
            provider_name="conceptnet",
            execution_time_ms=execution_time_ms,
            result=result.enhanced_data if result.success else None,
            error_message=result.error_message if not result.success else None,
            metadata={
                "confidence_score": result.confidence_score.overall if result.success else None,
                "cache_key": result.cache_key.generate_key() if result.success else None,
                "api_calls": result.enhanced_data.get("api_calls", 0) if result.success else 0
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        provider_manager.error_count += 1
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        logger.error(f"Error expanding concept with ConceptNet: {e}")
        
        return ProviderResponse(
            success=False,
            provider_name="conceptnet",
            execution_time_ms=execution_time_ms,
            error_message=str(e)
        )


@app.get("/providers/conceptnet/metadata")
async def get_provider_metadata():
    """Get ConceptNet provider metadata."""
    provider = provider_manager.get_provider()
    metadata = provider.metadata
    
    return {
        "name": metadata.name,
        "provider_type": metadata.provider_type,
        "version": metadata.version,
        "context_aware": metadata.context_aware,
        "strengths": metadata.strengths,
        "weaknesses": metadata.weaknesses,
        "best_for": metadata.best_for
    }


@app.post("/providers/conceptnet/health")
async def check_provider_health():
    """Check ConceptNet provider health."""
    provider = provider_manager.get_provider()
    
    try:
        health_status = await provider.health_check()
        
        return {
            "provider": "conceptnet",
            "health": health_status,
            "metadata": {
                "name": provider.metadata.name,
                "type": provider.metadata.provider_type,
                "version": provider.metadata.version
            }
        }
        
    except Exception as e:
        return {
            "provider": "conceptnet",
            "health": {
                "status": "error",
                "error": str(e)
            }
        }


if __name__ == "__main__":
    import uvicorn
    
    # Run the service
    uvicorn.run(
        "src.services.provider_services.conceptnet_service:app",
        host="0.0.0.0",
        port=settings.CONCEPTNET_SERVICE_PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    )