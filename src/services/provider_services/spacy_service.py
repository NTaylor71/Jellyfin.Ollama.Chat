"""
SpaCy Provider Service - Service for SpaCy NLP and temporal processing.

This service provides SpaCy NLP functionality with temporal processing
capabilities for entity extraction and temporal intelligence.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

from src.shared.config import get_settings
from src.services.base_service import BaseService
from src.shared.model_manager import ModelManager, ModelStatus

logger = logging.getLogger(__name__)


class ProviderRequest(BaseModel):
    """Request to SpaCy provider."""
    concept: str = Field(..., min_length=1, max_length=500)
    media_context: str = Field(default="movie")
    max_concepts: int = Field(default=10, ge=1, le=50)
    field_name: str = Field(default="concept")
    options: Dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    """Response from SpaCy provider."""
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
    models_ready: bool = False


class ModelInfo(BaseModel):
    """Information about a model."""
    name: str
    package: str
    storage_path: str
    size_mb: int
    required: bool
    status: str
    error_message: Optional[str] = None


class ModelStatusResponse(BaseModel):
    """Response for model status check."""
    success: bool
    models: Dict[str, ModelInfo]
    summary: Dict[str, Any]


class ModelDownloadRequest(BaseModel):
    """Request to download models."""
    model_ids: Optional[List[str]] = Field(default=None, description="Specific model IDs to download, or None for all required")
    force_download: bool = Field(default=False, description="Force re-download existing models")


class ModelDownloadResponse(BaseModel):
    """Response for model download."""
    success: bool
    downloaded_models: List[str]
    failed_models: List[str]
    error_message: Optional[str] = None


class SpacyProviderManager(BaseService):
    """Manages SpaCy provider instance."""
    
    def __init__(self):
        self.provider = None
        self.initialization_error = None
        self.request_count = 0
        self.error_count = 0
        self.start_time = None
        self.initialization_state = "starting"
    
    async def initialize_provider(self):
        """Initialize SpaCy provider."""
        logger.info("Initializing SpaCy provider...")
        self.start_time = asyncio.get_event_loop().time()
        self.initialization_state = "initializing"
        
        try:
            from src.providers.nlp.spacy_temporal_provider import SpacyTemporalProvider
            
            self.provider = SpacyTemporalProvider()
            if await self.provider.initialize():
                self.initialization_state = "ready"
                logger.info("✅ SpaCy provider initialized successfully")
            else:
                self.initialization_error = "Provider initialization failed"
                self.initialization_state = "failed"
                logger.error("❌ SpaCy provider initialization failed")
                
        except ImportError as e:
            self.initialization_error = f"Import error: {e}"
            self.initialization_state = "failed"
            logger.error(f"❌ SpaCy provider import failed: {e}")
        except Exception as e:
            self.initialization_error = str(e)
            self.initialization_state = "failed"
            logger.error(f"❌ SpaCy provider error: {e}")
    
    async def cleanup_provider(self):
        """Cleanup SpaCy provider."""
        logger.info("Cleaning up SpaCy provider...")
        if self.provider and hasattr(self.provider, 'cleanup'):
            try:
                await self.provider.cleanup()
                logger.info("✅ SpaCy provider cleaned up")
            except Exception as e:
                logger.error(f"❌ Error cleaning up SpaCy provider: {e}")
    
    def get_provider(self):
        """Get the SpaCy provider instance."""
        if self.provider is None:
            raise HTTPException(
                status_code=503,
                detail="SpaCy provider not available. Service may be initializing or failed to start."
            )
        return self.provider
    
    async def check_models_ready(self) -> bool:
        """Check if all required models are available."""
        try:
            # Call our own models status endpoint
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8007/models/status")
                if response.status_code == 200:
                    status_data = response.json()
                    missing_required = status_data.get("summary", {}).get("missing_required", 0)
                    return missing_required == 0
                return False
        except Exception as e:
            logger.error(f"Error checking models ready: {e}")
            return False
    
    def get_health_status(self) -> ServiceHealth:
        """Get service health status."""
        current_time = asyncio.get_event_loop().time()
        uptime = current_time - self.start_time if self.start_time else 0
        
        providers = {}
        if self.provider:
            try:
                metadata = self.provider.metadata
                providers["spacy_temporal"] = {
                    "status": "healthy",
                    "type": metadata.provider_type,
                    "version": metadata.version,
                    "initialized": True
                }
            except Exception as e:
                providers["spacy_temporal"] = {
                    "status": "error",
                    "error": str(e)
                }
        elif self.initialization_error:
            providers["spacy_temporal"] = {
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
provider_manager = SpacyProviderManager()


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
    title="SpaCy Provider Service",
    description="Service for SpaCy NLP and temporal processing functionality",
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
    health = provider_manager.get_health_status()
    health.models_ready = await provider_manager.check_models_ready()
    return health


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
            provider_info["spacy_temporal"] = {
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
        "available_providers": ["spacy_temporal"] if provider_manager.provider else [],
        "providers": provider_info,
        "initialization_error": provider_manager.initialization_error
    }


@app.post("/providers/spacy_temporal/expand", response_model=ProviderResponse)
async def expand_concept(request: ProviderRequest):
    """Expand a concept using SpaCy temporal processing."""
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
            provider_name="spacy_temporal",
            execution_time_ms=execution_time_ms,
            result=result.enhanced_data if result.success else None,
            error_message=result.error_message if not result.success else None,
            metadata={
                "confidence_score": result.confidence_score.overall if result.success else None,
                "cache_key": result.cache_key.generate_key() if result.success else None,
                "temporal_analysis": result.enhanced_data.get("temporal_expressions", {}) if result.success else None
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        provider_manager.error_count += 1
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        logger.error(f"Error expanding concept with SpaCy: {e}")
        
        return ProviderResponse(
            success=False,
            provider_name="spacy_temporal",
            execution_time_ms=execution_time_ms,
            error_message=str(e)
        )


@app.get("/providers/spacy_temporal/metadata")
async def get_provider_metadata():
    """Get SpaCy provider metadata."""
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


@app.post("/providers/spacy_temporal/health")
async def check_provider_health():
    """Check SpaCy provider health."""
    provider = provider_manager.get_provider()
    
    try:
        health_status = await provider.health_check()
        
        return {
            "provider": "spacy_temporal",
            "health": health_status,
            "metadata": {
                "name": provider.metadata.name,
                "type": provider.metadata.provider_type,
                "version": provider.metadata.version
            }
        }
        
    except Exception as e:
        return {
            "provider": "spacy_temporal",
            "health": {
                "status": "error",
                "error": str(e)
            }
        }


# =============================================================================
# MODEL MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/models/status", response_model=ModelStatusResponse)
async def get_models_status():
    """Get status of SpaCy models."""
    try:
        # Create ModelManager to check status
        model_manager = ModelManager(models_base_path="/app/models")
        
        # Filter to only SpaCy models
        spacy_models = {
            k: v for k, v in model_manager.models.items()
            if v.package == "spacy"
        }
        
        # Check status of SpaCy models
        models_info = {}
        for model_id, model_info in spacy_models.items():
            status = await model_manager._check_model_status(model_info)
            models_info[model_id] = ModelInfo(
                name=model_info.name,
                package=model_info.package,
                storage_path=model_info.storage_path,
                size_mb=model_info.size_mb,
                required=model_info.required,
                status=status.value,
                error_message=model_info.error_message
            )
        
        # Get summary
        available_count = sum(1 for m in models_info.values() if m.status == "available")
        required_count = sum(1 for m in models_info.values() if m.required)
        
        summary = {
            "total_models": len(models_info),
            "available_models": available_count,
            "required_models": required_count,
            "missing_required": required_count - available_count
        }
        
        return ModelStatusResponse(
            success=True,
            models=models_info,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Failed to get models status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/download", response_model=ModelDownloadResponse)
async def download_models(request: ModelDownloadRequest):
    """Download specific SpaCy models."""
    try:
        logger.info(f"Downloading SpaCy models - model_ids: {request.model_ids}, force: {request.force_download}")
        
        # Create ModelManager for downloading
        model_manager = ModelManager(models_base_path="/app/models")
        
        # Filter to only SpaCy models
        spacy_models = {
            k: v for k, v in model_manager.models.items()
            if v.package == "spacy"
        }
        
        # If specific model IDs requested, filter further
        if request.model_ids:
            spacy_models = {
                k: v for k, v in spacy_models.items()
                if k in request.model_ids
            }
        
        downloaded_models = []
        failed_models = []
        
        # Download each model
        for model_id, model_info in spacy_models.items():
            if not model_info.required and not request.force_download:
                continue
                
            try:
                # Check if model needs downloading
                current_status = await model_manager._check_model_status(model_info)
                
                if request.force_download or current_status == ModelStatus.MISSING:
                    logger.info(f"Downloading {model_info.package}/{model_info.name}")
                    success = await model_manager._download_model(model_info)
                    
                    if success:
                        downloaded_models.append(model_id)
                        logger.info(f"✅ Downloaded {model_id}")
                    else:
                        failed_models.append(model_id)
                        logger.error(f"❌ Failed to download {model_id}")
                else:
                    logger.info(f"Model {model_id} already available")
                    
            except Exception as e:
                logger.error(f"Error downloading {model_id}: {e}")
                failed_models.append(model_id)
        
        success = len(failed_models) == 0
        
        return ModelDownloadResponse(
            success=success,
            downloaded_models=downloaded_models,
            failed_models=failed_models,
            error_message=None if success else f"Failed to download {len(failed_models)} models"
        )
        
    except Exception as e:
        logger.error(f"Failed to download models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/ready")
async def models_ready_check():
    """Check if all required models are available."""
    models_ready = await provider_manager.check_models_ready()
    if models_ready:
        return {"models_ready": True}
    else:
        return {"models_ready": False}, 503


if __name__ == "__main__":
    import uvicorn
    
    # Run the service
    uvicorn.run(
        "src.services.provider_services.spacy_service:app",
        host="0.0.0.0",
        port=8007,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    )