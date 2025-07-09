"""
Minimal LLM Provider Service - Just demonstrates the architecture works.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

from src.shared.config import get_settings
from src.services.base_service import BaseService
from src.shared.model_manager import ModelManager, ModelStatus

# Import will be added when config loader is stable
try:
    from src.shared.model_config_loader import get_model_config_loader
    CONFIG_LOADER_AVAILABLE = True
except ImportError:
    CONFIG_LOADER_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMRequest(BaseModel):
    """Request to LLM provider."""
    concept: str = Field(..., min_length=1, max_length=500)
    media_context: str = Field(default="movie")
    max_concepts: int = Field(default=10, ge=1, le=50)
    field_name: str = Field(default="concept")
    options: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Response from LLM provider."""
    success: bool
    execution_time_ms: float
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMServiceHealth(BaseModel):
    """LLM service health status."""
    status: str
    uptime_seconds: float
    llm_provider: Dict[str, Any]
    total_requests: int
    failed_requests: int
    models_available: List[str]
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


class MinimalLLMManager(BaseService):
    """LLM manager with real Ollama integration."""
    
    def __init__(self):
        self.http_client = None
        self.request_count = 0
        self.error_count = 0
        self.start_time = asyncio.get_event_loop().time()
        self.settings = get_settings()
        self.ollama_url = self.settings.OLLAMA_CHAT_BASE_URL
        self.model = self._get_chat_model()
        self.initialization_state = "starting"  # starting -> connecting_ollama -> ready
        self.initialization_progress = {}
        logger.info(f"Initialized LLM manager with Ollama URL: {self.ollama_url}, model: {self.model}")
    
    def _get_chat_model(self) -> str:
        """Get chat model from YAML config, fallback to env var."""
        if CONFIG_LOADER_AVAILABLE:
            try:
                config_loader = get_model_config_loader()
                ollama_models = config_loader.get_models_for_service('ollama')
                
                # Look for chat model in YAML config
                chat_model_config = ollama_models.get('chat_model')
                if chat_model_config:
                    logger.info(f"Using chat model from YAML config: {chat_model_config.name}")
                    return chat_model_config.name
                    
            except Exception as e:
                logger.warning(f"Failed to load chat model from YAML config: {e}")
        
        # Fallback to environment variable or default
        try:
            model_name = self.settings.OLLAMA_CHAT_MODEL
            logger.info(f"Using chat model from environment variable: {model_name}")
            return model_name
        except AttributeError:
            # Final fallback if env var is not defined
            default_model = "mistral:latest"
            logger.info(f"Using default chat model: {default_model}")
            return default_model
    
    async def initialize_provider(self):
        """Initialize real Ollama LLM provider."""
        logger.info("Initializing Ollama LLM provider...")
        self.initialization_state = "connecting_ollama"
        self.initialization_progress = {"phase": "connecting_ollama", "current_task": "establishing HTTP client"}
        
        try:
            self.http_client = httpx.AsyncClient(timeout=30.0)
            self.initialization_progress["current_task"] = "connecting to Ollama API"
            
            # Test Ollama connection
            response = await self.http_client.get(f"{self.ollama_url}/api/tags")
            if response.status_code != 200:
                self.initialization_progress["error"] = f"Ollama not available: {response.status_code}"
                logger.error(f"Ollama not available: {response.status_code}")
                return False
                
            # Skip model downloading - Model Manager Service will handle this
            logger.info("Skipping model check/download - Model Manager Service will orchestrate this")
                
            self.initialization_state = "ready"
            self.initialization_progress = {
                "phase": "completed",
                "status": "ready",
                "model_download_status": "delegated_to_model_manager",
                "ollama_url": self.ollama_url
            }
            logger.info("✅ Ollama LLM Provider initialization complete.")
            return True
            
        except Exception as e:
            self.initialization_progress["error"] = f"Error initializing Ollama provider: {e}"
            logger.error(f"Error initializing Ollama provider: {e}")
            if self.http_client:
                await self.http_client.aclose()
                self.http_client = None
            return False
    
    def get_provider(self):
        """Get the LLM provider."""
        if not self.http_client:
            raise HTTPException(
                status_code=503,
                detail="Ollama LLM provider not available"
            )
        return self
    
    async def check_models_ready(self) -> bool:
        """Check if all required models are available."""
        try:
            # Call our own models status endpoint
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8002/models/status")
                if response.status_code == 200:
                    status_data = response.json()
                    missing_required = status_data.get("summary", {}).get("missing_required", 0)
                    return missing_required == 0
                return False
        except Exception as e:
            logger.error(f"Error checking models ready: {e}")
            return False
    
    async def get_health_status(self) -> LLMServiceHealth:
        """Get service health status."""
        current_time = asyncio.get_event_loop().time()
        uptime = current_time - self.start_time
        
        provider_status = {}
        models_available = []
        
        if self.http_client:
            try:
                # Test Ollama connection
                response = await self.http_client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    models_data = response.json()
                    models_available = [model.get("name", "") for model in models_data.get("models", [])]
                    
                    provider_status = {
                        "status": "healthy",
                        "type": "ollama",
                        "version": "1.0.0",
                        "initialized": True,
                        "model": self.model,
                        "backend_url": self.ollama_url
                    }
                else:
                    provider_status = {
                        "status": "unhealthy",
                        "type": "ollama",
                        "error": f"HTTP {response.status_code}"
                    }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                provider_status = {
                    "status": "unhealthy",
                    "type": "ollama",
                    "error": str(e)
                }
        else:
            provider_status = {
                "status": "not_initialized",
                "type": "ollama"
            }
        
        # Determine overall service status based on initialization state
        service_status = "starting"
        if self.initialization_state == "ready":
            service_status = "healthy" if self.http_client and provider_status.get("status") == "healthy" else "degraded"
        elif self.initialization_state in ["connecting_ollama", "checking_model"]:
            service_status = "initializing"
        else:
            service_status = "starting"
        
        return LLMServiceHealth(
            status=service_status,
            uptime_seconds=uptime,
            llm_provider=provider_status,
            total_requests=self.request_count,
            failed_requests=self.error_count,
            models_available=models_available,
            models_ready=False  # This will be set by the endpoint
        )


# Global provider manager
provider_manager = MinimalLLMManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    try:
        initialization_success = await provider_manager.initialize_provider()
        if not initialization_success:
            logger.warning("Ollama provider initialization failed - service will run in degraded mode")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        
    yield
    
    # Shutdown - cleanup HTTP client
    try:
        if provider_manager.http_client:
            await provider_manager.http_client.aclose()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title="Minimal LLM Provider Service",
    description="Minimal FastAPI service for testing architecture",
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


@app.get("/health", response_model=LLMServiceHealth)
async def health_check():
    """Get service health status."""
    health = await provider_manager.get_health_status()
    # Add models status to health check
    health.models_ready = await provider_manager.check_models_ready()
    return health


@app.get("/health/detailed")
async def detailed_health_check():
    """Get detailed health status with initialization progress."""
    health = await provider_manager.get_health_status()
    
    return {
        "status": health.status,
        "initialization_state": provider_manager.initialization_state,
        "initialization_progress": provider_manager.initialization_progress,
        "uptime_seconds": health.uptime_seconds,
        "llm_provider": health.llm_provider,
        "total_requests": health.total_requests,
        "failed_requests": health.failed_requests,
        "models_available": health.models_available,
        "ready": provider_manager.initialization_state == "ready"
    }


@app.get("/health/ready")
async def readiness_check():
    """Simple readiness check for Docker health checks."""
    if provider_manager.initialization_state == "ready":
        return {"ready": True, "status": "healthy"}
    else:
        return {"ready": False, "status": provider_manager.initialization_state}, 503


@app.get("/providers")
async def get_provider_info():
    """Get LLM provider information."""
    if not provider_manager.http_client:
        raise HTTPException(
            status_code=503,
            detail="Ollama LLM provider not available"
        )
    
    try:
        # Get available models
        response = await provider_manager.http_client.get(f"{provider_manager.ollama_url}/api/tags")
        models_available = []
        status = "unknown"
        
        if response.status_code == 200:
            models_data = response.json()
            models_available = [model.get("name", "") for model in models_data.get("models", [])]
            status = "healthy"
        
        return {
            "name": "Ollama LLM Provider",
            "type": "ollama",
            "version": "1.0.0",
            "model": provider_manager.model,
            "backend_url": provider_manager.ollama_url,
            "available_models": models_available,
            "status": status
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Error getting provider info: {str(e)}"
        )


@app.post("/providers/llm/expand", response_model=LLMResponse)
async def expand_concept(request: LLMRequest):
    """Expand a concept using the real Ollama LLM provider."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        provider_manager.get_provider()  # Verify provider is available
        provider_manager.request_count += 1
        
        # Build prompt for concept expansion
        prompt = f"""You are a {request.media_context} expert. For the concept "{request.concept}", list {request.max_concepts} related concepts that would help users find similar {request.media_context}s.

Guidelines:
- Focus on {request.media_context}-specific themes and elements
- Include synonyms, related genres, and common characteristics
- Consider what audiences who like "{request.concept}" would also enjoy
- Be specific to {request.media_context} content

Format: Return only a comma-separated list of concepts, no explanations.

Concept: {request.concept}
Related {request.media_context} concepts:"""
        
        # Make Ollama API call
        payload = {
            "model": provider_manager.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Focused temperature for concept expansion
                "num_predict": 200,  # Reasonable limit for concept lists
                "stop": ["\n\n", "Explanation:", "Note:"]
            }
        }
        
        # Get response from Ollama
        response = await provider_manager.http_client.post(
            f"{provider_manager.ollama_url}/api/generate",
            json=payload
        )
        
        # If model not found, return error (Model Manager Service handles downloads)
        if response.status_code == 404:
            logger.error(f"Model {provider_manager.model} not found. Models should be downloaded via Model Manager Service.")
            return LLMResponse(
                success=False,
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                error_message=f"Model {provider_manager.model} not available. Please ensure models are downloaded via Model Manager Service."
            )
        
        if response.status_code == 200:
            response_data = response.json()
            concepts_text = response_data.get("response", "").strip()
            
            # Parse concepts from response text
            expanded_concepts = [
                concept.strip() 
                for concept in concepts_text.split(',') 
                if concept.strip()
            ][:request.max_concepts]
            
            result = {
                "expanded_concepts": expanded_concepts,
                "confidence_scores": [0.9] * len(expanded_concepts),  # Ollama doesn't provide confidence
                "llm_reasoning": f"Generated {len(expanded_concepts)} concepts related to '{request.concept}' in {request.media_context} context using {provider_manager.model}"
            }
            
            success = True
            error_message = None
        else:
            result = {
                "expanded_concepts": [],
                "confidence_scores": [],
                "llm_reasoning": f"Failed to generate concepts: HTTP {response.status_code}"
            }
            success = False
            error_message = f"Ollama API returned status {response.status_code}"
        
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Normalize Unicode text in the result
        normalized_result = provider_manager.normalize_text(result)
        
        return LLMResponse(
            success=success,
            execution_time_ms=execution_time_ms,
            result=normalized_result,
            error_message=error_message,
            metadata={
                "real_ollama_response": True,
                "model_used": provider_manager.model,
                "request_concept": request.concept,
                "backend_url": provider_manager.ollama_url
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        provider_manager.error_count += 1
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        logger.error(f"Error expanding concept with Ollama: {e}")
        
        return LLMResponse(
            success=False,
            execution_time_ms=execution_time_ms,
            error_message=str(e)
        )


# =============================================================================
# MODEL MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/models/status", response_model=ModelStatusResponse)
async def get_models_status():
    """Get status of all Ollama models."""
    try:
        # Create a temporary ModelManager to check status
        # This doesn't download models, just checks their status
        model_manager = ModelManager(models_base_path="/app/models")
        
        # Filter to only Ollama models
        ollama_models = {
            k: v for k, v in model_manager.models.items()
            if v.package == "ollama"
        }
        
        # Check status of Ollama models
        models_info = {}
        for model_id, model_info in ollama_models.items():
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
    """Download specific Ollama models."""
    try:
        logger.info(f"Downloading Ollama models - model_ids: {request.model_ids}, force: {request.force_download}")
        
        # Create ModelManager for model info
        model_manager = ModelManager(models_base_path="/app/models")
        
        # Filter to only Ollama models
        ollama_models = {
            k: v for k, v in model_manager.models.items()
            if v.package == "ollama"
        }
        
        # If specific model IDs requested, filter further
        if request.model_ids:
            ollama_models = {
                k: v for k, v in ollama_models.items()
                if k in request.model_ids
            }
        
        downloaded_models = []
        failed_models = []
        
        # Download each model using Ollama API
        for model_id, model_info in ollama_models.items():
            if not model_info.required and not request.force_download:
                continue
                
            try:
                # Check if model needs downloading
                current_status = await model_manager._check_model_status(model_info)
                
                if request.force_download or current_status == ModelStatus.MISSING:
                    logger.info(f"Pulling Ollama model {model_info.name}")
                    
                    # Use Ollama API to pull model
                    pull_response = await provider_manager.http_client.post(
                        f"{provider_manager.ollama_url}/api/pull",
                        json={"name": model_info.name},
                        timeout=600.0  # 10 minutes for model download
                    )
                    
                    if pull_response.status_code == 200:
                        downloaded_models.append(model_id)
                        logger.info(f"✅ Downloaded {model_id}")
                    else:
                        failed_models.append(model_id)
                        logger.error(f"❌ Failed to pull {model_id}: {pull_response.status_code}")
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
        "src.services.provider_services.minimal_llm_service:app",
        host="0.0.0.0",
        port=8002,
        reload=settings.is_development,
        log_level=settings.LOG_LEVEL.lower()
    )