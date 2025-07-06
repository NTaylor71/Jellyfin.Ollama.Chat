"""
LLM Provider Service - FastAPI service hosting Ollama/LLM operations.

Dedicated service for LLM operations, separate from NLP providers for
better resource isolation and scaling.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.shared.config import get_settings
from src.concept_expansion.providers.llm.llm_provider import LLMProvider

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


class LLMProviderManager:
    """Manages LLM provider instance and its lifecycle."""
    
    def __init__(self):
        self.provider: Optional[LLMProvider] = None
        self.initialization_error: Optional[str] = None
        self.request_count = 0
        self.error_count = 0
        self.start_time = asyncio.get_event_loop().time()
    
    async def initialize_provider(self):
        """Initialize LLM provider."""
        logger.info("Initializing LLM provider...")
        
        try:
            from src.concept_expansion.providers.llm.llm_provider import LLMProvider
            self.provider = LLMProvider()
            if await self.provider.initialize():
                logger.info("✅ LLM provider initialized")
            else:
                self.initialization_error = "LLM provider initialization failed"
                logger.error("❌ LLM provider initialization failed")
        except ImportError as e:
            self.initialization_error = f"Import error: {e}"
            logger.error(f"❌ LLM provider import failed: {e}")
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"❌ LLM provider error: {e}")
        
        logger.info("LLM Provider initialization complete.")
    
    async def cleanup_provider(self):
        """Cleanup LLM provider."""
        logger.info("Cleaning up LLM provider...")
        if self.provider:
            try:
                if hasattr(self.provider, 'cleanup'):
                    await self.provider.cleanup()
                logger.info("✅ LLM provider cleaned up")
            except Exception as e:
                logger.error(f"❌ Error cleaning up LLM provider: {e}")
    
    def get_provider(self) -> LLMProvider:
        """Get the LLM provider."""
        if not self.provider:
            raise HTTPException(
                status_code=503, 
                detail=f"LLM provider not available. Error: {self.initialization_error}"
            )
        return self.provider
    
    def get_health_status(self) -> LLMServiceHealth:
        """Get service health status."""
        current_time = asyncio.get_event_loop().time()
        uptime = current_time - self.start_time
        
        provider_status = {}
        models_available = []
        
        if self.provider:
            try:
                metadata = self.provider.metadata
                provider_status = {
                    "status": "healthy",
                    "type": metadata.provider_type,
                    "version": metadata.version,
                    "initialized": getattr(self.provider, '_initialized', False)
                }
                
                # Get available models if possible
                if hasattr(self.provider, 'get_available_models'):
                    models_available = self.provider.get_available_models()
                
            except Exception as e:
                provider_status = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            provider_status = {
                "status": "failed_to_initialize",
                "error": self.initialization_error or "Unknown error"
            }
        
        return LLMServiceHealth(
            status="healthy" if self.provider else "degraded",
            uptime_seconds=uptime,
            llm_provider=provider_status,
            total_requests=self.request_count,
            failed_requests=self.error_count,
            models_available=models_available
        )


# Global provider manager
provider_manager = LLMProviderManager()


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
    title="LLM Provider Service",
    description="FastAPI service hosting Ollama/LLM operations",
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


@app.get("/health", response_model=LLMServiceHealth)
async def health_check():
    """Get service health status."""
    return provider_manager.get_health_status()


@app.get("/provider")
async def get_provider_info():
    """Get LLM provider information."""
    provider = provider_manager.get_provider()
    metadata = provider.metadata
    
    return {
        "name": metadata.name,
        "type": metadata.provider_type,
        "version": metadata.version,
        "strengths": metadata.strengths,
        "best_for": metadata.best_for,
        "context_aware": metadata.context_aware,
        "initialization_error": provider_manager.initialization_error
    }


@app.post("/expand", response_model=LLMResponse)
async def expand_concept(
    request: LLMRequest,
    background_tasks: BackgroundTasks
):
    """Expand a concept using the LLM provider."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        provider = provider_manager.get_provider()
        provider_manager.request_count += 1
        
        # Import ExpansionRequest here to avoid import issues
        from src.concept_expansion.providers.base_provider import ExpansionRequest
        
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
        
        return LLMResponse(
            success=result.success,
            execution_time_ms=execution_time_ms,
            result=result.enhanced_data if result.success else None,
            error_message=result.error_message if not result.success else None,
            metadata={
                "confidence_score": result.confidence_score.overall if result.success else None,
                "cache_key": result.cache_key.generate_key() if result.success else None,
                "model_used": getattr(result.plugin_metadata, 'model_used', None)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        provider_manager.error_count += 1
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        logger.error(f"Error expanding concept with LLM: {e}")
        
        return LLMResponse(
            success=False,
            execution_time_ms=execution_time_ms,
            error_message=str(e)
        )


@app.post("/provider/health")
async def check_provider_health():
    """Check health of LLM provider."""
    try:
        provider = provider_manager.get_provider()
        
        if hasattr(provider, 'health_check'):
            health_status = await provider.health_check()
        else:
            health_status = {
                "status": "healthy",
                "initialized": getattr(provider, '_initialized', False)
            }
        
        return {
            "provider": "llm",
            "health": health_status,
            "metadata": {
                "name": provider.metadata.name,
                "type": provider.metadata.provider_type,
                "version": provider.metadata.version
            }
        }
        
    except Exception as e:
        return {
            "provider": "llm",
            "health": {
                "status": "error",
                "error": str(e)
            }
        }


@app.get("/models")
async def list_available_models():
    """List available LLM models."""
    try:
        provider = provider_manager.get_provider()
        
        if hasattr(provider, 'get_available_models'):
            models = provider.get_available_models()
            return {
                "available_models": models,
                "current_model": getattr(provider, 'current_model', None)
            }
        else:
            return {
                "available_models": [],
                "current_model": None,
                "note": "Model listing not supported by this provider"
            }
            
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {
            "available_models": [],
            "error": str(e)
        }


@app.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """Load a specific model (if supported by provider)."""
    try:
        provider = provider_manager.get_provider()
        
        if hasattr(provider, 'load_model'):
            success = await provider.load_model(model_name)
            return {
                "success": success,
                "model": model_name,
                "message": f"Model {model_name} {'loaded' if success else 'failed to load'}"
            }
        else:
            return {
                "success": False,
                "model": model_name,
                "error": "Dynamic model loading not supported by this provider"
            }
            
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return {
            "success": False,
            "model": model_name,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    
    # Run the service
    uvicorn.run(
        "src.services.llm_provider_service:app",
        host="0.0.0.0",
        port=8002,
        reload=False,  # Disable reload for testing
        log_level=settings.LOG_LEVEL.lower()
    )