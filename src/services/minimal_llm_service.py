"""
Minimal LLM Provider Service - Just demonstrates the architecture works.
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


class MinimalLLMManager:
    """Minimal LLM manager for testing."""
    
    def __init__(self):
        self.provider = None
        self.request_count = 0
        self.error_count = 0
        self.start_time = asyncio.get_event_loop().time()
    
    async def initialize_provider(self):
        """Initialize demo LLM provider."""
        logger.info("Initializing minimal LLM provider...")
        
        # Mock provider for demonstration
        self.provider = {
            "name": "Mock LLM Provider",
            "type": "llm",
            "version": "1.0.0",
            "status": "ready",
            "models": ["mock-model-1", "mock-model-2"]
        }
        
        logger.info("LLM Provider initialization complete.")
    
    def get_provider(self):
        """Get the LLM provider."""
        if not self.provider:
            raise HTTPException(
                status_code=503,
                detail="LLM provider not available"
            )
        return self.provider
    
    def get_health_status(self) -> LLMServiceHealth:
        """Get service health status."""
        current_time = asyncio.get_event_loop().time()
        uptime = current_time - self.start_time
        
        provider_status = {}
        models_available = []
        
        if self.provider:
            provider_status = {
                "status": "healthy",
                "type": self.provider["type"],
                "version": self.provider["version"],
                "initialized": True
            }
            models_available = self.provider.get("models", [])
        else:
            provider_status = {
                "status": "not_initialized"
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
provider_manager = MinimalLLMManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await provider_manager.initialize_provider()
    yield
    # Shutdown - nothing to cleanup for minimal version


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
    return provider_manager.get_health_status()


@app.get("/provider")
async def get_provider_info():
    """Get LLM provider information."""
    provider = provider_manager.get_provider()
    
    return {
        "name": provider["name"],
        "type": provider["type"],
        "version": provider["version"],
        "models": provider.get("models", [])
    }


@app.post("/expand", response_model=LLMResponse)
async def expand_concept(request: LLMRequest):
    """Expand a concept using the LLM provider."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        provider = provider_manager.get_provider()
        provider_manager.request_count += 1
        
        # Mock LLM expansion result
        result = {
            "expanded_concepts": [
                f"llm_expanded_{request.concept}_1",
                f"llm_expanded_{request.concept}_2",
                f"llm_expanded_{request.concept}_3"
            ],
            "confidence_scores": [0.95, 0.85, 0.75],
            "llm_reasoning": f"Generated concepts related to '{request.concept}' in {request.media_context} context"
        }
        
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return LLMResponse(
            success=True,
            execution_time_ms=execution_time_ms,
            result=result,
            metadata={
                "mock_response": True,
                "model_used": "mock-model-1",
                "request_concept": request.concept
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


if __name__ == "__main__":
    import uvicorn
    
    # Run the service
    uvicorn.run(
        "src.services.minimal_llm_service:app",
        host="0.0.0.0",
        port=8002,
        reload=settings.is_development,
        log_level=settings.LOG_LEVEL.lower()
    )