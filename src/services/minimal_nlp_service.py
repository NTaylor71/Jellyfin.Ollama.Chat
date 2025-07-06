"""
Minimal NLP Provider Service - Just demonstrates the architecture works.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class ProviderRequest(BaseModel):
    """Request to a provider."""
    concept: str = Field(..., min_length=1, max_length=500)
    media_context: str = Field(default="movie")
    max_concepts: int = Field(default=10, ge=1, le=50)
    field_name: str = Field(default="concept")
    options: Dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    """Response from a provider."""
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


class MinimalNLPManager:
    """Minimal manager for testing."""
    
    def __init__(self):
        self.providers = {}
        self.request_count = 0
        self.error_count = 0
        self.start_time = asyncio.get_event_loop().time()
    
    async def initialize_providers(self):
        """Initialize demo providers."""
        logger.info("Initializing minimal NLP providers...")
        
        # Add mock providers for demonstration
        self.providers["mock_gensim"] = {
            "name": "Mock Gensim Provider",
            "type": "gensim", 
            "version": "1.0.0",
            "status": "ready"
        }
        
        self.providers["mock_spacy"] = {
            "name": "Mock SpaCy Provider",
            "type": "spacy",
            "version": "1.0.0", 
            "status": "ready"
        }
        
        logger.info(f"NLP Provider initialization complete. {len(self.providers)} providers available.")
    
    def get_provider(self, provider_name: str):
        """Get a provider by name."""
        if provider_name not in self.providers:
            raise HTTPException(
                status_code=404,
                detail=f"Provider '{provider_name}' not available. Available providers: {list(self.providers.keys())}"
            )
        return self.providers[provider_name]
    
    def get_health_status(self) -> ServiceHealth:
        """Get service health status."""
        current_time = asyncio.get_event_loop().time()
        uptime = current_time - self.start_time
        
        provider_statuses = {}
        for name, provider in self.providers.items():
            provider_statuses[name] = {
                "status": "healthy",
                "type": provider["type"],
                "version": provider["version"],
                "initialized": True
            }
        
        return ServiceHealth(
            status="healthy" if self.providers else "degraded",
            uptime_seconds=uptime,
            providers=provider_statuses,
            total_requests=self.request_count,
            failed_requests=self.error_count
        )


# Global provider manager
provider_manager = MinimalNLPManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await provider_manager.initialize_providers()
    yield
    # Shutdown - nothing to cleanup for minimal version


# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title="Minimal NLP Provider Service",
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


@app.get("/health", response_model=ServiceHealth)
async def health_check():
    """Get service health status."""
    return provider_manager.get_health_status()


@app.get("/providers")
async def list_providers():
    """List available providers."""
    return {
        "available_providers": list(provider_manager.providers.keys()),
        "providers": provider_manager.providers
    }


@app.post("/providers/{provider_name}/expand", response_model=ProviderResponse)
async def expand_concept(
    provider_name: str,
    request: ProviderRequest
):
    """Expand a concept using the specified provider."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        provider = provider_manager.get_provider(provider_name)
        provider_manager.request_count += 1
        
        # Mock expansion result
        result = {
            "expanded_concepts": [
                f"{request.concept}_related_1",
                f"{request.concept}_related_2", 
                f"{request.concept}_related_3"
            ],
            "confidence_scores": [0.9, 0.8, 0.7],
            "provider_used": provider_name
        }
        
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return ProviderResponse(
            success=True,
            provider_name=provider_name,
            execution_time_ms=execution_time_ms,
            result=result,
            metadata={
                "mock_response": True,
                "request_concept": request.concept
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        provider_manager.error_count += 1
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        logger.error(f"Error expanding concept with {provider_name}: {e}")
        
        return ProviderResponse(
            success=False,
            provider_name=provider_name,
            execution_time_ms=execution_time_ms,
            error_message=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    
    # Run the service
    uvicorn.run(
        "src.services.minimal_nlp_service:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.is_development,
        log_level=settings.LOG_LEVEL.lower()
    )