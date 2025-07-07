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
    """LLM manager with real Ollama integration."""
    
    def __init__(self):
        self.http_client = None
        self.request_count = 0
        self.error_count = 0
        self.start_time = asyncio.get_event_loop().time()
        self.settings = get_settings()
        self.ollama_url = self.settings.OLLAMA_CHAT_BASE_URL
        self.model = self.settings.OLLAMA_CHAT_MODEL
        logger.info(f"Initialized LLM manager with Ollama URL: {self.ollama_url}, model: {self.model}")
    
    async def initialize_provider(self):
        """Initialize real Ollama LLM provider."""
        logger.info("Initializing Ollama LLM provider...")
        
        try:
            self.http_client = httpx.AsyncClient(timeout=30.0)
            
            # Test Ollama connection
            response = await self.http_client.get(f"{self.ollama_url}/api/tags")
            if response.status_code != 200:
                logger.error(f"Ollama not available: {response.status_code}")
                return False
                
            # Check if model is available
            models_data = response.json()
            available_models = [model.get("name", "") for model in models_data.get("models", [])]
            model_available = any(self.model in model_name for model_name in available_models)
            
            if not model_available:
                logger.warning(f"Model {self.model} not found. Available: {available_models}")
                logger.info(f"Automatically pulling model {self.model}...")
                
                # Automatically pull the missing model
                try:
                    pull_response = await self.http_client.post(
                        f"{self.ollama_url}/api/pull",
                        json={"name": self.model},
                        timeout=600.0  # 10 minutes for model download
                    )
                    
                    if pull_response.status_code == 200:
                        logger.info(f"✅ Successfully pulled model {self.model}")
                    else:
                        logger.error(f"❌ Failed to pull model {self.model}: {pull_response.status_code}")
                        return False
                        
                except Exception as e:
                    logger.error(f"❌ Error pulling model {self.model}: {e}")
                    return False
                
            logger.info("Ollama LLM Provider initialization complete.")
            return True
            
        except Exception as e:
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
        
        return LLMServiceHealth(
            status="healthy" if self.http_client and provider_status.get("status") == "healthy" else "degraded",
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
    return await provider_manager.get_health_status()


@app.get("/provider")
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


@app.post("/expand", response_model=LLMResponse)
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
        
        # If model not found, try to pull it automatically
        if response.status_code == 404:
            logger.warning(f"Model {provider_manager.model} not found, attempting to pull...")
            try:
                pull_response = await provider_manager.http_client.post(
                    f"{provider_manager.ollama_url}/api/pull",
                    json={"name": provider_manager.model},
                    timeout=600.0
                )
                
                if pull_response.status_code == 200:
                    logger.info(f"✅ Successfully pulled model {provider_manager.model}, retrying generation...")
                    # Retry the original request
                    response = await provider_manager.http_client.post(
                        f"{provider_manager.ollama_url}/api/generate",
                        json=payload
                    )
                else:
                    logger.error(f"❌ Failed to pull model: {pull_response.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ Error pulling model: {e}")
        
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
        
        return LLMResponse(
            success=success,
            execution_time_ms=execution_time_ms,
            result=result,
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