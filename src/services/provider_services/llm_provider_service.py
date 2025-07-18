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
from src.providers.llm.llm_provider import LLMProvider

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
        self.initialization_state = "starting"  # starting -> connecting_ollama -> ready
        self.initialization_progress = {}
    
    async def initialize_provider(self):
        """Initialize LLM provider."""
        logger.info("Initializing LLM provider...")
        self.initialization_state = "connecting_ollama"
        self.initialization_progress = {"phase": "connecting_ollama", "current_task": "importing LLM provider"}
        
        try:
            from src.providers.llm.llm_provider import LLMProvider
            self.initialization_progress["current_task"] = "creating LLM provider instance"
            
            self.provider = LLMProvider()
            self.initialization_progress["current_task"] = "connecting to Ollama service"
            
            if await self.provider.initialize():
                self.initialization_state = "ready"
                self.initialization_progress = {
                    "phase": "completed",
                    "status": "ready"
                }
                logger.info("✅ LLM provider initialized")
            else:
                self.initialization_error = "LLM provider initialization failed"
                self.initialization_progress["error"] = "LLM provider initialization failed"
                logger.error("❌ LLM provider initialization failed")
        except ImportError as e:
            self.initialization_error = f"Import error: {e}"
            self.initialization_progress["error"] = f"Import error: {e}"
            logger.error(f"❌ LLM provider import failed: {e}")
        except Exception as e:
            self.initialization_error = str(e)
            self.initialization_progress["error"] = str(e)
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
        
        # Determine overall service status based on initialization state
        service_status = "starting"
        if self.initialization_state == "ready":
            service_status = "healthy" if self.provider else "degraded"
        elif self.initialization_state == "connecting_ollama":
            service_status = "initializing"
        else:
            service_status = "starting"
        
        return LLMServiceHealth(
            status=service_status,
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


@app.get("/health/detailed")
async def detailed_health_check():
    """Get detailed health status with initialization progress."""
    health = provider_manager.get_health_status()
    
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


@app.post("/providers/llm/expand", response_model=LLMResponse)
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


@app.post("/providers/llm/health")
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


# ============================================================================
# NEW ENDPOINTS FOR HTTP-ONLY PLUGIN ARCHITECTURE
# ============================================================================

class KeywordExpansionRequest(BaseModel):
    """Request for LLM keyword expansion."""
    keywords: List[str] = Field(..., min_items=1, max_items=10)
    context: str = Field(default="")
    field_name: str = Field(default="keywords")
    max_concepts: int = Field(default=15, ge=1, le=50)
    expansion_style: str = Field(default="semantic_related")
    prompt_template: str = Field(default="")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    smart_retry_until: str = Field(default="list")


class KeywordExpansionResponse(BaseModel):
    """Response from LLM keyword expansion."""
    concepts: List[str]
    metadata: Dict[str, Any]


@app.post("/providers/llm/keywords/expand", response_model=KeywordExpansionResponse)
async def expand_keywords(request: KeywordExpansionRequest):
    """Expand keywords using LLM semantic understanding."""
    try:
        provider = provider_manager.get_provider()
        provider_manager.request_count += 1
        
        start_time = asyncio.get_event_loop().time()
        
        # Build prompt for keyword expansion
        if request.prompt_template:
            # Use custom prompt template
            keywords_text = ", ".join(request.keywords)
            if request.context:
                prompt = request.prompt_template.format(value=f"{keywords_text} (Context: {request.context})")
            else:
                prompt = request.prompt_template.format(value=keywords_text)
        else:
            # Default prompt based on field name and expansion style
            keywords_text = ", ".join(request.keywords)
            if request.expansion_style == "semantic_related":
                prompt = f"Given these keywords: {keywords_text}, provide {request.max_concepts} semantically related concepts, themes, and terms that would help categorize and find similar content. Return as a simple list."
            elif request.expansion_style == "genre_expansion":
                prompt = f"Given these genres/categories: {keywords_text}, provide {request.max_concepts} related genres, subgenres, and thematic categories. Return as a simple list."
            else:
                prompt = f"Expand these keywords: {keywords_text} with {request.max_concepts} related concepts. Return as a simple list."
            
            if request.context:
                prompt += f" Context: {request.context}"
        
        # Create expansion request for provider
        from src.providers.nlp.base_provider import ExpansionRequest
        expansion_request = ExpansionRequest(
            concept=prompt,
            media_context=request.field_name,
            max_concepts=request.max_concepts,
            field_name=request.field_name
        )
        
        # Call LLM provider
        result = await provider.expand_concept(expansion_request)
        
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Extract concepts from result
        if isinstance(result, dict) and "concepts" in result:
            concepts = result["concepts"]
        elif isinstance(result, list):
            concepts = result
        else:
            # Fallback: try to parse as text
            concepts = str(result).split("\n") if result else []
        
        # Clean up concepts
        cleaned_concepts = []
        for concept in concepts:
            if isinstance(concept, str):
                clean = concept.strip().strip("-").strip("*").strip()
                if clean and len(clean) > 1:
                    cleaned_concepts.append(clean)
            elif isinstance(concept, dict) and "concept" in concept:
                clean = str(concept["concept"]).strip()
                if clean and len(clean) > 1:
                    cleaned_concepts.append(clean)
        
        # Limit results
        cleaned_concepts = cleaned_concepts[:request.max_concepts]
        
        return KeywordExpansionResponse(
            concepts=cleaned_concepts,
            metadata={
                "execution_time_ms": execution_time_ms,
                "expansion_style": request.expansion_style,
                "input_keywords": request.keywords,
                "result_count": len(cleaned_concepts),
                "field_name": request.field_name,
                "temperature": request.temperature
            }
        )
        
    except Exception as e:
        provider_manager.error_count += 1
        logger.error(f"LLM keyword expansion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Run the service
    uvicorn.run(
        "src.services.provider_services.llm_provider_service:app",
        host="0.0.0.0",
        port=8002,
        reload=False,  # Disable reload for testing
        log_level=settings.LOG_LEVEL.lower()
    )