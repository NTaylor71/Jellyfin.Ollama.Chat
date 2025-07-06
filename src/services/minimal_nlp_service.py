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
from prometheus_fastapi_instrumentator import Instrumentator

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
        """Initialize NLP providers - download models if needed during startup."""
        logger.info("Initializing NLP providers...")
        
        # Download models as needed during service startup
        from src.shared.model_manager import ModelManager, ModelStatus
        from pathlib import Path
        
        models_path = Path("/app/models")
        model_manager = ModelManager(models_base_path=models_path)
        
        logger.info("Ensuring all required models are available...")
        model_success = await model_manager.ensure_all_models()
        
        if not model_success:
            logger.error("❌ Failed to download required models")
            # Don't raise exception - let service start but report unhealthy
        
        # Check which models are available and initialize accordingly
        model_status = await model_manager.check_all_models()
        
        # Initialize providers based on what's actually available
        gensim_status = model_status.get("gensim_word2vec", ModelStatus.MISSING)
        if gensim_status == ModelStatus.AVAILABLE:
            self.providers["gensim"] = {
                "name": "Gensim Word2Vec Provider",
                "type": "gensim", 
                "version": "1.0.0",
                "status": "ready"
            }
            logger.info("✅ Gensim provider initialized")
        else:
            logger.warning(f"⚠️ Gensim provider not available: {gensim_status}")
        
        spacy_status = model_status.get("spacy_en_core", ModelStatus.MISSING)
        if spacy_status == ModelStatus.AVAILABLE:
            self.providers["spacy"] = {
                "name": "SpaCy NLP Provider",
                "type": "spacy",
                "version": "1.0.0", 
                "status": "ready"
            }
            logger.info("✅ SpaCy provider initialized")
        else:
            logger.warning(f"⚠️ SpaCy provider not available: {spacy_status}")
        
        # Check NLTK models too (they don't need a provider but should be available)
        nltk_punkt = model_status.get("nltk_punkt", ModelStatus.MISSING)
        nltk_stopwords = model_status.get("nltk_stopwords", ModelStatus.MISSING) 
        nltk_wordnet = model_status.get("nltk_wordnet", ModelStatus.MISSING)
        
        if all(status == ModelStatus.AVAILABLE for status in [nltk_punkt, nltk_stopwords, nltk_wordnet]):
            logger.info("✅ NLTK models available")
        else:
            logger.warning(f"⚠️ NLTK models missing - punkt: {nltk_punkt}, stopwords: {nltk_stopwords}, wordnet: {nltk_wordnet}")
        
        # Initialize HeidelTime provider (Java-based, no model download needed)
        try:
            self.providers["heideltime"] = {
                "name": "HeidelTime Temporal Provider",
                "type": "heideltime",
                "version": "1.0.0",
                "status": "ready"
            }
            logger.info("✅ HeidelTime provider initialized")
        except Exception as e:
            logger.warning(f"⚠️ HeidelTime provider not available: {e}")
        
        logger.info(f"✅ NLP Provider initialization complete. {len(self.providers)} providers ready.")
    
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
        
        # Service is healthy only if we have ALL required providers ready
        # This ensures nlp-service is only healthy when all models are downloaded
        expected_providers = ["gensim", "spacy"]  # NLTK models don't need a provider, just availability
        available_providers = list(self.providers.keys())
        all_providers_ready = all(provider in available_providers for provider in expected_providers)
        
        service_status = "healthy" if all_providers_ready else "unhealthy"
        
        return ServiceHealth(
            status=service_status,
            uptime_seconds=uptime,
            providers=provider_statuses,
            total_requests=self.request_count,
            failed_requests=self.error_count
        )


# Global provider manager
provider_manager = MinimalNLPManager()


# Real provider functions
async def _expand_with_gensim(request: ProviderRequest) -> Dict[str, Any]:
    """Expand concept using real Gensim Word2Vec provider."""
    try:
        from src.concept_expansion.providers.gensim_provider import GensimProvider
        
        provider = GensimProvider(model_name="word2vec-google-news-300")
        
        # Check health and initialize if needed
        health_status = await provider.health_check()
        if health_status.get("status") != "healthy":
            logger.warning(f"Gensim provider not healthy: {health_status}")
            
        # Create expansion request
        from src.concept_expansion.providers.base_provider import ExpansionRequest
        expansion_request = ExpansionRequest(
            concept=request.concept,
            media_context=request.media_context,
            max_concepts=request.max_concepts
        )
        
        # Use the provider's expand_concept method
        expansion_result = await provider.expand_concept(expansion_request)
        
        if expansion_result and expansion_result.success:
            concepts = expansion_result.enhanced_data.get("expanded_concepts", [])
            scores = expansion_result.enhanced_data.get("confidence_scores", [])
            
            return {
                "expanded_concepts": concepts[:request.max_concepts],
                "confidence_scores": scores[:request.max_concepts],
                "provider_used": "gensim",
                "total_found": len(concepts)
            }
        else:
            error_msg = expansion_result.error_message if expansion_result else "No result returned"
            return {
                "expanded_concepts": [],
                "confidence_scores": [],
                "provider_used": "gensim",
                "error": error_msg
            }
        
    except Exception as e:
        logger.error(f"Gensim expansion failed: {e}")
        return {
            "expanded_concepts": [],
            "confidence_scores": [],
            "provider_used": "gensim",
            "error": str(e)
        }


async def _expand_with_spacy(request: ProviderRequest) -> Dict[str, Any]:
    """Expand concept using real SpaCy NLP provider."""
    try:
        from src.concept_expansion.providers.spacy_temporal_provider import SpacyTemporalProvider
        
        provider = SpacyTemporalProvider(model_name="en_core_web_sm")
        
        # Check health and initialize if needed
        health_status = await provider.health_check()
        if health_status.get("status") != "healthy":
            logger.warning(f"SpaCy provider not healthy: {health_status}")
            
        # Create expansion request
        from src.concept_expansion.providers.base_provider import ExpansionRequest
        expansion_request = ExpansionRequest(
            concept=request.concept,
            media_context=request.media_context,
            max_concepts=request.max_concepts
        )
        
        # Use the provider's expand_concept method
        expansion_result = await provider.expand_concept(expansion_request)
        
        if expansion_result and expansion_result.success:
            concepts = expansion_result.enhanced_data.get("expanded_concepts", [])
            scores = expansion_result.enhanced_data.get("confidence_scores", [])
            analysis = expansion_result.enhanced_data.get("analysis", {})
            
            return {
                "expanded_concepts": concepts[:request.max_concepts],
                "confidence_scores": scores[:request.max_concepts],
                "provider_used": "spacy",
                "analysis": analysis
            }
        else:
            error_msg = expansion_result.error_message if expansion_result else "No result returned"
            return {
                "expanded_concepts": [],
                "confidence_scores": [],
                "provider_used": "spacy",
                "error": error_msg
            }
        
    except Exception as e:
        logger.error(f"SpaCy expansion failed: {e}")
        return {
            "expanded_concepts": [],
            "confidence_scores": [],
            "provider_used": "spacy", 
            "error": str(e)
        }


async def _expand_with_heideltime(request: ProviderRequest) -> Dict[str, Any]:
    """Expand concept using real HeidelTime temporal provider."""
    try:
        from src.concept_expansion.providers.heideltime_provider import HeidelTimeProvider
        
        provider = HeidelTimeProvider(language="english")
        
        # Check health and initialize if needed
        health_status = await provider.health_check()
        if health_status.get("status") != "healthy":
            logger.warning(f"HeidelTime provider not healthy: {health_status}")
            
        # Create expansion request
        from src.concept_expansion.providers.base_provider import ExpansionRequest
        expansion_request = ExpansionRequest(
            concept=request.concept,
            media_context=request.media_context,
            max_concepts=request.max_concepts
        )
        
        # Use the provider's expand_concept method
        expansion_result = await provider.expand_concept(expansion_request)
        
        if expansion_result and expansion_result.success:
            concepts = expansion_result.enhanced_data.get("expanded_concepts", [])
            scores = expansion_result.enhanced_data.get("confidence_scores", [])
            temporal_data = expansion_result.enhanced_data.get("temporal_expressions", {})
            
            return {
                "expanded_concepts": concepts[:request.max_concepts],
                "confidence_scores": scores[:request.max_concepts],
                "provider_used": "heideltime",
                "temporal_analysis": temporal_data
            }
        else:
            error_msg = expansion_result.error_message if expansion_result else "No result returned"
            return {
                "expanded_concepts": [],
                "confidence_scores": [],
                "provider_used": "heideltime",
                "error": error_msg
            }
        
    except Exception as e:
        logger.error(f"HeidelTime expansion failed: {e}")
        return {
            "expanded_concepts": [],
            "confidence_scores": [],
            "provider_used": "heideltime", 
            "error": str(e)
        }


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

# Prometheus metrics (creates /metrics endpoint)
if settings.ENABLE_METRICS:
    Instrumentator().instrument(app).expose(app)


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
        
        # Call real provider for expansion
        if provider_name == "gensim":
            result = await _expand_with_gensim(request)
        elif provider_name == "spacy":
            result = await _expand_with_spacy(request)
        elif provider_name == "heideltime":
            result = await _expand_with_heideltime(request)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Provider {provider_name} expansion not implemented"
            )
        
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return ProviderResponse(
            success=True,
            provider_name=provider_name,
            execution_time_ms=execution_time_ms,
            result=result,
            metadata={
                "real_provider_response": True,
                "request_concept": request.concept,
                "media_context": request.media_context
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