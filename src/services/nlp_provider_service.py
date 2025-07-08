"""
NLP Provider Service - FastAPI service hosting SpaCy, HeidelTime, and Gensim providers.

Lightweight service that handles all heavy NLP operations, allowing the worker
to remain lightweight and focused on orchestration.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
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


class NLPProviderManager:
    """Manages NLP provider instances and their lifecycle."""
    
    def __init__(self):
        self.providers: Dict[str, Any] = {}
        self.initialization_errors: Dict[str, str] = {}
        self.request_count = 0
        self.error_count = 0
        self.start_time = None  # Will be set during initialization
    
    async def initialize_providers(self):
        """Initialize all NLP providers."""
        logger.info("Initializing NLP providers...")
        self.start_time = asyncio.get_event_loop().time()
        
        # Initialize Gensim Provider
        try:
            from src.concept_expansion.providers.gensim_provider import GensimProvider
            gensim_provider = GensimProvider()
            if await gensim_provider.initialize():
                self.providers["gensim"] = gensim_provider
                logger.info("✅ Gensim provider initialized")
            else:
                self.initialization_errors["gensim"] = "Initialization failed"
                logger.error("❌ Gensim provider initialization failed")
        except ImportError as e:
            self.initialization_errors["gensim"] = f"Import error: {e}"
            logger.error(f"❌ Gensim provider import failed: {e}")
        except Exception as e:
            self.initialization_errors["gensim"] = str(e)
            logger.error(f"❌ Gensim provider error: {e}")
        
        # Initialize SpaCy Temporal Provider
        try:
            from src.concept_expansion.providers.spacy_temporal_provider import SpacyTemporalProvider
            spacy_provider = SpacyTemporalProvider()
            if await spacy_provider.initialize():
                self.providers["spacy_temporal"] = spacy_provider
                logger.info("✅ SpaCy temporal provider initialized")
            else:
                self.initialization_errors["spacy_temporal"] = "Initialization failed"
                logger.error("❌ SpaCy temporal provider initialization failed")
        except ImportError as e:
            self.initialization_errors["spacy_temporal"] = f"Import error: {e}"
            logger.error(f"❌ SpaCy temporal provider import failed: {e}")
        except Exception as e:
            self.initialization_errors["spacy_temporal"] = str(e)
            logger.error(f"❌ SpaCy temporal provider error: {e}")
        
        # Initialize HeidelTime Provider
        try:
            from src.concept_expansion.providers.heideltime_provider import HeidelTimeProvider
            heideltime_provider = HeidelTimeProvider()
            if await heideltime_provider.initialize():
                self.providers["heideltime"] = heideltime_provider
                logger.info("✅ HeidelTime provider initialized")
            else:
                self.initialization_errors["heideltime"] = "Initialization failed"
                logger.error("❌ HeidelTime provider initialization failed")
        except ImportError as e:
            self.initialization_errors["heideltime"] = f"Import error: {e}"
            logger.error(f"❌ HeidelTime provider import failed: {e}")
        except Exception as e:
            self.initialization_errors["heideltime"] = str(e)
            logger.error(f"❌ HeidelTime provider error: {e}")
        
        # Initialize ConceptNet Provider
        try:
            from src.concept_expansion.providers.conceptnet_provider import ConceptNetProvider
            conceptnet_provider = ConceptNetProvider()
            if await conceptnet_provider.initialize():
                self.providers["conceptnet"] = conceptnet_provider
                logger.info("✅ ConceptNet provider initialized")
            else:
                self.initialization_errors["conceptnet"] = "Initialization failed"
                logger.error("❌ ConceptNet provider initialization failed")
        except ImportError as e:
            self.initialization_errors["conceptnet"] = f"Import error: {e}"
            logger.error(f"❌ ConceptNet provider import failed: {e}")
        except Exception as e:
            self.initialization_errors["conceptnet"] = str(e)
            logger.error(f"❌ ConceptNet provider error: {e}")
        
        # Initialize SUTime Provider
        try:
            from src.concept_expansion.providers.sutime_provider import SUTimeProvider
            sutime_provider = SUTimeProvider()
            if await sutime_provider.initialize():
                self.providers["sutime"] = sutime_provider
                logger.info("✅ SUTime provider initialized")
            else:
                self.initialization_errors["sutime"] = "Initialization failed"
                logger.error("❌ SUTime provider initialization failed")
        except ImportError as e:
            self.initialization_errors["sutime"] = f"Import error: {e}"
            logger.error(f"❌ SUTime provider import failed: {e}")
        except Exception as e:
            self.initialization_errors["sutime"] = str(e)
            logger.error(f"❌ SUTime provider error: {e}")
        
        logger.info(f"NLP Provider initialization complete. {len(self.providers)} providers available.")
    
    async def cleanup_providers(self):
        """Cleanup all providers."""
        logger.info("Cleaning up NLP providers...")
        for name, provider in self.providers.items():
            try:
                if hasattr(provider, 'cleanup'):
                    await provider.cleanup()
                logger.info(f"✅ {name} provider cleaned up")
            except Exception as e:
                logger.error(f"❌ Error cleaning up {name} provider: {e}")
    
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
            try:
                metadata = provider.metadata
                provider_statuses[name] = {
                    "status": "healthy",
                    "type": metadata.provider_type,
                    "version": metadata.version,
                    "initialized": getattr(provider, '_initialized', False)
                }
            except Exception as e:
                provider_statuses[name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Add initialization errors
        for name, error in self.initialization_errors.items():
            if name not in provider_statuses:
                provider_statuses[name] = {
                    "status": "failed_to_initialize",
                    "error": error
                }
        
        return ServiceHealth(
            status="healthy" if self.providers else "degraded",
            uptime_seconds=uptime,
            providers=provider_statuses,
            total_requests=self.request_count,
            failed_requests=self.error_count
        )


# Global provider manager
provider_manager = NLPProviderManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await provider_manager.initialize_providers()
    yield
    # Shutdown
    await provider_manager.cleanup_providers()


# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title="NLP Provider Service",
    description="FastAPI service hosting SpaCy, HeidelTime, and Gensim providers",
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
    providers_info = {}
    for name, provider in provider_manager.providers.items():
        metadata = provider.metadata
        providers_info[name] = {
            "name": metadata.name,
            "type": metadata.provider_type,
            "version": metadata.version,
            "strengths": metadata.strengths,
            "best_for": metadata.best_for,
            "context_aware": metadata.context_aware
        }
    
    return {
        "available_providers": list(provider_manager.providers.keys()),
        "providers": providers_info,
        "initialization_errors": provider_manager.initialization_errors
    }


@app.post("/providers/{provider_name}/expand", response_model=ProviderResponse)
async def expand_concept(
    provider_name: str,
    request: ProviderRequest,
    background_tasks: BackgroundTasks
):
    """Expand a concept using the specified provider."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        provider = provider_manager.get_provider(provider_name)
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
        
        return ProviderResponse(
            success=result.success,
            provider_name=provider_name,
            execution_time_ms=execution_time_ms,
            result=result.enhanced_data if result.success else None,
            error_message=result.error_message if not result.success else None,
            metadata={
                "confidence_score": result.confidence_score.overall if result.success else None,
                "cache_key": result.cache_key.generate_key() if result.success else None
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


@app.get("/providers/{provider_name}/metadata")
async def get_provider_metadata(provider_name: str):
    """Get metadata for a specific provider."""
    provider = provider_manager.get_provider(provider_name)
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


@app.post("/providers/{provider_name}/health")
async def check_provider_health(provider_name: str):
    """Check health of a specific provider."""
    provider = provider_manager.get_provider(provider_name)
    
    try:
        if hasattr(provider, 'health_check'):
            health_status = await provider.health_check()
        else:
            health_status = {
                "status": "healthy",
                "initialized": getattr(provider, '_initialized', False)
            }
        
        return {
            "provider": provider_name,
            "health": health_status,
            "metadata": {
                "name": provider.metadata.name,
                "type": provider.metadata.provider_type,
                "version": provider.metadata.version
            }
        }
        
    except Exception as e:
        return {
            "provider": provider_name,
            "health": {
                "status": "error",
                "error": str(e)
            }
        }


# ============================================================================
# NEW ENDPOINTS FOR HTTP-ONLY PLUGIN ARCHITECTURE
# ============================================================================

class GensimSimilarityRequest(BaseModel):
    """Request for Gensim similarity search."""
    keywords: List[str] = Field(..., min_items=1, max_items=10)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_similar: int = Field(default=8, ge=1, le=20)
    model_type: str = Field(default="word2vec")
    include_scores: bool = Field(default=True)
    filter_duplicates: bool = Field(default=True)


class GensimSimilarityResponse(BaseModel):
    """Response from Gensim similarity search."""
    similar_terms: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class GensimCompareRequest(BaseModel):
    """Request to compare similarity between two terms."""
    term1: str = Field(..., min_length=1)
    term2: str = Field(..., min_length=1)
    model_type: str = Field(default="word2vec")


@app.post("/gensim/similarity", response_model=GensimSimilarityResponse)
async def gensim_similarity_search(request: GensimSimilarityRequest):
    """Find similar terms using Gensim word vectors."""
    try:
        provider = provider_manager.get_provider("gensim")
        provider_manager.request_count += 1
        
        start_time = asyncio.get_event_loop().time()
        
        # Convert to similarity search format
        similar_terms = []
        for keyword in request.keywords:
            # Call provider's similarity method
            if hasattr(provider, 'find_similar_terms'):
                similar = await provider.find_similar_terms(
                    keyword, 
                    threshold=request.similarity_threshold,
                    max_results=request.max_similar
                )
                
                if request.include_scores:
                    # Keep score information
                    for term_data in similar:
                        if isinstance(term_data, dict):
                            similar_terms.append(term_data)
                        else:
                            similar_terms.append({"term": str(term_data), "score": 1.0})
                else:
                    # Extract just terms
                    for term_data in similar:
                        if isinstance(term_data, dict):
                            similar_terms.append({"term": term_data.get("term", str(term_data))})
                        else:
                            similar_terms.append({"term": str(term_data)})
        
        # Remove duplicates if requested
        if request.filter_duplicates:
            seen_terms = set()
            filtered_terms = []
            for term_data in similar_terms:
                term = term_data.get("term", "")
                if term and term not in seen_terms:
                    seen_terms.add(term)
                    filtered_terms.append(term_data)
            similar_terms = filtered_terms
        
        # Limit results
        similar_terms = similar_terms[:request.max_similar]
        
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return GensimSimilarityResponse(
            similar_terms=similar_terms,
            metadata={
                "execution_time_ms": execution_time_ms,
                "model_type": request.model_type,
                "threshold": request.similarity_threshold,
                "input_keywords": request.keywords,
                "result_count": len(similar_terms)
            }
        )
        
    except Exception as e:
        provider_manager.error_count += 1
        logger.error(f"Gensim similarity search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gensim/compare")
async def gensim_compare_similarity(request: GensimCompareRequest):
    """Compare similarity between two specific terms."""
    try:
        provider = provider_manager.get_provider("gensim")
        provider_manager.request_count += 1
        
        start_time = asyncio.get_event_loop().time()
        
        # Call provider's compare method
        if hasattr(provider, 'compare_similarity'):
            similarity_score = await provider.compare_similarity(request.term1, request.term2)
        else:
            # Fallback: use similarity search
            similar_terms = await provider.find_similar_terms(request.term1, max_results=100)
            similarity_score = 0.0
            for term_data in similar_terms:
                if isinstance(term_data, dict):
                    if term_data.get("term", "").lower() == request.term2.lower():
                        similarity_score = term_data.get("score", 0.0)
                        break
                elif str(term_data).lower() == request.term2.lower():
                    similarity_score = 0.8  # Default score for exact match
                    break
        
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return {
            "similarity": similarity_score,
            "metadata": {
                "execution_time_ms": execution_time_ms,
                "model_type": request.model_type,
                "term1": request.term1,
                "term2": request.term2
            }
        }
        
    except Exception as e:
        provider_manager.error_count += 1
        logger.error(f"Gensim similarity comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Run the service
    uvicorn.run(
        "src.services.nlp_provider_service:app",
        host="0.0.0.0",
        port=8001,
        reload=False,  # Disable reload for testing
        log_level=settings.LOG_LEVEL.lower()
    )