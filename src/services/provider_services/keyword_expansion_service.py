"""
Keyword Expansion Service - FastAPI service for ConceptNet and other keyword providers.

Dedicated service for keyword expansion operations, focused on linguistic relationships
and fast keyword expansion via ConceptNet API.
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


class ConceptNetExpansionRequest(BaseModel):
    """Request for ConceptNet keyword expansion."""
    keywords: List[str] = Field(..., min_items=1, max_items=10)
    max_concepts: int = Field(default=10, ge=1, le=50)
    relation_types: List[str] = Field(default=["RelatedTo", "IsA", "PartOf"])
    language: str = Field(default="en")


class ConceptNetExpansionResponse(BaseModel):
    """Response from ConceptNet keyword expansion."""
    concepts: List[str]
    metadata: Dict[str, Any]


class KeywordServiceHealth(BaseModel):
    """Keyword service health status."""
    status: str
    uptime_seconds: float
    providers: Dict[str, Dict[str, Any]]
    total_requests: int
    failed_requests: int


class KeywordProviderManager:
    """Manages keyword provider instances and their lifecycle."""
    
    def __init__(self):
        self.providers: Dict[str, Any] = {}
        self.initialization_errors: Dict[str, str] = {}
        self.request_count = 0
        self.error_count = 0
        self.start_time = None
    
    async def initialize_providers(self):
        """Initialize keyword expansion providers."""
        logger.info("Initializing keyword providers...")
        self.start_time = asyncio.get_event_loop().time()
        
        # Initialize ConceptNet Provider
        try:
            from src.providers.knowledge.conceptnet_provider import ConceptNetProvider
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
        
        logger.info("Keyword provider initialization complete.")
    
    async def cleanup_providers(self):
        """Cleanup all providers."""
        logger.info("Cleaning up keyword providers...")
        for name, provider in self.providers.items():
            try:
                if hasattr(provider, 'cleanup'):
                    await provider.cleanup()
                logger.info(f"✅ {name} provider cleaned up")
            except Exception as e:
                logger.error(f"❌ Error cleaning up {name} provider: {e}")
    
    def get_provider(self, provider_name: str):
        """Get a specific provider."""
        if provider_name not in self.providers:
            error_msg = self.initialization_errors.get(provider_name, "Provider not found")
            raise HTTPException(
                status_code=503, 
                detail=f"Provider '{provider_name}' not available. Error: {error_msg}"
            )
        return self.providers[provider_name]
    
    def get_health_status(self) -> KeywordServiceHealth:
        """Get current service health status."""
        current_time = asyncio.get_event_loop().time()
        uptime = current_time - (self.start_time or current_time)
        
        providers_health = {}
        for name, provider in self.providers.items():
            try:
                providers_health[name] = {
                    "status": "healthy",
                    "initialized": getattr(provider, '_initialized', True),
                    "type": getattr(provider, 'metadata', {}).get('provider_type', 'unknown')
                }
            except Exception as e:
                providers_health[name] = {
                    "status": "error", 
                    "error": str(e)
                }
        
        return KeywordServiceHealth(
            status="healthy" if self.providers else "degraded",
            uptime_seconds=uptime,
            providers=providers_health,
            total_requests=self.request_count,
            failed_requests=self.error_count
        )


# Global provider manager
provider_manager = KeywordProviderManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting Keyword Expansion Service...")
    await provider_manager.initialize_providers()
    logger.info("Keyword Expansion Service startup complete.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Keyword Expansion Service...")
    await provider_manager.cleanup_providers()
    logger.info("Keyword Expansion Service shutdown complete.")


# Create FastAPI app
app = FastAPI(
    title="Keyword Expansion Service",
    description="Service for keyword expansion using ConceptNet and other providers",
    version="1.0.0",
    lifespan=lifespan
)

# Configure settings
settings = get_settings()

# Add CORS middleware if enabled
if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=settings.CORS_METHODS,
        allow_headers=settings.CORS_HEADERS,
    )


@app.get("/health", response_model=KeywordServiceHealth)
async def health_check():
    """Get service health status."""
    return provider_manager.get_health_status()


@app.get("/providers")
async def list_providers():
    """List available keyword providers."""
    providers_info = {}
    for name, provider in provider_manager.providers.items():
        if hasattr(provider, 'metadata'):
            metadata = provider.metadata
            providers_info[name] = {
                "name": metadata.name,
                "type": metadata.provider_type,
                "version": metadata.version,
                "strengths": getattr(metadata, 'strengths', []),
                "best_for": getattr(metadata, 'best_for', [])
            }
        else:
            providers_info[name] = {
                "name": name,
                "type": "unknown",
                "version": "1.0.0"
            }
    
    return {
        "available_providers": list(provider_manager.providers.keys()),
        "providers": providers_info,
        "initialization_errors": provider_manager.initialization_errors
    }


@app.post("/providers/conceptnet/expand", response_model=ConceptNetExpansionResponse)
async def expand_conceptnet_keywords(request: ConceptNetExpansionRequest):
    """Expand keywords using ConceptNet linguistic relationships."""
    try:
        provider = provider_manager.get_provider("conceptnet")
        provider_manager.request_count += 1
        
        start_time = asyncio.get_event_loop().time()
        
        # Process each keyword
        all_concepts = []
        for keyword in request.keywords:
            # Create expansion request
            from src.providers.nlp.base_provider import ExpansionRequest
            expansion_request = ExpansionRequest(
                concept=keyword,
                media_context="general",
                max_concepts=request.max_concepts,
                field_name="keyword"
            )
            
            # Call ConceptNet provider
            result = await provider.expand_concept(expansion_request)
            
            # Extract concepts from result
            if isinstance(result, dict) and "concepts" in result:
                concepts = result["concepts"]
            elif isinstance(result, list):
                concepts = result
            else:
                concepts = []
            
            # Add concepts to the list
            for concept in concepts:
                if isinstance(concept, str):
                    clean_concept = concept.strip().lower()
                    if clean_concept and clean_concept not in all_concepts:
                        all_concepts.append(clean_concept)
                elif isinstance(concept, dict) and "concept" in concept:
                    clean_concept = str(concept["concept"]).strip().lower()
                    if clean_concept and clean_concept not in all_concepts:
                        all_concepts.append(clean_concept)
        
        # Limit final results
        final_concepts = all_concepts[:request.max_concepts]
        
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return ConceptNetExpansionResponse(
            concepts=final_concepts,
            metadata={
                "execution_time_ms": execution_time_ms,
                "provider": "conceptnet",
                "input_keywords": request.keywords,
                "result_count": len(final_concepts),
                "relation_types": request.relation_types,
                "language": request.language
            }
        )
        
    except Exception as e:
        provider_manager.error_count += 1
        logger.error(f"ConceptNet keyword expansion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Run the service
    uvicorn.run(
        "src.services.provider_services.keyword_expansion_service:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    )