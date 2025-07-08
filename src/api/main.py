"""
FastAPI application entry point.
Clean foundation for the media intelligence system.
"""

import asyncio
import uvicorn
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
import httpx
import yaml

from src.shared.config import get_settings
from src.ingestion_manager import IngestionManager, MediaData


# Request/Response models
class MediaIngestionRequest(BaseModel):
    """Request for media ingestion."""
    media_type: str = "movie"
    media_items: List[Dict[str, Any]]
    skip_enrichment: bool = False
    batch_size: int = 5


class JellyfinIngestionRequest(BaseModel):
    """Request for Jellyfin media ingestion."""
    media_type: str = "movie"
    limit: Optional[int] = None
    item_names: Optional[List[str]] = None
    skip_enrichment: bool = False
    batch_size: int = 5


class IngestionResponse(BaseModel):
    """Response for ingestion operations."""
    status: str
    message: str
    media_type: str
    items_processed: int
    enriched: int
    errors: List[str] = Field(default_factory=list)
    task_id: Optional[str] = None


class EnrichmentRequest(BaseModel):
    """Request for media enrichment."""
    media_type: str = "movie"
    media_data: Dict[str, Any]
    fields: Optional[List[str]] = None


class EnrichmentResponse(BaseModel):
    """Response for enrichment operations."""
    status: str
    media_type: str
    enriched_data: Optional[Dict[str, Any]] = None
    enriched_fields: List[str] = Field(default_factory=list)
    execution_time_ms: float
    errors: List[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    """Request for media search."""
    media_type: str = "movie"
    query: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Response for search operations."""
    media_type: str
    results: List[Dict[str, Any]]
    total_found: int
    execution_time_ms: float


# Global ingestion managers by media type
ingestion_managers: Dict[str, IngestionManager] = {}


async def get_or_create_manager(media_type: str) -> IngestionManager:
    """Get or create an ingestion manager for the specified media type."""
    if media_type not in ingestion_managers:
        manager = IngestionManager(media_type=media_type)
        await manager.connect()
        ingestion_managers[media_type] = manager
    return ingestion_managers[media_type]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    settings = get_settings()
    
    # Startup
    print(f"🚀 Starting API server on {settings.ENV} environment")
    print(f"📡 Service URLs: {settings.get_service_urls()}")
    
    # Initialize default movie manager
    movie_manager = IngestionManager(media_type="movie")
    await movie_manager.connect()
    ingestion_managers["movie"] = movie_manager
    
    yield
    
    # Shutdown
    print("🔄 Shutting down API server")
    for manager in ingestion_managers.values():
        await manager.disconnect()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Production RAG System",
        description="Media Intelligence System with Plugin Architecture",
        version="2.0.0",
        docs_url="/docs" if settings.ENABLE_API_DOCS else None,
        redoc_url="/redoc" if settings.ENABLE_API_DOCS else None,
        lifespan=lifespan
    )
    
    # CORS middleware
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
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint for Docker and monitoring."""
        settings = get_settings()
        return {
            "status": "healthy",
            "environment": settings.ENV,
            "services": settings.get_health_status()
        }
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with system information."""
        settings = get_settings()
        return {
            "message": "Production RAG System API",
            "version": "2.0.0",
            "environment": settings.ENV,
            "docs_url": "/docs" if settings.ENABLE_API_DOCS else None
        }
    
    # ==========================================================================
    # MEDIA INGESTION ENDPOINTS
    # ==========================================================================
    
    @app.post("/ingest/media", response_model=IngestionResponse)
    async def ingest_media_items(
        request: MediaIngestionRequest,
        background_tasks: BackgroundTasks
    ):
        """Ingest media items with enrichment."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Get or create manager for media type
            manager = await get_or_create_manager(request.media_type)
            
            # Convert to MediaData objects
            media_items = []
            errors = []
            
            for item_data in request.media_items:
                try:
                    media_item = manager.dynamic_model(**item_data)
                    media_items.append(media_item)
                except Exception as e:
                    errors.append(f"Invalid data for {item_data.get('Name', 'unknown')}: {str(e)}")
            
            if not media_items:
                return IngestionResponse(
                    status="failed",
                    message="No valid media items to process",
                    media_type=request.media_type,
                    items_processed=0,
                    enriched=0,
                    errors=errors
                )
            
            # Process ingestion
            await manager.ingest_media(
                media_items,
                batch_size=request.batch_size,
                skip_enrichment=request.skip_enrichment
            )
            
            return IngestionResponse(
                status="success",
                message=f"Successfully ingested {len(media_items)} {request.media_type} items",
                media_type=request.media_type,
                items_processed=len(media_items),
                enriched=len(media_items) if not request.skip_enrichment else 0,
                errors=errors
            )
            
        except Exception as e:
            return IngestionResponse(
                status="error",
                message=f"Ingestion failed: {str(e)}",
                media_type=request.media_type,
                items_processed=0,
                enriched=0,
                errors=[str(e)]
            )
    
    @app.post("/ingest/jellyfin", response_model=IngestionResponse)
    async def ingest_from_jellyfin(
        request: JellyfinIngestionRequest,
        background_tasks: BackgroundTasks
    ):
        """Ingest media from Jellyfin API."""
        try:
            # Get or create manager for media type
            manager = await get_or_create_manager(request.media_type)
            
            # Load from Jellyfin
            media_items = await manager.load_media_from_jellyfin(
                limit=request.limit,
                item_names=request.item_names
            )
            
            if not media_items:
                return IngestionResponse(
                    status="success",
                    message=f"No {request.media_type} items found in Jellyfin",
                    media_type=request.media_type,
                    items_processed=0,
                    enriched=0
                )
            
            # Process ingestion
            await manager.ingest_media(
                media_items,
                batch_size=request.batch_size,
                skip_enrichment=request.skip_enrichment
            )
            
            return IngestionResponse(
                status="success",
                message=f"Successfully ingested {len(media_items)} {request.media_type} items from Jellyfin",
                media_type=request.media_type,
                items_processed=len(media_items),
                enriched=len(media_items) if not request.skip_enrichment else 0
            )
            
        except Exception as e:
            return IngestionResponse(
                status="error",
                message=f"Jellyfin ingestion failed: {str(e)}",
                media_type=request.media_type,
                items_processed=0,
                enriched=0,
                errors=[str(e)]
            )
    
    @app.post("/enrich", response_model=EnrichmentResponse)
    async def enrich_media_item(request: EnrichmentRequest):
        """Enrich a single media item."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Get or create manager for media type
            manager = await get_or_create_manager(request.media_type)
            
            # Convert to MediaData
            media_item = manager.dynamic_model(**request.media_data)
            
            # Enrich the item
            enriched_data = await manager.enrich_media_item(media_item)
            
            # Calculate execution time
            execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Get enriched fields
            original_fields = set(request.media_data.keys())
            enriched_fields = [field for field in enriched_data.keys() if field not in original_fields]
            
            return EnrichmentResponse(
                status="success",
                media_type=request.media_type,
                enriched_data=enriched_data,
                enriched_fields=enriched_fields,
                execution_time_ms=execution_time_ms
            )
            
        except Exception as e:
            execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            return EnrichmentResponse(
                status="error",
                media_type=request.media_type,
                execution_time_ms=execution_time_ms,
                errors=[str(e)]
            )
    
    @app.get("/media-types")
    async def list_media_types():
        """List available media type configurations."""
        try:
            config_dir = Path("config/media_types")
            if not config_dir.exists():
                return {"media_types": [], "error": "Configuration directory not found"}
            
            media_types = []
            for config_file in config_dir.glob("*.yaml"):
                if config_file.name != "movie_new_format.yaml":  # Skip template files
                    media_type = config_file.stem
                    try:
                        with open(config_file, 'r') as f:
                            config_data = yaml.safe_load(f)
                        media_types.append({
                            "media_type": media_type,
                            "name": config_data.get("name", media_type),
                            "description": config_data.get("description", ""),
                            "fields": list(config_data.get("fields", {}).keys())
                        })
                    except Exception as e:
                        print(f"Error loading {config_file}: {e}")
            
            return {"media_types": media_types}
            
        except Exception as e:
            return {"media_types": [], "error": str(e)}
    
    @app.get("/verify/{media_type}")
    async def verify_ingestion(media_type: str):
        """Verify ingestion results for a media type."""
        try:
            manager = await get_or_create_manager(media_type)
            results = await manager.verify_ingestion()
            return {"status": "success", **results}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD and settings.is_localhost,
        workers=settings.API_WORKERS if not settings.is_development else 1,
        log_level=settings.LOG_LEVEL.lower()
    )