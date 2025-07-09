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
from src.api.routes import media_router, ingestion_router, search_router


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
    
    # Include API routers
    app.include_router(media_router)
    app.include_router(ingestion_router)
    app.include_router(search_router)
    
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
    
    # Legacy endpoints for backward compatibility
    @app.get("/media-types")
    async def list_media_types():
        """List available media type configurations (legacy endpoint)."""
        try:
            config_dir = Path("config/media_types")
            if not config_dir.exists():
                return {"media_types": [], "error": "Configuration directory not found"}
            
            media_types = []
            for config_file in config_dir.glob("*.yaml"):
                if config_file.name not in ["movie_new_format.yaml", "media_detection.yaml"]:  # Skip template files
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
        """Verify ingestion results for a media type (legacy endpoint)."""
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