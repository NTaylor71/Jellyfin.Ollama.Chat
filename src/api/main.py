"""
FastAPI application entry point.
Clean foundation for the media intelligence system.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.shared.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    settings = get_settings()
    
    # Startup
    print(f"🚀 Starting API server on {settings.ENV} environment")
    print(f"📡 Service URLs: {settings.get_service_urls()}")
    
    yield
    
    # Shutdown
    print("🔄 Shutting down API server")


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
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD and settings.is_development,
        workers=settings.API_WORKERS if not settings.is_development else 1,
        log_level=settings.LOG_LEVEL.lower()
    )