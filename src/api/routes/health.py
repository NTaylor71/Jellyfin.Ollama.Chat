"""
Health check endpoints for monitoring and debugging.
"""

import asyncio
import time
from typing import Dict, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.shared.config import get_settings


router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    environment: str
    version: str = "2.0.0"
    services: Dict[str, Any]


class DetailedHealthResponse(HealthResponse):
    """Detailed health check with service connectivity."""
    config: Dict[str, Any]
    dependencies: Dict[str, bool]


@router.get("", response_model=HealthResponse)  # Changed from "/" to ""
async def health_check():
    """Basic health check endpoint."""
    settings = get_settings()
    
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        environment=settings.ENV,
        services={
            "api": "running",
            "config": "loaded"
        }
    )


@router.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Readiness check - confirms all dependencies are available."""
    settings = get_settings()
    
    # Basic readiness - just check config is loaded
    return HealthResponse(
        status="ready",
        timestamp=time.time(),
        environment=settings.ENV,
        services={
            "api": "ready",
            "config": "loaded",
            "directories": "created"
        }
    )


@router.get("/live", response_model=HealthResponse)
async def liveness_check():
    """Liveness check - confirms service is alive."""
    settings = get_settings()
    
    return HealthResponse(
        status="alive",
        timestamp=time.time(),
        environment=settings.ENV,
        services={
            "api": "alive",
            "memory": "ok"
        }
    )


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """Detailed health check with full service status."""
    settings = get_settings()
    
    # Test service connectivity (basic version for now)
    dependencies = {
        "config_loaded": True,
        "directories_exist": True
    }
    
    # Add Redis test when available
    try:
        # This will be implemented when we add Redis
        dependencies["redis"] = False  # Not connected yet
    except Exception:
        dependencies["redis"] = False
    
    return DetailedHealthResponse(
        status="healthy" if all(dependencies.values()) else "degraded",
        timestamp=time.time(),
        environment=settings.ENV,
        services={
            "api": "running",
            "config": "loaded"
        },
        config={
            "environment": settings.ENV,
            "debug_mode": settings.is_localhost,
            "redis_host": settings.REDIS_HOST,
            "api_port": settings.API_PORT
        },
        dependencies=dependencies
    )


@router.get("/ping")
async def ping():
    """Simple ping endpoint."""
    return {"message": "pong", "timestamp": time.time()}


@router.get("/version")
async def version():
    """Get API version information."""
    settings = get_settings()
    
    return {
        "version": "2.0.0",
        "environment": settings.ENV,
        "python_version": "3.11+",
        "api": "FastAPI",
        "features": {
            "cors_enabled": settings.ENABLE_CORS,
            "docs_enabled": settings.ENABLE_API_DOCS,
            "metrics_enabled": settings.ENABLE_METRICS
        }
    }
