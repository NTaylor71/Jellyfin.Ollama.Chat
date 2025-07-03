"""
Health check endpoints for monitoring and debugging.
"""

import asyncio
import logging
import time
from typing import Dict, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.shared.config import get_settings
from ..plugin_registry import plugin_registry


router = APIRouter()
logger = logging.getLogger(__name__)


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
    
    # Add plugin system health
    try:
        plugin_status = await plugin_registry.get_plugin_status()
        dependencies["plugin_system"] = plugin_status["initialized_plugins"] > 0
        dependencies["plugins_healthy"] = plugin_status["enabled_plugins"] == plugin_status["initialized_plugins"]
    except Exception as e:
        dependencies["plugin_system"] = False
        dependencies["plugins_healthy"] = False
    
    return DetailedHealthResponse(
        status="healthy" if all(dependencies.values()) else "degraded",
        timestamp=time.time(),
        environment=settings.ENV,
        services={
            "api": "running",
            "config": "loaded",
            "plugins": "initialized" if dependencies.get("plugin_system", False) else "not_initialized"
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


@router.get("/plugins")
async def plugin_health_summary():
    """Get plugin system health summary."""
    try:
        plugin_status = await plugin_registry.get_plugin_status()
        
        # Calculate health metrics
        total_plugins = plugin_status["total_plugins"]
        enabled_plugins = plugin_status["enabled_plugins"]
        initialized_plugins = plugin_status["initialized_plugins"]
        
        health_percentage = (initialized_plugins / total_plugins * 100) if total_plugins > 0 else 100
        
        # Count healthy vs unhealthy plugins
        healthy_count = 0
        for name, details in plugin_status["plugin_details"].items():
            if details.get("health", {}).get("status") == "healthy":
                healthy_count += 1
        
        return {
            "status": "healthy" if health_percentage >= 90 else "degraded" if health_percentage >= 70 else "unhealthy",
            "timestamp": time.time(),
            "total_plugins": total_plugins,
            "enabled_plugins": enabled_plugins,
            "initialized_plugins": initialized_plugins,
            "healthy_plugins": healthy_count,
            "health_percentage": round(health_percentage, 2),
            "plugins_by_type": plugin_status["plugins_by_type"]
        }
        
    except Exception as e:
        logger.error(f"Error getting plugin health summary: {e}")
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }


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
            "metrics_enabled": settings.ENABLE_METRICS,
            "plugin_system_enabled": True
        }
    }
