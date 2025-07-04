"""
Simple FastAPI application for RAG system with Prometheus metrics.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.shared.config import get_settings
from src.api.routes import health, chat, plugins, admin
from src.api.plugin_registry import plugin_registry
from src.api.plugin_watcher import plugin_watcher
from src.api.plugin_metrics import get_plugin_metrics

# Get settings
settings = get_settings()
logger = logging.getLogger(__name__)


async def update_plugin_metrics_periodically():
    """Background task to update plugin metrics for Prometheus."""
    metrics_collector = get_plugin_metrics()
    
    while True:
        try:
            await metrics_collector.update_plugin_health_metrics(plugin_registry)
            await asyncio.sleep(30)  # Update every 30 seconds
        except asyncio.CancelledError:
            logger.info("📊 Plugin metrics update task cancelled")
            break
        except Exception as e:
            logger.error(f"Error updating plugin metrics: {e}")
            await asyncio.sleep(30)  # Continue after error


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("🚀 Starting RAG API service")
    logger.info(f"Environment: {settings.ENV}")
    logger.info(f"API URL: {settings.API_URL}")
    
    # Initialize plugin system
    metrics_task = None
    try:
        logger.info("🔌 Initializing plugin system...")
        await plugin_registry.initialize()
        
        # Start plugin file watcher for hot-reload
        if getattr(settings, 'ENABLE_PLUGIN_HOT_RELOAD', True):
            await plugin_watcher.start_watching()
            logger.info("👁️ Plugin hot-reload watcher started")
        else:
            logger.info("⚠️ Plugin hot-reload disabled")
            
        plugin_status = await plugin_registry.get_plugin_status()
        logger.info(f"✅ Plugin system initialized: {plugin_status['enabled_plugins']}/{plugin_status['total_plugins']} plugins active")
        
        # Start plugin metrics collection if Prometheus is enabled
        if getattr(settings, 'ENABLE_METRICS', True):
            metrics_task = asyncio.create_task(update_plugin_metrics_periodically())
            logger.info("📊 Plugin metrics collection started")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize plugin system: {e}")
        logger.warning("🔄 Continuing without plugin system...")

    yield

    # Cleanup plugin system
    logger.info("🛑 Shutting down RAG API service")
    try:
        # Cancel metrics task
        if metrics_task and not metrics_task.done():
            metrics_task.cancel()
            try:
                await metrics_task
            except asyncio.CancelledError:
                pass
        
        if plugin_watcher.is_watching:
            await plugin_watcher.stop_watching()
            logger.info("👁️ Plugin watcher stopped")
        
        await plugin_registry.cleanup()
        logger.info("🔌 Plugin system cleaned up")
    except Exception as e:
        logger.error(f"❌ Error during plugin cleanup: {e}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    # Create FastAPI app
    app = FastAPI(
        title="Production RAG System API",
        description="Fast, scalable RAG system with FAISS vector search",
        version="2.0.0",
        docs_url="/docs" if settings.ENABLE_API_DOCS else None,
        redoc_url="/redoc" if settings.ENABLE_API_DOCS else None,
        lifespan=lifespan
    )

    # Add Prometheus metrics
    if getattr(settings, 'ENABLE_METRICS', True):  # Default to enabled
        instrumentator = Instrumentator(
            should_group_status_codes=False,
            should_ignore_untemplated=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=["/metrics"],
        )

        instrumentator.instrument(app).expose(app)
        logger.info("✅ Prometheus metrics enabled at /metrics")
    else:
        logger.info("⚠️ Prometheus metrics disabled")

    # Add CORS middleware
    if settings.ENABLE_CORS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=settings.CORS_METHODS,
            allow_headers=settings.CORS_HEADERS,
        )

    # Include routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(chat.router, prefix="/chat", tags=["Chat"])
    app.include_router(plugins.router, prefix="/plugins", tags=["Plugins"])
    app.include_router(admin.router, tags=["Admin"])

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "Production RAG System API",
            "version": "2.0.0",
            "environment": settings.ENV,
            "docs": "/docs" if settings.ENABLE_API_DOCS else None,
            "metrics": "/metrics",
            "health": "/health",
            "status": "running"
        }

    return app


# Create the app instance
app = create_app()


def main():
    """Main entry point for the API service."""

    # Setup basic logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("🎯 Starting Production RAG API")
    logger.info(f"Environment: {settings.ENV}")
    logger.info(f"Debug mode: {settings.is_localhost}")

    # Run the server
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.is_localhost,  # Only reload in localhost mode
        log_level=settings.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()
