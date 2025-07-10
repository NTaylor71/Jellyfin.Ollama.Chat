"""
Model Manager Service - Centralized model management with REST API.

Provides centralized model operations following the /providers/{provider}/{action} pattern.
Handles model downloads, status checks, updates, and cleanup for all services.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

from src.shared.config import get_settings
from src.shared.metrics import (
    track_mongodb_metrics,
    mongodb_operations_total,
    plugin_execution_total,
    plugin_execution_duration
)

logger = logging.getLogger(__name__)

# =============================================================================
# RESPONSE MODELS
# =============================================================================

class ModelInfo(BaseModel):
    """Information about a model."""
    name: str
    package: str
    storage_path: str
    size_mb: int
    required: bool
    status: str
    error_message: Optional[str] = None

class ModelStatusResponse(BaseModel):
    """Response for model status check."""
    success: bool
    execution_time_ms: float
    models: Dict[str, ModelInfo]
    summary: Dict[str, Any]
    error_message: Optional[str] = None

class ModelDownloadRequest(BaseModel):
    """Request to download models."""
    model_ids: Optional[List[str]] = Field(default=None, description="Specific model IDs to download, or None for all required")
    force_download: bool = Field(default=False, description="Force re-download existing models")

class ModelDownloadResponse(BaseModel):
    """Response for model download."""
    success: bool
    execution_time_ms: float
    downloaded_models: List[str]
    failed_models: List[str]
    error_message: Optional[str] = None

class ModelVerifyResponse(BaseModel):
    """Response for model verification."""
    success: bool
    execution_time_ms: float
    verification_results: Dict[str, Dict[str, Any]]
    error_message: Optional[str] = None

class ModelUpdateRequest(BaseModel):
    """Request to update models."""
    model_ids: Optional[List[str]] = Field(default=None, description="Specific model IDs to update, or None for all")
    dry_run: bool = Field(default=False, description="Show what would be updated without actually updating")

class ModelUpdateResponse(BaseModel):
    """Response for model update."""
    success: bool
    execution_time_ms: float
    update_results: Dict[str, Dict[str, Any]]
    error_message: Optional[str] = None

class ModelCleanupRequest(BaseModel):
    """Request to clean up models."""
    cleanup_cache: bool = Field(default=False, description="Also clean up package caches")
    dry_run: bool = Field(default=False, description="Show what would be cleaned without actually cleaning")

class ModelCleanupResponse(BaseModel):
    """Response for model cleanup."""
    success: bool
    execution_time_ms: float
    cleaned_files: int
    space_freed_mb: float
    error_message: Optional[str] = None

class ServiceHealth(BaseModel):
    """Service health status."""
    status: str
    uptime_seconds: float
    model_manager_ready: bool
    models_status: Dict[str, Dict[str, Any]]
    total_requests: int
    failed_requests: int

# =============================================================================
# GLOBAL STATE
# =============================================================================

# Service state
service_start_time = time.time()
total_requests = 0
failed_requests = 0
http_client: Optional[httpx.AsyncClient] = None

# Service endpoints for coordination - build dynamically from environment
def get_service_endpoints():
    """Get service endpoints for microservices architecture."""
    import os
    
    endpoints = {}
    
    # Use dynamic service discovery instead of hard-coded endpoints
    try:
        from src.shared.dynamic_service_discovery import get_service_discovery
        import asyncio
        
        # Get or create event loop (avoid deprecated get_event_loop)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Discover services dynamically
        async def discover_endpoints():
            discovery = await get_service_discovery()
            discovered = await discovery.discover_all_services()
            return {name: info.base_url for name, info in discovered.items()}
        
        if loop.is_running():
            # If in async context, use the services directly
            logger.info("Using dynamic service discovery")
        else:
            # If not in async context, run discovery
            dynamic_endpoints = loop.run_until_complete(discover_endpoints())
            endpoints.update(dynamic_endpoints)
            logger.info(f"Dynamically discovered services: {list(dynamic_endpoints.keys())}")
            
    except Exception as e:
        logger.warning(f"Dynamic service discovery failed: {e}")
        
        # No fallback - fail gracefully if discovery fails
        logger.error("Service discovery failed and no fallback configured")
        logger.error("Services will be discovered on-demand when accessed")
    
    return endpoints

SERVICE_ENDPOINTS = get_service_endpoints()

# Initialization state tracking (following NLP service pattern)
initialization_state = "starting"  # starting -> coordinating_services -> downloading_models -> ready
initialization_progress = {}  # Track what's being initialized

# =============================================================================
# LIFECYCLE MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global http_client, initialization_state, initialization_progress
    
    logger.info("🚀 Starting Model Manager Service")
    
    try:
        # Start initialization
        initialization_state = "starting"
        initialization_progress = {"phase": "starting", "current_task": "initializing HTTP client"}
        
        # Initialize HTTP client
        http_client = httpx.AsyncClient(timeout=30.0)
        
        # Coordinate with services
        initialization_state = "coordinating_services"
        initialization_progress = {"phase": "coordinating_services", "current_task": "waiting for services to be ready"}
        
        logger.info("Waiting for services to be ready...")
        await wait_for_services_ready()
        
        # Check what models are needed
        initialization_state = "downloading_models"
        initialization_progress = {"phase": "downloading_models", "current_task": "checking model status across services"}
        
        logger.info("Checking model status across all services...")
        models_status = await get_all_models_status()
        
        # Check if we found any services
        services_found = len(models_status)
        if services_found == 0:
            logger.warning("⚠️ No services discovered for model coordination")
            logger.warning("⚠️ Service discovery may be failing - no models will be downloaded")
            initialization_progress["model_download_status"] = "skipped_no_services"
            initialization_progress["services_found"] = 0
        else:
            logger.info(f"Found {services_found} services for model coordination")
            
            # Count missing required models
            total_missing = sum(status.get("summary", {}).get("missing_required", 0) for status in models_status.values())
            
            if total_missing > 0:
                initialization_progress["current_task"] = f"orchestrating download of {total_missing} missing models"
                logger.info(f"Orchestrating download of {total_missing} missing required models...")
                
                download_success = await orchestrate_model_downloads()
                if download_success:
                    initialization_progress["model_download_status"] = "completed"
                    logger.info("✅ All required models downloaded successfully")
                else:
                    initialization_progress["model_download_status"] = "partial"
                    logger.warning("⚠️ Some model downloads failed, but service will continue")
            else:
                initialization_progress["model_download_status"] = "not_needed"
                logger.info("✅ All required models already available")
        
        # Mark as ready
        initialization_state = "ready"
        initialization_progress = {
            "phase": "ready",
            "services_coordinated": list(SERVICE_ENDPOINTS.keys()),
            "startup_complete": True
        }
        
        logger.info(f"Model Manager ready - orchestrated {len(SERVICE_ENDPOINTS)} services")
        
        yield
        
    except Exception as e:
        initialization_state = "error"
        initialization_progress = {"phase": "error", "error": str(e)}
        logger.error(f"Failed to initialize Model Manager Service: {e}")
        raise
    finally:
        logger.info("🔄 Shutting down Model Manager Service")
        if http_client:
            await http_client.aclose()

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Model Manager Service",
    description="Centralized model management service for Universal Media Ingestion Framework",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def track_request():
    """Track request metrics."""
    global total_requests
    total_requests += 1

def track_failed_request():
    """Track failed request metrics."""
    global failed_requests
    failed_requests += 1

def get_execution_time_ms(start_time: float) -> float:
    """Get execution time in milliseconds."""
    return (time.time() - start_time) * 1000


async def wait_for_services_ready():
    """Wait for all services to be ready."""
    max_retries = 60  # Wait up to 60 seconds
    retry_delay = 1.0
    
    for service_name, service_url in SERVICE_ENDPOINTS.items():
        logger.info(f"Waiting for {service_name} to be ready...")
        
        for attempt in range(max_retries):
            try:
                response = await http_client.get(f"{service_url}/health/ready")
                if response.status_code == 200:
                    logger.info(f"✅ {service_name} is ready")
                    break
                else:
                    logger.debug(f"⏳ {service_name} not ready (HTTP {response.status_code}), retrying...")
            except Exception as e:
                logger.debug(f"⏳ {service_name} not ready ({e}), retrying...")
            
            await asyncio.sleep(retry_delay)
        else:
            raise Exception(f"Service {service_name} did not become ready within {max_retries} seconds")


async def get_all_models_status() -> Dict[str, Dict[str, Any]]:
    """Get models status from all services."""
    models_status = {}
    
    for service_name, service_url in SERVICE_ENDPOINTS.items():
        try:
            response = await http_client.get(f"{service_url}/models/status")
            if response.status_code == 200:
                models_status[service_name] = response.json()
                logger.info(f"✅ Got models status from {service_name}")
            else:
                logger.error(f"❌ Failed to get models status from {service_name}: HTTP {response.status_code}")
                models_status[service_name] = {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            logger.error(f"❌ Error getting models status from {service_name}: {e}")
            models_status[service_name] = {"success": False, "error": str(e)}
    
    return models_status


async def orchestrate_model_downloads(force_download: bool = False) -> bool:
    """Orchestrate model downloads across all services."""
    success = True
    
    for service_name, service_url in SERVICE_ENDPOINTS.items():
        try:
            logger.info(f"Requesting {service_name} to download models...")
            
            # Request service to download its models
            response = await http_client.post(
                f"{service_url}/models/download",
                json={"force_download": force_download},
                timeout=600.0  # 10 minutes for downloads
            )
            
            if response.status_code == 200:
                result = response.json()
                downloaded = result.get("downloaded_models", [])
                failed = result.get("failed_models", [])
                
                if downloaded:
                    logger.info(f"✅ {service_name} downloaded: {', '.join(downloaded)}")
                if failed:
                    logger.error(f"❌ {service_name} failed to download: {', '.join(failed)}")
                    success = False
                
                if not downloaded and not failed:
                    logger.info(f"ℹ️ {service_name} had no models to download")
                    
            else:
                logger.error(f"❌ {service_name} model download failed: HTTP {response.status_code}")
                success = False
                
        except Exception as e:
            logger.error(f"❌ Error requesting {service_name} to download models: {e}")
            success = False
    
    return success

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=ServiceHealth)
async def health_check():
    """Comprehensive health check endpoint with initialization status."""
    global initialization_state, initialization_progress
    track_request()
    
    uptime = time.time() - service_start_time
    
    models_status = {}
    manager_ready = initialization_state == "ready"
    
    if manager_ready and http_client:
        try:
            # Get models status from all services
            all_models_status = await get_all_models_status()
            models_status = {
                service: status.get("summary", {})
                for service, status in all_models_status.items()
            }
        except Exception as e:
            logger.warning(f"Failed to check models in health check: {e}")
            manager_ready = False
    
    return {
        "status": "healthy" if manager_ready else initialization_state,
        "uptime_seconds": uptime,
        "model_manager_ready": manager_ready,
        "models_status": models_status,
        "total_requests": total_requests,
        "failed_requests": failed_requests,
        "initialization_state": initialization_state,
        "initialization_progress": initialization_progress,
        "ready": initialization_state == "ready"
    }


@app.get("/health/ready")
async def readiness_check():
    """Simple readiness check for Docker health checks."""
    global initialization_state
    
    if initialization_state == "ready":
        return {"ready": True, "status": "healthy"}
    else:
        return {"ready": False, "status": initialization_state}, 503

@app.get("/services/discover")
async def discover_services():
    """Service discovery endpoint showing all configured services."""
    track_request()
    
    return {
        "success": True,
        "architecture": "microservices",
        "services": SERVICE_ENDPOINTS,
        "total_services": len(SERVICE_ENDPOINTS)
    }

@app.get("/providers/models/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Get status of all models across services."""
    track_request()
    start_time = time.time()
    
    if not http_client:
        track_failed_request()
        raise HTTPException(status_code=503, detail="Model Manager not initialized")
    
    try:
        # Get models status from all services
        all_models_status = await get_all_models_status()
        
        # Aggregate models info
        models_info = {}
        total_models = 0
        available_models = 0
        required_models = 0
        
        for service_name, service_status in all_models_status.items():
            if service_status.get("success", False):
                service_models = service_status.get("models", {})
                service_summary = service_status.get("summary", {})
                
                # Add service models to aggregated info
                for model_id, model_data in service_models.items():
                    models_info[f"{service_name}_{model_id}"] = ModelInfo(
                        name=model_data["name"],
                        package=model_data["package"],
                        storage_path=model_data["storage_path"],
                        size_mb=model_data["size_mb"],
                        required=model_data["required"],
                        status=model_data["status"],
                        error_message=model_data.get("error_message")
                    )
                
                # Aggregate summary
                total_models += service_summary.get("total_models", 0)
                available_models += service_summary.get("available_models", 0)
                required_models += service_summary.get("required_models", 0)
        
        summary = {
            "total_models": total_models,
            "available_models": available_models,
            "required_models": required_models,
            "missing_required": required_models - available_models
        }
        
        execution_time_ms = get_execution_time_ms(start_time)
        
        # Track metrics
        plugin_execution_total.labels(
            plugin_name="model_manager",
            media_type="system",
            status="success"
        ).inc()
        
        plugin_execution_duration.labels(
            plugin_name="model_manager",
            media_type="system"
        ).observe(execution_time_ms / 1000)
        
        return ModelStatusResponse(
            success=True,
            execution_time_ms=execution_time_ms,
            models=models_info,
            summary=summary
        )
        
    except Exception as e:
        track_failed_request()
        execution_time_ms = get_execution_time_ms(start_time)
        
        plugin_execution_total.labels(
            plugin_name="model_manager",
            media_type="system",
            status="error"
        ).inc()
        
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/providers/models/download", response_model=ModelDownloadResponse)
async def download_models(request: ModelDownloadRequest):
    """Download models across services."""
    track_request()
    start_time = time.time()
    
    if not http_client:
        track_failed_request()
        raise HTTPException(status_code=503, detail="Model Manager not initialized")
    
    try:
        logger.info(f"Orchestrating model downloads - force_download: {request.force_download}")
        
        # Orchestrate downloads across services
        success = await orchestrate_model_downloads(force_download=request.force_download)
        
        # Get updated status to determine what was downloaded
        all_models_status = await get_all_models_status()
        
        downloaded_models = []
        failed_models = []
        
        for service_name, service_status in all_models_status.items():
            if service_status.get("success", False):
                service_models = service_status.get("models", {})
                for model_id, model_data in service_models.items():
                    full_model_id = f"{service_name}_{model_id}"
                    
                    if request.model_ids is None or full_model_id in request.model_ids:
                        if model_data["status"] == "available":
                            downloaded_models.append(full_model_id)
                        elif model_data["status"] == "error":
                            failed_models.append(full_model_id)
        
        execution_time_ms = get_execution_time_ms(start_time)
        
        # Track metrics
        plugin_execution_total.labels(
            plugin_name="model_manager",
            media_type="system",
            status="success" if success else "error"
        ).inc()
        
        plugin_execution_duration.labels(
            plugin_name="model_manager",
            media_type="system"
        ).observe(execution_time_ms / 1000)
        
        # Hard-fail if any service is missing (per requirements)
        if not success:
            # Determine which services failed
            failed_services = []
            for service_name, service_status in all_models_status.items():
                if not service_status.get("success", False):
                    failed_services.append(service_name)
            
            # Build dynamic service URLs for health checks
            service_health_urls = {}
            for service_name, service_url in SERVICE_ENDPOINTS.items():
                service_health_urls[service_name] = f"{service_url}/health"
            
            error_detail = {
                "error": "Model download orchestration failed - missing services detected",
                "failed_services": failed_services,
                "remedy": "Ensure all required services are running and accessible. Check service health endpoints and Docker container status.",
                "required_services": list(SERVICE_ENDPOINTS.keys()),
                "service_urls": service_health_urls
            }
            
            logger.error(f"Orchestration hard-fail: {error_detail}")
            track_failed_request()
            raise HTTPException(status_code=503, detail=error_detail)
        
        return ModelDownloadResponse(
            success=success,
            execution_time_ms=execution_time_ms,
            downloaded_models=downloaded_models,
            failed_models=failed_models
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        track_failed_request()
        execution_time_ms = get_execution_time_ms(start_time)
        
        plugin_execution_total.labels(
            plugin_name="model_manager",
            media_type="system",
            status="error"
        ).inc()
        
        logger.error(f"Failed to download models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/providers/models/verify", response_model=ModelVerifyResponse)
async def verify_models():
    """Verify model integrity and functionality."""
    global model_manager
    track_request()
    start_time = time.time()
    
    if not model_manager:
        track_failed_request()
        raise HTTPException(status_code=503, detail="Model Manager not initialized")
    
    try:
        logger.info("Verifying model integrity...")
        
        # Verify all models
        verification_results = await model_manager.verify_models()
        
        # Check if all verifications passed
        all_passed = all(result["valid"] for result in verification_results.values())
        
        execution_time_ms = get_execution_time_ms(start_time)
        
        # Track metrics
        plugin_execution_total.labels(
            plugin_name="model_manager",
            media_type="system",
            status="success"
        ).inc()
        
        plugin_execution_duration.labels(
            plugin_name="model_manager",
            media_type="system"
        ).observe(execution_time_ms / 1000)
        
        return ModelVerifyResponse(
            success=all_passed,
            execution_time_ms=execution_time_ms,
            verification_results=verification_results
        )
        
    except Exception as e:
        track_failed_request()
        execution_time_ms = get_execution_time_ms(start_time)
        
        plugin_execution_total.labels(
            plugin_name="model_manager",
            media_type="system",
            status="error"
        ).inc()
        
        logger.error(f"Failed to verify models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/providers/models/update", response_model=ModelUpdateResponse)
async def update_models(request: ModelUpdateRequest):
    """Update models to latest versions."""
    global model_manager
    track_request()
    start_time = time.time()
    
    if not model_manager:
        track_failed_request()
        raise HTTPException(status_code=503, detail="Model Manager not initialized")
    
    try:
        logger.info(f"Updating models - dry_run: {request.dry_run}")
        
        # Update models
        update_results = await model_manager.update_all_models(dry_run=request.dry_run)
        
        # Check if all updates succeeded
        all_succeeded = all(result["updated"] for result in update_results.values())
        
        execution_time_ms = get_execution_time_ms(start_time)
        
        # Track metrics
        plugin_execution_total.labels(
            plugin_name="model_manager",
            media_type="system",
            status="success"
        ).inc()
        
        plugin_execution_duration.labels(
            plugin_name="model_manager",
            media_type="system"
        ).observe(execution_time_ms / 1000)
        
        return ModelUpdateResponse(
            success=all_succeeded,
            execution_time_ms=execution_time_ms,
            update_results=update_results
        )
        
    except Exception as e:
        track_failed_request()
        execution_time_ms = get_execution_time_ms(start_time)
        
        plugin_execution_total.labels(
            plugin_name="model_manager",
            media_type="system",
            status="error"
        ).inc()
        
        logger.error(f"Failed to update models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/providers/models/cleanup", response_model=ModelCleanupResponse)
async def cleanup_models(request: ModelCleanupRequest):
    """Clean up unused models and cache files."""
    global model_manager
    track_request()
    start_time = time.time()
    
    if not model_manager:
        track_failed_request()
        raise HTTPException(status_code=503, detail="Model Manager not initialized")
    
    try:
        logger.info(f"Cleaning up models - cleanup_cache: {request.cleanup_cache}, dry_run: {request.dry_run}")
        
        # Clean up models
        cleaned_files, space_freed = await model_manager.cleanup_models(
            cleanup_cache=request.cleanup_cache,
            dry_run=request.dry_run
        )
        
        execution_time_ms = get_execution_time_ms(start_time)
        
        # Track metrics
        plugin_execution_total.labels(
            plugin_name="model_manager",
            media_type="system",
            status="success"
        ).inc()
        
        plugin_execution_duration.labels(
            plugin_name="model_manager",
            media_type="system"
        ).observe(execution_time_ms / 1000)
        
        return ModelCleanupResponse(
            success=True,
            execution_time_ms=execution_time_ms,
            cleaned_files=cleaned_files,
            space_freed_mb=space_freed
        )
        
    except Exception as e:
        track_failed_request()
        execution_time_ms = get_execution_time_ms(start_time)
        
        plugin_execution_total.labels(
            plugin_name="model_manager",
            media_type="system",
            status="error"
        ).inc()
        
        logger.error(f"Failed to cleanup models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    settings = get_settings()
    
    logger.info("Starting Model Manager Service...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
        log_level="info"
    )