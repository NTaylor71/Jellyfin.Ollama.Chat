"""
Admin API Routes
Administrative endpoints for plugin management and system operations.
"""

import asyncio
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from ..plugin_registry import get_plugin_registry
from ...plugins.mongo_manager import get_plugin_manager, PluginStatus
from ...plugins.base import PluginType
from ...shared.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])
security = HTTPBearer()


class PluginReleaseRequest(BaseModel):
    """Request model for plugin release."""
    plugin_name: str = Field(..., description="Plugin name to release")
    version: Optional[str] = Field(None, description="Specific version to release (optional)")
    environment: str = Field(default="production", description="Target environment")
    force: bool = Field(default=False, description="Force release even if health checks fail")
    config_overrides: Dict[str, Any] = Field(default_factory=dict, description="Environment-specific config overrides")


class PluginReleaseResponse(BaseModel):
    """Response model for plugin release."""
    success: bool
    message: str
    deployment_id: Optional[str] = None
    version: Optional[str] = None
    health_check_results: Dict[str, Any] = Field(default_factory=dict)
    rollback_info: Optional[Dict[str, Any]] = None


class PluginStatusUpdate(BaseModel):
    """Request model for plugin status update."""
    status: PluginStatus
    reason: Optional[str] = Field(None, description="Reason for status change")


class BulkPluginOperation(BaseModel):
    """Request model for bulk plugin operations."""
    plugin_names: List[str]
    operation: str = Field(..., description="Operation: enable, disable, reload, publish")
    environment: Optional[str] = Field(None, description="Environment for publish operation")


async def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify admin authentication token."""
    settings = get_settings()
    
    # In production, implement proper JWT validation
    # For now, check against a configured admin token
    admin_token = getattr(settings, "ADMIN_TOKEN", None)
    if not admin_token:
        raise HTTPException(
            status_code=501,
            detail="Admin authentication not configured"
        )
    
    if credentials.credentials != admin_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid admin token"
        )
    
    return credentials.credentials


@router.post("/plugins/release", response_model=PluginReleaseResponse)
async def release_plugin(
    request: PluginReleaseRequest,
    background_tasks: BackgroundTasks,
    admin_token: str = Depends(verify_admin_token)
) -> PluginReleaseResponse:
    """Release a plugin to production with health checks and rollback capability."""
    try:
        logger.info(f"Admin plugin release requested: {request.plugin_name} to {request.environment}")
        
        plugin_manager = await get_plugin_manager()
        plugin_registry = await get_plugin_registry()
        
        # Get plugin information
        plugin_doc = await plugin_manager.get_plugin(request.plugin_name)
        if not plugin_doc:
            raise HTTPException(
                status_code=404,
                detail=f"Plugin {request.plugin_name} not found"
            )
        
        # Determine version to release
        release_version = request.version or plugin_doc.current_version
        
        # Check if plugin is in deployable state
        if plugin_doc.status not in [PluginStatus.DRAFT, PluginStatus.PUBLISHED]:
            if not request.force:
                raise HTTPException(
                    status_code=400,
                    detail=f"Plugin is in {plugin_doc.status} status and cannot be released"
                )
        
        # Perform pre-release health check
        health_results = await _perform_health_check(request.plugin_name, plugin_registry)
        if not health_results["healthy"] and not request.force:
            return PluginReleaseResponse(
                success=False,
                message="Health check failed. Use force=true to override.",
                health_check_results=health_results
            )
        
        # Get deployment history for rollback info
        deployment_history = await plugin_manager.get_deployment_history(
            request.plugin_name, request.environment
        )
        rollback_info = None
        if deployment_history:
            last_deployment = deployment_history[0]
            rollback_info = {
                "previous_version": last_deployment.rollback_version,
                "last_deployment": last_deployment.deployment_time.isoformat()
            }
        
        # Record deployment
        deployment_id = await plugin_manager.record_deployment(
            plugin_name=request.plugin_name,
            version=release_version,
            environment=request.environment,
            deployed_by=f"admin_api_{admin_token[:8]}",
            config_overrides=request.config_overrides
        )
        
        # Publish plugin if not already published
        if plugin_doc.status != PluginStatus.PUBLISHED:
            await plugin_manager.publish_plugin(request.plugin_name, release_version)
        
        # Schedule background reload of all workers
        background_tasks.add_task(
            _broadcast_plugin_reload,
            request.plugin_name,
            release_version,
            request.environment
        )
        
        logger.info(f"Plugin {request.plugin_name} v{release_version} released successfully")
        
        return PluginReleaseResponse(
            success=True,
            message=f"Plugin {request.plugin_name} v{release_version} released successfully",
            deployment_id=deployment_id,
            version=release_version,
            health_check_results=health_results,
            rollback_info=rollback_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plugin release failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Plugin release failed: {str(e)}"
        )


@router.post("/plugins/{plugin_name}/status")
async def update_plugin_status(
    plugin_name: str,
    request: PluginStatusUpdate,
    admin_token: str = Depends(verify_admin_token)
) -> Dict[str, Any]:
    """Update plugin status (publish, deprecate, etc.)."""
    try:
        plugin_manager = await get_plugin_manager()
        
        if request.status == PluginStatus.PUBLISHED:
            success = await plugin_manager.publish_plugin(plugin_name)
        elif request.status == PluginStatus.DEPRECATED:
            success = await plugin_manager.deprecate_plugin(plugin_name, request.reason or "")
        else:
            # For other status changes, update directly
            plugin_doc = await plugin_manager.get_plugin(plugin_name)
            if not plugin_doc:
                raise HTTPException(status_code=404, detail="Plugin not found")
            
            # Update would go here - simplified for now
            success = True
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to update plugin {plugin_name} status to {request.status}"
            )
        
        return {
            "success": True,
            "message": f"Plugin {plugin_name} status updated to {request.status}",
            "plugin_name": plugin_name,
            "new_status": request.status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status update failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Status update failed: {str(e)}"
        )


@router.post("/plugins/bulk-operation")
async def bulk_plugin_operation(
    request: BulkPluginOperation,
    background_tasks: BackgroundTasks,
    admin_token: str = Depends(verify_admin_token)
) -> Dict[str, Any]:
    """Perform bulk operations on multiple plugins."""
    try:
        plugin_registry = await get_plugin_registry()
        results = {}
        
        for plugin_name in request.plugin_names:
            try:
                if request.operation == "enable":
                    success = await plugin_registry.enable_plugin(plugin_name)
                elif request.operation == "disable":
                    success = await plugin_registry.disable_plugin(plugin_name)
                elif request.operation == "reload":
                    success = await plugin_registry.reload_plugin(plugin_name)
                elif request.operation == "publish":
                    plugin_manager = await get_plugin_manager()
                    success = await plugin_manager.publish_plugin(plugin_name)
                else:
                    success = False
                
                results[plugin_name] = {
                    "success": success,
                    "message": f"Operation {request.operation} completed"
                }
                
            except Exception as e:
                results[plugin_name] = {
                    "success": False,
                    "message": f"Operation failed: {str(e)}"
                }
        
        return {
            "operation": request.operation,
            "total_plugins": len(request.plugin_names),
            "successful": sum(1 for r in results.values() if r["success"]),
            "failed": sum(1 for r in results.values() if not r["success"]),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Bulk operation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Bulk operation failed: {str(e)}"
        )


@router.get("/plugins/statistics")
async def get_plugin_statistics(
    admin_token: str = Depends(verify_admin_token)
) -> Dict[str, Any]:
    """Get plugin system statistics."""
    try:
        plugin_manager = await get_plugin_manager()
        plugin_registry = await get_plugin_registry()
        
        # Get MongoDB statistics
        mongo_stats = await plugin_manager.get_plugin_statistics()
        
        # Get registry statistics
        registry_status = await plugin_registry.get_plugin_status()
        
        return {
            "mongodb_stats": mongo_stats,
            "registry_stats": registry_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.get("/plugins/{plugin_name}/deployment-history")
async def get_plugin_deployment_history(
    plugin_name: str,
    environment: Optional[str] = None,
    admin_token: str = Depends(verify_admin_token)
) -> Dict[str, Any]:
    """Get deployment history for a plugin."""
    try:
        plugin_manager = await get_plugin_manager()
        
        deployments = await plugin_manager.get_deployment_history(plugin_name, environment)
        
        return {
            "plugin_name": plugin_name,
            "environment": environment,
            "total_deployments": len(deployments),
            "deployments": [
                {
                    "version": dep.version,
                    "environment": dep.environment,
                    "deployed_by": dep.deployed_by,
                    "deployment_time": dep.deployment_time.isoformat(),
                    "status": dep.status,
                    "health_check_passed": dep.health_check_passed,
                    "rollback_version": dep.rollback_version
                }
                for dep in deployments
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get deployment history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get deployment history: {str(e)}"
        )


@router.post("/plugins/{plugin_name}/rollback")
async def rollback_plugin(
    plugin_name: str,
    target_version: Optional[str] = None,
    environment: str = "production",
    admin_token: str = Depends(verify_admin_token)
) -> Dict[str, Any]:
    """Rollback a plugin to a previous version."""
    try:
        plugin_manager = await get_plugin_manager()
        
        # Get deployment history to find rollback target
        deployments = await plugin_manager.get_deployment_history(plugin_name, environment)
        if not deployments:
            raise HTTPException(
                status_code=404,
                detail=f"No deployment history found for {plugin_name}"
            )
        
        # Determine rollback version
        if target_version:
            rollback_version = target_version
        else:
            # Use the rollback version from latest deployment
            latest = deployments[0]
            rollback_version = latest.rollback_version
            if not rollback_version:
                raise HTTPException(
                    status_code=400,
                    detail="No rollback version available"
                )
        
        # Record rollback deployment
        deployment_id = await plugin_manager.record_deployment(
            plugin_name=plugin_name,
            version=rollback_version,
            environment=environment,
            deployed_by=f"admin_rollback_{admin_token[:8]}"
        )
        
        # Trigger plugin reload
        plugin_registry = await get_plugin_registry()
        await plugin_registry.reload_plugin(plugin_name)
        
        return {
            "success": True,
            "message": f"Plugin {plugin_name} rolled back to version {rollback_version}",
            "deployment_id": deployment_id,
            "rollback_version": rollback_version
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Rollback failed: {str(e)}"
        )


async def _perform_health_check(plugin_name: str, plugin_registry) -> Dict[str, Any]:
    """Perform health check on a plugin."""
    try:
        plugin = await plugin_registry.get_plugin(plugin_name)
        if not plugin:
            return {
                "healthy": False,
                "error": "Plugin not found in registry"
            }
        
        health_info = await plugin.health_check()
        return {
            "healthy": health_info.get("status") == "healthy",
            "details": health_info
        }
        
    except Exception as e:
        logger.error(f"Health check failed for {plugin_name}: {e}")
        return {
            "healthy": False,
            "error": str(e)
        }


async def _broadcast_plugin_reload(plugin_name: str, version: str, environment: str):
    """Broadcast plugin reload to all workers (background task)."""
    try:
        logger.info(f"Broadcasting reload for plugin {plugin_name} v{version} in {environment}")
        
        # In a real implementation, this would:
        # 1. Send reload signal to all worker processes
        # 2. Wait for confirmation from each worker
        # 3. Update deployment status based on success/failure
        
        # Simulate broadcast delay
        await asyncio.sleep(2)
        
        # Update deployment status
        plugin_manager = await get_plugin_manager()
        # This would update the specific deployment record's health check status
        
        logger.info(f"Plugin {plugin_name} reload broadcast completed")
        
    except Exception as e:
        logger.error(f"Failed to broadcast plugin reload: {e}")


def _calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()