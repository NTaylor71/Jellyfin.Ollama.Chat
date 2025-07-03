"""
Plugin Management Routes
Provides API endpoints for managing and monitoring plugins.
"""

import logging
import time
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..plugin_registry import plugin_registry
from ..plugin_watcher import plugin_watcher
from ...plugins.base import PluginType, ExecutionPriority

logger = logging.getLogger(__name__)

router = APIRouter()


class PluginStatusResponse(BaseModel):
    """Response model for plugin status."""
    total_plugins: int
    enabled_plugins: int
    initialized_plugins: int
    plugins_by_type: Dict[str, int]
    plugin_details: Dict[str, Any]


class PluginActionRequest(BaseModel):
    """Request model for plugin actions."""
    plugin_name: str = Field(..., min_length=1, max_length=100)


class PluginReloadResponse(BaseModel):
    """Response model for plugin reload operations."""
    success: bool
    message: str
    reloaded_plugins: Optional[List[str]] = None


@router.get("/status", response_model=PluginStatusResponse)
async def get_plugin_status():
    """Get status of all plugins."""
    try:
        status = await plugin_registry.get_plugin_status()
        return PluginStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting plugin status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get plugin status: {str(e)}")


@router.get("/list")
async def list_plugins(
    plugin_type: Optional[PluginType] = Query(None, description="Filter by plugin type"),
    enabled_only: bool = Query(True, description="Show only enabled plugins")
):
    """List all plugins with optional filtering."""
    try:
        status = await plugin_registry.get_plugin_status()
        plugins = []
        
        for name, details in status["plugin_details"].items():
            if enabled_only and not details["enabled"]:
                continue
                
            plugin_info = {
                "name": name,
                "enabled": details["enabled"],
                "initialized": details["initialized"],
                "file_path": details["file_path"],
                "health": details.get("health", {}),
                "initialization_error": details.get("initialization_error")
            }
            
            # Get plugin metadata if available
            plugin = await plugin_registry.get_plugin(name)
            if plugin:
                metadata = plugin.metadata
                plugin_info.update({
                    "type": metadata.plugin_type.value,
                    "version": metadata.version,
                    "description": metadata.description,
                    "author": metadata.author,
                    "tags": metadata.tags,
                    "priority": metadata.execution_priority.value
                })
                
                # Filter by type if requested
                if plugin_type and metadata.plugin_type != plugin_type:
                    continue
            
            plugins.append(plugin_info)
        
        return {"plugins": plugins, "total": len(plugins)}
        
    except Exception as e:
        logger.error(f"Error listing plugins: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list plugins: {str(e)}")


@router.get("/health/{plugin_name}")
async def get_plugin_health(plugin_name: str):
    """Get health status for a specific plugin."""
    try:
        plugin = await plugin_registry.get_plugin(plugin_name)
        if not plugin:
            raise HTTPException(status_code=404, detail=f"Plugin '{plugin_name}' not found")
        
        health = await plugin.health_check()
        return {
            "plugin_name": plugin_name,
            "health": health,
            "metadata": {
                "name": plugin.metadata.name,
                "version": plugin.metadata.version,
                "type": plugin.metadata.plugin_type.value,
                "description": plugin.metadata.description
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting plugin health for {plugin_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get plugin health: {str(e)}")


@router.post("/reload/{plugin_name}", response_model=PluginReloadResponse)
async def reload_plugin(plugin_name: str):
    """Reload a specific plugin."""
    try:
        success = await plugin_registry.reload_plugin(plugin_name)
        
        if success:
            return PluginReloadResponse(
                success=True,
                message=f"Plugin '{plugin_name}' reloaded successfully",
                reloaded_plugins=[plugin_name]
            )
        else:
            return PluginReloadResponse(
                success=False,
                message=f"Failed to reload plugin '{plugin_name}'"
            )
            
    except Exception as e:
        logger.error(f"Error reloading plugin {plugin_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload plugin: {str(e)}")


@router.post("/reload-all", response_model=PluginReloadResponse)
async def reload_all_plugins():
    """Reload all plugins."""
    try:
        success_count = await plugin_registry.reload_all_plugins()
        total_plugins = len(plugin_registry._plugins)
        
        return PluginReloadResponse(
            success=success_count > 0,
            message=f"Reloaded {success_count}/{total_plugins} plugins",
            reloaded_plugins=list(plugin_registry._plugins.keys())
        )
        
    except Exception as e:
        logger.error(f"Error reloading all plugins: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload plugins: {str(e)}")


@router.post("/enable/{plugin_name}")
async def enable_plugin(plugin_name: str):
    """Enable a plugin."""
    try:
        success = await plugin_registry.enable_plugin(plugin_name)
        
        if success:
            return {"success": True, "message": f"Plugin '{plugin_name}' enabled"}
        else:
            return {"success": False, "message": f"Plugin '{plugin_name}' is already enabled or not found"}
            
    except Exception as e:
        logger.error(f"Error enabling plugin {plugin_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to enable plugin: {str(e)}")


@router.post("/disable/{plugin_name}")
async def disable_plugin(plugin_name: str):
    """Disable a plugin."""
    try:
        success = await plugin_registry.disable_plugin(plugin_name)
        
        if success:
            return {"success": True, "message": f"Plugin '{plugin_name}' disabled"}
        else:
            return {"success": False, "message": f"Plugin '{plugin_name}' is already disabled or not found"}
            
    except Exception as e:
        logger.error(f"Error disabling plugin {plugin_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to disable plugin: {str(e)}")


@router.get("/watcher/status")
async def get_watcher_status():
    """Get plugin file watcher status."""
    try:
        status = plugin_watcher.get_watch_status()
        return {
            "watcher": status,
            "message": "Plugin file watcher is running" if status["is_watching"] else "Plugin file watcher is stopped"
        }
        
    except Exception as e:
        logger.error(f"Error getting watcher status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get watcher status: {str(e)}")


@router.post("/watcher/force-reload")
async def force_reload_watcher():
    """Force reload all plugins through the watcher."""
    try:
        success_count = await plugin_watcher.force_reload_all()
        return {
            "success": success_count > 0,
            "message": f"Force reloaded {success_count} plugins",
            "reloaded_count": success_count
        }
        
    except Exception as e:
        logger.error(f"Error force reloading plugins: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to force reload: {str(e)}")


@router.get("/health")
async def get_plugin_health_overview():
    """Get health overview of all plugins with performance metrics."""
    try:
        status = await plugin_registry.get_plugin_status()
        
        # Calculate aggregate metrics
        total_plugins = status["total_plugins"]
        healthy_plugins = 0
        total_executions = 0
        total_failures = 0
        avg_response_time = 0
        aggregate_resource_usage = {
            "total_memory_used_mb": 0,
            "total_cpu_usage_percent": 0,
            "plugins_using_gpu": 0
        }
        
        plugin_health_details = []
        
        for name, details in status["plugin_details"].items():
            health_info = details.get("health", {})
            metrics = health_info.get("metrics", {})
            resource_usage = health_info.get("resource_usage", {})
            
            # Count healthy plugins
            if health_info.get("status") == "healthy":
                healthy_plugins += 1
            
            # Aggregate performance metrics
            executions = metrics.get("executions", 0)
            failures = metrics.get("failed_executions", 0)
            total_executions += executions
            total_failures += failures
            
            # Aggregate resource usage
            plugin_memory = resource_usage.get("memory_used_mb", 0)
            plugin_cpu = resource_usage.get("cpu_percent", 0)
            aggregate_resource_usage["total_memory_used_mb"] += plugin_memory
            aggregate_resource_usage["total_cpu_usage_percent"] += plugin_cpu
            
            # Check if plugin uses GPU
            requirements = resource_usage.get("resource_requirements", {})
            if requirements.get("requires_gpu", False):
                aggregate_resource_usage["plugins_using_gpu"] += 1
            
            # Calculate plugin-specific metrics
            success_rate = 0
            if executions > 0:
                success_rate = ((executions - failures) / executions) * 100
            
            plugin_health_details.append({
                "name": name,
                "status": health_info.get("status", "unknown"),
                "initialized": details.get("initialized", False),
                "success_rate_percent": round(success_rate, 2),
                "total_executions": executions,
                "avg_execution_time_ms": round(metrics.get("avg_execution_time_ms", 0), 2),
                "last_error": details.get("initialization_error"),
                "resource_usage": {
                    "memory_used_mb": plugin_memory,
                    "cpu_percent": plugin_cpu,
                    "num_threads": resource_usage.get("num_threads", 0),
                    "requires_gpu": requirements.get("requires_gpu", False)
                },
                "last_health_check": health_info.get("last_health_check", 0)
            })
        
        # Calculate overall metrics
        if total_executions > 0:
            overall_success_rate = ((total_executions - total_failures) / total_executions) * 100
        else:
            overall_success_rate = 100
            
        # Calculate average response time across all plugins
        total_response_time = 0
        plugins_with_metrics = 0
        for detail in plugin_health_details:
            if detail["avg_execution_time_ms"] > 0:
                total_response_time += detail["avg_execution_time_ms"]
                plugins_with_metrics += 1
        
        if plugins_with_metrics > 0:
            avg_response_time = total_response_time / plugins_with_metrics
        
        return {
            "overall_health": {
                "status": "healthy" if healthy_plugins == total_plugins else "degraded",
                "healthy_plugins": healthy_plugins,
                "total_plugins": total_plugins,
                "health_percentage": round((healthy_plugins / total_plugins) * 100, 2) if total_plugins > 0 else 0
            },
            "performance_metrics": {
                "total_executions": total_executions,
                "total_failures": total_failures,
                "overall_success_rate_percent": round(overall_success_rate, 2),
                "avg_response_time_ms": round(avg_response_time, 2)
            },
            "resource_usage": aggregate_resource_usage,
            "plugin_health_details": plugin_health_details,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting plugin health overview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get plugin health overview: {str(e)}")


@router.get("/types")
async def get_plugin_types():
    """Get available plugin types."""
    return {
        "plugin_types": [
            {
                "name": plugin_type.value,
                "description": f"{plugin_type.value.replace('_', ' ').title()} plugin type"
            }
            for plugin_type in PluginType
        ],
        "execution_priorities": [
            {
                "name": priority.value,
                "description": f"{priority.value.title()} execution priority"
            }
            for priority in ExecutionPriority
        ]
    }


@router.get("/by-type/{plugin_type}")
async def get_plugins_by_type(plugin_type: PluginType):
    """Get all plugins of a specific type."""
    try:
        plugins = await plugin_registry.get_plugins_by_type(plugin_type)
        
        plugin_info = []
        for plugin in plugins:
            metadata = plugin.metadata
            health = await plugin.health_check()
            
            plugin_info.append({
                "name": metadata.name,
                "version": metadata.version,
                "description": metadata.description,
                "author": metadata.author,
                "tags": metadata.tags,
                "priority": metadata.execution_priority.value,
                "health": health
            })
        
        return {
            "plugin_type": plugin_type.value,
            "plugins": plugin_info,
            "count": len(plugin_info)
        }
        
    except Exception as e:
        logger.error(f"Error getting plugins by type {plugin_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get plugins by type: {str(e)}")