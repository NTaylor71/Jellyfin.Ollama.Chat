"""
Health Monitor - Monitors plugin and service health for the worker.

Provides health checking, failure detection, and recovery mechanisms
for both local plugins and remote services.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json

from src.worker.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)


class HealthStatus:
    """Health status information."""
    
    def __init__(self, name: str):
        self.name = name
        self.status = "unknown"
        self.last_check = None
        self.consecutive_failures = 0
        self.last_success = None
        self.error_history = []
        self.metadata = {}
    
    def mark_success(self):
        """Mark a successful health check."""
        self.status = "healthy"
        self.last_check = datetime.utcnow()
        self.last_success = datetime.utcnow()
        self.consecutive_failures = 0
    
    def mark_failure(self, error: str):
        """Mark a failed health check."""
        self.status = "unhealthy"
        self.last_check = datetime.utcnow()
        self.consecutive_failures += 1
        self.error_history.append({
            "timestamp": datetime.utcnow(),
            "error": error
        })
        
        # Keep only recent errors
        if len(self.error_history) > 10:
            self.error_history = self.error_history[-10:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "consecutive_failures": self.consecutive_failures,
            "recent_errors": [
                {
                    "timestamp": err["timestamp"].isoformat(),
                    "error": err["error"]
                } for err in self.error_history[-3:]  # Only recent errors
            ],
            "metadata": self.metadata
        }


class HealthMonitor:
    """
    Monitors health of plugins and services.
    
    Features:
    - Periodic health checks
    - Failure detection and alerting
    - Health status reporting
    - Recovery recommendations
    """
    
    def __init__(self, plugin_loader: PluginLoader):
        self.plugin_loader = plugin_loader
        self.health_statuses: Dict[str, HealthStatus] = {}
        self.check_interval = 60  # seconds
        self.monitor_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start_monitoring(self):
        """Start the health monitoring loop."""
        if self.running:
            logger.warning("Health monitor already running")
            return
        
        logger.info("Starting health monitor...")
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("✅ Health monitor started")
    
    async def stop_monitoring(self):
        """Stop the health monitoring loop."""
        if not self.running:
            return
        
        logger.info("Stopping health monitor...")
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Health monitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(10)  # Brief pause on error
    
    async def _perform_health_checks(self):
        """Perform health checks on all plugins and services."""
        logger.debug("Performing health checks...")
        
        # Get list of available plugins
        plugins_info = self.plugin_loader.list_available_plugins()
        
        # Check local plugins
        for plugin_name in plugins_info.get("local_plugins", {}):
            await self._check_plugin_health(plugin_name)
        
        # Check service plugins
        for plugin_name in plugins_info.get("service_plugins", {}):
            await self._check_service_plugin_health(plugin_name)
        
        # Log summary
        healthy_count = len([h for h in self.health_statuses.values() if h.status == "healthy"])
        total_count = len(self.health_statuses)
        
        if total_count > 0:
            logger.info(f"Health check complete: {healthy_count}/{total_count} plugins healthy")
    
    async def _check_plugin_health(self, plugin_name: str):
        """Check health of a local plugin."""
        if plugin_name not in self.health_statuses:
            self.health_statuses[plugin_name] = HealthStatus(plugin_name)
        
        status = self.health_statuses[plugin_name]
        
        try:
            health_data = await self.plugin_loader.get_plugin_health(plugin_name)
            
            if health_data.get("status") == "healthy":
                status.mark_success()
                status.metadata = health_data
            else:
                error_msg = health_data.get("error", "Unknown health check failure")
                status.mark_failure(error_msg)
                
        except Exception as e:
            status.mark_failure(str(e))
            logger.warning(f"Health check failed for {plugin_name}: {e}")
    
    async def _check_service_plugin_health(self, plugin_name: str):
        """Check health of a service-based plugin."""
        if plugin_name not in self.health_statuses:
            self.health_statuses[plugin_name] = HealthStatus(plugin_name)
        
        status = self.health_statuses[plugin_name]
        
        try:
            health_data = await self.plugin_loader.get_plugin_health(plugin_name)
            
            if health_data.get("status") == "healthy":
                status.mark_success()
                status.metadata = health_data
            else:
                error_msg = health_data.get("error", "Service health check failed")
                status.mark_failure(error_msg)
                
        except Exception as e:
            status.mark_failure(str(e))
            logger.warning(f"Service health check failed for {plugin_name}: {e}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        now = datetime.utcnow()
        
        healthy_plugins = [h for h in self.health_statuses.values() if h.status == "healthy"]
        unhealthy_plugins = [h for h in self.health_statuses.values() if h.status == "unhealthy"]
        
        # Calculate overall health score
        total_plugins = len(self.health_statuses)
        health_score = len(healthy_plugins) / total_plugins if total_plugins > 0 else 0
        
        # Determine overall status
        if health_score >= 0.8:
            overall_status = "healthy"
        elif health_score >= 0.5:
            overall_status = "degraded"
        else:
            overall_status = "critical"
        
        return {
            "overall_status": overall_status,
            "health_score": health_score,
            "total_plugins": total_plugins,
            "healthy_plugins": len(healthy_plugins),
            "unhealthy_plugins": len(unhealthy_plugins),
            "last_check": now.isoformat(),
            "details": {
                "healthy": [h.name for h in healthy_plugins],
                "unhealthy": [h.name for h in unhealthy_plugins]
            }
        }
    
    def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information for all plugins."""
        return {
            "summary": self.get_health_summary(),
            "plugins": {
                name: status.to_dict() 
                for name, status in self.health_statuses.items()
            }
        }
    
    def get_plugin_health(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get health information for a specific plugin."""
        if plugin_name in self.health_statuses:
            return self.health_statuses[plugin_name].to_dict()
        return None
    
    def get_recovery_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for recovering unhealthy plugins."""
        recommendations = []
        
        for name, status in self.health_statuses.items():
            if status.status == "unhealthy":
                # Basic recovery recommendations
                if status.consecutive_failures >= 5:
                    recommendations.append({
                        "plugin": name,
                        "severity": "high",
                        "action": "restart_plugin",
                        "reason": f"Plugin has failed {status.consecutive_failures} consecutive health checks"
                    })
                elif status.consecutive_failures >= 3:
                    recommendations.append({
                        "plugin": name,
                        "severity": "medium", 
                        "action": "check_dependencies",
                        "reason": f"Plugin has failed {status.consecutive_failures} health checks"
                    })
                else:
                    recommendations.append({
                        "plugin": name,
                        "severity": "low",
                        "action": "monitor",
                        "reason": "Plugin recently became unhealthy"
                    })
        
        return recommendations
    
    async def attempt_plugin_recovery(self, plugin_name: str) -> bool:
        """Attempt to recover a failed plugin."""
        logger.info(f"Attempting recovery for plugin: {plugin_name}")
        
        try:
            # For local plugins, try reloading
            plugins_info = self.plugin_loader.list_available_plugins()
            
            if plugin_name in plugins_info.get("local_plugins", {}):
                # Try reloading the plugin
                # This would require plugin loader to support reloading
                logger.info(f"Would attempt to reload local plugin: {plugin_name}")
                return False  # Not implemented yet
            
            elif plugin_name in plugins_info.get("service_plugins", {}):
                # For service plugins, the service itself needs to recover
                logger.info(f"Service plugin {plugin_name} recovery depends on service health")
                return False  # Service recovery not implemented here
            
            else:
                logger.warning(f"Unknown plugin type for recovery: {plugin_name}")
                return False
                
        except Exception as e:
            logger.error(f"Recovery attempt failed for {plugin_name}: {e}")
            return False
    
    def export_health_report(self) -> str:
        """Export detailed health report as JSON string."""
        health_data = self.get_detailed_health()
        return json.dumps(health_data, indent=2, default=str)