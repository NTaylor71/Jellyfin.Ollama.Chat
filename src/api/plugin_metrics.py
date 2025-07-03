"""
Plugin Metrics for Prometheus
Exposes plugin performance and health metrics to Prometheus.
"""

import logging
import time
from typing import Dict, Any

from prometheus_client import Counter, Histogram, Gauge, Info

logger = logging.getLogger(__name__)

# Plugin execution metrics
plugin_executions_total = Counter(
    'plugin_executions_total',
    'Total number of plugin executions',
    ['plugin_name', 'plugin_type', 'status']
)

plugin_execution_duration_seconds = Histogram(
    'plugin_execution_duration_seconds',
    'Plugin execution duration in seconds',
    ['plugin_name', 'plugin_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Plugin health metrics
plugin_health_status = Gauge(
    'plugin_health_status',
    'Plugin health status (1=healthy, 0=unhealthy)',
    ['plugin_name', 'plugin_type']
)

plugin_initialization_status = Gauge(
    'plugin_initialization_status',
    'Plugin initialization status (1=initialized, 0=not_initialized)',
    ['plugin_name', 'plugin_type']
)

# Plugin resource usage metrics
plugin_memory_usage_bytes = Gauge(
    'plugin_memory_usage_bytes',
    'Plugin memory usage in bytes',
    ['plugin_name', 'plugin_type']
)

plugin_cpu_usage_percent = Gauge(
    'plugin_cpu_usage_percent',
    'Plugin CPU usage percentage',
    ['plugin_name', 'plugin_type']
)

plugin_thread_count = Gauge(
    'plugin_thread_count',
    'Number of threads used by plugin',
    ['plugin_name', 'plugin_type']
)

# System-wide plugin metrics
total_plugins = Gauge(
    'plugins_total',
    'Total number of registered plugins'
)

enabled_plugins = Gauge(
    'plugins_enabled_total',
    'Total number of enabled plugins'
)

initialized_plugins = Gauge(
    'plugins_initialized_total',
    'Total number of initialized plugins'
)

healthy_plugins = Gauge(
    'plugins_healthy_total',
    'Total number of healthy plugins'
)

# Plugin registry metrics
plugin_registry_info = Info(
    'plugin_registry_info',
    'Plugin registry information'
)


class PluginMetricsCollector:
    """Collects and updates plugin metrics for Prometheus."""
    
    def __init__(self):
        self._last_collection_time = 0
        self._collection_interval = 30  # seconds
        
    def record_plugin_execution(self, plugin_name: str, plugin_type: str, 
                              execution_time_ms: float, success: bool) -> None:
        """Record a plugin execution event."""
        status = "success" if success else "failure"
        
        # Update counters
        plugin_executions_total.labels(
            plugin_name=plugin_name,
            plugin_type=plugin_type,
            status=status
        ).inc()
        
        # Update duration histogram
        plugin_execution_duration_seconds.labels(
            plugin_name=plugin_name,
            plugin_type=plugin_type
        ).observe(execution_time_ms / 1000.0)  # Convert ms to seconds
        
        logger.debug(f"Recorded plugin execution: {plugin_name} ({status}, {execution_time_ms}ms)")
    
    async def update_plugin_health_metrics(self, plugin_registry) -> None:
        """Update plugin health and resource metrics."""
        try:
            current_time = time.time()
            
            # Rate limit collection to avoid overhead
            if current_time - self._last_collection_time < self._collection_interval:
                return
            
            self._last_collection_time = current_time
            
            # Get plugin status
            status = await plugin_registry.get_plugin_status()
            
            # Update system-wide metrics
            total_plugins.set(status["total_plugins"])
            enabled_plugins.set(status["enabled_plugins"])
            initialized_plugins.set(status["initialized_plugins"])
            
            # Count healthy plugins and update individual metrics
            healthy_count = 0
            
            for plugin_name, details in status["plugin_details"].items():
                # Get plugin instance for metadata
                plugin_instance = await plugin_registry.get_plugin(plugin_name)
                plugin_type = "unknown"
                
                if plugin_instance:
                    plugin_type = plugin_instance.metadata.plugin_type.value
                
                # Health status
                health_info = details.get("health", {})
                is_healthy = health_info.get("status") == "healthy"
                plugin_health_status.labels(
                    plugin_name=plugin_name,
                    plugin_type=plugin_type
                ).set(1 if is_healthy else 0)
                
                if is_healthy:
                    healthy_count += 1
                
                # Initialization status
                is_initialized = details.get("initialized", False)
                plugin_initialization_status.labels(
                    plugin_name=plugin_name,
                    plugin_type=plugin_type
                ).set(1 if is_initialized else 0)
                
                # Resource usage metrics
                resource_usage = health_info.get("resource_usage", {})
                
                # Memory usage (convert MB to bytes)
                memory_mb = resource_usage.get("memory_used_mb", 0)
                plugin_memory_usage_bytes.labels(
                    plugin_name=plugin_name,
                    plugin_type=plugin_type
                ).set(memory_mb * 1024 * 1024)
                
                # CPU usage
                cpu_percent = resource_usage.get("cpu_percent", 0)
                plugin_cpu_usage_percent.labels(
                    plugin_name=plugin_name,
                    plugin_type=plugin_type
                ).set(cpu_percent)
                
                # Thread count
                thread_count = resource_usage.get("num_threads", 0)
                plugin_thread_count.labels(
                    plugin_name=plugin_name,
                    plugin_type=plugin_type
                ).set(thread_count)
            
            # Update healthy plugins count
            healthy_plugins.set(healthy_count)
            
            # Update registry info
            plugin_registry_info.info({
                'total_plugins': str(status["total_plugins"]),
                'enabled_plugins': str(status["enabled_plugins"]),
                'initialized_plugins': str(status["initialized_plugins"]),
                'healthy_plugins': str(healthy_count),
                'last_update': str(current_time)
            })
            
            logger.debug(f"Updated plugin metrics: {healthy_count}/{status['total_plugins']} healthy plugins")
            
        except Exception as e:
            logger.error(f"Error updating plugin health metrics: {e}")
    
    def reset_plugin_metrics(self, plugin_name: str, plugin_type: str) -> None:
        """Reset metrics for a specific plugin (useful during plugin reload)."""
        try:
            # Reset health metrics
            plugin_health_status.labels(
                plugin_name=plugin_name,
                plugin_type=plugin_type
            ).set(0)
            
            plugin_initialization_status.labels(
                plugin_name=plugin_name,
                plugin_type=plugin_type
            ).set(0)
            
            # Reset resource metrics
            plugin_memory_usage_bytes.labels(
                plugin_name=plugin_name,
                plugin_type=plugin_type
            ).set(0)
            
            plugin_cpu_usage_percent.labels(
                plugin_name=plugin_name,
                plugin_type=plugin_type
            ).set(0)
            
            plugin_thread_count.labels(
                plugin_name=plugin_name,
                plugin_type=plugin_type
            ).set(0)
            
            logger.debug(f"Reset metrics for plugin: {plugin_name}")
            
        except Exception as e:
            logger.error(f"Error resetting plugin metrics for {plugin_name}: {e}")


# Global metrics collector instance
plugin_metrics = PluginMetricsCollector()


def get_plugin_metrics() -> PluginMetricsCollector:
    """Get the global plugin metrics collector."""
    return plugin_metrics