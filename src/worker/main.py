"""
Redis worker main entry point.
Processes tasks from Redis queue system.
"""

import asyncio
import signal
import sys
import logging
from typing import Dict, Any
from datetime import datetime
import threading
import time
from prometheus_client import Counter, Histogram, Gauge, start_http_server

from src.shared.config import get_settings
from src.shared.hardware_config_loader import get_hardware_config_loader
from src.worker.resource_queue_manager import ResourceAwareQueueManager
from src.worker.resource_manager import ResourcePool, ResourceRequirement, create_resource_pool_from_config
from src.worker.plugin_loader import PluginLoader
from src.worker.health_monitor import HealthMonitor
from src.plugins.base import PluginExecutionContext


logger = logging.getLogger(__name__)


TASKS_PROCESSED = Counter('worker_tasks_processed_total', 'Total number of tasks processed')
TASKS_FAILED = Counter('worker_tasks_failed_total', 'Total number of tasks that failed')
TASK_DURATION = Histogram('worker_task_duration_seconds', 'Time spent processing tasks')
WORKER_UPTIME = Gauge('worker_uptime_seconds', 'Worker uptime in seconds')
PLUGIN_EXECUTIONS = Counter('worker_plugin_executions_total', 'Total number of plugin executions')
SERVICE_CALLS = Counter('worker_service_calls_total', 'Total number of service calls')


class WorkerService:
    """Main worker service for processing queue tasks."""
    
    def __init__(self):
        self.settings = get_settings()
        

        hardware_loader = get_hardware_config_loader()
        config_name = getattr(self.settings, "HARDWARE_CONFIG", "default")
        
        try:
            resource_config = hardware_loader.get_resource_pool_config(config_name)
            logger.info(f"Loaded hardware config '{config_name}': {resource_config}")
        except Exception as e:
            logger.warning(f"Failed to load hardware config '{config_name}': {e}")
            logger.warning("Falling back to default configuration")

            resource_config = {
                "cpu_cores": getattr(self.settings, "WORKER_CPU_CORES", 3),
                "cpu_threads": getattr(self.settings, "WORKER_CPU_THREADS", 6),
                "gpu_count": getattr(self.settings, "WORKER_GPU_COUNT", 1),
                "memory_mb": getattr(self.settings, "WORKER_MEMORY_MB", 8192)
            }
        
        self.resource_pool = create_resource_pool_from_config(resource_config, worker_id=f"worker_{id(self)}")
        
        
        self.queue_manager = ResourceAwareQueueManager(self.resource_pool)
        self.plugin_loader = PluginLoader()
        self.health_monitor = None
        self.running = False
        self.metrics_server = None
        self.stats = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "uptime_start": None,
            "plugin_executions": 0,
            "service_calls": 0,
            "health_checks": 0
        }
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        def signal_handler(signum, frame):
            logger.info(f"\n🔄 Received signal {signum}, shutting down gracefully...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def process_task(self, task_data: Dict[str, Any]) -> None:
        """
        Process a single task from the queue using direct plugin execution with resource management.
        
        Args:
            task_data: Task data from queue
        """
        task_id = task_data.get("task_id")
        task_type = task_data.get("task_type")
        

        resource_req = ResourceRequirement(**task_data.get("resource_requirements", {}))
        allocation = self.resource_pool.allocate(task_id, resource_req)
        
        logger.info(f"🔄 Processing task {task_id} of type {task_type} (CPU={resource_req.cpu_cores}, GPU={resource_req.gpu_count}, Mem={resource_req.memory_mb}MB)")
        
        start_time = time.time()
        
        try:

            plugin_result = await self.plugin_loader.route_task_to_plugin(task_type, task_data)
            

            duration = time.time() - start_time
            TASK_DURATION.observe(duration)
            
            if plugin_result.success:

                result = {
                    "task_id": task_id,
                    "task_type": task_type,
                    "success": True,
                    "data": plugin_result.data,
                    "execution_time_ms": plugin_result.execution_time_ms,
                    "metadata": plugin_result.metadata,
                    "processed_at": datetime.utcnow().isoformat(),
                    "worker_id": f"worker_{id(self)}"
                }
                

                await self.queue_manager.complete_task(task_id, result)
                self.stats["tasks_processed"] += 1
                TASKS_PROCESSED.inc()
                
                
                if plugin_result.metadata and plugin_result.metadata.get("via_service"):
                    self.stats["service_calls"] += 1
                    SERVICE_CALLS.inc()
                else:
                    self.stats["plugin_executions"] += 1
                    PLUGIN_EXECUTIONS.inc()
                
                logger.info(f"✅ Completed task {task_id} in {plugin_result.execution_time_ms:.1f}ms")
                
            else:

                error_result = {
                    "task_id": task_id,
                    "task_type": task_type,
                    "success": False,
                    "error": plugin_result.error_message,
                    "execution_time_ms": plugin_result.execution_time_ms,
                    "processed_at": datetime.utcnow().isoformat(),
                    "worker_id": f"worker_{id(self)}"
                }
                
                await self.queue_manager.fail_task(task_data, plugin_result.error_message)
                self.stats["tasks_failed"] += 1
                TASKS_FAILED.inc()
                
                logger.error(f"❌ Task {task_id} failed: {plugin_result.error_message}")
            
        except Exception as e:
            logger.error(f"❌ Task {task_id} processing error: {e}")
            
            error_result = {
                "task_id": task_id,
                "task_type": task_type,
                "success": False,
                "error": str(e),
                "processed_at": datetime.utcnow().isoformat(),
                "worker_id": f"worker_{id(self)}"
            }
            
            await self.queue_manager.fail_task(task_data, str(e))
            self.stats["tasks_failed"] += 1
            TASKS_FAILED.inc()
        
        finally:

            self.resource_pool.release(task_id)
            logger.debug(f"Released resources for task {task_id}")
    
    async def worker_loop(self):
        """Main worker processing loop."""
        logger.info(f"🚀 Worker started on {self.settings.ENV} environment")
        logger.info(f"💾 Resource pool: {self.resource_pool.total_cpu_cores} CPU cores, {self.resource_pool.total_gpus} GPUs, {self.resource_pool.total_memory_mb}MB memory")
        logger.info(f"📊 Worker stats: {self.stats}")
        
        
        if not await self.plugin_loader.initialize():
            logger.error("❌ Failed to initialize plugin loader")
            return
        
        
        self.health_monitor = HealthMonitor(self.plugin_loader)
        await self.health_monitor.start_monitoring()
        
        self.stats["uptime_start"] = datetime.utcnow()
        

        try:
            start_http_server(8004)
            logger.info("📊 Prometheus metrics server started on port 8004")
        except Exception as e:
            logger.warning(f"⚠️ Failed to start metrics server: {e}")
        
        while self.running:
            
            if self.stats["uptime_start"]:
                uptime = (datetime.utcnow() - self.stats["uptime_start"]).total_seconds()
                WORKER_UPTIME.set(uptime)
            try:
                
                if not await self.queue_manager.health_check():
                    logger.warning("⚠️ Redis connection unhealthy, retrying...")
                    await asyncio.sleep(5)
                    continue
                

                task_data = await self.queue_manager.dequeue_task(timeout=10)
                
                if task_data:
                    await self.process_task(task_data)
                else:

                    await asyncio.sleep(1)
                    

                    if self.stats["tasks_processed"] > 0 and self.stats["tasks_processed"] % 10 == 0:
                        health_summary = self.health_monitor.get_health_summary() if self.health_monitor else {"overall_status": "unknown"}
                        logger.info(f"📊 Processed: {self.stats['tasks_processed']}, Failed: {self.stats['tasks_failed']}, Plugin: {self.stats['plugin_executions']}, Service: {self.stats['service_calls']}, Health: {health_summary['overall_status']}")
                    
            except Exception as e:
                logger.error(f"❌ Worker loop error: {e}")
                await asyncio.sleep(5)
    
    async def start(self):
        """Start the worker service."""
        self.running = True
        self.setup_signal_handlers()
        
        try:

            await self.worker_loop()
        finally:

            await self.cleanup()
        
        logger.info("🔄 Worker service stopped")
    
    async def cleanup(self):
        """Cleanup worker resources."""
        logger.info("🧹 Cleaning up worker resources...")
        

        if self.health_monitor:
            await self.health_monitor.stop_monitoring()
        

        if hasattr(self, 'plugin_loader'):
            await self.plugin_loader.cleanup()
        

        if hasattr(self, 'queue_manager'):
            await self.queue_manager.cleanup()
        

        uptime = (datetime.utcnow() - self.stats.get("uptime_start", datetime.utcnow())).total_seconds()
        resource_summary = self.resource_pool.get_status_summary()
        logger.info(f"📊 Final stats - Processed: {self.stats['tasks_processed']}, Failed: {self.stats['tasks_failed']}, Uptime: {uptime:.1f}s")
        logger.info(f"💾 Final resource usage: {resource_summary['current_usage']}")
        

        if resource_summary['running_tasks']['total'] > 0:
            logger.warning(f"⚠️ {resource_summary['running_tasks']['total']} tasks still marked as running during cleanup")
        
        logger.info("✅ Worker cleanup complete")


async def main():
    """Main entry point for the worker service."""
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    worker = WorkerService()
    await worker.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🔄 Worker interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Worker failed to start: {e}")
        sys.exit(1)