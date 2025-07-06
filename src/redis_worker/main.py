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
from src.redis_worker.queue_manager import RedisQueueManager
from src.redis_worker.plugin_loader import PluginLoader
from src.redis_worker.health_monitor import HealthMonitor
from src.plugins.base import PluginExecutionContext


logger = logging.getLogger(__name__)

# Prometheus metrics
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
        self.queue_manager = RedisQueueManager()
        self.plugin_loader = PluginLoader()
        self.health_monitor = None  # Will be initialized after plugin_loader
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
        Process a single task from the queue using plugin-based routing.
        
        Args:
            task_data: Task data from queue
        """
        task_id = task_data.get("task_id")
        task_type = task_data.get("task_type")
        
        logger.info(f"🔄 Processing task {task_id} of type {task_type}")
        
        start_time = time.time()
        
        try:
            # Route task to appropriate plugin
            plugin_result = await self.plugin_loader.route_task_to_plugin(task_type, task_data)
            
            # Record task duration
            duration = time.time() - start_time
            TASK_DURATION.observe(duration)
            
            if plugin_result.success:
                # Prepare successful result
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
                
                # Mark as completed
                self.queue_manager.complete_task(task_id, result)
                self.stats["tasks_processed"] += 1
                TASKS_PROCESSED.inc()
                
                # Update stats based on execution type
                if plugin_result.metadata and plugin_result.metadata.get("via_service"):
                    self.stats["service_calls"] += 1
                    SERVICE_CALLS.inc()
                else:
                    self.stats["plugin_executions"] += 1
                    PLUGIN_EXECUTIONS.inc()
                
                logger.info(f"✅ Completed task {task_id} in {plugin_result.execution_time_ms:.1f}ms")
                
            else:
                # Handle plugin execution failure
                error_result = {
                    "task_id": task_id,
                    "task_type": task_type,
                    "success": False,
                    "error": plugin_result.error_message,
                    "execution_time_ms": plugin_result.execution_time_ms,
                    "processed_at": datetime.utcnow().isoformat(),
                    "worker_id": f"worker_{id(self)}"
                }
                
                self.queue_manager.fail_task(task_data, plugin_result.error_message)
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
            
            self.queue_manager.fail_task(task_data, str(e))
            self.stats["tasks_failed"] += 1
            TASKS_FAILED.inc()
    
    async def worker_loop(self):
        """Main worker processing loop."""
        logger.info(f"🚀 Worker started on {self.settings.ENV} environment")
        logger.info(f"📊 Worker stats: {self.stats}")
        
        # Initialize plugin loader
        if not await self.plugin_loader.initialize():
            logger.error("❌ Failed to initialize plugin loader")
            return
        
        # Initialize health monitor
        self.health_monitor = HealthMonitor(self.plugin_loader)
        await self.health_monitor.start_monitoring()
        
        self.stats["uptime_start"] = datetime.utcnow()
        
        # Start metrics server on a different port for worker
        try:
            start_http_server(8004)  # Worker metrics on port 8004
            logger.info("📊 Prometheus metrics server started on port 8004")
        except Exception as e:
            logger.warning(f"⚠️ Failed to start metrics server: {e}")
        
        while self.running:
            # Update uptime metric
            if self.stats["uptime_start"]:
                uptime = (datetime.utcnow() - self.stats["uptime_start"]).total_seconds()
                WORKER_UPTIME.set(uptime)
            try:
                # Check Redis health
                if not self.queue_manager.health_check():
                    logger.warning("⚠️ Redis connection unhealthy, retrying...")
                    await asyncio.sleep(5)
                    continue
                
                # Dequeue task with timeout
                task_data = self.queue_manager.dequeue_task(timeout=10)
                
                if task_data:
                    await self.process_task(task_data)
                else:
                    # No tasks available, brief pause
                    await asyncio.sleep(1)
                    
                    # Periodic stats and health logging
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
            # Start worker loop
            await self.worker_loop()
        finally:
            # Cleanup
            await self.cleanup()
        
        logger.info("🔄 Worker service stopped")
    
    async def cleanup(self):
        """Cleanup worker resources."""
        logger.info("🧹 Cleaning up worker resources...")
        
        # Stop health monitoring
        if self.health_monitor:
            await self.health_monitor.stop_monitoring()
        
        # Cleanup plugin loader
        if hasattr(self, 'plugin_loader'):
            await self.plugin_loader.cleanup()
        
        # Final stats
        uptime = (datetime.utcnow() - self.stats.get("uptime_start", datetime.utcnow())).total_seconds()
        logger.info(f"📊 Final stats - Processed: {self.stats['tasks_processed']}, Failed: {self.stats['tasks_failed']}, Uptime: {uptime:.1f}s")
        
        logger.info("✅ Worker cleanup complete")


async def main():
    """Main entry point for the worker service."""
    # Setup logging
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