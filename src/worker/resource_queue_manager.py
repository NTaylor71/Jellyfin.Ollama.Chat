"""
Resource-Aware Queue Manager - Manages Redis queues with CPU/GPU resource awareness.

Extends the basic queue manager to respect resource constraints and prevent
resource contention between tasks.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import redis.asyncio as redis

from src.shared.config import get_settings
from src.worker.resource_manager import ResourcePool, ResourceRequirement, get_plugin_resource_requirements

logger = logging.getLogger(__name__)


class ResourceAwareQueueManager:
    """
    Queue manager that respects resource constraints when scheduling tasks.
    
    Features:
    - Separate queues for CPU and GPU tasks
    - Resource-aware task dequeuing
    - Priority scheduling within resource constraints
    - Dead letter queue for failed tasks
    - Task retry with exponential backoff
    """
    
    def __init__(self, resource_pool: ResourcePool):
        self.settings = get_settings()
        self.resource_pool = resource_pool
        self.redis_client = self._create_redis_client()
        
        # Queue names
        self.cpu_queue = f"{self.settings.REDIS_QUEUE}:cpu"
        self.gpu_queue = f"{self.settings.REDIS_QUEUE}:gpu"
        self.dead_letter_queue = self.settings.REDIS_DEAD_LETTER_QUEUE
        
        # Metrics
        self.tasks_enqueued = 0
        self.tasks_dequeued = 0
        self.tasks_failed = 0
    
    def _create_redis_client(self) -> redis.Redis:
        """Create async Redis client connection."""
        return redis.Redis(
            host=self.settings.redis_host,
            port=self.settings.redis_port,
            password=self.settings.REDIS_PASSWORD,
            db=self.settings.REDIS_DB,
            decode_responses=True,
            socket_timeout=10,
            retry_on_timeout=True
        )
    
    async def health_check(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False
    
    async def enqueue_task(
        self, 
        task_type: str, 
        data: Dict[str, Any], 
        plugin_name: Optional[str] = None,
        resource_req: Optional[ResourceRequirement] = None,
        priority: int = 0
    ) -> str:
        """
        Enqueue a task with resource requirements.
        
        Args:
            task_type: Type of task (e.g., 'plugin_execution')
            data: Task data payload
            plugin_name: Name of plugin to execute (for resource calculation)
            resource_req: Explicit resource requirements (overrides plugin defaults)
            priority: Task priority (higher = more urgent)
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        # Determine resource requirements
        if resource_req is None:
            if plugin_name:
                resource_req = get_plugin_resource_requirements(plugin_name)
            else:
                # Default requirements for unknown tasks
                resource_req = ResourceRequirement()
        
        task_payload = {
            "task_id": task_id,
            "task_type": task_type,
            "data": data,
            "plugin_name": plugin_name,
            "resource_requirements": {
                "cpu_cores": resource_req.cpu_cores,
                "gpu_count": resource_req.gpu_count,
                "memory_mb": resource_req.memory_mb,
                "exclusive_gpu": resource_req.exclusive_gpu,
                "estimated_runtime_seconds": resource_req.estimated_runtime_seconds
            },
            "priority": priority,
            "created_at": datetime.utcnow().isoformat(),
            "attempts": 0,
            "max_attempts": self.settings.WORKER_MAX_RETRIES
        }
        
        # Route to appropriate queue based on resource requirements
        if resource_req.gpu_count > 0:
            queue_name = self.gpu_queue
        else:
            queue_name = self.cpu_queue
        
        # Calculate score for sorted set (priority + timestamp for FIFO within priority)
        score = priority + (int(datetime.utcnow().timestamp()) / 1000000)
        
        await self.redis_client.zadd(
            queue_name,
            {json.dumps(task_payload): score}
        )
        
        self.tasks_enqueued += 1
        
        logger.info(f"Enqueued task {task_id} to {queue_name} (priority={priority}, CPU={resource_req.cpu_cores}, GPU={resource_req.gpu_count})")
        
        return task_id
    
    async def dequeue_task(self, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """
        Dequeue the highest priority task that fits available resources.
        
        Args:
            timeout: Maximum time to wait for a runnable task
            
        Returns:
            Task data or None if timeout
        """
        end_time = time.time() + timeout
        last_log_time = 0
        
        while time.time() < end_time:
            # Try to get a runnable task
            task = await self._find_runnable_task()
            
            if task:
                self.tasks_dequeued += 1
                logger.info(f"Dequeued task {task['task_id']} from queue")
                return task
            
            # Log resource status occasionally
            current_time = time.time()
            if current_time - last_log_time > 10:  # Every 10 seconds
                await self._log_queue_status()
                last_log_time = current_time
            
            # No runnable tasks, wait briefly
            await asyncio.sleep(0.5)
        
        logger.debug(f"No runnable tasks found within {timeout}s timeout")
        return None
    
    async def _find_runnable_task(self) -> Optional[Dict[str, Any]]:
        """
        Find the highest priority task that can run with current resources.
        
        Strategy:
        1. Check GPU queue first (usually higher priority)
        2. Then check CPU queue
        3. For each queue, examine tasks in priority order
        4. Return first task that fits available resources
        """
        # Try GPU queue first
        task = await self._try_dequeue_from_queue(self.gpu_queue)
        if task:
            return task
        
        # Try CPU queue
        task = await self._try_dequeue_from_queue(self.cpu_queue)
        if task:
            return task
        
        return None
    
    async def _try_dequeue_from_queue(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """
        Try to get a runnable task from a specific queue.
        
        Args:
            queue_name: Name of the queue to check
            
        Returns:
            Task data if found and runnable, None otherwise
        """
        # Get top tasks (highest priority) without removing them
        # We check multiple tasks because the highest priority might not be runnable
        tasks = await self.redis_client.zrevrange(queue_name, 0, 9, withscores=False)
        
        for task_json in tasks:
            try:
                task = json.loads(task_json)
                req = ResourceRequirement(**task["resource_requirements"])
                
                # Check if this task can run with current resources
                if self.resource_pool.can_schedule(req):
                    # Remove from queue atomically
                    removed = await self.redis_client.zrem(queue_name, task_json)
                    if removed > 0:
                        logger.debug(f"Found runnable task {task['task_id']} in {queue_name}")
                        return task
                    else:
                        # Task was already taken by another worker
                        logger.debug(f"Task {task['task_id']} was taken by another worker")
                        continue
                else:
                    logger.debug(f"Task {task['task_id']} cannot run: insufficient resources")
                    
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Invalid task data in queue {queue_name}: {e}")
                # Remove malformed task
                await self.redis_client.zrem(queue_name, task_json)
        
        return None
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        """
        Mark task as completed and store result.
        
        Args:
            task_id: Task identifier
            result: Task result data
        """
        result_data = {
            "task_id": task_id,
            "result": result,
            "completed_at": datetime.utcnow().isoformat(),
            "status": "completed"
        }
        
        # Store result with TTL
        await self.redis_client.setex(
            f"result:{task_id}",
            timedelta(hours=24),
            json.dumps(result_data)
        )
        
        logger.debug(f"Stored completion result for task {task_id}")
    
    async def fail_task(self, task_data: Dict[str, Any], error: str) -> None:
        """
        Handle task failure with retry logic.
        
        Args:
            task_data: Original task data
            error: Error message
        """
        task_data["attempts"] += 1
        task_data["last_error"] = error
        task_data["failed_at"] = datetime.utcnow().isoformat()
        
        self.tasks_failed += 1
        
        # Retry if under max attempts
        if task_data["attempts"] < task_data["max_attempts"]:
            # Exponential backoff
            delay = (2 ** task_data["attempts"]) * self.settings.WORKER_RETRY_DELAY
            retry_time = datetime.utcnow() + timedelta(seconds=delay)
            
            # Re-enqueue with delay (use retry time as score)
            resource_req = ResourceRequirement(**task_data["resource_requirements"])
            queue_name = self.gpu_queue if resource_req.gpu_count > 0 else self.cpu_queue
            
            await self.redis_client.zadd(
                queue_name,
                {json.dumps(task_data): retry_time.timestamp()}
            )
            
            logger.info(f"Retrying task {task_data['task_id']} in {delay}s (attempt {task_data['attempts']}/{task_data['max_attempts']})")
        else:
            # Move to dead letter queue
            await self.redis_client.lpush(
                self.dead_letter_queue,
                json.dumps(task_data)
            )
            
            logger.error(f"Task {task_data['task_id']} failed permanently after {task_data['attempts']} attempts")
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics."""
        cpu_pending = await self.redis_client.zcard(self.cpu_queue)
        gpu_pending = await self.redis_client.zcard(self.gpu_queue)
        failed_tasks = await self.redis_client.llen(self.dead_letter_queue)
        
        # Get resource utilization
        utilization = self.resource_pool.get_utilization()
        
        return {
            "queues": {
                "cpu_pending": cpu_pending,
                "gpu_pending": gpu_pending,
                "total_pending": cpu_pending + gpu_pending,
                "failed_tasks": failed_tasks
            },
            "metrics": {
                "tasks_enqueued": self.tasks_enqueued,
                "tasks_dequeued": self.tasks_dequeued,
                "tasks_failed": self.tasks_failed
            },
            "resources": {
                "pool_status": self.resource_pool.get_status_summary(),
                "utilization": utilization
            },
            "redis_info": await self._get_redis_info()
        }
    
    async def _get_redis_info(self) -> Dict[str, Any]:
        """Get Redis connection and memory info."""
        try:
            info = await self.redis_client.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "total_connections_received": info.get("total_connections_received", 0)
            }
        except Exception as e:
            logger.warning(f"Failed to get Redis info: {e}")
            return {"error": str(e)}
    
    async def _log_queue_status(self):
        """Log current queue and resource status."""
        stats = await self.get_queue_stats()
        
        logger.info(
            f"Queue status: CPU={stats['queues']['cpu_pending']}, "
            f"GPU={stats['queues']['gpu_pending']}, "
            f"Failed={stats['queues']['failed_tasks']}, "
            f"CPU usage={stats['resources']['utilization']['cpu_utilization']:.1f}%, "
            f"GPU usage={stats['resources']['utilization']['gpu_utilization']:.1f}%, "
            f"Active tasks={stats['resources']['utilization']['active_tasks']}"
        )
    
    async def purge_queues(self) -> Dict[str, int]:
        """
        Purge all tasks from queues (for testing/debugging).
        
        Returns:
            Dictionary with count of tasks removed from each queue
        """
        cpu_count = await self.redis_client.zcard(self.cpu_queue)
        gpu_count = await self.redis_client.zcard(self.gpu_queue)
        dead_count = await self.redis_client.llen(self.dead_letter_queue)
        
        await self.redis_client.delete(self.cpu_queue)
        await self.redis_client.delete(self.gpu_queue)
        await self.redis_client.delete(self.dead_letter_queue)
        
        logger.warning(f"Purged all queues: CPU={cpu_count}, GPU={gpu_count}, Dead={dead_count}")
        
        return {
            "cpu_tasks_removed": cpu_count,
            "gpu_tasks_removed": gpu_count,
            "dead_tasks_removed": dead_count
        }
    
    async def cleanup(self):
        """Clean up resources."""
        if self.redis_client:
            await self.redis_client.close()
            logger.debug("Closed Redis connection")