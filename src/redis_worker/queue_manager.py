"""
Redis queue manager for handling distributed tasks.
"""

import redis
import json
import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from src.shared.config import get_settings


class RedisQueueManager:
    """Manages Redis-based task queues for the RAG system."""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = self._create_redis_client()
    
    def _create_redis_client(self) -> redis.Redis:
        """Create Redis client connection."""
        return redis.Redis(
            host=self.settings.redis_host,
            port=self.settings.redis_port,
            password=self.settings.REDIS_PASSWORD,
            db=self.settings.REDIS_DB,
            decode_responses=True,
            socket_timeout=10,
            retry_on_timeout=True
        )
    
    def health_check(self) -> bool:
        """Check if Redis connection is healthy."""
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False
    
    def enqueue_task(self, task_type: str, data: Dict[str, Any], priority: int = 0) -> str:
        """
        Enqueue a task for processing.
        
        Args:
            task_type: Type of task (e.g., 'plugin_execution', 'media_analysis')
            data: Task data payload
            priority: Task priority (higher = more urgent)
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task_payload = {
            "task_id": task_id,
            "task_type": task_type,
            "data": data,
            "priority": priority,
            "created_at": datetime.utcnow().isoformat(),
            "attempts": 0,
            "max_attempts": self.settings.WORKER_MAX_RETRIES
        }
        
        # Use priority score for sorted set
        score = priority + (int(datetime.utcnow().timestamp()) / 1000000)
        
        self.redis_client.zadd(
            self.settings.REDIS_QUEUE,
            {json.dumps(task_payload): score}
        )
        
        return task_id
    
    def dequeue_task(self, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """
        Dequeue the highest priority task.
        
        Args:
            timeout: Blocking timeout in seconds
            
        Returns:
            Task data or None if timeout
        """
        try:
            # Get highest priority task (lowest score)
            result = self.redis_client.bzpopmin(self.settings.REDIS_QUEUE, timeout=timeout)
            
            if result:
                queue_name, task_json, score = result
                return json.loads(task_json)
                
            return None
            
        except Exception as e:
            print(f"Error dequeuing task: {e}")
            return None
    
    def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
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
        self.redis_client.setex(
            f"result:{task_id}",
            timedelta(hours=24),
            json.dumps(result_data)
        )
    
    def fail_task(self, task_data: Dict[str, Any], error: str) -> None:
        """
        Handle task failure with retry logic.
        
        Args:
            task_data: Original task data
            error: Error message
        """
        task_data["attempts"] += 1
        task_data["last_error"] = error
        task_data["failed_at"] = datetime.utcnow().isoformat()
        
        # Retry if under max attempts
        if task_data["attempts"] < task_data["max_attempts"]:
            # Exponential backoff
            delay = (2 ** task_data["attempts"]) * self.settings.WORKER_RETRY_DELAY
            retry_time = datetime.utcnow() + timedelta(seconds=delay)
            
            # Re-enqueue with delay
            self.redis_client.zadd(
                self.settings.REDIS_QUEUE,
                {json.dumps(task_data): retry_time.timestamp()}
            )
        else:
            # Move to dead letter queue
            self.redis_client.lpush(
                self.settings.REDIS_DEAD_LETTER_QUEUE,
                json.dumps(task_data)
            )
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "pending_tasks": self.redis_client.zcard(self.settings.REDIS_QUEUE),
            "failed_tasks": self.redis_client.llen(self.settings.REDIS_DEAD_LETTER_QUEUE),
            "redis_info": {
                "memory_usage": self.redis_client.info("memory"),
                "connected_clients": self.redis_client.info("clients")["connected_clients"]
            }
        }