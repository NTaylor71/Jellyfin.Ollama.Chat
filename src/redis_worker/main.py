"""
Redis worker main entry point.
Processes tasks from Redis queue system.
"""

import asyncio
import signal
import sys
from typing import Dict, Any

from src.shared.config import get_settings
from src.redis_worker.queue_manager import RedisQueueManager


class WorkerService:
    """Main worker service for processing queue tasks."""
    
    def __init__(self):
        self.settings = get_settings()
        self.queue_manager = RedisQueueManager()
        self.running = False
        self.stats = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "uptime_start": None
        }
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        def signal_handler(signum, frame):
            print(f"\n🔄 Received signal {signum}, shutting down gracefully...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def process_task(self, task_data: Dict[str, Any]) -> None:
        """
        Process a single task from the queue.
        
        Args:
            task_data: Task data from queue
        """
        task_id = task_data.get("task_id")
        task_type = task_data.get("task_type")
        
        print(f"🔄 Processing task {task_id} of type {task_type}")
        
        try:
            # TODO: Implement plugin-based task processing
            # For now, just simulate processing
            result = {
                "task_id": task_id,
                "task_type": task_type,
                "processed_at": "simulation",
                "message": f"Task {task_type} processed successfully"
            }
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            # Mark as completed
            self.queue_manager.complete_task(task_id, result)
            self.stats["tasks_processed"] += 1
            
            print(f"✅ Completed task {task_id}")
            
        except Exception as e:
            print(f"❌ Task {task_id} failed: {e}")
            self.queue_manager.fail_task(task_data, str(e))
            self.stats["tasks_failed"] += 1
    
    async def worker_loop(self):
        """Main worker processing loop."""
        print(f"🚀 Worker started on {self.settings.ENV} environment")
        print(f"📊 Worker stats: {self.stats}")
        
        while self.running:
            try:
                # Check Redis health
                if not self.queue_manager.health_check():
                    print("⚠️ Redis connection unhealthy, retrying...")
                    await asyncio.sleep(5)
                    continue
                
                # Dequeue task with timeout
                task_data = self.queue_manager.dequeue_task(timeout=10)
                
                if task_data:
                    await self.process_task(task_data)
                else:
                    # No tasks available, brief pause
                    await asyncio.sleep(1)
                    
            except Exception as e:
                print(f"❌ Worker loop error: {e}")
                await asyncio.sleep(5)
    
    async def start(self):
        """Start the worker service."""
        self.running = True
        self.setup_signal_handlers()
        
        # Start worker loop
        await self.worker_loop()
        
        print("🔄 Worker service stopped")


async def main():
    """Main entry point for the worker service."""
    worker = WorkerService()
    await worker.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🔄 Worker interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Worker failed to start: {e}")
        sys.exit(1)