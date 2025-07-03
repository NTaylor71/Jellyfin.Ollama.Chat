"""
Queue service for API - interfaces with Redis queue system.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from src.redis_worker.queue_manager import RedisQueueManager


class QueueService:
    """Queue service for handling job submission and status checking."""

    def __init__(self):
        self.queue_manager = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize the queue service."""
        try:
            self.queue_manager = RedisQueueManager()
            self.logger.info("✅ Queue service initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize queue service: {e}")
            # For development, we can continue without Redis
            # In production, you might want to fail here
            self.queue_manager = None

    async def submit_chat_job(
            self,
            query: str,
            context: Optional[Dict[str, Any]] = None,
            max_results: int = 5,
            priority: str = "normal"
    ) -> str:
        """
        Submit a chat job to the queue.

        Args:
            query: User query
            context: Additional context
            max_results: Maximum number of results
            priority: Job priority ("high", "normal", "low")

        Returns:
            Job ID
        """
        if not self.queue_manager:
            raise RuntimeError("Queue service not available")

        job_data = {
            "query": query,
            "context": context or {},
            "max_results": max_results
        }

        job_id = self.queue_manager.enqueue(
            job_type="chat",
            data=job_data,
            priority=priority
        )

        self.logger.info(f"📝 Submitted chat job {job_id[:8]}... query: {query[:50]}...")
        return job_id

    async def submit_ingest_job(
            self,
            data: Dict[str, Any],
            source: str = "unknown",
            priority: str = "low"
    ) -> str:
        """
        Submit an ingestion job to the queue.

        Args:
            data: Data to ingest
            source: Data source identifier
            priority: Job priority

        Returns:
            Job ID
        """
        if not self.queue_manager:
            raise RuntimeError("Queue service not available")

        job_data = {
            "data": data,
            "source": source
        }

        job_id = self.queue_manager.enqueue(
            job_type="ingest",
            data=job_data,
            priority=priority
        )

        self.logger.info(f"📚 Submitted ingest job {job_id[:8]}... source: {source}")
        return job_id

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and result."""
        if not self.queue_manager:
            return None

        return self.queue_manager.get_job_status(job_id)

    async def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job result if completed."""
        status = await self.get_job_status(job_id)

        if not status:
            return None

        if status["status"] == "completed":
            return status.get("result")

        return None

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job (if still pending).

        Note: This is a simplified implementation.
        A full implementation would need to handle jobs in different states.
        """
        if not self.queue_manager:
            return False

        # For now, just mark as failed
        # TODO: Implement proper cancellation logic
        job = self.queue_manager._get_job(job_id)
        if job and job.status == "pending":
            self.queue_manager.fail_job(job_id, "Cancelled by user", retry=False)
            return True

        return False

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        if not self.queue_manager:
            return {"available": False}

        health = self.queue_manager.health_check()
        return {
            "available": True,
            "redis_connected": health["redis_connected"],
            "queue_stats": health["queue_stats"],
            "total_pending": health["total_pending"]
        }

    async def get_recent_jobs(self, limit: int = 10) -> list:
        """Get recent jobs for debugging."""
        if not self.queue_manager:
            return []

        return self.queue_manager.get_recent_jobs(limit)

    async def health_check(self) -> Dict[str, Any]:
        """Check queue service health."""
        if not self.queue_manager:
            return {
                "status": "unavailable",
                "redis_connected": False,
                "error": "Queue manager not initialized"
            }

        health = self.queue_manager.health_check()

        return {
            "status": "healthy" if health["redis_connected"] else "degraded",
            "redis_connected": health["redis_connected"],
            "queue_stats": health.get("queue_stats", {}),
            "total_pending": health.get("total_pending", 0)
        }

    async def shutdown(self):
        """Shutdown the queue service."""
        if self.queue_manager:
            self.queue_manager.close()
            self.logger.info("✅ Queue service shutdown")


# Global instance - will be initialized by the API on startup
queue_service: Optional[QueueService] = None


def get_queue_service() -> QueueService:
    """Get the global queue service instance."""
    global queue_service
    if queue_service is None:
        raise RuntimeError("Queue service not initialized")
    return queue_service


async def initialize_queue_service() -> QueueService:
    """Initialize the global queue service."""
    global queue_service
    queue_service = QueueService()
    await queue_service.initialize()
    return queue_service
