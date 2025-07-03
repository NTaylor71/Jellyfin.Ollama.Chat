"""
Redis queue manager for handling job processing.
"""

import json
import time
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import redis
from src.shared.config import get_settings


@dataclass
class Job:
    """Job data structure."""
    id: str
    type: str  # "chat", "ingest", etc.
    data: Dict[str, Any]
    status: str = "pending"  # pending, processing, completed, failed
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create Job from dictionary."""
        return cls(**data)


class RedisQueueManager:
    """Redis-based queue manager with retry logic and job persistence."""

    def __init__(self):
        self.settings = get_settings()
        self.redis_client = None
        self._connect()

    def _connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.Redis(
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                db=0,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.redis_client.ping()
            print(f"✅ Connected to Redis at {self.settings.REDIS_HOST}:{self.settings.REDIS_PORT}")
        except Exception as e:
            print(f"❌ Failed to connect to Redis: {e}")
            raise

    def enqueue(self, job_type: str, data: Dict[str, Any], priority: str = "normal") -> str:
        """
        Add a job to the queue.

        Args:
            job_type: Type of job ("chat", "ingest", etc.)
            data: Job data
            priority: "high", "normal", "low"

        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())

        job = Job(
            id=job_id,
            type=job_type,
            data=data
        )

        # Store job data
        self._store_job(job)

        # Add to appropriate queue based on priority
        queue_name = f"queue:{priority}"
        self.redis_client.lpush(queue_name, job_id)

        print(f"📝 Enqueued job {job_id[:8]}... to {queue_name}")
        return job_id

    def dequeue(self, timeout: int = 10) -> Optional[Job]:
        """
        Get the next job from the queue (blocking).

        Args:
            timeout: How long to wait for a job (seconds)

        Returns:
            Job or None if timeout
        """
        # Try high priority first, then normal, then low
        queues = ["queue:high", "queue:normal", "queue:low"]

        result = self.redis_client.brpop(queues, timeout=timeout)
        if not result:
            return None

        queue_name, job_id = result

        # Get job data
        job = self._get_job(job_id)
        if not job:
            print(f"⚠️ Job {job_id} not found in storage")
            return None

        # Mark as processing
        job.status = "processing"
        job.started_at = time.time()
        self._store_job(job)

        print(f"📥 Dequeued job {job_id[:8]}... from {queue_name}")
        return job

    def complete_job(self, job_id: str, result: Dict[str, Any]):
        """Mark job as completed with result."""
        job = self._get_job(job_id)
        if not job:
            return

        job.status = "completed"
        job.completed_at = time.time()
        job.result = result
        self._store_job(job)

        print(f"✅ Completed job {job_id[:8]}...")

    def fail_job(self, job_id: str, error: str, retry: bool = True):
        """Mark job as failed, optionally retry."""
        job = self._get_job(job_id)
        if not job:
            return

        job.error = error
        job.retry_count += 1

        if retry and job.retry_count <= job.max_retries:
            # Retry with exponential backoff
            delay = min(300, 2 ** job.retry_count)  # Max 5 minutes
            retry_time = time.time() + delay

            job.status = "pending"
            job.started_at = None
            self._store_job(job)

            # Add back to queue with delay (using sorted set for scheduling)
            self.redis_client.zadd("queue:retry", {job_id: retry_time})

            print(f"🔄 Retrying job {job_id[:8]}... in {delay}s (attempt {job.retry_count})")
        else:
            # Max retries reached or no retry requested
            job.status = "failed"
            job.completed_at = time.time()
            self._store_job(job)

            # Move to dead letter queue
            self.redis_client.lpush("queue:dead", job_id)

            print(f"💀 Job {job_id[:8]}... failed permanently")

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and result."""
        job = self._get_job(job_id)
        if not job:
            return None

        return job.to_dict()

    def process_retry_queue(self):
        """Process jobs scheduled for retry."""
        current_time = time.time()

        # Get jobs ready for retry
        ready_jobs = self.redis_client.zrangebyscore(
            "queue:retry",
            0,
            current_time,
            withscores=False
        )

        for job_id in ready_jobs:
            # Remove from retry queue
            self.redis_client.zrem("queue:retry", job_id)

            # Add back to normal queue
            self.redis_client.lpush("queue:normal", job_id)

            print(f"⏰ Moved retry job {job_id[:8]}... back to queue")

    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        return {
            "high_priority": self.redis_client.llen("queue:high"),
            "normal_priority": self.redis_client.llen("queue:normal"),
            "low_priority": self.redis_client.llen("queue:low"),
            "retry_scheduled": self.redis_client.zcard("queue:retry"),
            "dead_letter": self.redis_client.llen("queue:dead")
        }

    def get_recent_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent jobs for debugging."""
        # Get all job keys
        job_keys = self.redis_client.keys("job:*")
        jobs = []

        for key in job_keys[-limit:]:  # Get last N jobs
            job_data = self.redis_client.get(key)
            if job_data:
                try:
                    job_dict = json.loads(job_data)
                    jobs.append(job_dict)
                except json.JSONDecodeError:
                    continue

        # Sort by created_at descending
        jobs.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return jobs[:limit]

    def clear_completed_jobs(self, older_than_hours: int = 24):
        """Clean up old completed jobs."""
        cutoff_time = time.time() - (older_than_hours * 3600)
        job_keys = self.redis_client.keys("job:*")
        deleted = 0

        for key in job_keys:
            job_data = self.redis_client.get(key)
            if job_data:
                try:
                    job_dict = json.loads(job_data)
                    if (job_dict.get("status") in ["completed", "failed"] and
                            job_dict.get("completed_at", float("inf")) < cutoff_time):
                        self.redis_client.delete(key)
                        deleted += 1
                except json.JSONDecodeError:
                    continue

        if deleted > 0:
            print(f"🧹 Cleaned up {deleted} old jobs")

    def health_check(self) -> Dict[str, Any]:
        """Check Redis health and queue status."""
        try:
            # Test Redis connection
            self.redis_client.ping()

            # Get queue stats
            stats = self.get_queue_stats()

            return {
                "redis_connected": True,
                "queue_stats": stats,
                "total_pending": sum(stats.values()) - stats["dead_letter"]
            }
        except Exception as e:
            return {
                "redis_connected": False,
                "error": str(e),
                "queue_stats": {},
                "total_pending": 0
            }

    def _store_job(self, job: Job):
        """Store job data in Redis."""
        key = f"job:{job.id}"
        data = json.dumps(job.to_dict())

        # Store with TTL (7 days)
        self.redis_client.setex(key, 7 * 24 * 3600, data)

    def _get_job(self, job_id: str) -> Optional[Job]:
        """Get job data from Redis."""
        key = f"job:{job_id}"
        data = self.redis_client.get(key)

        if not data:
            return None

        try:
            job_dict = json.loads(data)
            return Job.from_dict(job_dict)
        except json.JSONDecodeError:
            return None

    def close(self):
        """Close Redis connection."""
        if self.redis_client:
            self.redis_client.close()
