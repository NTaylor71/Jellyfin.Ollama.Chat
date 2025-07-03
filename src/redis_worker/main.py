"""
Redis queue worker service - processes jobs from Redis queues.
"""

import asyncio
import signal
import time
import logging
from typing import Dict, Any

from src.shared.config import get_settings
from src.redis_worker.queue_manager import RedisQueueManager, Job


class WorkerService:
    """Redis queue worker service."""

    def __init__(self):
        self.settings = get_settings()
        self.queue_manager = RedisQueueManager()
        self.running = False
        self.logger = logging.getLogger(__name__)

    async def start(self):
        """Start the worker service."""
        self.running = True
        self.logger.info("🚀 Starting Redis worker service")
        self.logger.info(f"Environment: {self.settings.ENV}")

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start background tasks
        tasks = [
            asyncio.create_task(self._job_processor()),
            asyncio.create_task(self._retry_processor()),
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._cleanup_task())
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Worker tasks cancelled")
        finally:
            self.queue_manager.close()

    async def _job_processor(self):
        """Main job processing loop."""
        self.logger.info("📥 Starting job processor")

        while self.running:
            try:
                # Get next job from queue (blocking with timeout)
                job = self.queue_manager.dequeue(timeout=5)

                if job:
                    await self._process_job(job)

            except Exception as e:
                self.logger.error(f"Error in job processor: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    async def _process_job(self, job: Job):
        """Process a single job."""
        self.logger.info(f"🔨 Processing job {job.id[:8]}... type: {job.type}")

        try:
            # Route job to appropriate processor
            if job.type == "chat":
                result = await self._process_chat_job(job)
            elif job.type == "ingest":
                result = await self._process_ingest_job(job)
            else:
                raise ValueError(f"Unknown job type: {job.type}")

            # Mark job as completed
            self.queue_manager.complete_job(job.id, result)

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Job {job.id[:8]}... failed: {error_msg}")

            # Mark job as failed (will retry if appropriate)
            self.queue_manager.fail_job(job.id, error_msg)

    async def _process_chat_job(self, job: Job) -> Dict[str, Any]:
        """Process a chat query job."""
        query = job.data.get("query", "")
        max_results = job.data.get("max_results", 5)

        self.logger.info(f"💬 Processing chat query: {query[:50]}...")

        # Simulate processing time
        await asyncio.sleep(0.5)

        # For now, return mock results
        # TODO: Replace with actual FAISS search + LLM processing
        mock_results = [
            {
                "title": f"Movie Result {i + 1}",
                "summary": f"This movie is relevant to your query: {query}",
                "relevance_score": 0.95 - (i * 0.1),
                "metadata": {
                    "genre": "Drama" if i % 2 == 0 else "Action",
                    "year": 2023 - i,
                    "duration": f"{120 - i * 5} min"
                }
            }
            for i in range(min(max_results, 3))
        ]

        return {
            "query": query,
            "results": mock_results,
            "response": f"Found {len(mock_results)} relevant movies for your query: '{query}'",
            "processing_time": 0.5,
            "total_results": len(mock_results),
            "processing_method": "mock_worker",
            "timestamp": time.time()
        }

    async def _process_ingest_job(self, job: Job) -> Dict[str, Any]:
        """Process a data ingestion job."""
        data = job.data.get("data", {})
        source = job.data.get("source", "unknown")

        self.logger.info(f"📚 Processing ingestion from {source}")

        # Simulate ingestion processing
        await asyncio.sleep(1.0)

        # TODO: Replace with actual FAISS indexing
        return {
            "source": source,
            "processed_items": 1,
            "status": "completed",
            "processing_time": 1.0,
            "timestamp": time.time()
        }

    async def _retry_processor(self):
        """Process retry queue periodically."""
        while self.running:
            try:
                self.queue_manager.process_retry_queue()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in retry processor: {e}")
                await asyncio.sleep(60)

    async def _health_monitor(self):
        """Monitor worker health and log stats."""
        while self.running:
            try:
                health = self.queue_manager.health_check()
                stats = health.get("queue_stats", {})

                if stats:
                    self.logger.info(
                        f"📊 Queue stats: "
                        f"High:{stats.get('high_priority', 0)} "
                        f"Normal:{stats.get('normal_priority', 0)} "
                        f"Low:{stats.get('low_priority', 0)} "
                        f"Retry:{stats.get('retry_scheduled', 0)} "
                        f"Dead:{stats.get('dead_letter', 0)}"
                    )

                await asyncio.sleep(60)  # Log every minute
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)

    async def _cleanup_task(self):
        """Periodic cleanup of old jobs."""
        while self.running:
            try:
                # Clean up completed jobs older than 24 hours
                self.queue_manager.clear_completed_jobs(older_than_hours=24)

                # Sleep for 1 hour before next cleanup
                await asyncio.sleep(3600)
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(3600)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False


def main():
    """Main entry point for the worker service."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)
    logger.info("🎯 Starting Production RAG Worker")

    # Create and start worker service
    worker = WorkerService()

    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        logger.info("🛑 Worker interrupted")
    except Exception as e:
        logger.error(f"❌ Worker failed: {e}")
    finally:
        logger.info("✅ Worker shutdown complete")


if __name__ == "__main__":
    main()
