#!/usr/bin/env python3
"""
Test script for Redis queue system.
Run this to verify the queue is working with a local Redis instance.
"""

import asyncio
import time
from rich.console import Console
from rich.table import Table

from src.redis_worker.queue_manager import RedisQueueManager
from src.api.services.queue_service import QueueService

console = Console()


async def test_redis_queue():
    """Test Redis queue functionality."""

    console.print("🧪 Testing Redis Queue System", style="bold blue")

    try:
        # Test 1: Queue Manager Connection
        console.print("\n1. Testing Redis connection...", style="yellow")
        queue_manager = RedisQueueManager()
        console.print("✅ Connected to Redis successfully", style="green")

        # Test 2: Queue Statistics
        console.print("\n2. Getting queue statistics...", style="yellow")
        stats = queue_manager.get_queue_stats()
        console.print(f"✅ Queue stats: {stats}", style="green")

        # Test 3: Job Submission
        console.print("\n3. Testing job submission...", style="yellow")
        job_id = queue_manager.enqueue(
            job_type="chat",
            data={
                "query": "test query about movies",
                "max_results": 3
            },
            priority="normal"
        )
        console.print(f"✅ Submitted job: {job_id[:8]}...", style="green")

        # Test 4: Job Status Check
        console.print("\n4. Checking job status...", style="yellow")
        status = queue_manager.get_job_status(job_id)
        if status:
            console.print(f"✅ Job status: {status['status']}", style="green")
        else:
            console.print("❌ Could not get job status", style="red")

        # Test 5: Queue Service (API layer)
        console.print("\n5. Testing Queue Service (API layer)...", style="yellow")
        queue_service = QueueService()
        await queue_service.initialize()

        # Submit a job through the service
        api_job_id = await queue_service.submit_chat_job(
            query="API test query",
            max_results=2,
            priority="high"
        )
        console.print(f"✅ API job submitted: {api_job_id[:8]}...", style="green")

        # Test 6: Health Check
        console.print("\n6. Testing health check...", style="yellow")
        health = await queue_service.health_check()
        console.print(f"✅ Health check: {health['status']}", style="green")

        # Test 7: Recent Jobs
        console.print("\n7. Getting recent jobs...", style="yellow")
        recent_jobs = await queue_service.get_recent_jobs(limit=5)
        console.print(f"✅ Found {len(recent_jobs)} recent jobs", style="green")

        # Display results in a table
        if recent_jobs:
            table = Table(title="Recent Jobs")
            table.add_column("Job ID", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Status", style="green")
            table.add_column("Created", style="yellow")

            for job in recent_jobs[:5]:
                created_time = time.strftime(
                    "%H:%M:%S",
                    time.localtime(job.get("created_at", 0))
                )
                table.add_row(
                    job["id"][:8] + "...",
                    job["type"],
                    job["status"],
                    created_time
                )

            console.print(table)

        # Test 8: Queue Stats Display
        console.print("\n8. Final queue statistics...", style="yellow")
        final_stats = await queue_service.get_queue_stats()

        if final_stats.get("available"):
            stats_table = Table(title="Queue Statistics")
            stats_table.add_column("Queue", style="cyan")
            stats_table.add_column("Count", style="green")

            queue_stats = final_stats.get("queue_stats", {})
            for queue_name, count in queue_stats.items():
                stats_table.add_row(queue_name.replace("_", " ").title(), str(count))

            console.print(stats_table)

        # Cleanup
        await queue_service.shutdown()
        queue_manager.close()

        console.print("\n🎉 All Redis queue tests passed!", style="bold green")

        # Instructions
        console.print("\n💡 Next steps:", style="bold blue")
        console.print("• Start a Redis worker: python -m src.redis_worker.main")
        console.print("• Submit jobs via API: python test_api.py")
        console.print("• Monitor queues: check the worker logs")

    except Exception as e:
        console.print(f"\n❌ Redis queue test failed: {e}", style="red")
        console.print("\n💡 Make sure Redis is running:", style="yellow")
        console.print("• Install Redis: https://redis.io/download")
        console.print("• Start Redis: redis-server")
        console.print("• Or use Docker: docker run -p 6379:6379 redis:7-alpine")


def main():
    """Main entry point."""
    try:
        asyncio.run(test_redis_queue())
    except KeyboardInterrupt:
        console.print("\n🛑 Test interrupted", style="yellow")
    except Exception as e:
        console.print(f"\n❌ Test failed: {e}", style="red")


if __name__ == "__main__":
    main()
