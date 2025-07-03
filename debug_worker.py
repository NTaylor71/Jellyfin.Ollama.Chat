#!/usr/bin/env python3
"""
Debug Redis Queue Worker - Clean version
"""
import sys
import time
import json
from pathlib import Path
import subprocess

import redis
from rich.console import Console
from rich.panel import Panel

# Add src to path and get config
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.shared.config import get_settings

    settings = get_settings()
    REDIS_HOST = settings.REDIS_HOST
    REDIS_PORT = settings.REDIS_PORT
    REDIS_QUEUE = settings.REDIS_QUEUE
except ImportError:
    # Fallback config
    import os
    from dotenv import load_dotenv

    load_dotenv()
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_QUEUE = os.getenv("REDIS_QUEUE", "chat:queue")

console = Console()


def main():
    """Main diagnostic function"""
    console.print("🔍 Redis Queue Worker Diagnostics", style="bold blue")
    console.print("=" * 50)

    console.print(f"🔧 Configuration:", style="blue")
    console.print(f"   Redis Host: {REDIS_HOST}", style="cyan")
    console.print(f"   Redis Port: {REDIS_PORT}", style="cyan")
    console.print(f"   Queue Name: {REDIS_QUEUE}", style="cyan")

    # 1. Check worker container
    console.print("\n1️⃣ Checking worker container...")
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            console.print("📦 Running containers:", style="blue")
            console.print(output, style="cyan")

            if "rag-worker" in output:
                console.print("✅ Worker container 'rag-worker' is running", style="green")
                worker_running = True
            else:
                console.print("❌ Worker container 'rag-worker' not found", style="red")
                worker_running = False
        else:
            console.print("❌ Failed to check containers", style="red")
            worker_running = False
    except Exception as e:
        console.print(f"❌ Error checking containers: {e}", style="red")
        worker_running = False

    # 2. Check worker logs
    if worker_running:
        console.print("\n2️⃣ Checking worker logs...")
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", "10", "rag-worker"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                logs = result.stdout.strip() if result.stdout.strip() else "No recent logs"
                console.print(Panel(logs, title="rag-worker Logs"))
            else:
                console.print("❌ Failed to get worker logs", style="red")
        except Exception as e:
            console.print(f"❌ Error getting logs: {e}", style="red")

    # 3. Test Redis connection
    console.print("\n3️⃣ Testing Redis connection...")
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        redis_client.ping()
        console.print(f"✅ Redis connection successful: {REDIS_HOST}:{REDIS_PORT}", style="green")

        info = redis_client.info()
        console.print(f"   Connected clients: {info.get('connected_clients')}", style="cyan")
    except Exception as e:
        console.print(f"❌ Redis connection failed: {e}", style="red")
        redis_client = None

    # 4. Inspect queue
    if redis_client:
        console.print("\n4️⃣ Inspecting Redis queue...")
        try:
            queue_len = redis_client.llen(REDIS_QUEUE)
            console.print(f"📊 Queue '{REDIS_QUEUE}' length: {queue_len}", style="blue")

            result_keys = redis_client.keys("chat:result:*")
            console.print(f"🔑 Result keys found: {len(result_keys)}", style="blue")

            if queue_len > 0:
                items = redis_client.lrange(REDIS_QUEUE, 0, 2)
                console.print("📋 Queue items:")
                for i, item in enumerate(items):
                    console.print(f"   [{i}] {item}", style="cyan")
        except Exception as e:
            console.print(f"❌ Queue inspection failed: {e}", style="red")

    # 5. Summary
    console.print("\n5️⃣ Summary:", style="bold blue")
    if worker_running and redis_client:
        console.print("✅ System ready! Worker and Redis are both operational", style="green")
        console.print("💡 Next step: python test_redis_queue_worker.py", style="cyan")
    elif not worker_running:
        console.print("❌ Worker not running. Start with:", style="red")
        console.print("   docker compose -f docker-compose.dev.yml up -d worker", style="white")
    elif not redis_client:
        console.print("❌ Redis connection failed", style="red")

    console.print("\n✅ Diagnostics complete!", style="bold green")


if __name__ == "__main__":
    main()
