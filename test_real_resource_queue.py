#!/usr/bin/env python3
"""
Real-world test of resource-aware queue system.
Tests the actual worker processing with resource constraints.
"""

import asyncio
import logging
import time
import httpx
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_real_worker_processing():
    """Test actual worker processing with resource-aware scheduling."""
    
    logger.info("🚀 Testing real worker with resource-aware queue...")
    
    # Connect to the running services
    base_url = "http://localhost:8003"  # Router service
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Check router health
            response = await client.get(f"{base_url}/health")
            if response.status_code != 200:
                logger.error("Router service not available")
                return
            
            logger.info("✅ Router service is healthy")
            
            # Submit CPU tasks (should run in parallel, up to 3)
            logger.info("\n🔄 Submitting CPU tasks...")
            cpu_tasks = []
            
            for i in range(5):  # More than our CPU limit
                task_data = {
                    "plugin_name": "ConceptNetKeywordPlugin",
                    "plugin_type": "field_enrichment", 
                    "data": {
                        "concept": f"test_concept_{i}",
                        "media_context": "movie",
                        "max_concepts": 5
                    }
                }
                
                response = await client.post(f"{base_url}/plugins/execute", json=task_data)
                if response.status_code == 200:
                    result = response.json()
                    cpu_tasks.append(result.get("task_id", f"task_{i}"))
                    logger.info(f"✅ Submitted CPU task {i}")
                else:
                    logger.error(f"❌ Failed to submit CPU task {i}: {response.text}")
            
            # Submit GPU task (should run exclusively)
            logger.info("\n🔄 Submitting GPU task...")
            gpu_task_data = {
                "plugin_name": "LLMKeywordPlugin",
                "plugin_type": "field_enrichment",
                "data": {
                    "concept": "action movie",
                    "media_context": "movie", 
                    "max_concepts": 10
                }
            }
            
            response = await client.post(f"{base_url}/plugins/execute", json=gpu_task_data)
            if response.status_code == 200:
                result = response.json()
                gpu_task_id = result.get("task_id", "gpu_task")
                logger.info(f"✅ Submitted GPU task: {gpu_task_id}")
            else:
                logger.error(f"❌ Failed to submit GPU task: {response.text}")
                gpu_task_id = None
            
            # Monitor processing
            logger.info("\n📊 Monitoring task processing...")
            start_time = time.time()
            timeout = 60  # 1 minute timeout
            
            while time.time() - start_time < timeout:
                # Check worker metrics
                try:
                    metrics_response = await client.get("http://localhost:8004/metrics")
                    if metrics_response.status_code == 200:
                        metrics_text = metrics_response.text
                        
                        # Extract task counts from metrics
                        processed_count = 0
                        failed_count = 0
                        
                        for line in metrics_text.split('\n'):
                            if 'worker_tasks_processed_total' in line and not line.startswith('#'):
                                processed_count = int(float(line.split()[-1]))
                            elif 'worker_tasks_failed_total' in line and not line.startswith('#'):
                                failed_count = int(float(line.split()[-1]))
                        
                        logger.info(f"📈 Worker metrics: {processed_count} processed, {failed_count} failed")
                        
                        # Stop when all tasks are processed
                        total_submitted = len(cpu_tasks) + (1 if gpu_task_id else 0)
                        if processed_count + failed_count >= total_submitted:
                            logger.info("✅ All tasks processed!")
                            break
                            
                except Exception as e:
                    logger.warning(f"Could not fetch metrics: {e}")
                
                await asyncio.sleep(2)
            
            # Final status
            logger.info("\n📊 Final status:")
            logger.info(f"CPU tasks submitted: {len(cpu_tasks)}")
            logger.info(f"GPU task submitted: {'Yes' if gpu_task_id else 'No'}")
            
        except Exception as e:
            logger.error(f"Test failed: {e}")


async def test_queue_stats():
    """Check queue statistics via Redis."""
    
    logger.info("\n📊 Checking queue statistics...")
    
    try:
        # Import here to avoid issues if Redis not available
        import redis.asyncio as redis
        from src.shared.config import get_settings
        
        settings = get_settings()
        redis_client = redis.Redis(
            host="localhost",
            port=6379,
            decode_responses=True
        )
        
        # Check queue sizes
        cpu_queue = "chat:queue:cpu"
        gpu_queue = "chat:queue:gpu"
        dead_letter = "chat:dead_letter"
        
        cpu_count = await redis_client.zcard(cpu_queue)
        gpu_count = await redis_client.zcard(gpu_queue)
        dead_count = await redis_client.llen(dead_letter)
        
        logger.info(f"📈 Queue status:")
        logger.info(f"  CPU queue: {cpu_count} tasks")
        logger.info(f"  GPU queue: {gpu_count} tasks")
        logger.info(f"  Dead letter: {dead_count} tasks")
        
        # Get some sample tasks
        if cpu_count > 0:
            sample_cpu = await redis_client.zrange(cpu_queue, 0, 2, withscores=False)
            logger.info(f"  Sample CPU tasks: {len(sample_cpu)} found")
            
        if gpu_count > 0:
            sample_gpu = await redis_client.zrange(gpu_queue, 0, 2, withscores=False)
            logger.info(f"  Sample GPU tasks: {len(sample_gpu)} found")
        
        await redis_client.close()
        
    except Exception as e:
        logger.warning(f"Could not check Redis queues: {e}")


async def test_service_health():
    """Check that all services are healthy."""
    
    logger.info("\n🏥 Checking service health...")
    
    services = [
        ("Router", "http://localhost:8003/health"),
        ("ConceptNet", "http://localhost:8001/health"),
        ("Gensim", "http://localhost:8006/health"),
        ("SpaCy", "http://localhost:8007/health"),
        ("HeidelTime", "http://localhost:8008/health"),
        ("LLM", "http://localhost:8002/health"),
        ("Worker Metrics", "http://localhost:8004/metrics")
    ]
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for name, url in services:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    logger.info(f"✅ {name} service healthy")
                else:
                    logger.warning(f"⚠️ {name} service returned {response.status_code}")
            except Exception as e:
                logger.error(f"❌ {name} service unreachable: {e}")


async def main():
    """Run all tests."""
    logger.info("🚀 Starting real-world resource queue tests...")
    
    await test_service_health()
    await test_queue_stats()
    await test_real_worker_processing()
    
    # Check final queue stats
    await test_queue_stats()
    
    logger.info("✅ Real-world tests completed!")


if __name__ == "__main__":
    asyncio.run(main())