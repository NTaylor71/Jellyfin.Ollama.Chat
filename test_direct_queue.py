#!/usr/bin/env python3
"""
Direct test of resource-aware queue by submitting tasks directly to Redis.
"""

import asyncio
import logging
import time
from src.worker.resource_manager import ResourcePool, ResourceRequirement
from src.worker.resource_queue_manager import ResourceAwareQueueManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_direct_queue_submission():
    """Test direct submission to the resource-aware queue."""
    
    logger.info("🚀 Testing direct queue submission...")
    

    resource_pool = ResourcePool(
        total_cpu_cores=3,
        total_gpus=1,
        total_memory_mb=8192,
        worker_id="test_direct"
    )
    

    queue_manager = ResourceAwareQueueManager(resource_pool)
    
    try:

        healthy = await queue_manager.health_check()
        if not healthy:
            logger.error("❌ Redis not healthy")
            return
        
        logger.info("✅ Redis is healthy")
        

        purged = await queue_manager.purge_queues()
        logger.info(f"🧹 Purged queues: {purged}")
        

        logger.info("\n🔄 Submitting CPU tasks directly to queue...")
        cpu_task_ids = []
        
        for i in range(3):
            task_id = await queue_manager.enqueue_task(
                task_type="plugin_execution",
                data={
                    "plugin_name": "ConceptNetKeywordPlugin",
                    "data": {"concept": f"test_concept_{i}"}
                },
                plugin_name="ConceptNetKeywordPlugin",
                priority=1
            )
            cpu_task_ids.append(task_id)
            logger.info(f"✅ Queued CPU task {i}: {task_id[:8]}...")
        

        logger.info("\n🔄 Submitting GPU task...")
        gpu_task_id = await queue_manager.enqueue_task(
            task_type="plugin_execution",
            data={
                "plugin_name": "LLMKeywordPlugin",
                "data": {"concept": "action movie"}
            },
            plugin_name="LLMKeywordPlugin",
            priority=10
        )
        logger.info(f"✅ Queued GPU task: {gpu_task_id[:8]}...")
        

        stats = await queue_manager.get_queue_stats()
        logger.info(f"\n📊 Queue stats after submission:")
        logger.info(f"  CPU queue: {stats['queues']['cpu_pending']} tasks")
        logger.info(f"  GPU queue: {stats['queues']['gpu_pending']} tasks")
        logger.info(f"  Total pending: {stats['queues']['total_pending']} tasks")
        

        logger.info("\n👀 Monitoring for worker processing (30s)...")
        start_time = time.time()
        
        while time.time() - start_time < 30:
            stats = await queue_manager.get_queue_stats()
            pending = stats['queues']['total_pending']
            failed = stats['queues']['failed_tasks']
            
            logger.info(f"📈 Pending: {pending}, Failed: {failed}")
            
            if pending == 0:
                logger.info("✅ All tasks processed!")
                break
                
            await asyncio.sleep(2)
        

        final_stats = await queue_manager.get_queue_stats()
        logger.info(f"\n📊 Final stats:")
        logger.info(f"  CPU queue: {final_stats['queues']['cpu_pending']} tasks")
        logger.info(f"  GPU queue: {final_stats['queues']['gpu_pending']} tasks") 
        logger.info(f"  Failed: {final_stats['queues']['failed_tasks']} tasks")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        
    finally:
        await queue_manager.cleanup()


async def main():
    await test_direct_queue_submission()


if __name__ == "__main__":
    asyncio.run(main())