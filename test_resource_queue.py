#!/usr/bin/env python3
"""
Test script for resource-aware queue system.
Demonstrates CPU/GPU task scheduling without conflicts.
"""

import asyncio
import logging
import time
from typing import Dict, Any

from src.worker.resource_manager import ResourcePool, ResourceRequirement
from src.worker.resource_queue_manager import ResourceAwareQueueManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_resource_aware_scheduling():
    """Test the resource-aware queue system with CPU and GPU tasks."""
    
    # Create resource pool for home machine (3 CPU, 1 GPU)
    resource_pool = ResourcePool(
        total_cpu_cores=3,
        total_gpus=1,
        total_memory_mb=8192,
        worker_id="test_worker"
    )
    
    # Create queue manager
    queue_manager = ResourceAwareQueueManager(resource_pool)
    
    try:
        # Test health check
        health_ok = await queue_manager.health_check()
        logger.info(f"Redis health check: {'✅ OK' if health_ok else '❌ Failed'}")
        
        if not health_ok:
            logger.error("Redis not available. Start Redis with: docker run -d -p 6379:6379 redis:alpine")
            return
        
        # Clear any existing tasks
        await queue_manager.purge_queues()
        
        # Enqueue test tasks
        logger.info("\n🔄 Enqueuing test tasks...")
        
        # CPU tasks (should run in parallel, up to 3)
        cpu_tasks = []
        for i in range(5):  # More than our limit
            task_id = await queue_manager.enqueue_task(
                task_type="plugin_execution",
                data={"test": f"cpu_task_{i}"},
                plugin_name="ConceptNetKeywordPlugin",  # CPU-only plugin
                priority=1
            )
            cpu_tasks.append(task_id)
            logger.info(f"Enqueued CPU task {i}: {task_id}")
        
        # GPU task (should run exclusively)
        gpu_task_id = await queue_manager.enqueue_task(
            task_type="plugin_execution", 
            data={"test": "gpu_task"},
            plugin_name="LLMKeywordPlugin",  # GPU plugin
            priority=10  # Higher priority
        )
        logger.info(f"Enqueued GPU task: {gpu_task_id}")
        
        # Show queue stats
        stats = await queue_manager.get_queue_stats()
        logger.info(f"\n📊 Queue stats: {stats['queues']}")
        logger.info(f"💾 Resource usage: {stats['resources']['utilization']}")
        
        # Simulate task processing
        logger.info("\n🔄 Simulating task processing...")
        
        processed_tasks = []
        start_time = time.time()
        timeout = 30  # 30 second test
        
        while time.time() - start_time < timeout:
            # Try to dequeue a task
            task = await queue_manager.dequeue_task(timeout=2)
            
            if task:
                task_id = task['task_id']
                req = ResourceRequirement(**task['resource_requirements'])
                
                logger.info(f"🔄 Processing {task_id} (CPU={req.cpu_cores}, GPU={req.gpu_count})")
                
                # Simulate task processing time
                processing_time = 3.0 if req.gpu_count > 0 else 1.5
                await asyncio.sleep(processing_time)
                
                # Complete the task
                result = {
                    "task_id": task_id,
                    "success": True,
                    "processing_time": processing_time,
                    "result": f"Completed {task['data'].get('test', 'unknown')}"
                }
                
                await queue_manager.complete_task(task_id, result)
                processed_tasks.append(task_id)
                
                # Release resources (normally done by worker)
                resource_pool.release(task_id)
                
                logger.info(f"✅ Completed {task_id}")
                
                # Show current resource usage
                utilization = resource_pool.get_utilization()
                logger.info(f"💾 Current usage: CPU={utilization['cpu_utilization']:.1f}%, GPU={utilization['gpu_utilization']:.1f}%, Active={utilization['active_tasks']}")
            else:
                logger.info("No runnable tasks available")
        
        # Final stats
        final_stats = await queue_manager.get_queue_stats()
        logger.info(f"\n📊 Final queue stats: {final_stats['queues']}")
        logger.info(f"✅ Processed {len(processed_tasks)} tasks")
        logger.info(f"📋 Processed task IDs: {processed_tasks}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        
    finally:
        await queue_manager.cleanup()


async def test_resource_constraints():
    """Test that resource constraints are properly enforced."""
    
    logger.info("\n🧪 Testing resource constraints...")
    
    # Small resource pool
    resource_pool = ResourcePool(
        total_cpu_cores=1,
        total_gpus=1, 
        total_memory_mb=1024,
        worker_id="constraint_test"
    )
    
    # Test CPU constraint
    req1 = ResourceRequirement(cpu_cores=0.8, memory_mb=400)
    req2 = ResourceRequirement(cpu_cores=0.5, memory_mb=400)  # Should fail (total > 1.0 CPU)
    
    assert resource_pool.can_schedule(req1), "Should be able to schedule first task"
    
    allocation1 = resource_pool.allocate("task1", req1)
    logger.info(f"✅ Allocated task1: CPU={req1.cpu_cores}")
    
    assert not resource_pool.can_schedule(req2), "Should NOT be able to schedule second task (CPU limit)"
    logger.info(f"✅ Correctly blocked second task (CPU constraint)")
    
    # Test GPU exclusivity
    gpu_req = ResourceRequirement(cpu_cores=0.1, gpu_count=1, exclusive_gpu=True)
    cpu_req_small = ResourceRequirement(cpu_cores=0.1)
    
    resource_pool.release("task1")  # Free up resources
    
    allocation_gpu = resource_pool.allocate("gpu_task", gpu_req)
    logger.info(f"✅ Allocated GPU task")
    
    assert not resource_pool.can_schedule(cpu_req_small), "Should NOT schedule CPU task when GPU is exclusive"
    logger.info(f"✅ Correctly blocked CPU task during exclusive GPU usage")
    
    resource_pool.release("gpu_task")
    logger.info(f"✅ Resource constraint tests passed!")


async def main():
    """Run all tests."""
    logger.info("🚀 Starting resource-aware queue system tests...")
    
    await test_resource_constraints()
    await test_resource_aware_scheduling()
    
    logger.info("✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())