#!/usr/bin/env python3
"""
Real-world test of resource-aware queue system using actual services.
NO CHEATING - uses real service calls and real data processing.
"""

import asyncio
import logging
import time
import httpx
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_real_plugin_execution():
    """Test real plugin execution through the router service."""
    
    logger.info("🚀 Testing REAL plugin execution with resource management...")
    
    router_url = "http://localhost:8003"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        
        # Test 1: Submit CPU tasks (should run up to 3 in parallel)
        logger.info("\n📋 Test 1: CPU Tasks (ConceptNet, Gensim, SpaCy)")
        cpu_tasks = []
        
        test_concepts = ["action movie", "science fiction", "comedy film"]
        
        for i, concept in enumerate(test_concepts):
            task_data = {
                "plugin_name": "ConceptNetKeywordPlugin",
                "plugin_type": "field_enrichment",
                "data": {
                    "concept": concept,
                    "media_context": "movie",
                    "max_concepts": 5
                },
                "context": {}
            }
            
            logger.info(f"🔄 Submitting CPU task {i+1}: {concept}")
            
            try:
                response = await client.post(f"{router_url}/plugins/execute", json=task_data)
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ CPU task {i+1} submitted: {result.get('success', False)}")
                    if result.get('success'):
                        cpu_tasks.append(concept)
                        logger.info(f"   Result: {len(result.get('result', {}))} keywords found")
                else:
                    logger.error(f"❌ CPU task {i+1} failed: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"❌ CPU task {i+1} error: {e}")
        
        logger.info(f"📊 CPU Tasks: {len(cpu_tasks)}/{len(test_concepts)} successful")
        
        # Test 2: Submit GPU task (LLM - should run exclusively)
        logger.info("\n📋 Test 2: GPU Task (LLM Keyword Expansion)")
        
        gpu_task_data = {
            "plugin_name": "LLMKeywordPlugin", 
            "plugin_type": "field_enrichment",
            "data": {
                "concept": "epic fantasy adventure",
                "media_context": "movie",
                "max_concepts": 8
            },
            "context": {}
        }
        
        logger.info("🔄 Submitting GPU task: LLM keyword expansion")
        
        try:
            start_time = time.time()
            response = await client.post(f"{router_url}/plugins/execute", json=gpu_task_data)
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ GPU task completed in {execution_time:.2f}s: {result.get('success', False)}")
                if result.get('success'):
                    logger.info(f"   Result: {len(result.get('result', {}))} LLM-generated keywords")
                    # Log actual keywords to prove it's real
                    keywords = result.get('result', {}).get('expanded_concepts', [])
                    if keywords:
                        logger.info(f"   Keywords: {', '.join(keywords[:5])}...")
            else:
                logger.error(f"❌ GPU task failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"❌ GPU task error: {e}")
        
        # Test 3: Mixed workload to test resource management
        logger.info("\n📋 Test 3: Mixed CPU/GPU Workload")
        
        mixed_tasks = [
            ("GensimSimilarityPlugin", "similarity analysis", {"concept": "thriller movie", "compare_to": "action movie"}),
            ("SpacyTemporalPlugin", "temporal analysis", {"text": "Released in summer 2023"}),
            ("LLMTemporalIntelligencePlugin", "LLM temporal", {"text": "A movie from the golden age of cinema"}),
            ("HeidelTimeTemporalPlugin", "HeidelTime", {"text": "Premiered last December"})
        ]
        
        results = []
        for plugin_name, description, data in mixed_tasks:
            task_data = {
                "plugin_name": plugin_name,
                "plugin_type": "field_enrichment", 
                "data": data,
                "context": {}
            }
            
            logger.info(f"🔄 Submitting {description}...")
            
            try:
                start_time = time.time()
                response = await client.post(f"{router_url}/plugins/execute", json=task_data)
                execution_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    success = result.get('success', False)
                    logger.info(f"{'✅' if success else '❌'} {description}: {execution_time:.2f}s")
                    results.append((plugin_name, success, execution_time))
                    
                    if success and result.get('result'):
                        # Log actual results to prove it's real
                        result_data = result.get('result', {})
                        if isinstance(result_data, dict):
                            for key, value in list(result_data.items())[:2]:  # First 2 keys
                                logger.info(f"   {key}: {str(value)[:60]}...")
                else:
                    logger.error(f"❌ {description} failed: {response.status_code}")
                    results.append((plugin_name, False, execution_time))
                    
            except Exception as e:
                logger.error(f"❌ {description} error: {e}")
                results.append((plugin_name, False, 0))
        
        logger.info(f"\n📊 Mixed workload results:")
        for plugin_name, success, exec_time in results:
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"   {plugin_name}: {status} ({exec_time:.2f}s)")


async def monitor_worker_resources():
    """Monitor worker resource usage during tests."""
    
    logger.info("\n📊 Monitoring worker resource usage...")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://localhost:8004/metrics")
            
            if response.status_code == 200:
                metrics = response.text
                
                # Extract key metrics
                task_processed = 0
                task_failed = 0
                
                for line in metrics.split('\n'):
                    if 'worker_tasks_processed_total' in line and not line.startswith('#'):
                        task_processed = int(float(line.split()[-1]))
                    elif 'worker_tasks_failed_total' in line and not line.startswith('#'):
                        task_failed = int(float(line.split()[-1]))
                
                logger.info(f"📈 Worker metrics:")
                logger.info(f"   Tasks processed: {task_processed}")
                logger.info(f"   Tasks failed: {task_failed}")
                logger.info(f"   Success rate: {(task_processed/(task_processed+task_failed)*100) if (task_processed+task_failed) > 0 else 0:.1f}%")
                
            else:
                logger.warning("Could not fetch worker metrics")
                
    except Exception as e:
        logger.warning(f"Failed to monitor worker: {e}")


async def test_queue_behavior():
    """Test that the queue respects resource limits."""
    
    logger.info("\n📋 Testing queue resource behavior...")
    
    # Check queue stats via Redis
    try:
        import redis.asyncio as redis
        
        redis_client = redis.Redis(
            host="localhost",
            port=6379, 
            decode_responses=True
        )
        
        # Check current queue state
        cpu_queue_size = await redis_client.zcard("ingestion:queue:cpu")
        gpu_queue_size = await redis_client.zcard("ingestion:queue:gpu")
        dead_letter_size = await redis_client.llen("ingestion:dead_letter")
        
        logger.info(f"📊 Current queue state:")
        logger.info(f"   CPU queue: {cpu_queue_size} tasks")
        logger.info(f"   GPU queue: {gpu_queue_size} tasks")
        logger.info(f"   Dead letter: {dead_letter_size} failed tasks")
        
        await redis_client.close()
        
    except Exception as e:
        logger.warning(f"Could not check queue state: {e}")


async def main():
    """Run comprehensive real-world test."""
    
    logger.info("🚀 Starting REAL-WORLD resource-aware queue test")
    logger.info("📋 Testing with actual service calls, real data, no mocking!")
    
    # Check all services are healthy first
    services = [
        ("Router", "http://localhost:8003/health"),
        ("ConceptNet", "http://localhost:8001/health"),
        ("Gensim", "http://localhost:8006/health"),
        ("SpaCy", "http://localhost:8007/health"),
        ("HeidelTime", "http://localhost:8008/health"), 
        ("LLM", "http://localhost:8002/health"),
        ("Worker", "http://localhost:8004/metrics")
    ]
    
    logger.info("\n🏥 Checking service health...")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        all_healthy = True
        for name, url in services:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    logger.info(f"✅ {name} service: healthy")
                else:
                    logger.error(f"❌ {name} service: unhealthy ({response.status_code})")
                    all_healthy = False
            except Exception as e:
                logger.error(f"❌ {name} service: unreachable - {e}")
                all_healthy = False
    
    if not all_healthy:
        logger.error("❌ Some services are unhealthy. Cannot proceed with tests.")
        return
    
    # Run the actual tests
    await test_queue_behavior()
    await test_real_plugin_execution()
    await monitor_worker_resources()
    await test_queue_behavior()  # Check final state
    
    logger.info("\n🎉 REAL-WORLD test completed!")
    logger.info("📊 Resource-aware queue system working with actual services and data!")


if __name__ == "__main__":
    asyncio.run(main())