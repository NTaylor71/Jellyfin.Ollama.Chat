#!/usr/bin/env python3
"""
Integration test for plugin system with Redis queue and hardware awareness.
Tests the core components built in Stage 3.2.69.
"""

import asyncio
import json
import logging
from typing import Dict, Any

from src.shared.config import get_settings
from src.redis_worker.queue_manager import RedisQueueManager
from src.plugins.concept_expansion_plugin import ConceptExpansionPlugin
from src.plugins.base import PluginExecutionContext
from src.shared.hardware_config import get_resource_limits

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_redis_connection():
    """Test Redis connection and queue manager."""
    logger.info("🔌 Testing Redis connection...")
    
    try:
        queue_manager = RedisQueueManager()
        is_healthy = queue_manager.health_check()
        
        if is_healthy:
            logger.info("✅ Redis connection healthy")
            
            # Test queue operations
            test_data = {"test": "data", "timestamp": "2025-01-05"}
            task_id = queue_manager.enqueue_task("test_task", test_data, priority=1)
            logger.info(f"✅ Task queued successfully: {task_id}")
            
            # Get queue stats
            stats = queue_manager.get_queue_stats()
            logger.info(f"📊 Queue stats: {stats}")
            
            return True
        else:
            logger.error("❌ Redis connection failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Redis test failed: {e}")
        return False


async def test_hardware_detection():
    """Test hardware detection and strategy selection."""
    logger.info("🖥️ Testing hardware detection...")
    
    try:
        hardware_limits = await get_resource_limits()
        logger.info(f"✅ Hardware detected: {hardware_limits}")
        
        # Test with different hardware scenarios
        contexts = [
            PluginExecutionContext(
                available_resources={"total_cpu_capacity": 1, "local_memory_gb": 1, "gpu_available": False}
            ),
            PluginExecutionContext(
                available_resources={"total_cpu_capacity": 4, "local_memory_gb": 4, "gpu_available": False}
            ),
            PluginExecutionContext(
                available_resources={"total_cpu_capacity": 8, "local_memory_gb": 8, "gpu_available": True}
            )
        ]
        
        plugin = ConceptExpansionPlugin()
        await plugin.initialize({})
        
        for i, context in enumerate(contexts):
            strategy = plugin._select_processing_strategy(context)
            logger.info(f"✅ Scenario {i+1}: {strategy.value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hardware detection test failed: {e}")
        return False


async def test_concept_expansion_plugin():
    """Test ConceptExpansionPlugin with actual data."""
    logger.info("🧠 Testing ConceptExpansionPlugin...")
    
    try:
        # Sample Jellyfin movie data
        test_data = {
            "Name": "The Matrix",
            "OriginalTitle": "The Matrix",
            "ProductionYear": 1999,
            "Genres": [{"Name": "Action"}, {"Name": "Science Fiction"}],
            "Taglines": ["Welcome to the Real World"],
            "Overview": "A computer hacker learns from mysterious rebels about the true nature of his reality.",
            "OfficialRating": "R"
        }
        
        # Initialize plugin
        plugin = ConceptExpansionPlugin()
        initialization_success = await plugin.initialize({})
        
        if not initialization_success:
            logger.error("❌ Plugin initialization failed")
            return False
        
        logger.info("✅ Plugin initialized successfully")
        
        # Test with low resource context (should use ConceptNet only)
        context = PluginExecutionContext(
            available_resources={"total_cpu_capacity": 1, "local_memory_gb": 1, "gpu_available": False}
        )
        
        logger.info("🔄 Testing concept expansion...")
        result_data = await plugin.embellish_embed_data(test_data, context)
        
        if "enhanced_fields" in result_data:
            enhanced = result_data["enhanced_fields"]
            logger.info("✅ Concept expansion successful!")
            logger.info(f"📊 Expansion metadata: {enhanced.get('expansion_metadata', {})}")
            
            # Log some example expansions
            expanded_concepts = enhanced.get("expanded_concepts", {})
            for concept, expansions in list(expanded_concepts.items())[:3]:
                logger.info(f"🎯 '{concept}' → {expansions[:5]}")
                
            return True
        else:
            logger.warning("⚠️ No enhanced_fields found in result")
            return False
            
    except Exception as e:
        logger.error(f"❌ ConceptExpansionPlugin test failed: {e}")
        return False
    finally:
        if 'plugin' in locals():
            await plugin.cleanup()


async def test_service_configuration():
    """Test service configuration and environment detection."""
    logger.info("⚙️ Testing service configuration...")
    
    try:
        settings = get_settings()
        logger.info(f"✅ Environment: {settings.ENV}")
        logger.info(f"🔗 Redis URL: {settings.redis_url}")
        logger.info(f"🔗 MongoDB URL: {settings.mongodb_url}")
        
        # Test service health
        health_status = settings.get_health_status()
        logger.info(f"🏥 Health status: {health_status}")
        
        return all(health_status.values())
        
    except Exception as e:
        logger.error(f"❌ Service configuration test failed: {e}")
        return False


async def main():
    """Run all integration tests."""
    logger.info("🚀 Starting Integration Tests for Stage 3.2.69")
    logger.info("=" * 60)
    
    tests = [
        ("Service Configuration", test_service_configuration),
        ("Redis Connection", test_redis_connection),
        ("Hardware Detection", test_hardware_detection),
        ("ConceptExpansionPlugin", test_concept_expansion_plugin),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            success = await test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"🏁 {test_name}: {status}")
            
        except Exception as e:
            logger.error(f"💥 {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {status} {test_name}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All integration tests PASSED! System is ready for Stage 3.2.69+")
    else:
        logger.info("⚠️ Some tests failed. Please check the logs above.")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(main())