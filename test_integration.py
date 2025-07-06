#!/usr/bin/env python3
"""
Integration test for plugin system with Redis queue and hardware awareness.
Tests the core components built in Stage 3.2.69.
"""

import asyncio
import json
from tests_shared import logger
from tests_shared import settings_to_console

from src.shared.config import get_settings
from src.redis_worker.queue_manager import RedisQueueManager
from src.plugins.concept_expansion_plugin import ConceptExpansionPlugin
from src.plugins.base import PluginExecutionContext
from src.shared.hardware_config import get_resource_limits
import subprocess


async def build_docker_image(image_name: str, dockerfile_path: str) -> bool:
    """Build a Docker image if missing."""
    logger.info(f"🔨 Building Docker image: {image_name}")
    try:
        result = subprocess.run(
            ["docker", "build", "-t", image_name, "-f", dockerfile_path, "."],
            capture_output=True, text=True, timeout=300  # 5 minutes for build
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Successfully built {image_name}")
            return True
        else:
            logger.error(f"❌ Failed to build {image_name}: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error building {image_name}: {e}")
        return False


async def start_docker_stack():
    """Start the full Docker stack using docker-compose."""
    logger.info("🐳 Starting Docker stack with docker-compose...")
    try:
        # Check if docker-compose file exists
        import os
        if not os.path.exists("docker-compose.dev.yml"):
            logger.error("❌ docker-compose.dev.yml not found")
            return False
            
        # Start the Docker stack (without optional profiles that aren't implemented yet)
        try:
            result = subprocess.run(
                ["docker", "compose", "-f", "docker-compose.dev.yml", "up", "-d"],
                capture_output=True, text=True, timeout=120
            )
        except FileNotFoundError:
            result = subprocess.run(
                ["docker-compose", "-f", "docker-compose.dev.yml", "up", "-d"],
                capture_output=True, text=True, timeout=120
            )
        
        if result.returncode == 0:
            logger.info("✅ Docker stack started successfully")
            logger.info("⏳ Waiting for services to be ready...")
            # Wait longer for all services to be ready
            import time
            time.sleep(10)
            return True
        else:
            logger.error(f"❌ Failed to start Docker stack")
            logger.error(f"Error output: {result.stderr}")
            
            # Try to build missing Docker images automatically
            if "jelly-faiss" in result.stderr and "does not exist" in result.stderr:
                logger.info("🔨 Building missing FAISS Docker image...")
                if await build_docker_image("jelly-faiss", "docker/faiss/Dockerfile"):
                    logger.info("✅ FAISS image built, retrying Docker stack...")
                    return await start_docker_stack()  # Retry after building
                    
            if "jelly-api" in result.stderr and "does not exist" in result.stderr:
                logger.info("🔨 Building missing API Docker image...")
                if await build_docker_image("jelly-api", "docker/api/Dockerfile"):
                    logger.info("✅ API image built, retrying Docker stack...")
                    return await start_docker_stack()  # Retry after building
                    
            if "jelly-worker" in result.stderr and "does not exist" in result.stderr:
                logger.info("🔨 Building missing Worker Docker image...")
                if await build_docker_image("jelly-worker", "docker/worker/Dockerfile"):
                    logger.info("✅ Worker image built, retrying Docker stack...")
                    return await start_docker_stack()  # Retry after building
                
            return False
            
    except Exception as e:
        logger.error(f"❌ Error starting Docker stack: {e}")
        return False


async def test_redis_connection():
    """Test Redis connection and queue manager - FAIL FAST if Redis unavailable."""
    logger.info("🔌 Testing Redis connection...")
    settings_to_console()
    
    # NO FALLBACKS - if Redis is down, test should fail hard
    queue_manager = RedisQueueManager()
    is_healthy = queue_manager.health_check()
    
    if not is_healthy:
        raise AssertionError("Redis connection failed - ensure Redis is running and accessible")
    
    logger.info("✅ Redis connection healthy")
    
    # Test queue operations - these should work or fail clearly
    test_data = {"test": "data", "timestamp": "2025-01-05"}
    task_id = queue_manager.enqueue_task("test_task", test_data, priority=1)
    
    if not task_id:
        raise AssertionError("Redis task queueing failed - Redis may be misconfigured")
    
    logger.info(f"✅ Task queued successfully: {task_id}")
    
    # Get queue stats
    stats = queue_manager.get_queue_stats()
    logger.info(f"📊 Queue stats: {stats}")
    
    return True


async def test_hardware_detection():
    """Test hardware detection and strategy selection - FAIL FAST if hardware detection broken."""
    logger.info("🖥️ Testing hardware detection...")
    settings_to_console()
    
    # NO FALLBACKS - if hardware detection is broken, test should fail hard
    hardware_limits = await get_resource_limits()
    
    if not hardware_limits or not isinstance(hardware_limits, dict):
        raise AssertionError("Hardware detection failed - get_resource_limits returned invalid data")
    
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
    initialization_success = await plugin.initialize({})
    
    if not initialization_success:
        raise AssertionError("ConceptExpansionPlugin initialization failed - plugin system is broken")
    
    for i, context in enumerate(contexts):
        strategy = plugin._select_processing_strategy(context)
        if not strategy:
            raise AssertionError(f"Strategy selection failed for scenario {i+1} - plugin logic is broken")
        logger.info(f"✅ Scenario {i+1}: {strategy.value}")
    
    return True


async def test_concept_expansion_plugin():
    """Test ConceptExpansionPlugin with actual data - FAIL FAST if broken."""
    logger.info("🧠 Testing ConceptExpansionPlugin...")
    settings_to_console()

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
    
    # Initialize plugin - NO FALLBACKS
    plugin = ConceptExpansionPlugin()
    initialization_success = await plugin.initialize({})
    
    if not initialization_success:
        raise AssertionError("ConceptExpansionPlugin initialization failed - plugin system is broken")
    
    logger.info("✅ Plugin initialized successfully")
    
    try:
        # Test with low resource context (should use ConceptNet only)
        context = PluginExecutionContext(
            available_resources={"total_cpu_capacity": 1, "local_memory_gb": 1, "gpu_available": False}
        )
        
        logger.info("🔄 Testing concept expansion...")
        result_data = await plugin.embellish_embed_data(test_data, context)
        
        # Validate we got actual enhancement results, not empty fallback
        if "enhanced_fields" not in result_data:
            raise AssertionError("ConceptExpansionPlugin returned no enhanced_fields - concept expansion is broken")
        
        enhanced = result_data["enhanced_fields"]
        if not enhanced or not enhanced.get("expanded_concepts"):
            raise AssertionError("ConceptExpansionPlugin returned empty enhanced_fields - no concept expansion occurred")
        
        logger.info("✅ Concept expansion successful!")
        logger.info(f"📊 Expansion metadata: {enhanced.get('expansion_metadata', {})}")
        
        # Log some example expansions
        expanded_concepts = enhanced.get("expanded_concepts", {})
        for concept, expansions in list(expanded_concepts.items())[:3]:
            logger.info(f"🎯 '{concept}' → {expansions[:5]}")
        
        return True
        
    finally:
        await plugin.cleanup()


async def test_service_configuration():
    """Test service configuration and environment detection - FAIL FAST if config broken."""
    logger.info("⚙️ Testing service configuration...")
    settings_to_console()

    # NO FALLBACKS - if configuration is broken, test should fail hard
    settings = get_settings()
    
    if not settings:
        raise AssertionError("get_settings() returned None - configuration system is broken")
    
    logger.info(f"✅ Environment: {settings.ENV}")
    logger.info(f"🔗 Redis URL: {settings.redis_url}")
    logger.info(f"🔗 MongoDB URL: {settings.mongodb_url}")
    
    # Test service health
    health_status = settings.get_health_status()
    logger.info(f"🏥 Health status: {health_status}")
    
    if not health_status or not isinstance(health_status, dict):
        raise AssertionError("get_health_status() returned invalid data - health check system is broken")
    
    # Check if all required services are healthy
    # Note: FAISS service not fully implemented yet, so it's optional for now
    required_services = ["ollama_chat", "redis"]
    optional_services = ["faiss"]
    failed_services = [service for service in required_services if not health_status.get(service, False)]
    
    # Check optional services and log their status
    failed_optional = [service for service in optional_services if not health_status.get(service, False)]
    if failed_optional:
        logger.info(f"ℹ️  Optional services not running: {failed_optional} (this is OK)")
    else:
        logger.info(f"✅ Optional services running: {optional_services}")
    
    if failed_services:
        logger.info(f"❌ Services not running: {failed_services}")
        logger.info("🚀 Starting required services...")
        
        # Start the full Docker stack to get all services
        await start_docker_stack()
        
        # Re-check health after starting services
        logger.info("🔄 Re-checking service health...")
        health_status = settings.get_health_status()
        failed_services = [service for service in required_services if not health_status.get(service, False)]
        
        if failed_services:
            logger.error(f"❌ Still failing after Docker startup: {failed_services}")
            
            # Provide specific guidance for each failing service
            for service in failed_services:
                if service == "faiss":
                    logger.error("💡 FAISS service may need Docker image built or profile enabled")
                    logger.error("   Try: docker build -t jelly-faiss -f docker/faiss/Dockerfile .")
                elif service == "redis":
                    logger.error("💡 Redis may be starting up - wait a few seconds and try again")
                elif service == "ollama_chat":
                    logger.error("💡 Ollama service not running - start with: ollama serve")
                    
            raise AssertionError(f"Integration test requires these services: {failed_services}")
    
    logger.info("✅ All required services are healthy")
    
    return True


async def main():
    """Run all integration tests."""
    logger.info("🚀 Starting Integration Tests for Stage 3.2.69")
    logger.info("=" * 60)
    settings_to_console()
    
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
        
        # NO FALLBACKS - if a test fails, the integration should fail hard
        success = await test_func()
        results[test_name] = success
        logger.info(f"🏁 {test_name}: ✅ PASSED")
    
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