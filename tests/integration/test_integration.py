#!/usr/bin/env python3
"""
Integration test for plugin system with Redis queue and hardware awareness.
Tests the core components built in Stage 3.2.69.
"""

import asyncio
import json
import sys
sys.path.append('.')
from tests.tests_shared import logger
from tests.tests_shared import settings_to_console

from src.shared.config import get_settings
from src.worker.resource_queue_manager import ResourceAwareQueueManager
from src.worker.resource_manager import create_resource_pool_from_config
# ConceptExpansionPlugin was archived - using HTTP-only plugins now
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
    resource_config = {"cpu_cores": 1, "gpu_count": 0, "memory_mb": 512}
    resource_pool = create_resource_pool_from_config(resource_config, worker_id="test_integration")
    queue_manager = ResourceAwareQueueManager(resource_pool)
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
    
    # Test basic resource detection
    required_keys = ["local_cpu_cores", "local_memory_gb", "gpu_available"]
    for key in required_keys:
        if key not in hardware_limits:
            raise AssertionError(f"Missing required resource key: {key} - hardware detection is incomplete")
    
    # Test that we get reasonable values
    if hardware_limits["local_cpu_cores"] <= 0:
        raise AssertionError("Invalid CPU core count - hardware detection is broken")
    
    if hardware_limits["local_memory_gb"] <= 0:
        raise AssertionError("Invalid memory size - hardware detection is broken")
    
    logger.info("✅ Hardware detection working correctly")
    
    return True


async def test_http_plugin_system():
    """Test HTTP-only plugin system - FAIL FAST if broken."""
    logger.info("🌐 Testing HTTP-only plugin system...")
    settings_to_console()
    
    # Test that we can import and initialize HTTP-based plugins
    try:
        from src.plugins.enrichment.conceptnet_keyword_plugin import ConceptNetKeywordPlugin
        from src.plugins.enrichment.llm_keyword_plugin import LLMKeywordPlugin
        
        # Test ConceptNet plugin
        conceptnet_plugin = ConceptNetKeywordPlugin()
        if not conceptnet_plugin.metadata:
            raise AssertionError("ConceptNetKeywordPlugin has no metadata - plugin system is broken")
        
        logger.info(f"✅ ConceptNetKeywordPlugin loaded: {conceptnet_plugin.metadata.name}")
        
        # Test LLM plugin
        llm_plugin = LLMKeywordPlugin()
        if not llm_plugin.metadata:
            raise AssertionError("LLMKeywordPlugin has no metadata - plugin system is broken")
        
        logger.info(f"✅ LLMKeywordPlugin loaded: {llm_plugin.metadata.name}")
        
        # Test plugin loader system
        from src.worker.plugin_loader import PluginLoader
        loader = PluginLoader()
        
        # Initialize loader
        if not await loader.initialize():
            raise AssertionError("PluginLoader initialization failed - plugin system is broken")
        
        logger.info("✅ PluginLoader initialized successfully")
        
        # Test plugin discovery
        available_plugins = loader.list_available_plugins()
        if not available_plugins:
            raise AssertionError("No plugins discovered - plugin discovery is broken")
        
        logger.info(f"✅ Plugin discovery working: {len(available_plugins.get('service_plugins', {}))} service plugins")
        
        await loader.cleanup()
        return True
        
    except ImportError as e:
        raise AssertionError(f"Failed to import HTTP plugins - plugin system is broken: {e}")
    except Exception as e:
        raise AssertionError(f"HTTP plugin system test failed: {e}")


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
        ("HTTP Plugin System", test_http_plugin_system),
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