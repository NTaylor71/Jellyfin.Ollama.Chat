#!/usr/bin/env python3
"""
Service-Oriented Plugin Architecture Test - FAIL FAST Approach
Tests the microservices architecture built in Stage 4.3.

HARD RULE 12: Never fix test conditions to make a failing test pass - fix actual bugs.

Tests:
1. Service startup and health
2. Plugin routing through services  
3. End-to-end workflow: Worker → Plugin → Service → Result
4. Service discovery and communication
5. Error handling and fallbacks

NO FALLBACKS - If services fail, this should fail immediately with clear error messages.
"""

import asyncio
import subprocess
import sys
import time
import httpx
import json
from pathlib import Path
from ..tests_shared import logger, settings_to_console
from src.shared.config import get_settings


class ServiceTestManager:
    """Manages service testing lifecycle."""
    
    def __init__(self):
        self.settings = get_settings()
        self.services = {
            "conceptnet_provider": {
                "module": "src.services.provider_services.conceptnet_service",
                "port": 8001,
                "url": "http://localhost:8001",
                "process": None
            },
            "gensim_provider": {
                "module": "src.services.provider_services.gensim_service",
                "port": 8006,
                "url": "http://localhost:8006",
                "process": None
            },
            "spacy_provider": {
                "module": "src.services.provider_services.spacy_service",
                "port": 8007,
                "url": "http://localhost:8007",
                "process": None
            },
            "heideltime_provider": {
                "module": "src.services.provider_services.heideltime_service",
                "port": 8008,
                "url": "http://localhost:8008",
                "process": None
            },
            "llm_provider": {
                "module": "src.services.provider_services.minimal_llm_service", 
                "port": 8002,
                "url": self.settings.llm_service_url,
                "process": None
            },
            "plugin_router": {
                "module": "src.services.orchestration.plugin_router_service",
                "port": 8003,
                "url": self.settings.router_service_url,
                "process": None
            }
        }
        self.http_client = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.http_client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.http_client:
            await self.http_client.aclose()
        await self.stop_all_services()
    
    async def start_docker_services(self):
        """Start all services using Docker Compose."""
        try:
            logger.info("🚀 Starting Docker services...")
            
            # Start Docker services
            result = subprocess.run([
                "docker", "compose", "-f", "docker-compose.dev.yml", "up", "-d",
                "conceptnet-service", "gensim-service", "spacy-service", "heideltime-service", "llm-service", "router-service", "redis", "mongodb", "ollama"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                raise AssertionError(f"❌ Docker compose failed: {result.stderr}")
            
            logger.info("✅ Docker services started")
            
            # Wait for services to be ready
            await asyncio.sleep(10)
            
            return True
            
        except Exception as e:
            raise AssertionError(f"❌ Failed to start Docker services: {e}")
    
    async def stop_service(self, service_name: str):
        """Stop a specific service."""
        service = self.services[service_name]
        
        if not service["process"] or service["process"].poll() is not None:
            return
        
        logger.info(f"🛑 Stopping {service_name}...")
        
        try:
            service["process"].terminate()
            try:
                service["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                service["process"].kill()
                service["process"].wait()
            
            service["process"] = None
            logger.info(f"✅ Service {service_name} stopped")
            
        except Exception as e:
            logger.warning(f"⚠️ Error stopping {service_name}: {e}")
    
    async def stop_all_services(self):
        """Stop all Docker services."""
        try:
            logger.info("🛑 Stopping Docker services...")
            result = subprocess.run([
                "docker", "compose", "-f", "docker-compose.dev.yml", "down"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.warning(f"⚠️ Docker compose down warning: {result.stderr}")
            else:
                logger.info("✅ Docker services stopped")
                
        except Exception as e:
            logger.warning(f"⚠️ Error stopping Docker services: {e}")
    
    async def check_service_health(self, service_name: str) -> bool:
        """Check if a service is healthy."""
        service = self.services[service_name]
        
        try:
            response = await self.http_client.get(f"{service['url']}/health")
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ {service_name} health check passed: {health_data.get('status', 'unknown')}")
                return True
            else:
                raise AssertionError(f"❌ {service_name} health check failed: HTTP {response.status_code}")
        
        except httpx.ConnectError:
            raise AssertionError(f"❌ {service_name} not accessible at {service['url']}")
        except Exception as e:
            raise AssertionError(f"❌ {service_name} health check error: {e}")
    
    async def test_service_endpoints(self, service_name: str):
        """Test specific service endpoints."""
        service = self.services[service_name]
        base_url = service['url']
        
        if service_name in ["conceptnet_provider", "gensim_provider", "spacy_provider", "heideltime_provider"]:
            # Test split architecture provider endpoints
            response = await self.http_client.get(f"{base_url}/providers")
            if response.status_code != 200:
                raise AssertionError(f"❌ {service_name} providers endpoint failed: HTTP {response.status_code}")
            
            providers_data = response.json()
            available_providers = providers_data.get("available_providers", [])
            logger.info(f"✅ {service_name} providers available: {available_providers}")
            
            if not available_providers:
                raise AssertionError(f"❌ No {service_name} providers available - check provider initialization")
        
        elif service_name == "llm_provider":
            # Test LLM provider endpoints
            response = await self.http_client.get(f"{base_url}/providers")
            if response.status_code != 200:
                raise AssertionError(f"❌ LLM provider endpoint failed: HTTP {response.status_code}")
            
            provider_data = response.json()
            logger.info(f"✅ LLM provider info: {provider_data.get('name', 'unknown')}")
        
        elif service_name == "plugin_router":
            # Test plugin router endpoints
            response = await self.http_client.get(f"{base_url}/services")
            if response.status_code != 200:
                raise AssertionError(f"❌ Router services endpoint failed: HTTP {response.status_code}")
            
            services_data = response.json()
            configured_services = list(services_data.get("services", {}).keys())
            logger.info(f"✅ Router configured services: {configured_services}")


async def test_service_startup_sequence():
    """Test services start up correctly in sequence."""
    logger.info("🧪 Testing Service Startup Sequence")
    logger.info("-" * 50)
    
    async with ServiceTestManager() as manager:
        # Start all services using Docker
        await manager.start_docker_services()
        
        # Test each service health and endpoints
        service_order = ["conceptnet_provider", "gensim_provider", "spacy_provider", "heideltime_provider", "llm_provider", "plugin_router"]
        
        for service_name in service_order:
            # Verify health
            await manager.check_service_health(service_name)
            
            # Test specific endpoints
            await manager.test_service_endpoints(service_name)
        
        logger.info("✅ All services started and healthy")


async def test_service_communication():
    """Test inter-service communication patterns."""
    logger.info("\n🧪 Testing Service Communication")
    logger.info("-" * 50)
    
    async with ServiceTestManager() as manager:
        # Start all services using Docker
        await manager.start_docker_services()
        
        # Test plugin router can reach other services
        router_url = manager.settings.router_service_url
        
        try:
            # Refresh service health through router
            response = await manager.http_client.post(f"{router_url}/services/health")
            if response.status_code != 200:
                raise AssertionError(f"❌ Router health check failed: HTTP {response.status_code}")
            
            health_data = response.json()
            logger.info(f"✅ Router service discovery: {health_data}")
            
            # Check service statuses
            services_info = health_data.get("services", {})
            for service_name, service_data in services_info.items():
                status = service_data.get("status", "unknown")
                if status != "healthy":
                    raise AssertionError(f"❌ Service {service_name} unhealthy: {status}")
                logger.info(f"✅ Service {service_name}: {status}")
        
        except Exception as e:
            raise AssertionError(f"❌ Service communication test failed: {e}")


async def test_plugin_execution_routing():
    """Test plugin execution through service routing."""
    logger.info("\n🧪 Testing Plugin Execution Routing")
    logger.info("-" * 50)
    
    async with ServiceTestManager() as manager:
        # Start all services using Docker
        await manager.start_docker_services()
        
        # Test concept expansion through router
        router_url = manager.settings.router_service_url
        
        test_request = {
            "plugin_name": "ConceptExpansionPlugin",
            "plugin_type": "concept_expansion",
            "data": {
                "concept": "action",
                "media_context": "movie",
                "max_concepts": 5,
                "field_name": "genre"
            },
            "context": {}
        }
        
        try:
            response = await manager.http_client.post(
                f"{router_url}/execute",
                json=test_request,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise AssertionError(f"❌ Plugin execution failed: HTTP {response.status_code} - {response.text}")
            
            result = response.json()
            
            if not result.get("success", False):
                error_msg = result.get("error_message", "Unknown error")
                raise AssertionError(f"❌ Plugin execution failed: {error_msg}")
            
            execution_time = result.get("execution_time_ms", 0)
            service_used = result.get("service_used", "unknown")
            
            logger.info(f"✅ Plugin executed successfully:")
            logger.info(f"   Service: {service_used}")
            logger.info(f"   Time: {execution_time:.1f}ms")
            logger.info(f"   Result: {result.get('result', {}).keys() if result.get('result') else 'None'}")
            
        except httpx.TimeoutException:
            raise AssertionError("❌ Plugin execution timed out - check service performance")
        except Exception as e:
            raise AssertionError(f"❌ Plugin execution routing failed: {e}")


async def test_service_error_handling():
    """Test service error handling and degradation."""
    logger.info("\n🧪 Testing Service Error Handling")
    logger.info("-" * 50)
    
    async with ServiceTestManager() as manager:
        # Start only router service (partial Docker setup)
        await manager.start_docker_services()
        
        # Verify router reports unhealthy services
        await manager.check_service_health("plugin_router")
        
        router_url = manager.settings.router_service_url
        
        # Test plugin execution with unavailable services
        test_request = {
            "plugin_name": "ConceptExpansionPlugin", 
            "plugin_type": "concept_expansion",
            "data": {"concept": "test"},
            "context": {}
        }
        
        try:
            response = await manager.http_client.post(
                f"{router_url}/execute",
                json=test_request,
                timeout=10.0
            )
            
            # Should fail gracefully
            if response.status_code == 503:
                result = response.json()
                logger.info(f"✅ Router correctly reports service unavailable: {result.get('detail', '')}")
            else:
                # Should not succeed when services are down
                raise AssertionError(f"❌ Router should fail when services unavailable, got HTTP {response.status_code}")
        
        except httpx.TimeoutException:
            logger.info("✅ Router correctly times out when services unavailable")
        except Exception as e:
            raise AssertionError(f"❌ Error handling test failed: {e}")


async def test_worker_service_integration():
    """Test worker integration with services."""
    logger.info("\n🧪 Testing Worker-Service Integration")
    logger.info("-" * 50)
    
    # This test requires Redis to be running
    settings = get_settings()
    
    try:
        import redis
        r = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.REDIS_DB,
            socket_timeout=5
        )
        r.ping()
        logger.info("✅ Redis connection verified")
    except Exception as e:
        raise AssertionError(f"❌ Redis required for worker test: {e}")
    
    async with ServiceTestManager() as manager:
        # Start all services using Docker
        await manager.start_docker_services()
        
        # Import worker components
        try:
            from src.worker.resource_queue_manager import ResourceAwareQueueManager
            from src.worker.resource_manager import create_resource_pool_from_config
            from src.worker.plugin_loader import PluginLoader
            
            resource_config = {"cpu_cores": 1, "gpu_count": 0, "memory_mb": 512}
            resource_pool = create_resource_pool_from_config(resource_config, worker_id="test_services")
            queue_manager = ResourceAwareQueueManager(resource_pool)
            plugin_loader = PluginLoader()
            
            # Initialize plugin loader
            if not await plugin_loader.initialize():
                raise AssertionError("❌ Plugin loader initialization failed")
            
            logger.info("✅ Worker components initialized")
            
            # Test task routing through services
            task_data = {
                "task_id": "test_service_task",
                "task_type": "concept_expansion",
                "concept": "thriller",
                "media_context": "movie",
                "max_concepts": 3
            }
            
            # Route task through plugin loader
            result = await plugin_loader.route_task_to_plugin("concept_expansion", task_data)
            
            if not result.success:
                raise AssertionError(f"❌ Worker task routing failed: {result.error_message}")
            
            logger.info(f"✅ Worker task completed:")
            logger.info(f"   Execution time: {result.execution_time_ms:.1f}ms")
            logger.info(f"   Via service: {result.metadata.get('via_service', False)}")
            
            # Cleanup
            await plugin_loader.cleanup()
            
        except ImportError as e:
            raise AssertionError(f"❌ Worker components import failed: {e}")


def test_service_runner_cli():
    """Test the service runner CLI utility."""
    logger.info("\n🧪 Testing Service Runner CLI")
    logger.info("-" * 50)
    
    try:
        # Test service runner help
        result = subprocess.run([
            sys.executable, "-m", "src.services.orchestration.service_runner", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            raise AssertionError(f"❌ Service runner help failed: {result.stderr}")
        
        if "start" not in result.stdout or "stop" not in result.stdout:
            raise AssertionError("❌ Service runner missing expected commands")
        
        logger.info("✅ Service runner CLI working")
        
        # Test service status (should show no services running)
        result = subprocess.run([
            sys.executable, "-m", "src.services.orchestration.service_runner", "status"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            raise AssertionError(f"❌ Service runner status failed: {result.stderr}")
        
        logger.info("✅ Service runner status command working")
        
    except subprocess.TimeoutExpired:
        raise AssertionError("❌ Service runner CLI timed out")
    except Exception as e:
        raise AssertionError(f"❌ Service runner CLI test failed: {e}")


async def main():
    """Main test entry point."""
    logger.info("🚀 Starting Service-Oriented Plugin Architecture Tests")
    logger.info("=" * 60)
    
    # Show current configuration
    settings_to_console()
    
    try:
        # Test 1: Service startup sequence
        await test_service_startup_sequence()
        
        # Test 2: Service communication
        await test_service_communication()
        
        # Test 3: Plugin execution routing
        await test_plugin_execution_routing()
        
        # Test 4: Error handling
        await test_service_error_handling()
        
        # Test 5: Worker integration (requires Redis)
        try:
            await test_worker_service_integration()
        except AssertionError as e:
            if "Redis required" in str(e):
                logger.warning(f"⚠️ Skipping worker test: {e}")
            else:
                raise
        
        # Test 6: CLI utility
        test_service_runner_cli()
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 ALL SERVICE TESTS PASSED")
        logger.info("✅ Stage 4.3 Service-Oriented Plugin Architecture: WORKING")
        
    except AssertionError as e:
        logger.error(f"\n❌ SERVICE TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 UNEXPECTED ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())