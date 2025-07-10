#!/usr/bin/env python3
"""
Docker Service Integration Test - FAIL FAST Approach
Tests the complete microservices architecture through Docker.

HARD RULE 12: Never fix test conditions to make a failing test pass - fix actual bugs.

This test builds and validates:
1. Service Docker images build successfully
2. Services start up in correct order
3. Health checks pass for all services
4. Inter-service communication works
5. End-to-end plugin execution through services

NO FALLBACKS - If any service fails, this should fail immediately with clear error messages.
"""

import asyncio
import subprocess
import sys
import time
import httpx
import json
from pathlib import Path
from tests.tests_shared import logger, settings_to_console


class DockerServiceManager:
    """Manages Docker services for testing."""
    
    def __init__(self):
        try:

            result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise AssertionError("Docker command failed")
            logger.info(f"✅ Docker available: {result.stdout.strip()}")
        except Exception as e:
            raise AssertionError(f"❌ Docker not available: {e}")
        
        self.services = [
            "redis",
            "mongodb", 
            "conceptnet-service",
            "gensim-service",
            "spacy-service",
            "heideltime-service",
            "llm-service",
            "router-service",
            "worker"
        ]
        
        self.service_urls = {
            "conceptnet-service": "http://localhost:8001",
            "gensim-service": "http://localhost:8006",
            "spacy-service": "http://localhost:8007",
            "heideltime-service": "http://localhost:8008",
            "llm-service": "http://localhost:8002", 
            "router-service": "http://localhost:8003"
        }
    
    def cleanup_containers(self):
        """Remove any existing containers."""
        logger.info("🧹 Cleaning up existing containers...")
        
        try:

            result = subprocess.run([
                "docker", "compose", "-f", "docker-compose.dev.yml", "down", "-v"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info("✅ Containers cleaned up")
            else:
                logger.warning(f"⚠️ Cleanup warning: {result.stderr}")
        
        except Exception as e:
            logger.warning(f"⚠️ Cleanup error: {e}")
    
    def build_services(self):
        """Build all service images."""
        logger.info("🔨 Building service images...")
        
        try:

            services_to_build = ["conceptnet-service", "gensim-service", "spacy-service", "heideltime-service", "llm-service", "router-service"]
            
            for service in services_to_build:
                logger.info(f"🔨 Building {service}...")
                
                result = subprocess.run([
                    "docker", "compose", "-f", "docker-compose.dev.yml", 
                    "build", service
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    raise AssertionError(f"❌ Failed to build {service}: {result.stderr}")
                
                logger.info(f"✅ {service} built successfully")
            
            logger.info("✅ All service images built")
            
        except subprocess.TimeoutExpired:
            raise AssertionError("❌ Service build timed out")
        except Exception as e:
            raise AssertionError(f"❌ Service build failed: {e}")
    
    def start_core_services(self):
        """Start core infrastructure services (Redis, MongoDB)."""
        logger.info("🚀 Starting core services...")
        
        try:
            result = subprocess.run([
                "docker", "compose", "-f", "docker-compose.dev.yml",
                "up", "-d", "redis", "mongodb"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                raise AssertionError(f"❌ Failed to start core services: {result.stderr}")
            

            logger.info("⏳ Waiting for core services to be healthy...")
            time.sleep(10)
            

            for service in ["redis", "mongodb"]:
                self._wait_for_container_health(service)
            
            logger.info("✅ Core services started and healthy")
            
        except Exception as e:
            raise AssertionError(f"❌ Core services startup failed: {e}")
    
    def start_microservices(self):
        """Start microservices in dependency order."""
        logger.info("🚀 Starting microservices...")
        
        try:

            service_order = ["conceptnet-service", "gensim-service", "spacy-service", "heideltime-service", "llm-service", "router-service"]
            
            for service in service_order:
                logger.info(f"🚀 Starting {service}...")
                
                result = subprocess.run([
                    "docker", "compose", "-f", "docker-compose.dev.yml",
                    "up", "-d", service
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode != 0:
                    raise AssertionError(f"❌ Failed to start {service}: {result.stderr}")
                

                logger.info(f"⏳ Waiting for {service} to be healthy...")
                time.sleep(15)
                
                self._wait_for_container_health(service, timeout=120)
                logger.info(f"✅ {service} started and healthy")
            
            logger.info("✅ All microservices started")
            
        except Exception as e:
            raise AssertionError(f"❌ Microservices startup failed: {e}")
    
    def _wait_for_container_health(self, service_name: str, timeout: int = 60):
        """Wait for a container to be healthy."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:

                result = subprocess.run([
                    "docker", "compose", "-f", "docker-compose.dev.yml",
                    "ps", service_name
                ], capture_output=True, text=True)
                
                if "healthy" in result.stdout:
                    return True
                elif "unhealthy" in result.stdout:
                    raise AssertionError(f"❌ {service_name} reported unhealthy")
                
                time.sleep(5)
                
            except Exception as e:
                logger.warning(f"⚠️ Health check error for {service_name}: {e}")
                time.sleep(5)
        
        raise AssertionError(f"❌ {service_name} failed to become healthy within {timeout}s")
    
    async def test_service_endpoints(self):
        """Test all service endpoints are responding."""
        logger.info("🧪 Testing service endpoints...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for service_name, url in self.service_urls.items():
                try:

                    response = await client.get(f"{url}/health")
                    
                    if response.status_code != 200:
                        raise AssertionError(f"❌ {service_name} health check failed: HTTP {response.status_code}")
                    
                    health_data = response.json()
                    logger.info(f"✅ {service_name} health: {health_data.get('status', 'unknown')}")
                    
                except httpx.ConnectError:
                    raise AssertionError(f"❌ {service_name} not accessible at {url}")
                except Exception as e:
                    raise AssertionError(f"❌ {service_name} endpoint test failed: {e}")
    
    async def test_service_communication(self):
        """Test inter-service communication."""
        logger.info("🧪 Testing service communication...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:

            router_url = self.service_urls["router-service"]
            
            try:

                response = await client.post(f"{router_url}/services/health")
                
                if response.status_code != 200:
                    raise AssertionError(f"❌ Router health check failed: HTTP {response.status_code}")
                
                health_data = response.json()
                services = health_data.get("services", {})
                
                for service_name, service_data in services.items():
                    status = service_data.get("status", "unknown")
                    if status != "healthy":
                        raise AssertionError(f"❌ Router reports {service_name} unhealthy: {status}")
                    logger.info(f"✅ Router → {service_name}: {status}")
                
            except Exception as e:
                raise AssertionError(f"❌ Service communication test failed: {e}")
    
    async def test_plugin_execution(self):
        """Test end-to-end plugin execution through services."""
        logger.info("🧪 Testing plugin execution through services...")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            router_url = self.service_urls["router-service"]
            

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
                response = await client.post(
                    f"{router_url}/execute",
                    json=test_request,
                    timeout=60.0
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
                
            except httpx.TimeoutException:
                raise AssertionError("❌ Plugin execution timed out")
            except Exception as e:
                raise AssertionError(f"❌ Plugin execution failed: {e}")
    
    def get_service_logs(self, service_name: str, tail: int = 50):
        """Get logs from a service for debugging."""
        try:
            result = subprocess.run([
                "docker", "compose", "-f", "docker-compose.dev.yml",
                "logs", "--tail", str(tail), service_name
            ], capture_output=True, text=True, timeout=30)
            
            return result.stdout
            
        except Exception as e:
            return f"Error getting logs: {e}"
    
    def cleanup(self):
        """Cleanup all resources."""
        logger.info("🧹 Cleaning up Docker services...")
        self.cleanup_containers()


async def test_docker_microservices():
    """Main test function for Docker microservices."""
    logger.info("🐳 Starting Docker Microservices Test")
    logger.info("=" * 60)
    
    manager = DockerServiceManager()
    
    try:

        manager.cleanup_containers()
        

        manager.build_services()
        

        manager.start_core_services()
        

        manager.start_microservices()
        

        await manager.test_service_endpoints()
        

        await manager.test_service_communication()
        

        await manager.test_plugin_execution()
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 ALL DOCKER SERVICE TESTS PASSED")
        logger.info("✅ Stage 4.3 Microservices Architecture: FULLY WORKING")
        
    except AssertionError as e:
        logger.error(f"\n❌ DOCKER SERVICE TEST FAILED: {e}")
        

        logger.error("\n📋 Service logs for debugging:")
        for service in ["conceptnet-service", "gensim-service", "spacy-service", "heideltime-service", "llm-service", "router-service"]:
            logger.error(f"\n--- {service} logs ---")
            logs = manager.get_service_logs(service)
            logger.error(logs)
        
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"\n💥 UNEXPECTED ERROR: {e}")
        sys.exit(1)
    
    finally:

        manager.cleanup()


def main():
    """Main entry point."""
    logger.info("🚀 Docker Service Integration Test")
    logger.info("Testing Stage 4.3 Service-Oriented Plugin Architecture")
    

    settings_to_console()
    

    asyncio.run(test_docker_microservices())


if __name__ == "__main__":
    main()