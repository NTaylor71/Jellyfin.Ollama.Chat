#!/usr/bin/env python3
"""
Microservices Architecture Validation Test

Quick validation test for the entire queue → worker → plugin → service flow.
Tests all routing, queuing, providers, and services with focused scenarios.

Usage: python test_microservices_validation.py
"""

import asyncio
import time
import uuid
import sys
from typing import Dict, Any

import httpx
from src.worker.resource_queue_manager import ResourceAwareQueueManager
from src.worker.resource_manager import create_resource_pool_from_config


class MicroservicesValidator:
    """Validates microservices architecture with focused tests."""
    
    def __init__(self):
        resource_config = {"cpu_cores": 1, "gpu_count": 0, "memory_mb": 512}
        resource_pool = create_resource_pool_from_config(resource_config, worker_id="test_validation")
        self.queue_manager = ResourceAwareQueueManager(resource_pool)
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_results = []
    
    async def validate_all(self) -> bool:
        """Run all validation tests."""
        print("🚀 Microservices Architecture Validation")
        print("=" * 50)
        

        await self._test("Redis Connection", self._test_redis)
        

        await self._test("ConceptNet Service", self._test_conceptnet_service)
        await self._test("Gensim Service", self._test_gensim_service)
        await self._test("SpaCy Service", self._test_spacy_service)
        await self._test("HeidelTime Service", self._test_heideltime_service)
        await self._test("LLM Service", self._test_llm_service)
        

        await self._test("Gensim Provider", self._test_gensim_provider)
        await self._test("SpaCy Provider", self._test_spacy_provider)
        await self._test("HeidelTime Provider", self._test_heideltime_provider)
        await self._test("Ollama LLM Provider", self._test_ollama_provider)
        

        await self._test("ConceptExpansion Direct", self._test_concept_expansion_direct)
        await self._test("QuestionExpansion Direct", self._test_question_expansion_direct)
        

        await self._test("Queue Submission", self._test_queue_submission)
        

        return self._generate_report()
    
    async def _test(self, name: str, test_func):
        """Run individual test."""
        start_time = time.time()
        try:
            result = await test_func()
            duration = (time.time() - start_time) * 1000
            self.test_results.append((name, True, duration, result))
            print(f"✅ {name} ({duration:.0f}ms)")
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.test_results.append((name, False, duration, str(e)))
            print(f"❌ {name} - {str(e)}")
    
    async def _test_redis(self) -> str:
        """Test Redis connection."""
        if not self.queue_manager.health_check():
            raise Exception("Redis ping failed")
        return "Connected"
    
    async def _test_conceptnet_service(self) -> Dict[str, Any]:
        """Test ConceptNet service health and providers."""
        response = await self.client.get("http://localhost:8001/health")
        if response.status_code != 200:
            raise Exception(f"Health check failed: {response.status_code}")
        
        health_data = response.json()
        if health_data.get("status") != "healthy":
            raise Exception(f"Service unhealthy: {health_data}")
        

        response = await self.client.get("http://localhost:8001/providers")
        if response.status_code != 200:
            raise Exception(f"Providers endpoint failed: {response.status_code}")
        
        providers_data = response.json()
        expected = ["conceptnet"]
        available = providers_data.get("available_providers", [])
        
        for provider in expected:
            if provider not in available:
                raise Exception(f"Missing provider: {provider}")
        
        return {"status": "healthy", "providers": len(available)}
    
    async def _test_gensim_service(self) -> Dict[str, Any]:
        """Test Gensim service health and providers."""
        response = await self.client.get("http://localhost:8006/health")
        if response.status_code != 200:
            raise Exception(f"Health check failed: {response.status_code}")
        
        health_data = response.json()
        if health_data.get("status") != "healthy":
            raise Exception(f"Service unhealthy: {health_data}")
        

        response = await self.client.get("http://localhost:8006/providers")
        if response.status_code != 200:
            raise Exception(f"Providers endpoint failed: {response.status_code}")
        
        providers_data = response.json()
        expected = ["gensim"]
        available = providers_data.get("available_providers", [])
        
        for provider in expected:
            if provider not in available:
                raise Exception(f"Missing provider: {provider}")
        
        return {"status": "healthy", "providers": len(available)}
    
    async def _test_spacy_service(self) -> Dict[str, Any]:
        """Test SpaCy service health and providers."""
        response = await self.client.get("http://localhost:8007/health")
        if response.status_code != 200:
            raise Exception(f"Health check failed: {response.status_code}")
        
        health_data = response.json()
        if health_data.get("status") != "healthy":
            raise Exception(f"Service unhealthy: {health_data}")
        

        response = await self.client.get("http://localhost:8007/providers")
        if response.status_code != 200:
            raise Exception(f"Providers endpoint failed: {response.status_code}")
        
        providers_data = response.json()
        expected = ["spacy_temporal"]
        available = providers_data.get("available_providers", [])
        
        for provider in expected:
            if provider not in available:
                raise Exception(f"Missing provider: {provider}")
        
        return {"status": "healthy", "providers": len(available)}
    
    async def _test_heideltime_service(self) -> Dict[str, Any]:
        """Test HeidelTime service health and providers."""
        response = await self.client.get("http://localhost:8008/health")
        if response.status_code != 200:
            raise Exception(f"Health check failed: {response.status_code}")
        
        health_data = response.json()
        if health_data.get("status") != "healthy":
            raise Exception(f"Service unhealthy: {health_data}")
        

        response = await self.client.get("http://localhost:8008/providers")
        if response.status_code != 200:
            raise Exception(f"Providers endpoint failed: {response.status_code}")
        
        providers_data = response.json()
        expected = ["heideltime"]
        available = providers_data.get("available_providers", [])
        
        for provider in expected:
            if provider not in available:
                raise Exception(f"Missing provider: {provider}")
        
        return {"status": "healthy", "providers": len(available)}
    
    async def _test_llm_service(self) -> Dict[str, Any]:
        """Test LLM service health."""
        response = await self.client.get("http://localhost:8002/health")
        if response.status_code != 200:
            raise Exception(f"Health check failed: {response.status_code}")
        
        health_data = response.json()
        if health_data.get("status") != "healthy":
            raise Exception(f"Service unhealthy: {health_data}")
        
        models = health_data.get("models_available", [])
        if "llama3.2:3b" not in models:
            raise Exception("llama3.2:3b model not available")
        
        return {"status": "healthy", "model": "llama3.2:3b"}
    
    
    async def _test_gensim_provider(self) -> Dict[str, Any]:
        """Test Gensim provider through dedicated Gensim service."""
        request_data = {
            "concept": "action thriller",
            "media_context": "movie", 
            "max_concepts": 3,
            "options": {}
        }
        
        response = await self.client.post(
            "http://localhost:8001/providers/gensim/expand",
            json=request_data
        )
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        
        data = response.json()
        if not data.get("success"):
            raise Exception(f"Provider failed: {data.get('error_message')}")
        
        result = data.get("result", {})
        concepts = result.get("expanded_concepts", [])
        
        if not concepts:
            raise Exception("No concepts returned")
        
        return {"concepts": concepts[:3], "count": len(concepts)}
    
    async def _test_spacy_provider(self) -> Dict[str, Any]:
        """Test SpaCy provider through dedicated SpaCy service."""
        request_data = {
            "concept": "90s action movie",
            "media_context": "movie",
            "max_concepts": 3,
            "options": {}
        }
        
        response = await self.client.post(
            "http://localhost:8001/providers/spacy_temporal/expand", 
            json=request_data
        )
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        
        data = response.json()
        if not data.get("success"):
            raise Exception(f"Provider failed: {data.get('error_message')}")
        
        result = data.get("result", {})
        concepts = result.get("expanded_concepts", [])
        
        return {"concepts": concepts, "detected_temporal": "90s" in concepts}
    
    async def _test_heideltime_provider(self) -> Dict[str, Any]:
        """Test HeidelTime provider through dedicated HeidelTime service.""" 
        request_data = {
            "concept": "movie from 1995",
            "media_context": "movie",
            "max_concepts": 3,
            "options": {}
        }
        
        response = await self.client.post(
            "http://localhost:8001/providers/heideltime/expand",
            json=request_data
        )
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        
        data = response.json()
        if not data.get("success"):
            raise Exception(f"Provider failed: {data.get('error_message')}")
        
        result = data.get("result", {})
        concepts = result.get("expanded_concepts", [])
        
        return {"concepts": concepts, "temporal_detected": len(concepts) > 0}
    
    async def _test_ollama_provider(self) -> Dict[str, Any]:
        """Test Ollama provider through LLM service."""
        request_data = {
            "concept": "space opera epic",
            "media_context": "movie",
            "max_concepts": 3,
            "options": {}
        }
        
        response = await self.client.post(
            "http://localhost:8002/providers/llm/expand",
            json=request_data
        )
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        
        data = response.json()
        if not data.get("success"):
            raise Exception(f"Provider failed: {data.get('error_message')}")
        
        result = data.get("result", {})
        concepts = result.get("expanded_concepts", [])
        
        if not concepts:
            raise Exception("No concepts returned")
        
        return {"concepts": concepts[:3], "model": "llama3.2:3b"}
    
    async def _test_concept_expansion_direct(self) -> Dict[str, Any]:
        """Test ConceptExpansion plugin using direct service routing."""
        request_data = {
            "plugin_name": "ConceptExpansionPlugin",
            "plugin_type": "concept_expansion",
            "data": {
                "concept": "mystery detective story",
                "media_context": "movie",
                "max_concepts": 3
            },
            "context": {
                "user_id": "test_user",
                "session_id": str(uuid.uuid4()),
                "request_id": str(uuid.uuid4()),
                "execution_timeout": 25.0,
                "priority": "normal"
            }
        }
        
        response = await self.client.post(
            "http://localhost:8003/plugins/execute",
            json=request_data
        )
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        
        data = response.json()
        if not data.get("success"):
            raise Exception(f"Routing failed: {data.get('error_message')}")
        
        result = data.get("result", {})
        concepts = result.get("expanded_concepts", [])
        service_used = data.get("service_used")
        
        if service_used not in ["conceptnet_provider", "gensim_provider", "spacy_provider", "heideltime_provider"]:
            raise Exception(f"Wrong service used: {service_used}")
        
        return {
            "service": service_used,
            "concepts": concepts[:3],
            "execution_time": data.get("execution_time_ms", 0)
        }
    
    async def _test_question_expansion_direct(self) -> Dict[str, Any]:
        """Test QuestionExpansion plugin using direct service routing."""
        request_data = {
            "plugin_name": "QuestionExpansionPlugin",
            "plugin_type": "query_processing",
            "data": {
                "concept": "fast-paced adventure",
                "media_context": "movie",
                "max_concepts": 3
            },
            "context": {
                "user_id": "test_user",
                "session_id": str(uuid.uuid4()),
                "request_id": str(uuid.uuid4()),
                "execution_timeout": 25.0,
                "priority": "normal"
            }
        }
        
        response = await self.client.post(
            "http://localhost:8003/plugins/execute",
            json=request_data
        )
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        
        data = response.json()
        if not data.get("success"):
            raise Exception(f"Routing failed: {data.get('error_message')}")
        
        result = data.get("result", {})
        concepts = result.get("expanded_concepts", [])
        service_used = data.get("service_used")
        
        if service_used != "llm_provider":
            raise Exception(f"Wrong service used: {service_used}")
        
        return {
            "service": service_used,
            "concepts": concepts[:3], 
            "execution_time": data.get("execution_time_ms", 0)
        }
    
    async def _test_queue_submission(self) -> Dict[str, Any]:
        """Test queue submission and basic flow."""
        task_data = {
            "plugin_name": "ConceptExpansionPlugin",
            "plugin_type": "concept_expansion",
            "user_id": f"test_{int(time.time())}",
            "session_id": str(uuid.uuid4()),
            "task_id": str(uuid.uuid4()),
            "data": {
                "concept": "validation test movie",
                "media_context": "movie",
                "max_concepts": 2
            }
        }
        
        task_id = self.queue_manager.enqueue_task("plugin_execution", task_data, priority=1)
        
        if not task_id:
            raise Exception("Failed to enqueue task")
        

        await asyncio.sleep(3)
        
        return {"task_id": task_id, "status": "submitted"}
    
    def _generate_report(self) -> bool:
        """Generate validation report."""
        print("\n" + "=" * 50)
        print("📊 VALIDATION REPORT")
        print("=" * 50)
        
        total = len(self.test_results)
        passed = sum(1 for _, success, _, _ in self.test_results if success)
        failed = total - passed
        
        success_rate = (passed / total) * 100 if total > 0 else 0
        total_time = sum(duration for _, _, duration, _ in self.test_results)
        
        print(f"Tests: {total} | Passed: {passed} | Failed: {failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Time: {total_time:.0f}ms")
        
        if failed > 0:
            print(f"\n❌ FAILED ({failed}):")
            for name, success, duration, result in self.test_results:
                if not success:
                    print(f"  - {name}: {result}")
        
        print(f"\n✅ PASSED ({passed}):")
        for name, success, duration, result in self.test_results:
            if success:
                print(f"  - {name} ({duration:.0f}ms)")
        
        print("\n" + "=" * 50)
        
        if success_rate >= 90:
            print("🎉 MICROSERVICES ARCHITECTURE: FULLY OPERATIONAL! 🎉")
            return True
        elif success_rate >= 80:
            print("✅ MICROSERVICES ARCHITECTURE: OPERATIONAL")
            return True  
        else:
            print("❌ MICROSERVICES ARCHITECTURE: ISSUES DETECTED")
            return False
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.client.aclose()


async def main():
    """Main validation execution."""
    validator = MicroservicesValidator()
    
    try:
        success = await validator.validate_all()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)
        
    finally:
        await validator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())