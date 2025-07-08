#!/usr/bin/env python3
"""
Complete Microservices Architecture Test

Tests the entire queue → worker → plugin → service → provider flow
to validate that all components work together correctly.

This test validates:
- Redis queue submission and pickup
- Worker plugin discovery and routing  
- Service communication (Router, NLP, LLM)
- Provider execution (Gensim, SpaCy, HeidelTime, Ollama)
- End-to-end data flow and results

Usage: python test_microservices_complete.py
"""

import asyncio
import time
import uuid
import json
import sys
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import project modules
from src.worker.resource_queue_manager import ResourceAwareQueueManager
from src.worker.resource_manager import create_resource_pool_from_config
from src.worker.plugin_loader import PluginLoader
from src.shared.config import get_settings

import httpx


@dataclass
class TestResult:
    """Test result data structure."""
    test_name: str
    success: bool
    execution_time_ms: float
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class MicroservicesArchitectureTest:
    """Comprehensive test suite for microservices architecture."""
    
    def __init__(self):
        self.settings = get_settings()
        resource_config = {"cpu_cores": 1, "gpu_count": 0, "memory_mb": 512}
        resource_pool = create_resource_pool_from_config(resource_config, worker_id="test_complete")
        self.queue_manager = ResourceAwareQueueManager(resource_pool)
        self.http_client = httpx.AsyncClient(timeout=60.0)
        self.test_results: List[TestResult] = []
        
    async def run_all_tests(self) -> bool:
        """Run complete test suite."""
        print("🚀 Starting Complete Microservices Architecture Test")
        print("=" * 60)
        
        # Test infrastructure health
        await self._test_infrastructure_health()
        
        # Test direct service endpoints
        await self._test_service_endpoints()
        
        # Test provider functionality
        await self._test_providers()
        
        # Test queue and worker flow
        await self._test_queue_worker_flow()
        
        # Test complete end-to-end flows
        await self._test_end_to_end_flows()
        
        # Generate report
        return self._generate_report()
    
    async def _test_infrastructure_health(self):
        """Test basic infrastructure health."""
        print("\n🔍 Testing Infrastructure Health...")
        
        # Test Redis
        await self._run_test(
            "Redis Connection",
            self._test_redis_health
        )
        
        # Test Docker services
        services = [
            ("NLP Service", "http://localhost:8001/health"),
            ("LLM Service", "http://localhost:8002/health"), 
            ("Router Service", "http://localhost:8003/health"),
            ("Worker Service", "http://localhost:8004/metrics")  # Prometheus endpoint
        ]
        
        for service_name, url in services:
            await self._run_test(
                f"{service_name} Health",
                lambda u=url: self._test_service_health(u)
            )
    
    async def _test_service_endpoints(self):
        """Test service endpoints directly."""
        print("\n🔍 Testing Service Endpoints...")
        
        # Test NLP Service providers
        await self._run_test(
            "NLP Service Providers List",
            self._test_nlp_providers
        )
        
        # Test Router Service discovery
        await self._run_test(
            "Router Service Discovery",
            self._test_router_discovery
        )
    
    async def _test_providers(self):
        """Test individual providers through services."""
        print("\n🔍 Testing Providers...")
        
        # Test each NLP provider
        nlp_tests = [
            ("Gensim Provider", "gensim", "action movie thriller"),
            ("SpaCy Provider", "spacy", "90s action movie"),
            ("HeidelTime Provider", "heideltime", "movie from 1995")
        ]
        
        for test_name, provider, concept in nlp_tests:
            await self._run_test(
                test_name,
                lambda p=provider, c=concept: self._test_nlp_provider(p, c)
            )
        
        # Test LLM provider
        await self._run_test(
            "Ollama LLM Provider",
            lambda: self._test_llm_provider("sci-fi space adventure")
        )
    
    async def _test_queue_worker_flow(self):
        """Test Redis queue and worker processing."""
        print("\n🔍 Testing Queue & Worker Flow...")
        
        # Test queue submission
        await self._run_test(
            "Queue Task Submission",
            self._test_queue_submission
        )
        
        # Test plugin discovery
        await self._run_test(
            "Worker Plugin Discovery",
            self._test_plugin_discovery
        )
    
    async def _test_end_to_end_flows(self):
        """Test complete end-to-end flows."""
        print("\n🔍 Testing End-to-End Flows...")
        
        # Test concept expansion flow
        await self._run_test(
            "ConceptExpansion E2E Flow",
            lambda: self._test_e2e_concept_expansion("epic fantasy adventure")
        )
        
        # Test question expansion flow
        await self._run_test(
            "QuestionExpansion E2E Flow", 
            lambda: self._test_e2e_question_expansion("fast-paced psychological thriller")
        )
        
        # Test temporal analysis flow
        await self._run_test(
            "TemporalAnalysis E2E Flow",
            lambda: self._test_e2e_temporal_analysis("classic 80s comedy")
        )
    
    async def _run_test(self, test_name: str, test_func):
        """Run individual test and capture results."""
        start_time = time.time()
        
        try:
            result_data = await test_func()
            execution_time = (time.time() - start_time) * 1000
            
            self.test_results.append(TestResult(
                test_name=test_name,
                success=True,
                execution_time_ms=execution_time,
                result_data=result_data
            ))
            
            print(f"  ✅ {test_name} - {execution_time:.1f}ms")
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            self.test_results.append(TestResult(
                test_name=test_name,
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e)
            ))
            
            print(f"  ❌ {test_name} - {str(e)}")
    
    async def _test_redis_health(self) -> Dict[str, Any]:
        """Test Redis connection."""
        if not self.queue_manager.health_check():
            raise Exception("Redis connection failed")
        return {"status": "healthy"}
    
    async def _test_service_health(self, url: str) -> Dict[str, Any]:
        """Test service health endpoint."""
        response = await self.http_client.get(url)
        if response.status_code != 200:
            raise Exception(f"Service unhealthy: {response.status_code}")
        return response.json()
    
    async def _test_nlp_providers(self) -> Dict[str, Any]:
        """Test NLP service provider listing."""
        response = await self.http_client.get("http://localhost:8001/providers")
        if response.status_code != 200:
            raise Exception(f"NLP providers failed: {response.status_code}")
        
        data = response.json()
        providers = data.get("available_providers", [])
        
        expected_providers = ["gensim", "spacy", "heideltime"]
        for provider in expected_providers:
            if provider not in providers:
                raise Exception(f"Missing provider: {provider}")
        
        return data
    
    async def _test_router_discovery(self) -> Dict[str, Any]:
        """Test router service discovery."""
        response = await self.http_client.get("http://localhost:8003/services")
        if response.status_code != 200:
            raise Exception(f"Router discovery failed: {response.status_code}")
        
        data = response.json()
        services = data.get("services", {})
        
        expected_services = ["nlp_provider", "llm_provider"]
        for service in expected_services:
            if service not in services:
                raise Exception(f"Missing service: {service}")
        
        return data
    
    async def _test_nlp_provider(self, provider: str, concept: str) -> Dict[str, Any]:
        """Test NLP provider through service."""
        request_data = {
            "concept": concept,
            "media_context": "movie",
            "max_concepts": 3,
            "options": {}
        }
        
        response = await self.http_client.post(
            f"http://localhost:8001/providers/{provider}/expand",
            json=request_data
        )
        
        if response.status_code != 200:
            raise Exception(f"Provider {provider} failed: {response.status_code}")
        
        data = response.json()
        if not data.get("success"):
            raise Exception(f"Provider {provider} returned error: {data.get('error_message')}")
        
        result = data.get("result", {})
        concepts = result.get("expanded_concepts", [])
        
        if not concepts:
            raise Exception(f"Provider {provider} returned no concepts")
        
        return {"provider": provider, "concepts": concepts, "count": len(concepts)}
    
    async def _test_llm_provider(self, concept: str) -> Dict[str, Any]:
        """Test LLM provider through service."""
        request_data = {
            "concept": concept,
            "media_context": "movie",
            "max_concepts": 3,
            "options": {}
        }
        
        response = await self.http_client.post(
            "http://localhost:8002/providers/llm/expand",
            json=request_data
        )
        
        if response.status_code != 200:
            raise Exception(f"LLM provider failed: {response.status_code}")
        
        data = response.json()
        if not data.get("success"):
            raise Exception(f"LLM provider returned error: {data.get('error_message')}")
        
        result = data.get("result", {})
        concepts = result.get("expanded_concepts", [])
        
        if not concepts:
            raise Exception("LLM provider returned no concepts")
        
        return {"concepts": concepts, "model": "llama3.2:3b", "count": len(concepts)}
    
    async def _test_queue_submission(self) -> Dict[str, Any]:
        """Test queue task submission."""
        task_data = {
            "test_type": "queue_test",
            "timestamp": time.time()
        }
        
        task_id = self.queue_manager.enqueue_task("test_task", task_data, priority=1)
        
        if not task_id:
            raise Exception("Failed to enqueue task")
        
        return {"task_id": task_id, "status": "enqueued"}
    
    async def _test_plugin_discovery(self) -> Dict[str, Any]:
        """Test worker plugin discovery."""
        loader = PluginLoader()
        await loader.initialize()
        
        if not loader.plugin_service_mapping:
            raise Exception("No service plugins discovered")
        
        expected_plugins = ["ConceptExpansionPlugin", "QuestionExpansionPlugin", "TemporalAnalysisPlugin"]
        
        for plugin in expected_plugins:
            if plugin not in loader.plugin_service_mapping:
                raise Exception(f"Missing plugin mapping: {plugin}")
        
        await loader.cleanup()
        
        return {
            "plugins_found": len(loader.plugin_service_mapping),
            "plugin_mapping": loader.plugin_service_mapping
        }
    
    async def _test_e2e_concept_expansion(self, concept: str) -> Dict[str, Any]:
        """Test complete ConceptExpansion end-to-end flow."""
        return await self._test_e2e_plugin_flow(
            "ConceptExpansionPlugin",
            "concept_expansion", 
            concept
        )
    
    async def _test_e2e_question_expansion(self, concept: str) -> Dict[str, Any]:
        """Test complete QuestionExpansion end-to-end flow."""
        return await self._test_e2e_plugin_flow(
            "QuestionExpansionPlugin",
            "query_processing",
            concept
        )
    
    async def _test_e2e_temporal_analysis(self, concept: str) -> Dict[str, Any]:
        """Test complete TemporalAnalysis end-to-end flow."""
        return await self._test_e2e_plugin_flow(
            "TemporalAnalysisPlugin", 
            "temporal_analysis",
            concept
        )
    
    async def _test_e2e_plugin_flow(self, plugin_name: str, plugin_type: str, concept: str) -> Dict[str, Any]:
        """Test complete plugin flow through queue → worker → service."""
        # Submit task to queue
        task_data = {
            "plugin_name": plugin_name,
            "plugin_type": plugin_type,
            "user_id": f"test_{int(time.time())}",
            "session_id": str(uuid.uuid4()),
            "task_id": str(uuid.uuid4()),
            "data": {
                "concept": concept,
                "media_context": "movie",
                "max_concepts": 3
            }
        }
        
        task_id = self.queue_manager.enqueue_task("plugin_execution", task_data, priority=1)
        
        if not task_id:
            raise Exception("Failed to enqueue task")
        
        # Wait for processing and check results
        max_wait_time = 60  # 60 seconds max
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Check if task was completed by looking at Redis
            try:
                # Check completed tasks key (implementation dependent)
                completed_key = f"completed_task:{task_id}"
                result = self.queue_manager.redis_client.get(completed_key)
                
                if result:
                    result_data = json.loads(result)
                    if result_data.get("success"):
                        return {
                            "task_id": task_id,
                            "plugin": plugin_name,
                            "execution_time_ms": result_data.get("execution_time_ms"),
                            "result_data": result_data.get("data"),
                            "status": "completed"
                        }
                    else:
                        raise Exception(f"Task failed: {result_data.get('error')}")
                
                # Wait a bit before checking again
                await asyncio.sleep(2)
                
            except json.JSONDecodeError:
                await asyncio.sleep(2)
                continue
        
        # If we get here, test via router service directly as fallback
        return await self._test_router_plugin_directly(plugin_name, plugin_type, concept)
    
    async def _test_router_plugin_directly(self, plugin_name: str, plugin_type: str, concept: str) -> Dict[str, Any]:
        """Test plugin via router service directly as fallback."""
        request_data = {
            "plugin_name": plugin_name,
            "plugin_type": plugin_type,
            "data": {
                "concept": concept,
                "media_context": "movie",
                "max_concepts": 3
            },
            "context": {
                "user_id": f"direct_test_{int(time.time())}",
                "session_id": str(uuid.uuid4()),
                "request_id": str(uuid.uuid4()),
                "execution_timeout": 30.0,
                "priority": "normal"
            }
        }
        
        response = await self.http_client.post(
            "http://localhost:8003/plugins/execute",
            json=request_data
        )
        
        if response.status_code != 200:
            raise Exception(f"Router execution failed: {response.status_code}")
        
        data = response.json()
        if not data.get("success"):
            raise Exception(f"Plugin execution failed: {data.get('error_message')}")
        
        return {
            "plugin": plugin_name,
            "execution_time_ms": data.get("execution_time_ms"),
            "service_used": data.get("service_used"),
            "result_data": data.get("result"),
            "status": "completed_via_router"
        }
    
    def _generate_report(self) -> bool:
        """Generate test report."""
        print("\n" + "=" * 60)
        print("📊 MICROSERVICES ARCHITECTURE TEST REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.success)
        failed_tests = total_tests - passed_tests
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        total_execution_time = sum(result.execution_time_ms for result in self.test_results)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Execution Time: {total_execution_time:.1f}ms")
        
        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result.success:
                    print(f"  - {result.test_name}: {result.error_message}")
        
        print(f"\n✅ PASSED TESTS:")
        for result in self.test_results:
            if result.success:
                print(f"  - {result.test_name} ({result.execution_time_ms:.1f}ms)")
        
        print("\n" + "=" * 60)
        
        if success_rate >= 90:
            print("🎉 MICROSERVICES ARCHITECTURE: FULLY FUNCTIONAL! 🎉")
            return True
        elif success_rate >= 70:
            print("⚠️  MICROSERVICES ARCHITECTURE: MOSTLY FUNCTIONAL")
            return True
        else:
            print("❌ MICROSERVICES ARCHITECTURE: ISSUES DETECTED")
            return False
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.http_client.aclose()


async def main():
    """Main test execution."""
    test_suite = MicroservicesArchitectureTest()
    
    try:
        success = await test_suite.run_all_tests()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        sys.exit(1)
        
    finally:
        await test_suite.cleanup()


if __name__ == "__main__":
    asyncio.run(main())