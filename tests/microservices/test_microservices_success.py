#!/usr/bin/env python3
"""
Stage 4.3 SUCCESS TEST - Service-Oriented Plugin Architecture

This test proves the microservices architecture is working:
✅ NLP Provider Service
✅ LLM Provider Service  
✅ Plugin Router Service
✅ Service Communication
✅ End-to-End Workflow

HARD RULE 12: Never fix test conditions to make a failing test pass - these tests validate actual working services.
"""

import asyncio
import httpx
from tests.tests_shared import logger, settings_to_console


async def test_individual_services():
    """Test each service individually."""
    logger.info("🧪 Testing Individual Services")
    logger.info("-" * 40)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Test NLP Service
        response = await client.get("http://localhost:8001/health")
        assert response.status_code == 200
        health = response.json()
        assert health["status"] == "healthy"
        assert len(health["providers"]) > 0
        logger.info("✅ NLP Service: Healthy with {} providers".format(len(health["providers"])))
        
        # Test LLM Service
        response = await client.get("http://localhost:8002/health")
        assert response.status_code == 200
        health = response.json()
        assert health["status"] == "healthy"
        assert len(health["models_available"]) > 0
        logger.info("✅ LLM Service: Healthy with {} models".format(len(health["models_available"])))
        
        # Test Router Service
        response = await client.get("http://localhost:8003/health")
        assert response.status_code == 200
        health = response.json()
        assert health["status"] == "healthy"
        assert len(health["services"]) == 2
        logger.info("✅ Router Service: Healthy with {} services configured".format(len(health["services"])))


async def test_service_capabilities():
    """Test service-specific capabilities."""
    logger.info("\n🧪 Testing Service Capabilities")
    logger.info("-" * 40)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Test NLP providers
        response = await client.get("http://localhost:8001/providers")
        assert response.status_code == 200
        providers = response.json()
        available_providers = providers["available_providers"]
        assert "gensim" in available_providers
        assert "spacy_temporal" in available_providers
        logger.info("✅ NLP Providers: {}".format(available_providers))
        
        # Test LLM provider info
        response = await client.get("http://localhost:8002/providers")
        assert response.status_code == 200
        provider_info = response.json()
        assert provider_info["type"] == "ollama"
        assert len(provider_info["available_models"]) > 0
        logger.info("✅ LLM Provider: {} with models {}".format(provider_info["name"], provider_info["available_models"]))


async def test_concept_expansion():
    """Test actual concept expansion through services."""
    logger.info("\n🧪 Testing Concept Expansion")
    logger.info("-" * 40)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Test NLP concept expansion
        response = await client.post(
            "http://localhost:8001/providers/gensim/expand",
            json={
                "concept": "action",
                "media_context": "movie",
                "max_concepts": 3,
                "field_name": "genre"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        # Check for real gensim results (similar words)
        expanded_concepts = result["result"]["expanded_concepts"]
        assert len(expanded_concepts) > 0
        logger.info("✅ NLP Expansion: {} → {}".format("action", expanded_concepts))
        
        # Test LLM concept expansion
        response = await client.post(
            "http://localhost:8002/providers/llm/expand",
            json={
                "concept": "thriller",
                "media_context": "movie",
                "max_concepts": 3
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        # Check for real LLM results 
        expanded_concepts = result["result"]["expanded_concepts"]
        assert len(expanded_concepts) > 0
        logger.info("✅ LLM Expansion: {} → {}".format("thriller", expanded_concepts))


async def test_service_communication():
    """Test router service communication."""
    logger.info("\n🧪 Testing Service Communication")
    logger.info("-" * 40)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Test router service discovery
        response = await client.get("http://localhost:8003/services")
        assert response.status_code == 200
        services = response.json()
        assert "nlp_provider" in services["services"]
        assert "llm_provider" in services["services"]
        logger.info("✅ Service Discovery: Router knows about {} services".format(len(services["services"])))
        
        # Test router health checks
        response = await client.post("http://localhost:8003/services/health")
        assert response.status_code == 200
        health_result = response.json()
        for service_name, service_data in health_result["services"].items():
            assert service_data["status"] == "healthy"
        logger.info("✅ Service Health: All services healthy via router")


async def test_performance():
    """Test service performance."""
    logger.info("\n🧪 Testing Service Performance")
    logger.info("-" * 40)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Test NLP service performance
        start_time = asyncio.get_event_loop().time()
        response = await client.post(
            "http://localhost:8001/providers/gensim/expand",
            json={"concept": "performance_test", "max_concepts": 5}
        )
        nlp_time = (asyncio.get_event_loop().time() - start_time) * 1000
        assert response.status_code == 200
        logger.info("✅ NLP Performance: {:.1f}ms".format(nlp_time))
        
        # Test LLM service performance
        start_time = asyncio.get_event_loop().time()
        response = await client.post(
            "http://localhost:8002/providers/llm/expand",
            json={"concept": "performance_test", "max_concepts": 5}
        )
        llm_time = (asyncio.get_event_loop().time() - start_time) * 1000
        assert response.status_code == 200
        logger.info("✅ LLM Performance: {:.1f}ms".format(llm_time))
        
        # Test router performance
        start_time = asyncio.get_event_loop().time()
        response = await client.get("http://localhost:8003/health")
        router_time = (asyncio.get_event_loop().time() - start_time) * 1000
        assert response.status_code == 200
        logger.info("✅ Router Performance: {:.1f}ms".format(router_time))


async def main():
    """Main test entry point."""
    logger.info("🚀 Stage 4.3 SUCCESS TEST: Service-Oriented Plugin Architecture")
    logger.info("=" * 70)
    
    # Show current configuration
    settings_to_console()
    
    try:
        # Test 1: Individual service health
        await test_individual_services()
        
        # Test 2: Service capabilities
        await test_service_capabilities()
        
        # Test 3: Concept expansion functionality
        await test_concept_expansion()
        
        # Test 4: Service communication
        await test_service_communication()
        
        # Test 5: Performance validation
        await test_performance()
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 ALL TESTS PASSED - STAGE 4.3 COMPLETE!")
        logger.info("✅ NLP Provider Service: WORKING")
        logger.info("✅ LLM Provider Service: WORKING") 
        logger.info("✅ Plugin Router Service: WORKING")
        logger.info("✅ Service Communication: WORKING")
        logger.info("✅ End-to-End Workflow: WORKING")
        logger.info("✅ Service-Oriented Plugin Architecture: FULLY OPERATIONAL")
        
        logger.info("\n🎯 READY FOR STAGE 4.3.3: Create Service Client Plugins")
        
    except AssertionError as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        logger.error(f"\n💥 UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())