"""
Test script for LLM provider Stage 3.2 implementation.
Tests the LLM concept expansion system with Ollama backend.
"""

import asyncio
import logging
from typing import List

from src.shared.concept_expander import get_concept_expander, ExpansionMethod
from src.providers.llm.llm_provider import LLMProvider
from src.providers.llm.ollama_backend_client import OllamaBackendClient
from src.providers.nlp.base_provider import ExpansionRequest
from src.api.cache_admin import clear_test_cache, print_cache_summary

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_ollama_backend_client():
    """Test the Ollama backend client directly."""
    print("🧪 Testing Ollama Backend Client...")
    
    client = OllamaBackendClient()
    
    try:
        # Test initialization
        print("🔧 Initializing Ollama client...")
        success = await client.initialize()
        
        if not success:
            print("❌ Failed to initialize Ollama client")
            print("   Make sure Ollama is running and the model is available")
            return False
        
        print("✅ Ollama client initialized successfully")
        
        # Test health check
        health = await client.health_check()
        print(f"🏥 Health status: {health['status']}")
        
        if health['status'] != 'healthy':
            print(f"⚠️  Health check issues: {health}")
            return False
        
        # Test model info
        model_info = client.get_model_info()
        print(f"🤖 Model: {model_info['name']} (Backend: {model_info['backend']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Ollama backend test failed: {e}")
        return False
    finally:
        await client.close()


async def test_llm_provider():
    """Test the LLM provider."""
    print("\n🧪 Testing LLM Provider...")
    
    provider = LLMProvider()
    
    try:
        # Test initialization
        print("🔧 Initializing LLM provider...")
        success = await provider.initialize()
        
        if not success:
            print("❌ Failed to initialize LLM provider")
            return False
        
        print("✅ LLM provider initialized successfully")
        
        # Test provider metadata
        metadata = provider.metadata
        print(f"📋 Provider: {metadata.name} (Type: {metadata.provider_type})")
        print(f"   Context-aware: {metadata.context_aware}")
        print(f"   Strengths: {', '.join(metadata.strengths[:3])}...")
        
        # Test concept expansion
        print("🎬 Testing concept expansion: 'action' movie")
        request = ExpansionRequest(
            concept="action",
            media_context="movie",
            max_concepts=5,
            field_name="genre"
        )
        
        result = await provider.expand_concept(request)
        
        if result and result.success:
            expanded_concepts = result.enhanced_data.get("expanded_concepts", [])
            confidence_scores = result.confidence_score.per_item
            
            print(f"✅ Success! Expanded 'action' into {len(expanded_concepts)} concepts:")
            for concept in expanded_concepts:
                confidence = confidence_scores.get(concept, 0.0)
                print(f"   • {concept} (confidence: {confidence:.2f})")
            
            print(f"📊 Execution time: {result.plugin_metadata.execution_time_ms:.1f}ms")
            print(f"🔧 Backend: {result.enhanced_data.get('backend', 'unknown')}")
            print(f"🤖 Model: {result.enhanced_data.get('model', 'unknown')}")
            
            return True
        else:
            print("❌ Concept expansion failed")
            if result:
                print(f"   Error: {result.error_message}")
            return False
        
    except Exception as e:
        print(f"❌ LLM provider test failed: {e}")
        return False
    finally:
        await provider.close()


async def test_concept_expander_llm():
    """Test ConceptExpander with LLM method."""
    print("\n🧪 Testing ConceptExpander with LLM method...")
    
    expander = get_concept_expander()
    
    try:
        # Test LLM capabilities
        llm_caps = expander.get_method_capabilities(ExpansionMethod.LLM)
        print(f"🧠 LLM capabilities (type: {llm_caps['type']}, context-aware: {llm_caps['context_aware']})")
        
        # Test method recommendation
        recommended = expander.get_recommended_method("psychological thriller", "movie")
        print(f"💡 Recommended method for 'psychological thriller': {recommended.value}")
        
        # Test concept expansion
        print("🎬 Testing: 'psychological thriller' + 'movie' context")
        result = await expander.expand_concept(
            concept="psychological thriller",
            media_context="movie",
            method=ExpansionMethod.LLM,
            max_concepts=8
        )
        
        if result and result.success:
            expanded_concepts = result.enhanced_data.get("expanded_concepts", [])
            confidence_scores = result.confidence_score.per_item
            
            print(f"✅ Success! Expanded 'psychological thriller' into {len(expanded_concepts)} concepts:")
            for concept in expanded_concepts:
                confidence = confidence_scores.get(concept, 0.0)
                print(f"   • {concept} (confidence: {confidence:.2f})")
            
            print(f"📊 Total execution time: {result.plugin_metadata.execution_time_ms:.1f}ms")
            print(f"🗃️ Cache key: {result.cache_key.generate_key()}")
            
            return True
        else:
            print("❌ Concept expansion failed")
            return False
            
    except Exception as e:
        print(f"❌ ConceptExpander LLM test failed: {e}")
        return False


async def test_llm_vs_conceptnet():
    """Compare LLM and ConceptNet results for the same concept."""
    print("\n🧪 Testing LLM vs ConceptNet comparison...")
    
    expander = get_concept_expander()
    concept = "action"
    media_context = "movie"
    
    try:
        # Test ConceptNet
        print("🔍 ConceptNet expansion:")
        conceptnet_result = await expander.expand_concept(
            concept=concept,
            media_context=media_context,
            method=ExpansionMethod.CONCEPTNET,
            max_concepts=5
        )
        
        if conceptnet_result and conceptnet_result.success:
            conceptnet_concepts = conceptnet_result.enhanced_data.get("expanded_concepts", [])
            print(f"   {', '.join(conceptnet_concepts)}")
            print(f"   Time: {conceptnet_result.plugin_metadata.execution_time_ms:.1f}ms")
        else:
            print("   ❌ Failed")
        
        # Test LLM
        print("🧠 LLM expansion:")
        llm_result = await expander.expand_concept(
            concept=concept,
            media_context=media_context,
            method=ExpansionMethod.LLM,
            max_concepts=5
        )
        
        if llm_result and llm_result.success:
            llm_concepts = llm_result.enhanced_data.get("expanded_concepts", [])
            print(f"   {', '.join(llm_concepts)}")
            print(f"   Time: {llm_result.plugin_metadata.execution_time_ms:.1f}ms")
        else:
            print("   ❌ Failed")
        
        # Compare results
        if conceptnet_result and llm_result and conceptnet_result.success and llm_result.success:
            conceptnet_set = set(conceptnet_concepts)
            llm_set = set(llm_concepts)
            
            overlap = conceptnet_set.intersection(llm_set)
            print(f"\n📊 Comparison:")
            print(f"   Common concepts: {len(overlap)} - {', '.join(overlap) if overlap else 'None'}")
            print(f"   ConceptNet unique: {len(conceptnet_set - llm_set)}")
            print(f"   LLM unique: {len(llm_set - conceptnet_set)}")
            
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False


async def test_cache_behavior():
    """Test that LLM results are properly cached."""
    print("\n🧪 Testing LLM cache behavior...")
    
    expander = get_concept_expander()
    
    try:
        concept = "horror"
        media_context = "movie"
        
        # First call (cache miss)
        print("🔄 First call (cache miss)...")
        start_time = asyncio.get_event_loop().time()
        result1 = await expander.expand_concept(
            concept=concept,
            media_context=media_context,
            method=ExpansionMethod.LLM,
            max_concepts=5
        )
        first_call_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        if not result1 or not result1.success:
            print("❌ First call failed")
            return False
        
        # Second call (cache hit)
        print("🔄 Second call (cache hit)...")
        start_time = asyncio.get_event_loop().time()
        result2 = await expander.expand_concept(
            concept=concept,
            media_context=media_context,
            method=ExpansionMethod.LLM,
            max_concepts=5
        )
        second_call_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        if not result2 or not result2.success:
            print("❌ Second call failed")
            return False
        
        # Compare results
        concepts1 = result1.enhanced_data.get("expanded_concepts", [])
        concepts2 = result2.enhanced_data.get("expanded_concepts", [])
        
        if concepts1 == concepts2:
            print("✅ Cache working correctly - identical results")
            print(f"📊 Performance improvement: {first_call_time:.1f}ms → {second_call_time:.1f}ms")
            speedup = first_call_time / second_call_time if second_call_time > 0 else float('inf')
            print(f"⚡ Speedup: {speedup:.1f}x faster")
            return True
        else:
            print("⚠️  Cache issue - results differ between calls")
            return False
            
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False


async def run_all_tests():
    """Run all LLM provider tests."""
    print("🚀 Starting LLM Provider Test Suite for Stage 3.2")
    print("=" * 60)
    
    tests = [
        ("Ollama Backend Client", test_ollama_backend_client),
        ("LLM Provider", test_llm_provider),
        ("ConceptExpander LLM Integration", test_concept_expander_llm),
        ("LLM vs ConceptNet Comparison", test_llm_vs_conceptnet),
        ("Cache Behavior", test_cache_behavior),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 40)
        
        try:
            success = await test_func()
            results.append((test_name, success))
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n❌ ERROR in {test_name}: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LLM provider system is working correctly.")
        print("\n🎓 Stage 3.2 LLM Concept Understanding: COMPLETED ✅")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
    
    return passed == total


if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n📝 Next: Update todo.md to mark Stage 3.2 as completed")
        print("🔜 Ready for Stage 3.3: Multi-Source Concept Fusion")
    else:
        print("\n🔧 Please fix the issues above before proceeding")