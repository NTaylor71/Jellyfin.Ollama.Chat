"""
Test script for ConceptExpander Stage 3.1 implementation.
Tests the examples from todo.md: "action" + "movie" → ["fight", "combat", "battle", "intense", "fast-paced"]
"""

import asyncio
import logging
from typing import List

from src.shared.concept_expander import get_concept_expander, ExpansionMethod
from src.api.cache_admin import clear_test_cache, print_cache_summary
from src.shared.test_data import get_concept_expansion_test_cases


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_concept_expansion():
    """Test basic ConceptExpander functionality with todo.md example."""
    print("🧪 Testing ConceptExpander with todo.md example...")
    
    expander = get_concept_expander()
    

    conceptnet_caps = expander.get_method_capabilities(ExpansionMethod.CONCEPTNET)
    llm_caps = expander.get_method_capabilities(ExpansionMethod.LLM)
    
    print(f"\n🔍 ConceptNet capabilities (type: {conceptnet_caps['type']}, context-aware: {conceptnet_caps['context_aware']})")
    print(f"   Strengths: {', '.join(conceptnet_caps['strengths'])}")
    print(f"   Weaknesses: {', '.join(conceptnet_caps['weaknesses'])}")
    
    print(f"\n🧠 LLM capabilities (type: {llm_caps['type']}, context-aware: {llm_caps['context_aware']})")
    print(f"   Strengths: {', '.join(llm_caps['strengths'])}")
    print(f"   Best for: {', '.join(llm_caps['best_for'])}")
    

    print("\n📋 Testing: 'action' + 'movie' context")
    recommended = expander.get_recommended_method("action", "movie")
    print(f"💡 Recommended method: {recommended.value} (but testing ConceptNet anyway)")
    
    result = await expander.expand_concept("action", "movie")
    
    if result and result.success:
        expanded_concepts = result.enhanced_data.get("expanded_concepts", [])
        confidence_scores = result.confidence_score.per_item
        
        print(f"✅ Success! Expanded 'action' into {len(expanded_concepts)} concepts:")
        for concept in expanded_concepts[:10]:
            confidence = confidence_scores.get(concept, 0.0)
            print(f"   • {concept} (confidence: {confidence:.2f})")
        

        expected = ["fight", "combat", "battle", "intense", "fast-paced"]
        found_expected = [exp for exp in expected if exp in expanded_concepts]
        print(f"\n🎯 Expected concepts found: {found_expected}")
        

        if not found_expected:
            print("💡 NOTE: ConceptNet's RelatedTo gives generic actions, not movie genre concepts.")
            print("      This is why Stage 3.2 LLM expansion will be important for better results.")
        

        cache_key = result.cache_key.generate_key()
        print(f"🗄️  Cached with key: {cache_key}")
        print(f"⏱️  Execution time: {result.plugin_metadata.execution_time_ms:.1f}ms")
        
    else:
        print("❌ Failed to expand 'action' concept")
        if result:
            print(f"   Error: {result.error_message}")


async def test_cache_behavior():
    """Test that cache-first behavior works correctly."""
    print("\n🗄️ Testing cache behavior...")
    
    expander = get_concept_expander()
    

    print("First call (cache miss expected):")
    result1 = await expander.expand_concept("samurai", "movie")
    
    if result1 and result1.success:
        time1 = result1.plugin_metadata.execution_time_ms
        print(f"   ✅ Success in {time1:.1f}ms (includes API call)")
    

    print("Second call (cache hit expected):")
    result2 = await expander.expand_concept("samurai", "movie")
    
    if result2 and result2.success:
        time2 = result2.plugin_metadata.execution_time_ms
        print(f"   ✅ Success in {time2:.1f}ms (cache hit)")
        
        if time2 < time1:
            print("   🚀 Cache hit was faster than API call!")
        else:
            print("   ⚠️  Cache hit not noticeably faster")


async def test_concept_expansion_test_cases():
    """Test with examples from test_data.py concept expansion cases."""
    print("\n🧪 Testing with test_data.py examples...")
    
    expander = get_concept_expander()
    test_cases = get_concept_expansion_test_cases()
    
    for i, case in enumerate(test_cases[:3], 1):
        concept = case["input_term"]
        media_context = case["media_context"]
        expected_concepts = case["expected_concepts"]
        
        print(f"\n{i}. REQUEST: Expand '{concept}' in '{media_context}' context")
        print(f"   Source: {case['source_movie']}")
        print(f"   Expected: {expected_concepts}")
        
        result = await expander.expand_concept(concept, media_context)
        
        print(f"   RESULT:")
        print(f"   - Result object: {type(result).__name__ if result else 'None'}")
        print(f"   - Is None: {result is None}")
        
        if result:
            print(f"   - Success: {result.success}")
            print(f"   - Enhanced data keys: {list(result.enhanced_data.keys()) if result.enhanced_data else 'None'}")
            
            if result.success and result.enhanced_data:
                expanded = result.enhanced_data.get("expanded_concepts", [])
                print(f"   - Expanded concepts count: {len(expanded)}")
                print(f"   - Actual concepts: {expanded}")
                

                found = [exp for exp in expected_concepts if exp in expanded]
                if found:
                    print(f"   - ✅ Expected matches found: {found}")
                else:
                    print(f"   - ❌ No matches with expected concepts")
            else:
                print(f"   - ❌ Failed expansion or no enhanced data")
        else:
            print(f"   - ❌ NULL RESULT - This is the actual failure!")


async def test_batch_expansion():
    """Test batch concept expansion."""
    print("\n📦 Testing batch concept expansion...")
    
    expander = get_concept_expander()
    concepts = ["action", "comedy", "horror", "drama"]
    
    print(f"REQUEST: Batch expand {len(concepts)} concepts: {concepts}")
    results = await expander.batch_expand_concepts(concepts, "movie")
    
    print(f"RESULTS:")
    successful = 0
    for concept, result in results.items():
        print(f"\n   '{concept}':")
        print(f"   - Result object: {type(result).__name__ if result else 'None'}")
        print(f"   - Is None: {result is None}")
        
        if result:
            print(f"   - Success: {result.success}")
            if result.success and result.enhanced_data:
                expanded = result.enhanced_data.get("expanded_concepts", [])
                print(f"   - Concepts ({len(expanded)}): {expanded}")
                successful += 1
            else:
                print(f"   - ❌ Success={result.success}, enhanced_data={bool(result.enhanced_data)}")
        else:
            print(f"   - ❌ NULL RESULT")
    
    print(f"📊 Batch results: {successful}/{len(concepts)} successful")


async def main():
    """Run all ConceptExpander tests."""
    print("🚀 ConceptExpander Stage 3.1 Test Suite")
    print("=" * 50)
    
    try:

        print("🧹 Cache management options:")
        print("   - Run: await clear_test_cache() to start fresh")
        print("   - Current cache:")
        await print_cache_summary()
        

        await test_basic_concept_expansion()
        

        await test_cache_behavior()
        

        await test_concept_expansion_test_cases()
        

        await test_batch_expansion()
        
        print("\n🎉 All tests completed!")
        print("\n📋 Next steps:")
        print("   1. Run this test with: python -m src.providers.knowledge.conceptnet_provider")
        print("   2. Check cache in MongoDB at: http://localhost:8081")
        print("   3. Use cache_admin.clear_test_cache() to reset between tests")
        print("   4. Ready for Stage 3.2 LLM integration")
        
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:

        expander = get_concept_expander()
        await expander.close()


if __name__ == "__main__":
    asyncio.run(main())