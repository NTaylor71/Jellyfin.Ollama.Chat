"""
Test script for Stage 3.2.5 new concept expansion providers.
Tests Gensim, SpaCy Temporal, HeidelTime, and SUTime providers.
"""

import asyncio
import logging
from typing import Dict, Any

from src.shared.concept_expander import ConceptExpander, ExpansionMethod
from src.shared.plugin_contracts import PluginResult


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_provider(
    expander: ConceptExpander,
    method: ExpansionMethod,
    concept: str,
    media_context: str = "movie"
) -> Dict[str, Any]:
    """
    Test a specific provider with a concept.
    
    Args:
        expander: ConceptExpander instance
        method: Expansion method to test
        concept: Concept to expand
        media_context: Media context
        
    Returns:
        Test result dictionary
    """
    print(f"\n=== Testing {method.value.upper()} Provider ===")
    print(f"Concept: '{concept}' (context: {media_context})")
    
    try:
        result = await expander.expand_concept(
            concept=concept,
            media_context=media_context,
            method=method,
            max_concepts=8
        )
        
        if result and result.success:
            expanded = result.enhanced_data.get("expanded_concepts", [])
            confidence_scores = result.confidence_score.per_item
            execution_time = result.plugin_metadata.execution_time_ms
            
            print(f"✅ Success! Found {len(expanded)} concepts in {execution_time:.1f}ms")
            
            for i, concept_item in enumerate(expanded[:5]):
                confidence = confidence_scores.get(concept_item, 0.0)
                print(f"  {i+1}. {concept_item} (confidence: {confidence:.3f})")
            
            if len(expanded) > 5:
                print(f"  ... and {len(expanded) - 5} more")
            
            return {
                "status": "success",
                "provider": method.value,
                "concept_count": len(expanded),
                "execution_time_ms": execution_time,
                "concepts": expanded[:5],
                "confidence_scores": {k: v for k, v in confidence_scores.items() if k in expanded[:5]}
            }
        else:
            print(f"❌ Failed: No result returned")
            return {
                "status": "failed",
                "provider": method.value,
                "error": "No result returned"
            }
            
    except Exception as e:
        print(f"💥 Error: {e}")
        return {
            "status": "error", 
            "provider": method.value,
            "error": str(e)
        }


async def test_all_new_providers():
    """Test all newly implemented providers."""
    print("🚀 Testing Stage 3.2.5 New Concept Providers")
    print("=" * 60)
    

    expander = ConceptExpander()
    

    test_cases = [

        {
            "method": ExpansionMethod.GENSIM,
            "concepts": [
                ("action", "movie"),
                ("thriller", "movie"),
                ("comedy", "book")
            ]
        },
        

        {
            "method": ExpansionMethod.SPACY_TEMPORAL,
            "concepts": [
                ("recent", "movie"),
                ("last year", "movie"),
                ("90s movies", "movie"),
                ("christmas release", "movie")
            ]
        },
        

        {
            "method": ExpansionMethod.HEIDELTIME,
            "concepts": [
                ("classic cinema", "movie"),
                ("modern films", "movie"),
                ("historical drama", "movie"),
                ("contemporary", "book")
            ]
        },
        

        {
            "method": ExpansionMethod.SUTIME,
            "concepts": [
                ("2020 release", "movie"),
                ("premiere date", "movie"),
                ("sequel timeline", "movie"),
                ("annual awards", "movie")
            ]
        }
    ]
    
    all_results = []
    
    for test_case in test_cases:
        method = test_case["method"]
        concepts = test_case["concepts"]
        
        print(f"\n🔍 Testing {method.value.upper()} Provider")
        print(f"Provider Type: {expander.get_method_capabilities(method)['type']}")
        print(f"Context Aware: {expander.get_method_capabilities(method)['context_aware']}")
        
        provider_results = []
        
        for concept, media_context in concepts:
            result = await test_provider(expander, method, concept, media_context)
            provider_results.append(result)
        
        all_results.extend(provider_results)
    

    print(f"\n📊 SUMMARY")
    print("=" * 60)
    
    success_count = sum(1 for r in all_results if r["status"] == "success")
    failed_count = sum(1 for r in all_results if r["status"] == "failed")
    error_count = sum(1 for r in all_results if r["status"] == "error")
    
    print(f"✅ Successful tests: {success_count}")
    print(f"❌ Failed tests: {failed_count}")
    print(f"💥 Error tests: {error_count}")
    print(f"📈 Success rate: {success_count / len(all_results) * 100:.1f}%")
    

    provider_stats = {}
    for result in all_results:
        provider = result["provider"]
        if provider not in provider_stats:
            provider_stats[provider] = {"success": 0, "total": 0}
        provider_stats[provider]["total"] += 1
        if result["status"] == "success":
            provider_stats[provider]["success"] += 1
    
    print(f"\n📋 Provider Performance:")
    for provider, stats in provider_stats.items():
        success_rate = stats["success"] / stats["total"] * 100
        print(f"  {provider.upper()}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
    

    await expander.close()
    
    return all_results


async def test_provider_integration():
    """Test provider integration and method selection."""
    print(f"\n🔗 Testing Provider Integration")
    print("=" * 60)
    
    expander = ConceptExpander()
    

    concept = "recent action movies"
    

    methods_to_test = [
        ExpansionMethod.CONCEPTNET,
        ExpansionMethod.LLM,
        ExpansionMethod.GENSIM,
        ExpansionMethod.SPACY_TEMPORAL
    ]
    
    print(f"Concept: '{concept}'")
    print(f"Testing {len(methods_to_test)} different expansion methods:")
    
    for method in methods_to_test:
        capabilities = expander.get_method_capabilities(method)
        print(f"\n{method.value.upper()}:")
        print(f"  Type: {capabilities['type']}")
        print(f"  Context-aware: {capabilities['context_aware']}")
        print(f"  Best for: {', '.join(capabilities['best_for'][:3])}")
        
        result = await test_provider(expander, method, concept)
        if result["status"] == "success":
            print(f"  ✅ Returned {result['concept_count']} concepts")
        else:
            print(f"  ❌ {result.get('error', 'Failed')}")
    

    recommended = expander.get_recommended_method(concept, "movie")
    print(f"\n🎯 Recommended method for '{concept}': {recommended.value.upper()}")
    
    await expander.close()


if __name__ == "__main__":
    async def main():
        try:
            await test_all_new_providers()
            await test_provider_integration()
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted by user")
        except Exception as e:
            print(f"\n💥 Test suite failed: {e}")
            logger.exception("Test suite error")
    
    asyncio.run(main())