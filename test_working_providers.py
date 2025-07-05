"""
Test working providers for Stage 3.2.5 without large downloads.
Tests ConceptNet and LLM providers that should work without additional downloads.
"""

import asyncio
from concept_expansion.concept_expander import ConceptExpander, ExpansionMethod

async def test_working_providers():
    """Test providers that should work out of the box."""
    print("🔍 Testing Working Providers (ConceptNet + LLM)")
    print("=" * 50)
    
    expander = ConceptExpander()
    
    # Test concepts that should work
    test_cases = [
        ("action", "movie", ExpansionMethod.CONCEPTNET),
        ("thriller", "movie", ExpansionMethod.LLM),
        ("comedy", "book", ExpansionMethod.CONCEPTNET),
        ("recent movies", "movie", ExpansionMethod.LLM)
    ]
    
    successful_tests = 0
    total_tests = len(test_cases)
    
    for concept, context, method in test_cases:
        print(f"\n🎯 Testing: '{concept}' ({context}) with {method.value.upper()}")
        
        try:
            result = await expander.expand_concept(
                concept=concept,
                media_context=context,
                method=method,
                max_concepts=5
            )
            
            if result and result.success:
                expanded = result.enhanced_data.get("expanded_concepts", [])
                confidence_scores = result.confidence_score.per_item
                execution_time = result.plugin_metadata.execution_time_ms
                
                print(f"✅ Success! Found {len(expanded)} concepts in {execution_time:.1f}ms")
                for i, concept_item in enumerate(expanded[:3]):  # Show top 3
                    confidence = confidence_scores.get(concept_item, 0.0)
                    print(f"   {i+1}. {concept_item} (confidence: {confidence:.3f})")
                
                successful_tests += 1
            else:
                print(f"❌ Failed: No valid result")
                
        except Exception as e:
            print(f"💥 Error: {e}")
    
    print(f"\n📊 RESULTS")
    print(f"✅ Successful: {successful_tests}/{total_tests}")
    print(f"📈 Success rate: {successful_tests/total_tests*100:.1f}%")
    
    await expander.close()

def test_provider_availability():
    """Test which providers are available without initialization."""
    print("🔧 Provider Availability Check")
    print("=" * 50)
    
    expander = ConceptExpander()
    
    availability = {
        "ConceptNet": True,  # Always available (no external deps)
        "LLM": True,        # Should work with Ollama
        "Gensim": False,    # Would require large model download
        "SpaCy Temporal": True,  # Using SpaCy for temporal parsing
        "HeidelTime": False, # Requires pip install py-heideltime  
        "SUTime": False     # Requires Java + SUTime setup
    }
    
    for method in ExpansionMethod:
        if method == ExpansionMethod.MULTI_SOURCE or method == ExpansionMethod.AUTO:
            continue
            
        provider = expander.providers.get(method)
        if provider:
            try:
                metadata = provider.metadata
                expected_available = availability.get(metadata.name, False)
                status = "✅ Ready" if expected_available else "⚠️ Needs setup"
                print(f"{metadata.name:12} ({metadata.provider_type:10}): {status}")
                if not expected_available:
                    print(f"             Requirements: {metadata.weaknesses[0] if metadata.weaknesses else 'External dependencies'}")
            except Exception as e:
                print(f"{method.value:12}: ❌ Error - {e}")

if __name__ == "__main__":
    print("🚀 Stage 3.2.5 Working Providers Test")
    print("=" * 60)
    
    try:
        test_provider_availability()
        print("\n")
        asyncio.run(test_working_providers())
        
        print(f"\n🎉 Tests completed!")
        print("Note: Gensim, HeidelTime, SUTime need additional setup")
        
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()