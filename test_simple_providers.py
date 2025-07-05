"""
Simple test for Stage 3.2.5 providers - tests basic functionality without large downloads.
"""

from concept_expansion.concept_expander import ConceptExpander, ExpansionMethod

def test_provider_initialization():
    """Test that all providers can be initialized."""
    print("🔍 Testing Provider Initialization")
    
    expander = ConceptExpander()
    
    # Test that all providers are created
    expected_providers = [
        ExpansionMethod.CONCEPTNET,
        ExpansionMethod.LLM, 
        ExpansionMethod.GENSIM,
        ExpansionMethod.SPACY_TEMPORAL,
        ExpansionMethod.HEIDELTIME,
        ExpansionMethod.SUTIME
    ]
    
    print(f"✅ ConceptExpander created with {len(expander.providers)} providers")
    
    for method in expected_providers:
        if method in expander.providers:
            provider = expander.providers[method]
            try:
                metadata = provider.metadata
                print(f"✅ {method.value}: {metadata.name} ({metadata.provider_type})")
                print(f"   Context-aware: {metadata.context_aware}")
                print(f"   Strengths: {', '.join(metadata.strengths[:2])}")
            except Exception as e:
                print(f"⚠️  {method.value}: Error getting metadata - {e}")
        else:
            print(f"❌ {method.value}: Not found in providers")
    
    # Test method capabilities
    print(f"\n🔧 Testing Method Capabilities")
    
    for method in expected_providers:
        try:
            capabilities = expander.get_method_capabilities(method)
            print(f"✅ {method.value}: {capabilities['type']} provider")
        except Exception as e:
            print(f"❌ {method.value}: {e}")
    
    # Test method recommendation
    print(f"\n🎯 Testing Method Recommendation")
    
    test_concepts = [
        ("action", "movie"),
        ("recent movies", "movie"), 
        ("90s films", "movie"),
        ("comedy books", "book")
    ]
    
    for concept, context in test_concepts:
        try:
            recommended = expander.get_recommended_method(concept, context)
            print(f"✅ '{concept}' ({context}) → {recommended.value}")
        except Exception as e:
            print(f"❌ '{concept}' ({context}): {e}")

def test_provider_support():
    """Test provider support checking."""
    print(f"\n📋 Testing Provider Support")
    
    expander = ConceptExpander()
    
    test_concepts = [
        ("action", "movie"),
        ("recent", "movie"),
        ("2020 release", "movie"),
        ("last year", "movie"),
        ("comedy", "book")
    ]
    
    providers_to_test = [
        ExpansionMethod.CONCEPTNET,
        ExpansionMethod.LLM,
        ExpansionMethod.GENSIM,
        ExpansionMethod.SPACY_TEMPORAL
    ]
    
    for concept, context in test_concepts:
        print(f"\nConcept: '{concept}' ({context})")
        for method in providers_to_test:
            try:
                provider = expander.providers[method]
                supports = provider.supports_concept(concept, context)
                print(f"  {method.value}: {'✅' if supports else '❌'}")
            except Exception as e:
                print(f"  {method.value}: ⚠️  Error - {e}")

if __name__ == "__main__":
    print("🚀 Simple Provider Test - Stage 3.2.5")
    print("=" * 50)
    
    try:
        test_provider_initialization()
        test_provider_support()
        print(f"\n🎉 Basic provider tests completed!")
        
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()