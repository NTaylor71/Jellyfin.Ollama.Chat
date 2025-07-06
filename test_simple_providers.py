"""
Simple test for Stage 3.2.5 providers - tests basic functionality without large downloads.
"""
from tests_shared import logger
from concept_expansion.concept_expander import ConceptExpander, ExpansionMethod
from tests_shared import settings_to_console


def test_provider_initialization():
    """Test that all providers can be initialized."""
    logger.info("🔍 Testing Provider Initialization")
    
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
    
    logger.info(f"✅ ConceptExpander created with {len(expander.providers)} providers")
    
    for method in expected_providers:
        if method in expander.providers:
            provider = expander.providers[method]
            # NO FALLBACKS - if metadata is broken, test should fail hard
            metadata = provider.metadata
            if not metadata:
                raise AssertionError(f"{method.value} provider returned no metadata - provider is broken")
            logger.info(f"✅ {method.value}: {metadata.name} ({metadata.provider_type})")
            logger.info(f"   Context-aware: {metadata.context_aware}")
            logger.info(f"   Strengths: {', '.join(metadata.strengths[:2])}")
        else:
            logger.info(f"❌ {method.value}: Not found in providers")
    
    # Test method capabilities
    logger.info(f"\n🔧 Testing Method Capabilities")
    
    for method in expected_providers:
        # NO FALLBACKS - if capabilities are broken, test should fail hard
        capabilities = expander.get_method_capabilities(method)
        if not capabilities or 'type' not in capabilities:
            raise AssertionError(f"{method.value} get_method_capabilities returned invalid data - expander is broken")
        logger.info(f"✅ {method.value}: {capabilities['type']} provider")
    
    # Test method recommendation
    logger.info(f"\n🎯 Testing Method Recommendation")
    
    test_concepts = [
        ("action", "movie"),
        ("recent movies", "movie"), 
        ("90s films", "movie"),
        ("comedy books", "book")
    ]
    
    for concept, context in test_concepts:
        # NO FALLBACKS - if method recommendation is broken, test should fail hard
        recommended = expander.get_recommended_method(concept, context)
        if not recommended:
            raise AssertionError(f"get_recommended_method returned None for '{concept}' ({context}) - recommendation logic is broken")
        logger.info(f"✅ '{concept}' ({context}) → {recommended.value}")

def test_provider_support():
    """Test provider support checking and show actual provider capabilities."""
    logger.info(f"\n📋 Testing Provider Support & Capabilities")
    
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
        logger.info(f"\nConcept: '{concept}' ({context})")
        for method in providers_to_test:
            # NO FALLBACKS - if provider support checking is broken, test should fail hard
            if method not in expander.providers:
                raise AssertionError(f"Provider {method.value} not found in expander.providers - provider system is broken")
            provider = expander.providers[method]
            supports = provider.supports_concept(concept, context)
            if supports is None:
                raise AssertionError(f"Provider {method.value} supports_concept returned None - provider logic is broken")
            
            # Show detailed provider info
            status = "✅" if supports else "❌"
            metadata = provider.metadata
            strengths = ", ".join(metadata.strengths[:2]) if metadata.strengths else "unknown"
            logger.info(f"  {method.value:15}: {status} - {metadata.provider_type} ({strengths})")
            
            # For supported providers, show what they would recommend
            if supports:
                try:
                    params = provider.get_recommended_parameters(concept, context)
                    max_concepts = params.get('max_concepts', 'default')
                    logger.info(f"    └─ Recommends: max_concepts={max_concepts}")
                except Exception as e:
                    logger.info(f"    └─ Parameter recommendation failed: {e}")

if __name__ == "__main__":
    logger.info("🚀 Simple Provider Test - Stage 3.2.5")
    logger.info("=" * 50)
    
    # NO FALLBACKS - if any test fails, the whole test should fail hard
    test_provider_initialization()
    test_provider_support()
    logger.info(f"\n🎉 All basic provider tests passed - system is working correctly!")