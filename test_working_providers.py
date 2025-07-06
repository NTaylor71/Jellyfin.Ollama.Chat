"""
Test working providers for Stage 3.2.5 without large downloads.
Tests ConceptNet and LLM providers that should work without additional downloads.
"""
from tests_shared import logger
from tests_shared import settings_to_console

import asyncio
from concept_expansion.concept_expander import ConceptExpander, ExpansionMethod


async def test_working_providers():
    """Test providers that should work out of the box."""
    logger.info("🔍 Testing Working Providers (ConceptNet + LLM)")
    logger.info("=" * 50)
    
    expander = ConceptExpander()
    
    # Test concepts that should work
    test_cases = [
        ("action", "movie", ExpansionMethod.CONCEPTNET),
        ("thriller", "movie", ExpansionMethod.LLM),
        ("comedy", "book", ExpansionMethod.CONCEPTNET),
        ("recent movies", "movie", ExpansionMethod.LLM)
    ]
    
    for concept, context, method in test_cases:
        logger.info(f"\n🎯 Testing: '{concept}' ({context}) with {method.value.upper()}")
        
        # NO FALLBACKS - if expansion fails, test should fail hard
        result = await expander.expand_concept(
            concept=concept,
            media_context=context,
            method=method,
            max_concepts=5
        )
        
        # Validate we got actual results, not empty fallback
        if not result or not result.success:
            raise AssertionError(f"Concept expansion failed for '{concept}' with {method.value} - provider may be down or broken")
        
        expanded = result.enhanced_data.get("expanded_concepts", [])
        if not expanded:
            raise AssertionError(f"Provider {method.value} returned no expanded concepts for '{concept}' - provider is broken")
        
        confidence_scores = result.confidence_score.per_item
        execution_time = result.plugin_metadata.execution_time_ms
        
        logger.info(f"✅ Success! Found {len(expanded)} concepts in {execution_time:.1f}ms")
        for i, concept_item in enumerate(expanded[:3]):  # Show top 3
            confidence = confidence_scores.get(concept_item, 0.0)
            logger.info(f"   {i+1}. {concept_item} (confidence: {confidence:.3f})")
    
    logger.info(f"\n✅ All {len(test_cases)} provider tests passed successfully!")
    
    await expander.close()

def test_provider_availability():
    """Test which providers are available without initialization."""
    logger.info("🔧 Provider Availability Check")
    logger.info("=" * 50)
    
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
            # NO FALLBACKS - if metadata is broken, test should fail hard
            metadata = provider.metadata
            if not metadata:
                raise AssertionError(f"Provider {method.value} returned no metadata - provider is broken")
            
            expected_available = availability.get(metadata.name, False)
            status = "✅ Ready" if expected_available else "⚠️ Needs setup"
            logger.info(f"{metadata.name:12} ({metadata.provider_type:10}): {status}")
            if not expected_available:
                logger.info(f"             Requirements: {metadata.weaknesses[0] if metadata.weaknesses else 'External dependencies'}")
        else:
            raise AssertionError(f"Provider {method.value} not found in expander.providers - provider system is broken")

if __name__ == "__main__":
    logger.info("🚀 Stage 3.2.5 Working Providers Test")
    logger.info("=" * 60)

    settings_to_console()
    
    # NO FALLBACKS - if any test fails, the whole test should fail hard
    test_provider_availability()
    logger.info("\n")
    asyncio.run(test_working_providers())
    
    logger.info(f"\n🎉 All working provider tests completed successfully!")
    logger.info("Note: Gensim, HeidelTime, SUTime need additional setup")