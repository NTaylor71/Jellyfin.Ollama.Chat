"""
Test the new procedural temporal intelligence architecture.
Demonstrates clean separation between pure parsing and LLM-driven intelligence.
"""

import asyncio
from src.shared.concept_expander import ConceptExpander, ExpansionMethod
from src.shared.temporal_concept_generator import TemporalConceptGenerator, TemporalConceptRequest
from tests.tests_shared import logger
from tests.tests_shared import settings_to_console


async def test_temporal_concept_generator():
    """Test the standalone TemporalConceptGenerator - FAIL FAST if broken."""
    logger.info("🧠 Testing TemporalConceptGenerator (Procedural Intelligence)")
    logger.info("=" * 60)
    
    # Check if Ollama is available first
    try:
        import ollama
        # Test basic connectivity
        ollama.list()
    except ImportError:
        raise AssertionError("Ollama not installed. Run: pip install ollama")
    except Exception as e:
        raise AssertionError(f"Ollama not accessible: {e}. Ensure Ollama is running.")
    
    generator = TemporalConceptGenerator()
    
    # Test temporal intelligence for different media contexts
    test_cases = [
        ("recent", "movie"),
        ("classic", "movie"),
        ("90s", "music"),
        ("vintage", "book"),
        ("modern", "tv")
    ]
    
    for temporal_term, media_context in test_cases:
        logger.info(f"\n🎯 Testing: '{temporal_term}' in {media_context} context")
        
        # NO FALLBACKS - if this fails, test should fail hard
        request = TemporalConceptRequest(
            temporal_term=temporal_term,
            media_context=media_context,
            max_concepts=5
        )
        
        result = await generator.generate_temporal_concepts(request)
        
        # Validate we got actual results, not empty fallback
        if not result or not result.success:
            raise AssertionError(f"TemporalConceptGenerator failed for '{temporal_term}' in {media_context} context - LLM may be down")
        
        concepts = result.enhanced_data.get("temporal_concepts", [])
        if not concepts:
            raise AssertionError(f"TemporalConceptGenerator returned no concepts for '{temporal_term}' - intelligence is broken")
        
        confidence_scores = result.confidence_score.per_item
        execution_time = result.plugin_metadata.execution_time_ms
        
        logger.info(f"✅ Generated {len(concepts)} intelligent concepts in {execution_time:.1f}ms")
        for i, concept in enumerate(concepts):
            confidence = confidence_scores.get(concept, 0.0)
            logger.info(f"   {i+1}. {concept} (confidence: {confidence:.3f})")
    
    logger.info(f"\n✅ All TemporalConceptGenerator tests passed")
    await generator.close()
    return len(test_cases)

async def test_hybrid_temporal_expansion():
    """Test the hybrid temporal expansion (parsing + intelligence) - FAIL FAST if broken."""
    logger.info(f"\n🔗 Testing Hybrid Temporal Expansion")
    logger.info("=" * 60)
    
    # Check dependencies upfront
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise AssertionError("SpaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    except ImportError:
        raise AssertionError("SpaCy not installed. Run: pip install spacy")
    
    expander = ConceptExpander()
    
    # Test temporal concepts that should trigger intelligence
    test_cases = [
        ("recent movies", "movie", ExpansionMethod.SPACY_TEMPORAL),
        ("classic cinema", "movie", ExpansionMethod.HEIDELTIME), 
        ("90s films", "movie", ExpansionMethod.SUTIME),
        ("modern books", "book", ExpansionMethod.SPACY_TEMPORAL)
    ]
    
    for concept, context, method in test_cases:
        logger.info(f"\n🎯 Testing: '{concept}' ({context}) with {method.value.upper()}")
        
        # NO FALLBACKS - if temporal providers are broken, test should fail hard
        result = await expander.expand_concept(
            concept=concept,
            media_context=context,
            method=method,
            max_concepts=6
        )
        
        # Validate we got actual results, not empty fallback
        if not result or not result.success:
            raise AssertionError(f"Hybrid temporal expansion failed for '{concept}' with {method.value} - provider may be broken")
        
        expanded = result.enhanced_data.get("expanded_concepts", [])
        if not expanded:
            raise AssertionError(f"Hybrid expansion returned no concepts for '{concept}' - temporal intelligence is broken")
        
        confidence_scores = result.confidence_score.per_item
        execution_time = result.plugin_metadata.execution_time_ms
        expansion_method = result.enhanced_data.get("expansion_method", "unknown")
        
        logger.info(f"✅ {expansion_method}: {len(expanded)} concepts in {execution_time:.1f}ms")
        logger.info(f"📋 All {len(expanded)} concepts generated:")
        for i, concept_item in enumerate(expanded):
            confidence = confidence_scores.get(concept_item, 0.0)
            logger.info(f"   {i+1}. {concept_item} (confidence: {confidence:.3f})")
    
    logger.info(f"\n✅ All hybrid temporal expansion tests passed")
    await expander.close()
    return len(test_cases)


async def main():
    logger.info("🚀 Procedural Temporal Intelligence Test")
    logger.info("Demonstrating the clean architecture: Pure Parsing + LLM Intelligence")
    logger.info("=" * 80)

    settings_to_console()
    
    # NO FALLBACKS - if any test fails, the whole test should fail hard
    # Test individual components
    await test_temporal_concept_generator()
    await test_hybrid_temporal_expansion()

    logger.info(f"\n🎉 FINAL RESULTS")
    logger.info(f"✅ TemporalConceptGenerator: Working")
    logger.info(f"✅ Hybrid Temporal Expansion: Working") 

    logger.info(f"\n🧠 INTELLIGENCE SUMMARY:")
    logger.info("• Temporal providers are now PURE parsers")
    logger.info("• TemporalConceptGenerator provides LLM-driven intelligence")  

    logger.info(f"\n✅ All temporal intelligence tests passed - system is working correctly")

if __name__ == "__main__":
    asyncio.run(main())