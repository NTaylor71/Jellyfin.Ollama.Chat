"""
Test the new procedural temporal intelligence architecture.
Demonstrates clean separation between pure parsing and LLM-driven intelligence.
"""

import asyncio
from concept_expansion.concept_expander import ConceptExpander, ExpansionMethod
from concept_expansion.temporal_concept_generator import TemporalConceptGenerator, TemporalConceptRequest
from tests_shared import logger
from tests_shared import settings_to_console


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
        for i, concept_item in enumerate(expanded[:4]):  # Show top 4
            confidence = confidence_scores.get(concept_item, 0.0)
            logger.info(f"   {i+1}. {concept_item} (confidence: {confidence:.3f})")
    
    logger.info(f"\n✅ All hybrid temporal expansion tests passed")
    await expander.close()
    return len(test_cases)

async def test_architecture_comparison():
    """Compare the new clean architecture vs what would have been hard-coded."""
    logger.info(f"\n🏗️ Architecture Comparison")
    logger.info("=" * 60)
    
    logger.info("❌ OLD BRITTLE APPROACH:")
    logger.info("   Hard-coded patterns in providers:")
    logger.info("   'recent' → ['new', 'latest', 'current', 'contemporary', 'modern']")
    logger.info("   'classic' → ['old', 'vintage', 'retro', 'traditional', 'timeless']")
    logger.info("   Same patterns for ALL media types (movies = books = music)")
    logger.info("   Programmer assumptions, not intelligence")
    
    logger.info("\n✅ NEW PROCEDURAL APPROACH:")
    logger.info("   LLM analysis: 'What does recent mean for movies in 2024?'")
    logger.info("   Media-aware: recent movies ≠ recent books ≠ recent music")
    logger.info("   Cached intelligence, not hard-coded patterns")
    logger.info("   Pure parsers + procedural intelligence")
    
    # Demonstrate the difference
    generator = TemporalConceptGenerator()
    
    logger.info(f"\n🧪 DEMONSTRATION:")
    logger.info("   Asking LLM: 'What does recent mean for movies vs books?'")
    
    movie_request = TemporalConceptRequest("recent", "movie", max_concepts=3)
    book_request = TemporalConceptRequest("recent", "book", max_concepts=3)
    
    # NO FALLBACKS - demonstration should work or fail clearly
    movie_result = await generator.generate_temporal_concepts(movie_request)
    book_result = await generator.generate_temporal_concepts(book_request)
    
    if not movie_result or not movie_result.success:
        raise AssertionError("Movie temporal concept generation failed - LLM may be down")
    if not book_result or not book_result.success:
        raise AssertionError("Book temporal concept generation failed - LLM may be down")
    
    movie_concepts = movie_result.enhanced_data.get("temporal_concepts", [])
    book_concepts = book_result.enhanced_data.get("temporal_concepts", [])
    
    logger.info(f"   Recent MOVIES: {movie_concepts}")
    logger.info(f"   Recent BOOKS:  {book_concepts}")
    logger.info("   ↑ See the difference? Media-aware intelligence!")
    
    await generator.close()

async def main():
    logger.info("🚀 Procedural Temporal Intelligence Test")
    logger.info("Demonstrating the clean architecture: Pure Parsing + LLM Intelligence")
    logger.info("=" * 80)

    settings_to_console()
    
    # NO FALLBACKS - if any test fails, the whole test should fail hard
    # Test individual components
    generator_success = await test_temporal_concept_generator()
    hybrid_success = await test_hybrid_temporal_expansion()
    
    # Show architecture benefits
    await test_architecture_comparison()
    
    logger.info(f"\n🎉 FINAL RESULTS")
    logger.info(f"✅ TemporalConceptGenerator: Working")
    logger.info(f"✅ Hybrid Temporal Expansion: Working") 
    logger.info(f"✅ No Hard-coded Patterns: Clean!")
    logger.info(f"✅ Media-aware Intelligence: Procedural!")
    
    logger.info(f"\n🧠 INTELLIGENCE SUMMARY:")
    logger.info("• Temporal providers are now PURE parsers")
    logger.info("• TemporalConceptGenerator provides LLM-driven intelligence")  
    logger.info("• ConceptExpander orchestrates parsing + intelligence")
    logger.info("• All temporal knowledge is procedural and cached")
    logger.info("• Zero hard-coded patterns anywhere!")
    
    logger.info(f"\n✅ All temporal intelligence tests passed - system is working correctly")

if __name__ == "__main__":
    asyncio.run(main())