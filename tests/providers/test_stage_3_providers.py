#!/usr/bin/env python3
"""
Test all Stage 3 concept expansion providers to verify they're working correctly.
"""

import asyncio
from tests.tests_shared import logger
from src.shared.concept_expander import ConceptExpander, ExpansionMethod
from tests.tests_shared import settings_to_console


async def test_conceptnet_provider(expander):
    """Test ConceptNet provider - FAIL FAST if broken."""
    logger.info("=== Testing ConceptNet Provider ===")
    
    # NO FALLBACKS - if this fails, test should fail hard
    result = await expander.expand_concept(
        concept="action",
        media_context="movie",
        method=ExpansionMethod.CONCEPTNET
    )
    
    # Validate we got actual results, not empty fallback
    if not result or not result.enhanced_data.get('expanded_concepts'):
        raise AssertionError("ConceptNet provider returned no expanded concepts - provider is broken")
    
    expanded_concepts = result.enhanced_data.get('expanded_concepts', [])
    logger.info(f"ConceptNet expansion for 'action': {expanded_concepts[:5]}")
    
    # Ensure we got meaningful results
    if len(expanded_concepts) < 2:
        raise AssertionError(f"ConceptNet provider returned too few concepts: {len(expanded_concepts)} - provider may be degraded")
    
    logger.info("✅ ConceptNet provider working correctly")
    return True

async def test_llm_provider(expander):
    """Test LLM provider - FAIL FAST if Ollama unavailable."""
    logger.info("=== Testing LLM Provider ===")
    
    # NO FALLBACKS - if Ollama is down or misconfigured, test should fail hard
    result = await expander.expand_concept(
        concept="horror",
        media_context="movie", 
        method=ExpansionMethod.LLM
    )
    
    # Validate we got actual LLM results, not empty fallback
    if not result or not result.enhanced_data.get('expanded_concepts'):
        raise AssertionError("LLM provider returned no expanded concepts - Ollama may be down or misconfigured")
    
    expanded_concepts = result.enhanced_data.get('expanded_concepts', [])
    logger.info(f"LLM expansion for 'horror': {expanded_concepts[:5]}")
    
    # LLM should generate meaningful concepts
    if len(expanded_concepts) < 3:
        raise AssertionError(f"LLM provider returned too few concepts: {len(expanded_concepts)} - Ollama may be degraded")
    
    logger.info("✅ LLM provider working correctly")
    return True

async def test_spacy_temporal_provider(expander):
    """Test SpaCy Temporal provider - FAIL FAST if SpaCy models missing."""
    logger.info("=== Testing SpaCy Temporal Provider ===")
    
    # First check if SpaCy is properly installed with models
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise AssertionError("SpaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    except ImportError:
        raise AssertionError("SpaCy not installed. Run: pip install spacy")
    
    # NO FALLBACKS - if SpaCy models are missing, test should fail hard
    result = await expander.expand_concept(
        concept="recent",
        media_context="movie",
        method=ExpansionMethod.SPACY_TEMPORAL
    )
    
    # Validate we got actual temporal results, not fallback
    if not result or not result.enhanced_data.get('expanded_concepts'):
        raise AssertionError("SpaCy Temporal provider returned no expanded concepts - temporal parsing may be broken")
    
    expanded_concepts = result.enhanced_data.get('expanded_concepts', [])
    logger.info(f"SpaCy Temporal expansion for 'recent': {expanded_concepts[:5]}")
    
    logger.info("✅ SpaCy Temporal provider working correctly")
    return True

async def test_gensim_provider(expander):
    """Test Gensim provider - FAIL FAST if Gensim models missing."""
    logger.info("=== Testing Gensim Provider ===")
    
    # First check if Gensim is properly installed
    try:
        import gensim
    except ImportError:
        raise AssertionError("Gensim not installed. Run: pip install gensim")
    
    # NO FALLBACKS - if Gensim models are missing or broken, test should fail hard
    result = await expander.expand_concept(
        concept="thriller",
        media_context="movie",
        method=ExpansionMethod.GENSIM
    )
    
    # Validate we got actual Gensim results, not fallback
    if not result or not result.enhanced_data.get('expanded_concepts'):
        raise AssertionError("Gensim provider returned no expanded concepts - word embeddings may be missing or broken")
    
    expanded_concepts = result.enhanced_data.get('expanded_concepts', [])
    logger.info(f"Gensim expansion for 'thriller': {expanded_concepts[:5]}")
    
    # Gensim should find semantic similarities
    if len(expanded_concepts) < 2:
        raise AssertionError(f"Gensim provider returned too few concepts: {len(expanded_concepts)} - word embedding model may be degraded")
    
    logger.info("✅ Gensim provider working correctly")
    return True

async def test_heideltime_provider(expander):
    """Test HeidelTime provider - FAIL FAST if Java/HeidelTime missing."""
    logger.info("=== Testing HeidelTime Provider ===")
    
    # First check if Java is available
    import subprocess
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            raise AssertionError("Java not found or not working. HeidelTime requires Java 17+")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        raise AssertionError("Java not found. HeidelTime requires Java 17+. Install with: sudo apt install openjdk-17-jdk")
    
    # Check if py-heideltime is installed
    try:
        from py_heideltime.py_heideltime import heideltime
    except ImportError:
        raise AssertionError("py-heideltime not installed. Run: pip install py-heideltime")
    
    # NO FALLBACKS - if HeidelTime is broken, test should fail hard
    result = await expander.expand_concept(
        concept="classic",
        media_context="movie",
        method=ExpansionMethod.HEIDELTIME
    )
    
    # Validate we got actual HeidelTime results, not fallback
    if not result or not result.enhanced_data.get('expanded_concepts'):
        raise AssertionError("HeidelTime provider returned no expanded concepts - temporal parsing may be broken")
    
    expanded_concepts = result.enhanced_data.get('expanded_concepts', [])
    logger.info(f"HeidelTime expansion for 'classic': {expanded_concepts[:5]}")
    
    logger.info("✅ HeidelTime provider working correctly")
    return True

async def main():
    """Run all provider tests."""
    logger.info("🧪 Testing all Stage 3 Concept Expansion Providers")
    
    tests = [
        ("ConceptNet", test_conceptnet_provider),
        ("LLM", test_llm_provider), 
        ("SpaCy Temporal", test_spacy_temporal_provider),
        ("Gensim", test_gensim_provider),
        ("HeidelTime", test_heideltime_provider),
    ]
    
    results = {}
    expander = None
    
    try:
        # Initialize ConceptExpander once for all tests
        logger.info("🔧 Initializing ConceptExpander...")
        expander = ConceptExpander()
        
        for name, test_func in tests:
            # NO FALLBACKS - if a test fails, the whole test suite should fail
            logger.info(f"🧪 Running {name} test...")
            success = await test_func(expander)
            results[name] = success
            logger.info(f"✅ {name}: PASSED")
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("📊 STAGE 3 PROVIDER TEST SUMMARY")
        logger.info("="*50)
        
        passed = sum(1 for success in results.values() if success)
        total = len(results)
        
        for name, success in results.items():
            status = "✅ WORKING" if success else "❌ FAILED"
            logger.info(f"{name:15} {status}")
        
        logger.info(f"\n🎯 Overall: {passed}/{total} providers working")
        
        if passed == total:
            logger.info("🎉 All Stage 3 providers are working correctly!")
            logger.info("🚀 Stage 3.2.5 cleanup + HeidelTime integration successful!")
        else:
            logger.warning("⚠️  Some providers need attention")
        
        return passed == total
        
    finally:
        # Clean up resources - cleanup failures should be logged but not hidden
        if expander:
            if hasattr(expander, 'llm_provider') and expander.llm_provider:
                await expander.llm_provider.close()
            logger.info("🧹 Resources cleaned up successfully")

if __name__ == "__main__":
    asyncio.run(main())