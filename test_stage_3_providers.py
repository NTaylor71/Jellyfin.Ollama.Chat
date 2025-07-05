#!/usr/bin/env python3
"""
Test all Stage 3 concept expansion providers to verify they're working correctly.
"""

import asyncio
import logging
from src.concept_expansion.concept_expander import ConceptExpander, ExpansionMethod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_conceptnet_provider(expander):
    """Test ConceptNet provider."""
    logger.info("=== Testing ConceptNet Provider ===")
    
    try:
        result = await expander.expand_concept(
            concept="action",
            media_context="movie",
            method=ExpansionMethod.CONCEPTNET
        )
        logger.info(f"ConceptNet expansion for 'action': {result.enhanced_data.get('expanded_concepts', [])[:5]}")
        return True
    except Exception as e:
        logger.error(f"ConceptNet provider failed: {e}")
        return False

async def test_llm_provider(expander):
    """Test LLM provider."""
    logger.info("=== Testing LLM Provider ===")
    
    try:
        result = await expander.expand_concept(
            concept="horror",
            media_context="movie", 
            method=ExpansionMethod.LLM
        )
        logger.info(f"LLM expansion for 'horror': {result.enhanced_data.get('expanded_concepts', [])[:5]}")
        return True
    except Exception as e:
        logger.error(f"LLM provider failed: {e}")
        return False

async def test_spacy_temporal_provider(expander):
    """Test SpaCy Temporal provider."""
    logger.info("=== Testing SpaCy Temporal Provider ===")
    
    try:
        result = await expander.expand_concept(
            concept="recent",
            media_context="movie",
            method=ExpansionMethod.SPACY_TEMPORAL
        )
        logger.info(f"SpaCy Temporal expansion for 'recent': {result.enhanced_data.get('expanded_concepts', [])[:5]}")
        return True
    except Exception as e:
        logger.error(f"SpaCy Temporal provider failed: {e}")
        return False

async def test_gensim_provider(expander):
    """Test Gensim provider."""
    logger.info("=== Testing Gensim Provider ===")
    
    try:
        result = await expander.expand_concept(
            concept="thriller",
            media_context="movie",
            method=ExpansionMethod.GENSIM
        )
        logger.info(f"Gensim expansion for 'thriller': {result.enhanced_data.get('expanded_concepts', [])[:5]}")
        return True
    except Exception as e:
        logger.error(f"Gensim provider failed: {e}")
        return False

async def test_heideltime_provider(expander):
    """Test HeidelTime provider."""
    logger.info("=== Testing HeidelTime Provider ===")
    
    try:
        result = await expander.expand_concept(
            concept="classic",
            media_context="movie",
            method=ExpansionMethod.HEIDELTIME
        )
        logger.info(f"HeidelTime expansion for 'classic': {result.enhanced_data.get('expanded_concepts', [])[:5]}")
        return True
    except Exception as e:
        logger.error(f"HeidelTime provider failed: {e}")
        return False

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
            try:
                success = await test_func(expander)
                results[name] = success
                logger.info(f"✅ {name}: {'PASSED' if success else 'FAILED'}")
            except Exception as e:
                logger.error(f"❌ {name}: ERROR - {e}")
                results[name] = False
        
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
        # Clean up resources
        if expander:
            try:
                if hasattr(expander, 'llm_provider') and expander.llm_provider:
                    await expander.llm_provider.close()
                logger.info("🧹 Resources cleaned up successfully")
            except Exception as e:
                logger.debug(f"Cleanup error: {e}")
                pass

if __name__ == "__main__":
    asyncio.run(main())