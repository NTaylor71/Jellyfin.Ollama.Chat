#!/usr/bin/env python3
"""
Test HeidelTime provider through ConceptExpander.
"""

import asyncio
from tests_shared import logger
from src.concept_expansion.concept_expander import ConceptExpander, ExpansionMethod
from tests_shared import settings_to_console


async def test_heideltime_provider():
    """Test HeidelTime provider - FAIL FAST if broken."""
    logger.info("=== Testing HeidelTime Provider ===")
    settings_to_console()
    
    # First check if Java is available
    import subprocess
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            raise AssertionError("Java not working properly. HeidelTime requires Java 17+")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        raise AssertionError("Java not found. HeidelTime requires Java 17+. Install with: sudo apt install openjdk-17-jdk")
    
    # Check if py-heideltime is installed
    try:
        from py_heideltime.py_heideltime import heideltime
    except ImportError:
        raise AssertionError("py-heideltime not installed. Run: pip install py-heideltime")
    
    expander = ConceptExpander()
    
    # Test multiple temporal concepts to show different results
    test_concepts = [
        ("recent", "movie"),
        ("last year", "movie"),
        ("2023", "movie"),
        ("old classic", "movie")
    ]
    
    for concept, media_context in test_concepts:
        logger.info(f"\n🔍 Testing HeidelTime expansion for: '{concept}' in context '{media_context}'")
        
        # NO FALLBACKS - if HeidelTime fails, test should fail hard
        result = await expander.expand_concept(
            concept=concept,
            media_context=media_context,
            method=ExpansionMethod.HEIDELTIME
        )
        
        # Log detailed request information
        logger.info(f"📤 Request: concept='{concept}', media_context='{media_context}', method=HEIDELTIME")
        
        # Validate we got actual results, not empty fallback
        if not result:
            raise AssertionError(f"HeidelTime provider returned no result for '{concept}' - provider is broken")
        
        if not result.success:
            error_msg = result.enhanced_data.get('error', 'unknown error') if result.enhanced_data else 'no error data'
            raise AssertionError(f"HeidelTime provider failed for '{concept}' - {error_msg}")
        
        # Log detailed response information
        logger.info(f"📥 Response received: success={result.success}")
        logger.info(f"   Cache key: {result.cache_key.generate_key()}")
        logger.info(f"   Cache type: {result.cache_key.cache_type.value}")
        logger.info(f"   Execution time: {result.plugin_metadata.execution_time_ms:.2f}ms")
        logger.info(f"   Plugin: {result.plugin_metadata.plugin_name} v{result.plugin_metadata.plugin_version}")
        if result.plugin_metadata.model_used:
            logger.info(f"   Model used: {result.plugin_metadata.model_used}")
        if result.plugin_metadata.api_endpoint:
            logger.info(f"   API endpoint: {result.plugin_metadata.api_endpoint}")
        
        concepts = result.enhanced_data.get('expanded_concepts', [])
        confidence_scores = result.confidence_score.per_item if result.confidence_score else {}
        
        if not concepts:
            raise AssertionError(f"HeidelTime provider returned no expanded concepts for '{concept}' - temporal parsing may be broken")
        
        logger.info(f"   📊 Found {len(concepts)} expanded concepts:")
        for i, expanded_concept in enumerate(concepts[:10]):  # Show first 10
            confidence = confidence_scores.get(expanded_concept, 0.0)
            logger.info(f"      {i+1}. '{expanded_concept}' (confidence: {confidence:.3f})")
        
        if len(concepts) > 10:
            logger.info(f"      ... and {len(concepts) - 10} more concepts")
        
        # Log additional metadata if available
        enhanced_data = result.enhanced_data
        if enhanced_data:
            provider_type = enhanced_data.get('provider_type', 'unknown')
            expansion_method = enhanced_data.get('expansion_method', 'unknown')
            logger.info(f"   🔧 Provider type: {provider_type}")
            logger.info(f"   🔧 Expansion method: {expansion_method}")
            
            # Log HeidelTime specific data
            if 'parsing_concepts' in enhanced_data:
                logger.info(f"   📝 Parsing concepts: {enhanced_data['parsing_concepts']}")
            if 'intelligent_concepts' in enhanced_data:
                logger.info(f"   🧠 Intelligent concepts: {enhanced_data['intelligent_concepts']}")
        
        logger.info(f"✅ HeidelTime expansion for '{concept}' successful!")
    
    logger.info(f"\n🎉 All HeidelTime tests passed successfully!")
    return True

if __name__ == "__main__":
    asyncio.run(test_heideltime_provider())