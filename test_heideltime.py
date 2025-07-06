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
        import heideltime
    except ImportError:
        raise AssertionError("py-heideltime not installed. Run: pip install py-heideltime")
    
    expander = ConceptExpander()
    
    # NO FALLBACKS - if HeidelTime fails, test should fail hard
    result = await expander.expand_concept(
        concept="recent",
        media_context="movie",
        method=ExpansionMethod.HEIDELTIME
    )
    
    # Validate we got actual results, not empty fallback
    if not result:
        raise AssertionError("HeidelTime provider returned no result - provider is broken")
    
    if not result.success:
        raise AssertionError(f"HeidelTime provider failed - {result.enhanced_data.get('error', 'unknown error')}")
    
    concepts = result.enhanced_data.get('expanded_concepts', [])
    if not concepts:
        raise AssertionError("HeidelTime provider returned no expanded concepts - temporal parsing may be broken")
    
    logger.info(f"✅ HeidelTime expansion successful")
    logger.info(f"HeidelTime expansion for 'recent': {concepts}")
    return True

if __name__ == "__main__":
    asyncio.run(test_heideltime_provider())