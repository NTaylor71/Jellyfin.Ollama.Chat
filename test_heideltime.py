#!/usr/bin/env python3
"""
Test HeidelTime provider through ConceptExpander.
"""

import asyncio
import logging
from src.concept_expansion.concept_expander import ConceptExpander, ExpansionMethod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_heideltime_provider():
    """Test HeidelTime provider."""
    logger.info("=== Testing HeidelTime Provider ===")
    expander = ConceptExpander()
    
    try:
        result = await expander.expand_concept(
            concept="recent",
            media_context="movie",
            method=ExpansionMethod.HEIDELTIME
        )
        
        if result:
            logger.info(f"✅ HeidelTime expansion successful")
            concepts = result.enhanced_data.get('expanded_concepts', [])
            logger.info(f"HeidelTime expansion for 'recent': {concepts}")
            return True
        else:
            logger.error("❌ No result returned")
            return False
    except Exception as e:
        logger.error(f"❌ HeidelTime provider failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_heideltime_provider())