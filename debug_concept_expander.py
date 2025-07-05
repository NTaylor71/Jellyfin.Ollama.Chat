#!/usr/bin/env python3
"""
Debug the ConceptExpander to trace where the malformed output comes from.
"""

import asyncio
import logging
from src.concept_expansion.concept_expander import ConceptExpander, ExpansionMethod

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_concept_expander():
    """Debug the ConceptExpander step by step."""
    logger.info("🔍 DEBUGGING ConceptExpander for SpaCy Temporal")
    
    expander = ConceptExpander()
    
    logger.info(f"📝 Testing concept: 'recent' with SpaCy Temporal method")
    
    try:
        result = await expander.expand_concept(
            concept="recent",
            media_context="movie",
            method=ExpansionMethod.SPACY_TEMPORAL
        )
        
        if result:
            logger.info(f"✅ Expansion successful")
            logger.info(f"Enhanced data: {result.enhanced_data}")
            
            concepts = result.enhanced_data.get('expanded_concepts', [])
            logger.info(f"Number of concepts: {len(concepts)}")
            
            for i, concept in enumerate(concepts):
                logger.info(f"  [{i}]: '{concept}' (type: {type(concept)}, len: {len(concept)})")
                
                # Show character-by-character for debugging
                logger.info(f"       chars: {[c for c in concept[:50]]}")
        else:
            logger.error("❌ No result returned")
            
    except Exception as e:
        logger.error(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_concept_expander())