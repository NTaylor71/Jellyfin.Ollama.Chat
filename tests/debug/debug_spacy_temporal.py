#!/usr/bin/env python3
"""
Debug the SpaCy Temporal provider to see exactly what's happening.
"""

import asyncio
import logging
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.providers.nlp.spacy_temporal_provider import SpacyTemporalProvider
from src.providers.nlp.base_provider import ExpansionRequest


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_spacy_temporal():
    """Debug SpaCy temporal provider step by step."""
    logger.info("🔍 DEBUGGING SpaCy Temporal Provider")
    
    provider = SpacyTemporalProvider()
    

    request = ExpansionRequest(
        concept="recent",
        media_context="movie", 
        field_name="concept",
        max_concepts=10
    )
    
    logger.info(f"📝 Testing concept: '{request.concept}'")
    
    try:

        if not await provider._ensure_initialized():
            logger.error("❌ Provider failed to initialize")
            return
        
        logger.info("✅ Provider initialized successfully")
        

        logger.info("🔍 Testing direct SpaCy parsing...")
        temporal_concepts = await provider._parse_temporal_concept(request.concept)
        logger.info(f"SpaCy parsing result: {temporal_concepts}")
        

        logger.info("🔍 Testing full expansion...")
        result = await provider.expand_concept(request)
        
        if result:
            logger.info(f"✅ Expansion successful")
            logger.info(f"Enhanced data keys: {list(result.enhanced_data.keys())}")
            
            concepts = result.enhanced_data.get('expanded_concepts', [])
            logger.info(f"Number of concepts: {len(concepts)}")
            
            for i, concept in enumerate(concepts):
                logger.info(f"  [{i}]: '{concept}' (type: {type(concept)}, len: {len(concept)})")
        else:
            logger.error("❌ No result returned")
            
    except Exception as e:
        logger.error(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_spacy_temporal())