#!/usr/bin/env python3
"""
Debug the SpaCy Temporal provider to see exactly what's happening.
"""

import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.providers.nlp.spacy_temporal_provider import SpacyTemporalProvider
from src.providers.nlp.base_provider import ExpansionRequest

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_spacy_temporal():
    """Debug SpaCy temporal provider step by step."""
    logger.info("🔍 DEBUGGING SpaCy Temporal Provider")
    
    provider = SpacyTemporalProvider()
    
    # Test the specific case that's failing
    request = ExpansionRequest(
        concept="recent",
        media_context="movie", 
        field_name="concept",
        max_concepts=10
    )
    
    logger.info(f"📝 Testing concept: '{request.concept}'")
    
    try:
        # Step 1: Check if provider initializes
        if not await provider._ensure_initialized():
            logger.error("❌ Provider failed to initialize")
            return
        
        logger.info("✅ Provider initialized successfully")
        
        # Step 2: Test direct temporal parsing
        logger.info("🔍 Testing direct SpaCy parsing...")
        temporal_concepts = await provider._parse_temporal_concept(request.concept)
        logger.info(f"SpaCy parsing result: {temporal_concepts}")
        
        # Step 3: Test full expansion
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