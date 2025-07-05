"""
Test script for generic FieldExpansionCache supporting all field expansion types.
Tests cache operations for ConceptNet, Gensim, Duckling, Tag expansion, etc.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from src.data.cache_manager import get_cache_manager, CacheConfig, CacheStrategy
from src.data.field_expansion_cache import get_field_expansion_cache
from src.shared.plugin_contracts import (
    PluginResult, CacheKey, CacheType, ConfidenceScore, PluginMetadata, PluginType,
    create_field_expansion_result
)
from src.shared.config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_conceptnet_expansion(field_name: str, input_value: str) -> PluginResult:
    """Create ConceptNet expansion result."""
    expansions = {
        "action": ["fight", "combat", "battle", "intense", "fast-paced"],
        "thriller": ["suspense", "mystery", "tension", "psychological"],
        "sci-fi": ["science", "future", "technology", "space", "alien"]
    }
    
    expanded = expansions.get(input_value, [f"concept_{input_value}_1", f"concept_{input_value}_2"])
    confidence_scores = {concept: 0.8 + (i * 0.05) for i, concept in enumerate(expanded)}
    
    return create_field_expansion_result(
        field_name=field_name,
        input_value=input_value,
        expansion_result={
            "expanded_concepts": expanded,
            "original_term": input_value,
            "source": "conceptnet"
        },
        confidence_scores=confidence_scores,
        plugin_name="ConceptNetPlugin",
        plugin_version="1.0.0",
        cache_type=CacheType.CONCEPTNET,
        execution_time_ms=120.0,
        plugin_type=PluginType.CONCEPT_EXPANSION,
        api_endpoint="http://api.conceptnet.io/related"
    )


async def create_gensim_similarity(field_name: str, input_value: str) -> PluginResult:
    """Create Gensim similarity expansion result."""
    similarities = {
        "thriller": ["suspense", "mystery", "drama", "crime"],
        "comedy": ["humor", "funny", "laughs", "entertainment"],
        "action": ["adventure", "excitement", "intense", "dynamic"]
    }
    
    similar = similarities.get(input_value, [f"similar_{input_value}_1", f"similar_{input_value}_2"])
    confidence_scores = {term: 0.75 + (i * 0.05) for i, term in enumerate(similar)}
    
    return create_field_expansion_result(
        field_name=field_name,
        input_value=input_value,
        expansion_result={
            "similar_terms": similar,
            "similarity_method": "word2vec",
            "model_version": "gensim-4.3.0"
        },
        confidence_scores=confidence_scores,
        plugin_name="GensimSimilarityPlugin",
        plugin_version="1.0.0",
        cache_type=CacheType.GENSIM_SIMILARITY,
        execution_time_ms=85.0,
        plugin_type=PluginType.ENHANCEMENT,
        model_used="word2vec-google-news-300"
    )


async def create_duckling_time_parse(field_name: str, input_value: str) -> PluginResult:
    """Create Duckling time parsing result."""
    time_parses = {
        "next friday": {
            "parsed_time": "2025-07-11T00:00:00Z",
            "time_type": "date",
            "confidence": 0.95
        },
        "tomorrow": {
            "parsed_time": "2025-07-06T00:00:00Z", 
            "time_type": "date",
            "confidence": 0.99
        },
        "in 2 hours": {
            "parsed_time": "2025-07-05T12:16:00Z",
            "time_type": "datetime",
            "confidence": 0.90
        }
    }
    
    parsed = time_parses.get(input_value, {
        "parsed_time": None,
        "time_type": "unknown",
        "confidence": 0.1
    })
    
    return create_field_expansion_result(
        field_name=field_name,
        input_value=input_value,
        expansion_result=parsed,
        confidence_scores={input_value: parsed["confidence"]},
        plugin_name="DucklingTimePlugin",
        plugin_version="1.0.0",
        cache_type=CacheType.DUCKLING_TIME,
        execution_time_ms=45.0,
        plugin_type=PluginType.ENHANCEMENT,
        api_endpoint="http://localhost:8000/parse"
    )


async def create_tag_expansion(field_name: str, input_value: str) -> PluginResult:
    """Create tag expansion result."""
    tag_expansions = {
        "sci-fi": ["science fiction", "futuristic", "cyberpunk", "space opera"],
        "horror": ["scary", "frightening", "supernatural", "thriller"],
        "romance": ["love story", "romantic", "relationship", "drama"]
    }
    
    expanded = tag_expansions.get(input_value, [f"tag_{input_value}_alt"])
    confidence_scores = {tag: 0.85 for tag in expanded}
    
    return create_field_expansion_result(
        field_name=field_name,
        input_value=input_value,
        expansion_result={
            "expanded_tags": expanded,
            "tag_category": "genre",
            "expansion_method": "synonym_lookup"
        },
        confidence_scores=confidence_scores,
        plugin_name="TagExpansionPlugin",
        plugin_version="1.0.0",
        cache_type=CacheType.TAG_EXPANSION,
        execution_time_ms=25.0,
        plugin_type=PluginType.ENHANCEMENT
    )


async def create_spacy_ner(field_name: str, input_value: str) -> PluginResult:
    """Create SpaCy NER result."""
    ner_results = {
        "Tom Cruise stars in this action movie": {
            "entities": [
                {"text": "Tom Cruise", "label": "PERSON", "start": 0, "end": 10}
            ],
            "people": ["Tom Cruise"],
            "organizations": [],
            "locations": []
        },
        "Set in New York during the 1980s": {
            "entities": [
                {"text": "New York", "label": "GPE", "start": 7, "end": 15},
                {"text": "1980s", "label": "DATE", "start": 27, "end": 32}
            ],
            "people": [],
            "organizations": [],
            "locations": ["New York"]
        }
    }
    
    result = ner_results.get(input_value, {
        "entities": [],
        "people": [],
        "organizations": [],
        "locations": []
    })
    
    confidence_scores = {entity["text"]: 0.92 for entity in result["entities"]}
    
    return create_field_expansion_result(
        field_name=field_name,
        input_value=input_value,
        expansion_result=result,
        confidence_scores=confidence_scores,
        plugin_name="SpacyNERPlugin",
        plugin_version="1.0.0",
        cache_type=CacheType.SPACY_NER,
        execution_time_ms=75.0,
        plugin_type=PluginType.ENHANCEMENT,
        model_used="en_core_web_sm"
    )


async def test_multiple_expansion_types():
    """Test caching with different field expansion types."""
    logger.info("Testing multiple field expansion types...")
    
    try:
        manager = get_cache_manager()
        await manager.initialize()
        
        # Test cases: (field_name, input_value, expansion_type, compute_func)
        test_cases = [
            ("tags", "action", CacheType.CONCEPTNET, lambda: create_conceptnet_expansion("tags", "action")),
            ("genres", "thriller", CacheType.GENSIM_SIMILARITY, lambda: create_gensim_similarity("genres", "thriller")),
            ("release_date", "next friday", CacheType.DUCKLING_TIME, lambda: create_duckling_time_parse("release_date", "next friday")),
            ("tags", "sci-fi", CacheType.TAG_EXPANSION, lambda: create_tag_expansion("tags", "sci-fi")),
            ("overview", "Tom Cruise stars in this action movie", CacheType.SPACY_NER, lambda: create_spacy_ner("overview", "Tom Cruise stars in this action movie"))
        ]
        
        results = {}
        
        for field_name, input_value, cache_type, compute_func in test_cases:
            logger.info(f"Testing {cache_type.value} expansion for {field_name}='{input_value}'")
            
            cache_key = manager.generate_cache_key(cache_type, field_name, input_value)
            
            # First call - should compute and cache
            result1 = await manager.get_or_compute(cache_key, compute_func)
            if result1:
                logger.info(f"✅ {cache_type.value} expansion computed and cached")
                results[f"{cache_type.value}_compute"] = True
            else:
                logger.error(f"❌ {cache_type.value} expansion failed")
                results[f"{cache_type.value}_compute"] = False
                continue
            
            # Second call - should be cache hit
            result2 = await manager.get_or_compute(cache_key, compute_func)
            if result2:
                logger.info(f"✅ {cache_type.value} expansion retrieved from cache")
                results[f"{cache_type.value}_cache_hit"] = True
            else:
                logger.error(f"❌ {cache_type.value} cache retrieval failed")
                results[f"{cache_type.value}_cache_hit"] = False
        
        # Check results
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        logger.info(f"Multiple expansion types test: {success_count}/{total_count} operations successful")
        return success_count == total_count
        
    except Exception as e:
        logger.error(f"❌ Multiple expansion types test failed: {e}")
        return False


async def test_cache_key_format():
    """Test that cache keys have correct format for field expansions."""
    logger.info("Testing cache key format...")
    
    try:
        manager = get_cache_manager()
        
        # Test different key formats
        test_cases = [
            (CacheType.CONCEPTNET, "tags", "action", "movie", "conceptnet:tags:action:movie"),
            (CacheType.GENSIM_SIMILARITY, "genres", "thriller", "book", "gensim_similarity:genres:thriller:book"),
            (CacheType.DUCKLING_TIME, "release_date", "tomorrow", "movie", "duckling_time:release_date:tomorrow:movie"),
            (CacheType.TAG_EXPANSION, "categories", "sci-fi", "book", "tag_expansion:categories:sci_fi:book")  # Note: "sci-fi" becomes "sci_fi"
        ]
        
        for cache_type, field_name, input_value, media_context, expected_key in test_cases:
            cache_key = manager.generate_cache_key(cache_type, field_name, input_value, media_context)
            actual_key = cache_key.generate_key()
            
            if actual_key == expected_key:
                logger.info(f"✅ Cache key format correct: {actual_key}")
            else:
                logger.error(f"❌ Cache key format incorrect. Expected: {expected_key}, Got: {actual_key}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cache key format test failed: {e}")
        return False


async def test_cache_warm_field_values():
    """Test cache warming for field values."""
    logger.info("Testing cache warming for field values...")
    
    try:
        manager = get_cache_manager()
        
        # Test warming tag values
        tag_values = ["action", "thriller", "comedy"]
        
        async def compute_tag_expansion(value: str) -> PluginResult:
            return await create_tag_expansion("tags", value)
        
        results = await manager.warm_cache_for_field_values(
            field_name="tags",
            values=tag_values,
            cache_type=CacheType.TAG_EXPANSION,
            compute_func=compute_tag_expansion
        )
        
        success_count = sum(1 for success in results.values() if success)
        if success_count == len(tag_values):
            logger.info(f"✅ Cache warming successful for {success_count} tag values")
            return True
        else:
            logger.error(f"❌ Cache warming failed. Only {success_count}/{len(tag_values)} successful")
            return False
            
    except Exception as e:
        logger.error(f"❌ Cache warming test failed: {e}")
        return False


async def test_actual_cache_documents():
    """Test that actual MongoDB documents have correct structure."""
    logger.info("Testing actual cache document structure...")
    
    try:
        cache = get_field_expansion_cache()
        
        # Create a test expansion
        result = await create_conceptnet_expansion("tags", "horror")
        await cache.store_result(result)
        
        # Retrieve and check structure
        retrieved = await cache.get_cached_result(result.cache_key)
        
        if retrieved:
            # Verify structure
            cache_doc = retrieved.to_cache_document()
            required_fields = [
                "cache_key", "field_name", "input_value", "media_type",
                "expansion_type", "expansion_result", "confidence_scores",
                "source_metadata", "created_at"
            ]
            
            missing_fields = [field for field in required_fields if field not in cache_doc]
            
            if not missing_fields:
                logger.info("✅ Cache document structure is correct")
                logger.info(f"Sample cache_key: {cache_doc['cache_key']}")
                logger.info(f"Field name: {cache_doc['field_name']}")
                logger.info(f"Input value: {cache_doc['input_value']}")
                logger.info(f"Expansion type: {cache_doc['expansion_type']}")
                return True
            else:
                logger.error(f"❌ Missing fields in cache document: {missing_fields}")
                return False
        else:
            logger.error("❌ Failed to retrieve cached result")
            return False
            
    except Exception as e:
        logger.error(f"❌ Cache document structure test failed: {e}")
        return False


async def run_all_generic_cache_tests():
    """Run all generic cache tests."""
    logger.info("=" * 60)
    logger.info("STARTING GENERIC FIELD EXPANSION CACHE TESTS")
    logger.info("=" * 60)
    
    # Show environment info
    settings = get_settings()
    logger.info(f"Environment: {settings.ENV}")
    logger.info(f"MongoDB URL: {settings.mongodb_url}")
    
    tests = [
        ("Multiple Expansion Types", test_multiple_expansion_types),
        ("Cache Key Format", test_cache_key_format),
        ("Cache Warming Field Values", test_cache_warm_field_values),
        ("Actual Cache Documents", test_actual_cache_documents)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = await test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("GENERIC CACHE TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Generic Field Expansion Cache is working!")
        return True
    else:
        logger.error("❌ Some tests failed - check logs for details")
        return False


if __name__ == "__main__":
    asyncio.run(run_all_generic_cache_tests())