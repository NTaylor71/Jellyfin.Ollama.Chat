"""
Test script for Stage 2.1: ConceptCache and CacheManager functionality.
Tests cache operations with actual MongoDB via docker environment.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from src.data.cache_manager import get_cache_manager, CacheConfig, CacheStrategy
from src.data.field_expansion_cache import get_concept_cache
from src.shared.plugin_contracts import (
    PluginResult, CacheKey, CacheType, ConfidenceScore, PluginMetadata, PluginType,
    create_concept_expansion_result
)
from src.shared.config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_test_plugin_result(term: str, cache_type: CacheType = CacheType.CONCEPTNET) -> PluginResult:
    """Create a test plugin result for caching."""
    
    # Simulate concept expansion results
    expanded_concepts = {
        "action": ["fight", "combat", "battle", "intense", "fast-paced"],
        "thriller": ["suspense", "mystery", "tension", "psychological", "dark"],
        "comedy": ["humor", "funny", "laugh", "light-hearted", "amusing"]
    }.get(term, [f"expanded_{term}_1", f"expanded_{term}_2"])
    
    confidence_scores = {concept: 0.8 + (i * 0.05) for i, concept in enumerate(expanded_concepts)}
    
    return create_concept_expansion_result(
        input_term=term,
        expanded_concepts=expanded_concepts,
        confidence_scores=confidence_scores,
        plugin_name="TestConceptPlugin",
        plugin_version="1.0.0",
        cache_type=cache_type,
        execution_time_ms=150.0,
        media_context="movie",
        api_endpoint="http://test.conceptnet.io/test",
        model_used="test_model"
    )


async def test_mongodb_connection():
    """Test MongoDB connection."""
    logger.info("Testing MongoDB connection...")
    
    try:
        cache = get_concept_cache()
        
        # Test sync connection
        if cache.client.test_sync_connection():
            logger.info("✅ MongoDB sync connection successful")
        else:
            logger.error("❌ MongoDB sync connection failed")
            return False
        
        # Test async connection
        if await cache.client.test_connection():
            logger.info("✅ MongoDB async connection successful")
        else:
            logger.error("❌ MongoDB async connection failed")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MongoDB connection test failed: {e}")
        return False


async def test_cache_initialization():
    """Test cache initialization."""
    logger.info("Testing cache initialization...")
    
    try:
        cache = get_concept_cache()
        
        # Initialize collection
        if await cache.initialize_collection():
            logger.info("✅ ConceptCache collection initialized")
        else:
            logger.error("❌ ConceptCache collection initialization failed")
            return False
        
        # Check collection info
        info = cache.get_collection_info()
        logger.info(f"Collection info: {info}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cache initialization test failed: {e}")
        return False


async def test_basic_cache_operations():
    """Test basic cache store/retrieve operations."""
    logger.info("Testing basic cache operations...")
    
    try:
        cache = get_concept_cache()
        
        # Create test result
        test_result = await create_test_plugin_result("action", CacheType.CONCEPTNET)
        
        # Store in cache
        if await cache.store_result(test_result):
            logger.info("✅ Successfully stored result in cache")
        else:
            logger.error("❌ Failed to store result in cache")
            return False
        
        # Retrieve from cache
        cache_key = test_result.cache_key
        retrieved = await cache.get_cached_result(cache_key)
        
        if retrieved:
            logger.info("✅ Successfully retrieved result from cache")
            logger.info(f"Retrieved: {retrieved.enhanced_data}")
        else:
            logger.error("❌ Failed to retrieve result from cache")
            return False
        
        # Test cache hit check
        hit = await cache.check_cache_hit(cache_key)
        if hit:
            logger.info("✅ Cache hit check successful")
        else:
            logger.error("❌ Cache hit check failed")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic cache operations test failed: {e}")
        return False


async def test_cache_manager():
    """Test CacheManager functionality."""
    logger.info("Testing CacheManager...")
    
    try:
        # Initialize cache manager
        config = CacheConfig(ttl_seconds=3600)
        manager = get_cache_manager(config)
        
        if await manager.initialize():
            logger.info("✅ CacheManager initialized successfully")
        else:
            logger.error("❌ CacheManager initialization failed")
            return False
        
        # Test cache-first operation
        async def mock_compute_func() -> PluginResult:
            return await create_test_plugin_result("thriller", CacheType.LLM)
        
        cache_key = manager.generate_cache_key(
            cache_type=CacheType.LLM,
            field_name="concept",
            input_value="thriller",
            media_context="movie"
        )
        
        # First call should be cache miss and compute
        result1 = await manager.get_or_compute(cache_key, mock_compute_func)
        if result1:
            logger.info("✅ First call successful (cache miss + compute)")
        else:
            logger.error("❌ First call failed")
            return False
        
        # Second call should be cache hit
        result2 = await manager.get_or_compute(cache_key, mock_compute_func)
        if result2:
            logger.info("✅ Second call successful (cache hit)")
        else:
            logger.error("❌ Second call failed")
            return False
        
        # Test cache metrics
        metrics = await manager.get_cache_metrics()
        logger.info(f"Cache metrics: {metrics}")
        
        # Test health check
        health = await manager.health_check()
        logger.info(f"Health check: {health}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ CacheManager test failed: {e}")
        return False


async def test_multiple_cache_types():
    """Test caching with different cache types."""
    logger.info("Testing multiple cache types...")
    
    try:
        manager = get_cache_manager()
        
        # Test different cache types
        test_cases = [
            (CacheType.CONCEPTNET, "action"),
            (CacheType.LLM, "psychological"),
            (CacheType.GENSIM, "similarity"),
            (CacheType.NLTK, "tokenize"),
            (CacheType.CUSTOM, "custom_analysis")
        ]
        
        for cache_type, term in test_cases:
            async def compute_func():
                return await create_test_plugin_result(term, cache_type)
            
            cache_key = manager.generate_cache_key(cache_type, "concept", term)
            result = await manager.get_or_compute(cache_key, compute_func)
            
            if result:
                logger.info(f"✅ {cache_type.value} cache test successful for term: {term}")
            else:
                logger.error(f"❌ {cache_type.value} cache test failed for term: {term}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Multiple cache types test failed: {e}")
        return False


async def test_cache_stats():
    """Test cache statistics and monitoring."""
    logger.info("Testing cache statistics...")
    
    try:
        cache = get_concept_cache()
        
        # Get cache stats
        stats = await cache.get_cache_stats()
        logger.info(f"Cache stats: {stats}")
        
        # Should have some documents from previous tests
        if stats.get("total_documents", 0) > 0:
            logger.info("✅ Cache contains documents from tests")
        else:
            logger.warning("⚠️ No documents found in cache")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cache stats test failed: {e}")
        return False


async def run_all_tests():
    """Run all cache tests."""
    logger.info("=" * 60)
    logger.info("STARTING STAGE 2.1 CACHE TESTS")
    logger.info("=" * 60)
    
    # Show environment info
    settings = get_settings()
    logger.info(f"Environment: {settings.ENV}")
    logger.info(f"MongoDB URL: {settings.mongodb_url}")
    
    tests = [
        ("MongoDB Connection", test_mongodb_connection),
        ("Cache Initialization", test_cache_initialization),
        ("Basic Cache Operations", test_basic_cache_operations),
        ("Cache Manager", test_cache_manager),
        ("Multiple Cache Types", test_multiple_cache_types),
        ("Cache Statistics", test_cache_stats)
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
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Stage 2.1 Cache Infrastructure is working!")
        return True
    else:
        logger.error("❌ Some tests failed - check logs for details")
        return False


if __name__ == "__main__":
    asyncio.run(run_all_tests())