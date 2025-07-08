"""
CacheManager class for Stage 2.2: Cache Management Service.
Implements the cache-first pattern for concept expansion and intelligence results.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from src.storage.field_expansion_cache import get_field_expansion_cache
from src.shared.plugin_contracts import PluginResult, CacheKey, CacheType, ConfidenceScore
from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies for different types of operations."""
    CACHE_FIRST = "cache_first"       # Check cache first, call API if miss
    CACHE_ONLY = "cache_only"         # Only return cached results
    BYPASS_CACHE = "bypass_cache"     # Skip cache, always call API
    REFRESH_CACHE = "refresh_cache"   # Force refresh cache entry


@dataclass
class CacheConfig:
    """Configuration for cache operations."""
    ttl_seconds: int = 3600           # Default 1 hour
    max_retries: int = 3              # Max retries for failed operations
    retry_delay_seconds: int = 1      # Delay between retries
    enable_cache_warming: bool = True # Enable cache warming
    cache_key_prefix: str = ""        # Optional prefix for cache keys


class CacheManager:
    """
    High-level cache management service implementing cache-first patterns.
    
    Handles:
    - Cache key generation and management
    - TTL management with configurable expiration
    - Cache warming for common terms
    - Graceful degradation when cache is unavailable
    - Metrics collection for cache performance
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.settings = get_settings()
        self.cache = get_field_expansion_cache()
        self._cache_metrics = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "cache_operations": 0
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the cache manager and ensure cache is ready.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize the field expansion cache collection
            success = await self.cache.initialize_collection()
            if success:
                logger.info("CacheManager initialized successfully")
                return True
            else:
                logger.error("Failed to initialize CacheManager")
                return False
        except Exception as e:
            logger.error(f"CacheManager initialization failed: {e}")
            return False
    
    async def get_or_compute(
        self,
        cache_key: Union[str, CacheKey],
        compute_func: Callable[[], Awaitable[PluginResult]],
        strategy: CacheStrategy = CacheStrategy.CACHE_FIRST,
        ttl_seconds: Optional[int] = None
    ) -> Optional[PluginResult]:
        """
        Core cache-first operation: check cache first, compute if miss.
        
        Args:
            cache_key: Cache key for the operation
            compute_func: Async function to compute result if cache miss
            strategy: Cache strategy to use
            ttl_seconds: TTL override for this operation
            
        Returns:
            PluginResult if successful, None if failed
        """
        try:
            self._cache_metrics["cache_operations"] += 1
            
            # Handle different cache strategies
            if strategy == CacheStrategy.CACHE_ONLY:
                return await self._get_cached_only(cache_key)
            elif strategy == CacheStrategy.BYPASS_CACHE:
                return await self._compute_and_cache(cache_key, compute_func, ttl_seconds)
            elif strategy == CacheStrategy.REFRESH_CACHE:
                return await self._refresh_cache(cache_key, compute_func, ttl_seconds)
            else:  # CACHE_FIRST
                return await self._cache_first(cache_key, compute_func, ttl_seconds)
                
        except Exception as e:
            logger.error(f"Cache operation failed: {e}")
            self._cache_metrics["errors"] += 1
            return None
    
    async def _cache_first(
        self,
        cache_key: Union[str, CacheKey],
        compute_func: Callable[[], Awaitable[PluginResult]],
        ttl_seconds: Optional[int]
    ) -> Optional[PluginResult]:
        """Execute cache-first strategy."""
        # Check cache first
        cached_result = await self.cache.get_cached_result(cache_key)
        
        if cached_result:
            logger.debug(f"Cache hit for key: {cache_key}")
            self._cache_metrics["hits"] += 1
            return cached_result
        
        # Cache miss - compute and cache
        logger.debug(f"Cache miss for key: {cache_key}")
        self._cache_metrics["misses"] += 1
        
        return await self._compute_and_cache(cache_key, compute_func, ttl_seconds)
    
    async def _get_cached_only(self, cache_key: Union[str, CacheKey]) -> Optional[PluginResult]:
        """Get result from cache only."""
        result = await self.cache.get_cached_result(cache_key)
        if result:
            self._cache_metrics["hits"] += 1
        else:
            self._cache_metrics["misses"] += 1
        return result
    
    async def _compute_and_cache(
        self,
        cache_key: Union[str, CacheKey],
        compute_func: Callable[[], Awaitable[PluginResult]],
        ttl_seconds: Optional[int]
    ) -> Optional[PluginResult]:
        """Compute result and cache it."""
        try:
            # Compute the result
            result = await compute_func()
            
            if result:
                # Set TTL if provided
                if ttl_seconds:
                    result.cache_ttl_seconds = ttl_seconds
                elif result.cache_ttl_seconds is None:
                    result.cache_ttl_seconds = self.config.ttl_seconds
                
                # Cache the result
                await self.cache.store_result(result)
                logger.debug(f"Computed and cached result for key: {cache_key}")
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to compute and cache result: {e}")
            return None
    
    async def _refresh_cache(
        self,
        cache_key: Union[str, CacheKey],
        compute_func: Callable[[], Awaitable[PluginResult]],
        ttl_seconds: Optional[int]
    ) -> Optional[PluginResult]:
        """Force refresh cache entry."""
        logger.debug(f"Refreshing cache for key: {cache_key}")
        return await self._compute_and_cache(cache_key, compute_func, ttl_seconds)
    
    def generate_cache_key(
        self,
        cache_type: CacheType,
        field_name: str,
        input_value: str,
        media_context: str = "movie",
        **kwargs
    ) -> CacheKey:
        """
        Generate consistent cache key for field expansion operations.
        
        Args:
            cache_type: Type of expansion (conceptnet, gensim, spacy_temporal, etc.)
            field_name: Name of field being expanded (tags, genres, people, etc.)
            input_value: Value being processed
            media_context: Media context (movie, book, etc.)
            **kwargs: Additional parameters for cache key
            
        Returns:
            CacheKey object
        """
        # Apply prefix if configured
        if self.config.cache_key_prefix:
            input_value = f"{self.config.cache_key_prefix}:{input_value}"
        
        return CacheKey(
            cache_type=cache_type,
            field_name=field_name,
            input_value=input_value,
            media_context=media_context
        )
    
    async def bulk_check_cache(self, cache_keys: List[Union[str, CacheKey]]) -> Dict[str, bool]:
        """
        Check multiple cache keys for existence.
        
        Args:
            cache_keys: List of cache keys to check
            
        Returns:
            Dictionary of cache_key -> hit status
        """
        results = {}
        
        for cache_key in cache_keys:
            key_str = cache_key.generate_key() if isinstance(cache_key, CacheKey) else cache_key
            hit = await self.cache.check_cache_hit(cache_key)
            results[key_str] = hit
        
        return results
    
    async def warm_cache_for_field_values(
        self,
        field_name: str,
        values: List[str],
        cache_type: CacheType,
        media_context: str = "movie",
        compute_func: Optional[Callable[[str], Awaitable[PluginResult]]] = None
    ) -> Dict[str, bool]:
        """
        Warm cache for a list of field values.
        
        Args:
            field_name: Name of field being warmed (tags, genres, etc.)
            values: List of values to warm cache for
            cache_type: Type of expansion operation
            media_context: Media context
            compute_func: Optional function to compute results for cache misses
            
        Returns:
            Dictionary of value -> success status
        """
        if not self.config.enable_cache_warming:
            logger.debug("Cache warming disabled")
            return {value: False for value in values}
        
        results = {}
        
        for value in values:
            cache_key = self.generate_cache_key(cache_type, field_name, value, media_context)
            
            # Check if already cached
            if await self.cache.check_cache_hit(cache_key):
                results[value] = True
                continue
            
            # Compute if function provided
            if compute_func:
                try:
                    result = await compute_func(value)
                    if result:
                        await self.cache.store_result(result)
                        results[value] = True
                    else:
                        results[value] = False
                except Exception as e:
                    logger.error(f"Failed to warm cache for {field_name} value '{value}': {e}")
                    results[value] = False
            else:
                results[value] = False
        
        logger.info(f"Cache warming completed for {len(values)} {field_name} values")
        return results
    
    async def clear_cache_by_prefix(self, prefix: str) -> int:
        """
        Clear cache entries by key prefix.
        
        Args:
            prefix: Prefix to match for deletion
            
        Returns:
            Number of entries cleared
        """
        # This would require a more complex MongoDB query
        # For now, we'll log the request and return 0
        logger.warning(f"Clear cache by prefix '{prefix}' not implemented yet")
        return 0
    
    async def get_cache_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics.
        
        Returns:
            Dictionary with cache metrics
        """
        cache_stats = await self.cache.get_cache_stats()
        
        # Add our own metrics
        total_operations = self._cache_metrics["hits"] + self._cache_metrics["misses"]
        hit_rate = (self._cache_metrics["hits"] / total_operations) if total_operations > 0 else 0.0
        
        return {
            "cache_stats": cache_stats,
            "performance_metrics": {
                "hit_rate": hit_rate,
                "total_operations": total_operations,
                "cache_hits": self._cache_metrics["hits"],
                "cache_misses": self._cache_metrics["misses"],
                "cache_errors": self._cache_metrics["errors"],
                "total_cache_operations": self._cache_metrics["cache_operations"]
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform cache health check.
        
        Returns:
            Health status information
        """
        try:
            # Test basic cache operations
            test_key = CacheKey(
                cache_type=CacheType.CUSTOM,
                field_name="health",
                input_value="health_check",
                media_context="test"
            )
            
            # Check if we can check cache hit
            can_check = await self.cache.check_cache_hit(test_key)
            
            return {
                "status": "healthy",
                "cache_accessible": True,
                "test_key_exists": can_check,
                "collection_info": self.cache.get_collection_info()
            }
            
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {
                "status": "unhealthy",
                "cache_accessible": False,
                "error": str(e)
            }


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(config: Optional[CacheConfig] = None) -> CacheManager:
    """Get singleton CacheManager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(config)
    return _cache_manager