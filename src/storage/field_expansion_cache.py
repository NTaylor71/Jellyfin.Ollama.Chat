"""
FieldExpansionCache collection for Stage 2: Never call expansion plugins twice for same field input.
Generic caching for all data ingestion field expansions: ConceptNet, Gensim, SpaCy Temporal, etc.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pymongo import IndexModel, ASCENDING, TEXT
from pymongo.errors import DuplicateKeyError
from motor.motor_asyncio import AsyncIOMotorCollection

from src.storage.mongodb_client import get_mongodb_client, get_async_db
from src.shared.plugin_contracts import PluginResult, CacheKey, CacheType

logger = logging.getLogger(__name__)


class FieldExpansionCache:
    """
    Generic field expansion cache for all data ingestion plugins.
    
    Supports any field expansion during data ingestion:
    {
      "_id": ObjectId,
      "cache_key": "conceptnet:tags:action:movie",
      "field_name": "tags",
      "input_value": "action",
      "media_type": "movie", 
      "expansion_type": "conceptnet",
      "expansion_result": {...},  // Generic result structure
      "confidence_scores": {"fight": 0.9, "combat": 0.85},
      "source_metadata": {...},
      "created_at": ISODate,
      "expires_at": ISODate
    }
    
    Examples of supported expansions:
    - ConceptNet: "conceptnet:tags:action:movie" 
    - Gensim: "gensim_similarity:genres:thriller:movie"
    - SpaCy Temporal: "spacy_temporal:release_date:next friday:movie"
    - Tag expansion: "tag_expansion:tags:sci-fi:movie"
    """
    
    COLLECTION_NAME = "field_expansion_cache"
    
    def __init__(self):
        self.client = get_mongodb_client()
        self._collection: Optional[AsyncIOMotorCollection] = None
        self._sync_collection = None
        
    @property
    async def collection(self) -> AsyncIOMotorCollection:
        """Get async collection instance."""
        if self._collection is None:
            db = await get_async_db()
            self._collection = db[self.COLLECTION_NAME]
        return self._collection
    
    @property
    def sync_collection(self):
        """Get sync collection instance."""
        if self._sync_collection is None:
            db = self.client.sync_db
            self._sync_collection = db[self.COLLECTION_NAME]
        return self._sync_collection
    
    async def initialize_collection(self) -> bool:
        """
        Initialize the FieldExpansionCache collection with proper indexes.
        
        Creates:
        - Unique index on cache_key (primary lookup)
        - Index on field_name + input_value (field-based queries)
        - Index on media_type (media-specific queries)
        - Index on expansion_type (plugin-specific queries)
        - TTL index on expires_at (automatic cleanup)
        - Text index on expansion_result (fuzzy search)
        """
        try:
            collection = await self.collection
            

            indexes = [

                IndexModel([("cache_key", ASCENDING)], unique=True, name="cache_key_unique"),
                

                IndexModel([("field_name", ASCENDING), ("input_value", ASCENDING)], name="field_value_idx"),
                

                IndexModel([("field_name", ASCENDING), ("media_type", ASCENDING)], name="field_media_idx"),
                

                IndexModel([("expansion_type", ASCENDING)], name="expansion_type_idx"),
                

                IndexModel([("expires_at", ASCENDING)], expireAfterSeconds=0, name="ttl_idx"),
                

                IndexModel([("created_at", ASCENDING)], name="created_at_idx"),
                

                IndexModel([("expansion_result", TEXT)], name="expansion_result_text"),
                

                IndexModel([("overall_confidence", ASCENDING)], name="confidence_idx")
            ]
            
            
            await collection.create_indexes(indexes)
            logger.info(f"FieldExpansionCache collection initialized with {len(indexes)} indexes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FieldExpansionCache collection: {e}")
            return False
    
    async def store_result(self, result: PluginResult) -> bool:
        """
        Store a plugin result in the cache.
        
        Args:
            result: PluginResult to cache
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            collection = await self.collection
            cache_doc = result.to_cache_document()
            

            await collection.replace_one(
                {"cache_key": cache_doc["cache_key"]},
                cache_doc,
                upsert=True
            )
            
            logger.debug(f"Cached result for key: {cache_doc['cache_key']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store cache result: {e}")
            return False
    
    async def get_cached_result(self, cache_key: Union[str, CacheKey]) -> Optional[PluginResult]:
        """
        Retrieve a cached result by cache key.
        
        Args:
            cache_key: Cache key string or CacheKey object
            
        Returns:
            PluginResult if found and not expired, None otherwise
        """
        try:
            collection = await self.collection
            

            key_str = cache_key.generate_key() if isinstance(cache_key, CacheKey) else cache_key
            

            doc = await collection.find_one({"cache_key": key_str})
            
            if not doc:
                logger.debug(f"No cached result found for key: {key_str}")
                return None
            
            
            if doc.get("expires_at") and doc["expires_at"] < datetime.utcnow().timestamp():
                logger.debug(f"Cached result expired for key: {key_str}")
                return None
            

            result = PluginResult.from_cache_document(doc)
            logger.debug(f"Retrieved cached result for key: {key_str}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached result: {e}")
            return None
    
    async def check_cache_hit(self, cache_key: Union[str, CacheKey]) -> bool:
        """
        Check if a cache key exists without retrieving the full result.
        
        Args:
            cache_key: Cache key string or CacheKey object
            
        Returns:
            True if cache hit, False otherwise
        """
        try:
            collection = await self.collection
            

            key_str = cache_key.generate_key() if isinstance(cache_key, CacheKey) else cache_key
            
            
            doc = await collection.find_one(
                {"cache_key": key_str},
                {"_id": 1, "expires_at": 1}
            )
            
            if not doc:
                return False
            
            
            if doc.get("expires_at") and doc["expires_at"] < datetime.utcnow().timestamp():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check cache hit: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            collection = await self.collection
            

            total_docs = await collection.count_documents({})
            

            pipeline = [
                {"$group": {
                    "_id": "$expansion_type",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$overall_confidence"}
                }}
            ]
            
            expansion_stats = {}
            async for doc in collection.aggregate(pipeline):
                expansion_stats[doc["_id"]] = {
                    "count": doc["count"],
                    "avg_confidence": doc.get("avg_confidence", 0.0)
                }
            

            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            recent_docs = await collection.count_documents({
                "created_at": {"$gte": recent_cutoff}
            })
            
            return {
                "total_documents": total_docs,
                "expansion_type_stats": expansion_stats,
                "recent_activity_24h": recent_docs,
                "collection_name": self.COLLECTION_NAME
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}
    
    async def clear_expired_entries(self) -> int:
        """
        Manually clear expired entries (TTL should handle this automatically).
        
        Returns:
            Number of entries cleared
        """
        try:
            collection = await self.collection
            

            current_time = datetime.utcnow().timestamp()
            result = await collection.delete_many({
                "expires_at": {"$lt": current_time}
            })
            
            if result.deleted_count > 0:
                logger.info(f"Cleared {result.deleted_count} expired cache entries")
            
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Failed to clear expired entries: {e}")
            return 0
    
    async def warm_cache(self, terms: List[str], media_type: str = "movie") -> Dict[str, bool]:
        """
        Pre-populate cache with common terms (for future use).
        
        Args:
            terms: List of terms to warm cache for
            media_type: Media type context
            
        Returns:
            Dictionary of term -> success status
        """


        logger.info(f"Cache warming requested for {len(terms)} terms (not implemented yet)")
        return {term: False for term in terms}
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get sync collection information."""
        return self.client.get_collection_info(self.COLLECTION_NAME)



_field_expansion_cache: Optional[FieldExpansionCache] = None


def get_field_expansion_cache() -> FieldExpansionCache:
    """Get singleton FieldExpansionCache instance."""
    global _field_expansion_cache
    if _field_expansion_cache is None:
        _field_expansion_cache = FieldExpansionCache()
    return _field_expansion_cache



def get_concept_cache() -> FieldExpansionCache:
    """Backward compatibility alias. Use get_field_expansion_cache() for new code."""
    return get_field_expansion_cache()