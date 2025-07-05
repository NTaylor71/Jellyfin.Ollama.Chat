"""
Cache administration utilities for testing and development.
Provides endpoints to clear, inspect, and manage the concept expansion cache.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.data.field_expansion_cache import get_field_expansion_cache
from src.data.cache_manager import get_cache_manager

logger = logging.getLogger(__name__)


class CacheAdmin:
    """
    Administrative interface for concept expansion cache management.
    
    Provides utilities for testing and development to clear, inspect,
    and manage cache entries.
    """
    
    def __init__(self):
        self.cache = get_field_expansion_cache()
        self.cache_manager = get_cache_manager()
    
    async def clear_all_cache(self) -> Dict[str, Any]:
        """
        Clear all cache entries.
        
        Returns:
            Dictionary with operation results
        """
        try:
            collection = await self.cache.collection
            result = await collection.delete_many({})
            
            logger.info(f"Cleared {result.deleted_count} cache entries")
            return {
                "success": True,
                "deleted_count": result.deleted_count,
                "message": f"Cleared {result.deleted_count} cache entries"
            }
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return {
                "success": False,
                "deleted_count": 0,
                "error": str(e)
            }
    
    async def clear_cache_by_pattern(self, pattern: str) -> Dict[str, Any]:
        """
        Clear cache entries matching a pattern.
        
        Args:
            pattern: Regex pattern to match cache keys
            
        Returns:
            Dictionary with operation results
        """
        try:
            collection = await self.cache.collection
            
            # Use regex to match cache keys
            query = {"cache_key": {"$regex": pattern, "$options": "i"}}
            result = await collection.delete_many(query)
            
            logger.info(f"Cleared {result.deleted_count} cache entries matching '{pattern}'")
            return {
                "success": True,
                "deleted_count": result.deleted_count,
                "pattern": pattern,
                "message": f"Cleared {result.deleted_count} entries matching '{pattern}'"
            }
        except Exception as e:
            logger.error(f"Failed to clear cache by pattern '{pattern}': {e}")
            return {
                "success": False,
                "deleted_count": 0,
                "pattern": pattern,
                "error": str(e)
            }
    
    async def clear_cache_by_method(self, expansion_type: str) -> Dict[str, Any]:
        """
        Clear cache entries for specific expansion method.
        
        Args:
            expansion_type: Type of expansion (conceptnet, llm, gensim, etc.)
            
        Returns:
            Dictionary with operation results
        """
        try:
            collection = await self.cache.collection
            
            query = {"expansion_type": expansion_type}
            result = await collection.delete_many(query)
            
            logger.info(f"Cleared {result.deleted_count} {expansion_type} cache entries")
            return {
                "success": True,
                "deleted_count": result.deleted_count,
                "expansion_type": expansion_type,
                "message": f"Cleared {result.deleted_count} {expansion_type} entries"
            }
        except Exception as e:
            logger.error(f"Failed to clear {expansion_type} cache: {e}")
            return {
                "success": False,
                "deleted_count": 0,
                "expansion_type": expansion_type,
                "error": str(e)
            }
    
    async def get_cache_summary(self) -> Dict[str, Any]:
        """
        Get summary of current cache contents.
        
        Returns:
            Dictionary with cache statistics and summary
        """
        try:
            collection = await self.cache.collection
            
            # Basic counts
            total_docs = await collection.count_documents({})
            
            # Count by expansion type
            pipeline = [
                {"$group": {
                    "_id": "$expansion_type",
                    "count": {"$sum": 1},
                    "avg_concepts": {"$avg": {"$size": "$expansion_result.expanded_concepts"}}
                }}
            ]
            
            expansion_stats = {}
            async for doc in collection.aggregate(pipeline):
                expansion_stats[doc["_id"]] = {
                    "count": doc["count"],
                    "avg_concepts": round(doc.get("avg_concepts", 0), 2)
                }
            
            # Recent entries
            recent_docs = []
            async for doc in collection.find({}, {
                "cache_key": 1, 
                "expansion_type": 1, 
                "created_at": 1,
                "expansion_result.expanded_concepts": 1
            }).sort("created_at", -1).limit(10):
                concepts = doc.get("expansion_result", {}).get("expanded_concepts", [])
                recent_docs.append({
                    "cache_key": doc["cache_key"],
                    "expansion_type": doc.get("expansion_type", "unknown"),
                    "concept_count": len(concepts),
                    "created_at": doc.get("created_at")
                })
            
            return {
                "success": True,
                "total_entries": total_docs,
                "expansion_type_stats": expansion_stats,
                "recent_entries": recent_docs,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache summary: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_cache_entries(self, limit: int = 20, skip: int = 0) -> Dict[str, Any]:
        """
        Get cache entries with pagination.
        
        Args:
            limit: Maximum entries to return
            skip: Number of entries to skip
            
        Returns:
            Dictionary with cache entries and pagination info
        """
        try:
            collection = await self.cache.collection
            
            # Get entries with pagination
            entries = []
            async for doc in collection.find({}).skip(skip).limit(limit).sort("created_at", -1):
                # Convert ObjectId to string for JSON serialization
                doc["_id"] = str(doc["_id"])
                entries.append(doc)
            
            total_count = await collection.count_documents({})
            
            return {
                "success": True,
                "entries": entries,
                "pagination": {
                    "total": total_count,
                    "limit": limit,
                    "skip": skip,
                    "returned": len(entries)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache entries: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Global admin instance
_cache_admin: Optional[CacheAdmin] = None


def get_cache_admin() -> CacheAdmin:
    """Get singleton CacheAdmin instance."""
    global _cache_admin
    if _cache_admin is None:
        _cache_admin = CacheAdmin()
    return _cache_admin


# Convenience functions for testing
async def clear_test_cache():
    """Clear all cache for fresh testing. Use at start of test suites."""
    admin = get_cache_admin()
    return await admin.clear_all_cache()


async def clear_conceptnet_cache():
    """Clear only ConceptNet cache entries."""
    admin = get_cache_admin()
    return await admin.clear_cache_by_method("conceptnet")


async def print_cache_summary():
    """Print formatted cache summary for debugging."""
    admin = get_cache_admin()
    summary = await admin.get_cache_summary()
    
    if summary["success"]:
        print(f"📊 Cache Summary ({summary['total_entries']} total entries)")
        print("=" * 50)
        
        for exp_type, stats in summary["expansion_type_stats"].items():
            print(f"  {exp_type}: {stats['count']} entries, avg {stats['avg_concepts']} concepts")
        
        print(f"\n🕒 Recent entries:")
        for entry in summary["recent_entries"][:5]:
            print(f"  {entry['cache_key']}: {entry['concept_count']} concepts ({entry['created_at']})")
    else:
        print(f"❌ Failed to get cache summary: {summary.get('error')}")


if __name__ == "__main__":
    # Example usage
    async def main():
        print("🧹 Cache Admin Demo")
        
        # Show current cache
        await print_cache_summary()
        
        # Clear all cache
        result = await clear_test_cache()
        print(f"\n🗑️  {result['message']}")
        
        # Show empty cache
        await print_cache_summary()
    
    asyncio.run(main())