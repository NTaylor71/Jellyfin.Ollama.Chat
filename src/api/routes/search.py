"""
Search-related API endpoints.
Handles intelligent search with query expansion and various search strategies.
"""

from typing import Dict, Any, List, Optional
import time
import asyncio

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient

from src.shared.config import get_settings
from src.ingestion_manager import IngestionManager
from src.shared.metrics import track_search_metrics


router = APIRouter(prefix="/api/v1/search", tags=["search"])


class SearchRequest(BaseModel):
    """Request for media search."""
    query: str = Field(..., description="Search query")
    media_type: str = Field("movie", description="Media type to search")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    search_strategy: str = Field("hybrid", description="Search strategy: text, semantic, hybrid")


class SearchResult(BaseModel):
    """Individual search result."""
    id: str
    name: str
    media_type: str
    score: float
    snippet: Optional[str] = None
    data: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response for search operations."""
    query: str
    media_type: str
    strategy: str
    results: List[SearchResult]
    total_found: int
    execution_time_ms: float
    expanded_query: Optional[str] = None



_managers: Dict[str, IngestionManager] = {}


async def get_or_create_manager(media_type: str) -> IngestionManager:
    """Get or create an ingestion manager for the specified media type."""
    if media_type not in _managers:
        manager = IngestionManager(media_type=media_type)
        await manager.connect()
        _managers[media_type] = manager
    return _managers[media_type]


def _serialize_media_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize MongoDB document for JSON response."""
    from datetime import datetime
    from bson import ObjectId
    

    if "_id" in item:
        item["_id"] = str(item["_id"])
    

    for key, value in item.items():
        if isinstance(value, datetime):
            item[key] = value.isoformat()
    
    return item


async def _expand_query(query: str, media_type: str) -> str:
    """Expand query using available enrichment plugins."""

    return query


async def _text_search(manager: IngestionManager, query: str, limit: int, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Perform text-based search using MongoDB text index."""
    collection_name = (
        manager.media_config.output.get("collection", f"{manager.media_type}_enriched")
        if manager.media_config.output
        else f"{manager.media_type}_enriched"
    )
    collection = manager.db[collection_name]
    
    
    search_query = {
        "_media_type": manager.media_type,
        "$text": {"$search": query}
    }
    
    
    if filters:
        search_query.update(filters)
    

    cursor = collection.find(
        search_query,
        {"score": {"$meta": "textScore"}}
    ).sort([("score", {"$meta": "textScore"})]).limit(limit)
    
    results = []
    async for doc in cursor:
        results.append({
            "item": doc,
            "score": doc.get("score", 0.0),
            "strategy": "text"
        })
    
    return results


async def _semantic_search(manager: IngestionManager, query: str, limit: int, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Perform semantic search using available embeddings."""

    return await _text_search(manager, query, limit, filters)


async def _hybrid_search(manager: IngestionManager, query: str, limit: int, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Perform hybrid search combining text and semantic strategies."""
    
    text_results = await _text_search(manager, query, limit, filters)
    semantic_results = await _semantic_search(manager, query, limit, filters)
    

    combined_results = {}
    
    
    for result in text_results:
        item_id = str(result["item"].get("_id", result["item"].get("Id")))
        combined_results[item_id] = result
        combined_results[item_id]["score"] = result["score"] * 0.6
    
    
    for result in semantic_results:
        item_id = str(result["item"].get("_id", result["item"].get("Id")))
        if item_id in combined_results:

            combined_results[item_id]["score"] += result["score"] * 0.4
            combined_results[item_id]["strategy"] = "hybrid"
        else:
            combined_results[item_id] = result
            combined_results[item_id]["score"] = result["score"] * 0.4
    

    sorted_results = sorted(combined_results.values(), key=lambda x: x["score"], reverse=True)
    return sorted_results[:limit]


@router.post("/", response_model=SearchResponse)
@track_search_metrics
async def search_media(request: SearchRequest):
    """Search for media items with intelligent query expansion."""
    start_time = time.time()
    
    try:
        
        manager = await get_or_create_manager(request.media_type)
        

        expanded_query = await _expand_query(request.query, request.media_type)
        

        if request.search_strategy == "text":
            results = await _text_search(manager, expanded_query, request.limit, request.filters)
        elif request.search_strategy == "semantic":
            results = await _semantic_search(manager, expanded_query, request.limit, request.filters)
        elif request.search_strategy == "hybrid":
            results = await _hybrid_search(manager, expanded_query, request.limit, request.filters)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown search strategy: {request.search_strategy}")
        

        search_results = []
        for result in results:
            item = result["item"]
            search_results.append(SearchResult(
                id=str(item.get("_id", item.get("Id"))),
                name=item.get("Name", "Unknown"),
                media_type=request.media_type,
                score=result["score"],
                snippet=item.get("Overview", "")[:200] + "..." if item.get("Overview") else None,
                data=_serialize_media_item(item)
            ))
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=request.query,
            media_type=request.media_type,
            strategy=request.search_strategy,
            results=search_results,
            total_found=len(search_results),
            execution_time_ms=execution_time_ms,
            expanded_query=expanded_query if expanded_query != request.query else None
        )
        
    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/{media_type}", response_model=SearchResponse)
async def search_media_get(
    media_type: str,
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    strategy: str = Query("hybrid", description="Search strategy: text, semantic, hybrid")
):
    """Search for media items using GET request (for convenience)."""
    request = SearchRequest(
        query=q,
        media_type=media_type,
        limit=limit,
        search_strategy=strategy
    )
    return await search_media(request)