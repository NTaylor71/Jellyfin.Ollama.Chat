"""
Media-related API endpoints.
Handles media retrieval, enrichment, and metadata operations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import time

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

from src.shared.config import get_settings
from src.ingestion_manager import IngestionManager
from src.shared.metrics import (
    media_retrieval_total, media_retrieval_duration,
    track_enrichment_metrics
)


router = APIRouter(prefix="/api/v1/media", tags=["media"])


class MediaResponse(BaseModel):
    """Response model for media item."""
    id: str
    name: str
    media_type: str
    data: Dict[str, Any]
    ingested_at: Optional[datetime] = None
    enrichment_version: Optional[str] = None


class MediaListResponse(BaseModel):
    """Response model for media list."""
    media_type: str
    items: List[MediaResponse]
    total: int
    page: int
    page_size: int


class EnrichmentRequest(BaseModel):
    """Request for media enrichment."""
    media_type: str = "movie"
    media_data: Dict[str, Any]
    fields: Optional[List[str]] = None


class EnrichmentResponse(BaseModel):
    """Response for enrichment operations."""
    status: str
    media_type: str
    enriched_data: Optional[Dict[str, Any]] = None
    enriched_fields: List[str] = Field(default_factory=list)
    execution_time_ms: float
    errors: List[str] = Field(default_factory=list)


# Global managers cache
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
    # Convert ObjectId to string
    if "_id" in item:
        item["_id"] = str(item["_id"])
    
    # Convert datetime objects to ISO strings
    for key, value in item.items():
        if isinstance(value, datetime):
            item[key] = value.isoformat()
    
    return item


@router.get("/{media_type}", response_model=MediaListResponse)
async def list_media(
    media_type: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    name_filter: Optional[str] = Query(None, description="Filter by name substring")
):
    """List media items of a specific type with pagination."""
    try:
        manager = await get_or_create_manager(media_type)
        
        # Build query
        query = {"_media_type": media_type}
        if name_filter:
            query["Name"] = {"$regex": name_filter, "$options": "i"}
        
        # Get collection
        collection_name = (
            manager.media_config.output.get("collection", f"{media_type}_enriched")
            if manager.media_config.output
            else f"{media_type}_enriched"
        )
        collection = manager.db[collection_name]
        
        # Get total count
        total = await collection.count_documents(query)
        
        # Get paginated results
        skip = (page - 1) * page_size
        cursor = collection.find(query).skip(skip).limit(page_size).sort("_ingested_at", -1)
        items = await cursor.to_list(length=page_size)
        
        # Convert to response format
        media_items = []
        for item in items:
            media_items.append(MediaResponse(
                id=str(item.get("_id", item.get("Id"))),
                name=item.get("Name", "Unknown"),
                media_type=media_type,
                data=_serialize_media_item(item),
                ingested_at=item.get("_ingested_at"),
                enrichment_version=item.get("_enrichment_version")
            ))
        
        return MediaListResponse(
            media_type=media_type,
            items=media_items,
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list {media_type}: {str(e)}")


@router.get("/{media_type}/{item_id}", response_model=MediaResponse)
async def get_media_item(media_type: str, item_id: str):
    """Get a specific media item by ID."""
    start_time = time.time()
    
    try:
        manager = await get_or_create_manager(media_type)
        
        # Get collection
        collection_name = (
            manager.media_config.output.get("collection", f"{media_type}_enriched")
            if manager.media_config.output
            else f"{media_type}_enriched"
        )
        collection = manager.db[collection_name]
        
        # Try to find by MongoDB ObjectId first, then by media Id
        query = {"_media_type": media_type}
        
        # Try ObjectId format first
        try:
            query["_id"] = ObjectId(item_id)
            item = await collection.find_one(query)
        except:
            # Fall back to string Id field
            del query["_id"]
            query["Id"] = item_id
            item = await collection.find_one(query)
        
        if not item:
            media_retrieval_total.labels(media_type=media_type, status='not_found').inc()
            raise HTTPException(status_code=404, detail=f"{media_type} item not found")
        
        # Track successful retrieval
        media_retrieval_total.labels(media_type=media_type, status='success').inc()
        duration = time.time() - start_time
        media_retrieval_duration.labels(media_type=media_type).observe(duration)
        
        return MediaResponse(
            id=str(item.get("_id", item.get("Id"))),
            name=item.get("Name", "Unknown"),
            media_type=media_type,
            data=_serialize_media_item(item),
            ingested_at=item.get("_ingested_at"),
            enrichment_version=item.get("_enrichment_version")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        media_retrieval_total.labels(media_type=media_type, status='error').inc()
        raise HTTPException(status_code=500, detail=f"Failed to get {media_type} item: {str(e)}")


@router.post("/{media_type}/enrich", response_model=EnrichmentResponse)
@track_enrichment_metrics
async def enrich_media_item(media_type: str, request: EnrichmentRequest):
    """Enrich a single media item."""
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Get or create manager for media type
        manager = await get_or_create_manager(media_type)
        
        # Convert to MediaData
        media_item = manager.dynamic_model(**request.media_data)
        
        # Enrich the item
        enriched_data = await manager.enrich_media_item(media_item)
        
        # Calculate execution time
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Get enriched fields
        original_fields = set(request.media_data.keys())
        enriched_fields = [field for field in enriched_data.keys() if field not in original_fields]
        
        return EnrichmentResponse(
            status="success",
            media_type=media_type,
            enriched_data=enriched_data,
            enriched_fields=enriched_fields,
            execution_time_ms=execution_time_ms
        )
        
    except Exception as e:
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        return EnrichmentResponse(
            status="error",
            media_type=media_type,
            execution_time_ms=execution_time_ms,
            errors=[str(e)]
        )


@router.post("/analyze", response_model=EnrichmentResponse)
@track_enrichment_metrics
async def analyze_media_item(request: EnrichmentRequest):
    """Analyze media item without storing to database."""
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Get or create manager for media type
        manager = await get_or_create_manager(request.media_type)
        
        # Convert to MediaData
        media_item = manager.dynamic_model(**request.media_data)
        
        # Enrich the item (without storing)
        enriched_data = await manager.enrich_media_item(media_item)
        
        # Calculate execution time
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Get enriched fields
        original_fields = set(request.media_data.keys())
        enriched_fields = [field for field in enriched_data.keys() if field not in original_fields]
        
        return EnrichmentResponse(
            status="success",
            media_type=request.media_type,
            enriched_data=enriched_data,
            enriched_fields=enriched_fields,
            execution_time_ms=execution_time_ms
        )
        
    except Exception as e:
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        return EnrichmentResponse(
            status="error",
            media_type=request.media_type,
            execution_time_ms=execution_time_ms,
            errors=[str(e)]
        )


class FieldEnrichmentRequest(BaseModel):
    """Request for enriching specific fields only."""
    media_type: str = "movie"
    media_data: Dict[str, Any]
    target_fields: List[str] = Field(..., description="Specific fields to enrich")


@router.post("/enrich/field", response_model=EnrichmentResponse)
@track_enrichment_metrics
async def enrich_specific_fields(request: FieldEnrichmentRequest):
    """Enrich only specific fields of a media item."""
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Get or create manager for media type
        manager = await get_or_create_manager(request.media_type)
        
        # Convert to MediaData
        media_item = manager.dynamic_model(**request.media_data)
        
        # Selective field enrichment - only run plugins for requested fields
        media_config = manager.media_config
        
        # Get field configurations for only the requested fields
        field_configs = {}
        for field in request.target_fields:
            if field in media_config.fields:
                field_configs[field] = media_config.fields[field]
        
        if not field_configs:
            # No valid fields requested, return original data
            filtered_data = request.media_data.copy()
        else:
            # Create temporary config with only target fields
            from src.shared.media_field_config import MediaFieldConfig
            temp_config = MediaFieldConfig(
                name=media_config.name,
                description=f"Selective enrichment for: {', '.join(request.target_fields)}",
                fields=field_configs,
                output=media_config.output
            )
            
            # Override manager's config temporarily
            original_config = manager.media_config
            manager.media_config = temp_config
            
            try:
                # Enrich with selective field configuration
                enriched_data = await manager.enrich_media_item(media_item)
                filtered_data = enriched_data
            finally:
                # Restore original config
                manager.media_config = original_config
        
        # Calculate execution time
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Get enriched fields (only the ones we targeted)
        original_fields = set(request.media_data.keys())
        enriched_fields = [field for field in request.target_fields if field in enriched_data and field not in original_fields]
        
        return EnrichmentResponse(
            status="success",
            media_type=request.media_type,
            enriched_data=filtered_data,
            enriched_fields=enriched_fields,
            execution_time_ms=execution_time_ms
        )
        
    except Exception as e:
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        return EnrichmentResponse(
            status="error",
            media_type=request.media_type,
            execution_time_ms=execution_time_ms,
            errors=[str(e)]
        )


@router.get("/{media_type}/verify")
async def verify_media_ingestion(media_type: str):
    """Verify ingestion results for a media type."""
    try:
        manager = await get_or_create_manager(media_type)
        results = await manager.verify_ingestion()
        return {"status": "success", **results}
    except Exception as e:
        return {"status": "error", "error": str(e)}