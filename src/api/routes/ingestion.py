"""
Ingestion-related API endpoints.
Handles media ingestion from various sources (JSON, Jellyfin, etc.).
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml
import time

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from src.ingestion_manager import IngestionManager
from src.shared.metrics import (
    media_ingestion_total, media_ingestion_duration,
    data_source_items_total, data_source_duration
)


router = APIRouter(prefix="/api/v1/ingest", tags=["ingestion"])


class MediaIngestionRequest(BaseModel):
    """Request for media ingestion."""
    media_type: str = "movie"
    media_items: List[Dict[str, Any]]
    skip_enrichment: bool = False
    batch_size: int = 5


class JellyfinIngestionRequest(BaseModel):
    """Request for Jellyfin media ingestion."""
    media_type: str = "movie"
    limit: Optional[int] = None
    item_names: Optional[List[str]] = None
    skip_enrichment: bool = False
    batch_size: int = 5


class IngestionResponse(BaseModel):
    """Response for ingestion operations."""
    status: str
    message: str
    media_type: str
    items_processed: int
    enriched: int
    errors: List[str] = Field(default_factory=list)
    task_id: Optional[str] = None


class MediaTypeInfo(BaseModel):
    """Information about a media type."""
    media_type: str
    name: str
    description: str
    fields: List[str]


class MediaTypesResponse(BaseModel):
    """Response for media types listing."""
    media_types: List[MediaTypeInfo]



_managers: Dict[str, IngestionManager] = {}


async def get_or_create_manager(media_type: str) -> IngestionManager:
    """Get or create an ingestion manager for the specified media type."""
    if media_type not in _managers:
        manager = IngestionManager(media_type=media_type)
        await manager.connect()
        _managers[media_type] = manager
    return _managers[media_type]


@router.post("/media", response_model=IngestionResponse)
async def ingest_media_items(
    request: MediaIngestionRequest,
    background_tasks: BackgroundTasks
):
    """Ingest media items with enrichment."""
    start_time = time.time()
    
    try:
        
        manager = await get_or_create_manager(request.media_type)
        

        media_items = []
        errors = []
        
        for item_data in request.media_items:
            try:
                media_item = manager.dynamic_model(**item_data)
                media_items.append(media_item)
            except Exception as e:
                errors.append(f"Invalid data for {item_data.get('Name', 'unknown')}: {str(e)}")
        
        if not media_items:
            return IngestionResponse(
                status="failed",
                message="No valid media items to process",
                media_type=request.media_type,
                items_processed=0,
                enriched=0,
                errors=errors
            )
        

        await manager.ingest_media(
            media_items,
            batch_size=request.batch_size,
            skip_enrichment=request.skip_enrichment
        )
        

        media_ingestion_total.labels(
            media_type=request.media_type,
            source='json',
            status='success'
        ).inc(len(media_items))
        
        duration = time.time() - start_time
        media_ingestion_duration.labels(
            media_type=request.media_type,
            source='json'
        ).observe(duration)
        
        return IngestionResponse(
            status="success",
            message=f"Successfully ingested {len(media_items)} {request.media_type} items",
            media_type=request.media_type,
            items_processed=len(media_items),
            enriched=len(media_items) if not request.skip_enrichment else 0,
            errors=errors
        )
        
    except Exception as e:

        media_ingestion_total.labels(
            media_type=request.media_type,
            source='json',
            status='error'
        ).inc()
        
        return IngestionResponse(
            status="error",
            message=f"Ingestion failed: {str(e)}",
            media_type=request.media_type,
            items_processed=0,
            enriched=0,
            errors=[str(e)]
        )


@router.post("/jellyfin", response_model=IngestionResponse)
async def ingest_from_jellyfin(
    request: JellyfinIngestionRequest,
    background_tasks: BackgroundTasks
):
    """Ingest media from Jellyfin API."""
    start_time = time.time()
    
    try:
        
        manager = await get_or_create_manager(request.media_type)
        

        media_items = await manager.load_media_from_api(
            limit=request.limit,
            item_names=request.item_names
        )
        
        if not media_items:
            return IngestionResponse(
                status="success",
                message=f"No {request.media_type} items found in Jellyfin",
                media_type=request.media_type,
                items_processed=0,
                enriched=0
            )
        

        await manager.ingest_media(
            media_items,
            batch_size=request.batch_size,
            skip_enrichment=request.skip_enrichment
        )
        

        media_ingestion_total.labels(
            media_type=request.media_type,
            source='jellyfin',
            status='success'
        ).inc(len(media_items))
        
        duration = time.time() - start_time
        media_ingestion_duration.labels(
            media_type=request.media_type,
            source='jellyfin'
        ).observe(duration)
        
        return IngestionResponse(
            status="success",
            message=f"Successfully ingested {len(media_items)} {request.media_type} items from Jellyfin",
            media_type=request.media_type,
            items_processed=len(media_items),
            enriched=len(media_items) if not request.skip_enrichment else 0
        )
        
    except Exception as e:

        media_ingestion_total.labels(
            media_type=request.media_type,
            source='jellyfin',
            status='error'
        ).inc()
        
        return IngestionResponse(
            status="error",
            message=f"Jellyfin ingestion failed: {str(e)}",
            media_type=request.media_type,
            items_processed=0,
            enriched=0,
            errors=[str(e)]
        )


@router.get("/media-types", response_model=MediaTypesResponse)
async def list_media_types():
    """List available media type configurations."""
    try:
        config_dir = Path("config/media_types")
        if not config_dir.exists():
            return MediaTypesResponse(media_types=[])
        
        media_types = []
        for config_file in config_dir.glob("*.yaml"):
            if config_file.name not in ["movie_new_format.yaml", "media_detection.yaml"]:
                media_type = config_file.stem
                try:
                    with open(config_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                    media_types.append(MediaTypeInfo(
                        media_type=media_type,
                        name=config_data.get("name", media_type),
                        description=config_data.get("description", ""),
                        fields=list(config_data.get("fields", {}).keys())
                    ))
                except Exception as e:
                    print(f"Error loading {config_file}: {e}")
        
        return MediaTypesResponse(media_types=media_types)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list media types: {str(e)}")