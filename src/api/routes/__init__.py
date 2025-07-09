"""API routes package."""

from .media import router as media_router
from .ingestion import router as ingestion_router
from .search import router as search_router

__all__ = ["media_router", "ingestion_router", "search_router"]