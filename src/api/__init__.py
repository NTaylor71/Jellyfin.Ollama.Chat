"""
API module for the production RAG system.
Includes FastAPI endpoints, cache administration, and API utilities.
"""

from src.api.cache_admin import get_cache_admin, clear_test_cache, print_cache_summary

__all__ = [
    "get_cache_admin",
    "clear_test_cache", 
    "print_cache_summary"
]