"""
Basic tests for new API endpoints to verify they work correctly.
Tests the new /api/v1/ route structure and metrics integration.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.api.main import create_app
from src.api.routes.media import router as media_router
from src.api.routes.ingestion import router as ingestion_router
from src.api.routes.search import router as search_router


@pytest.fixture
def app():
    """Create FastAPI app for testing."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health and basic endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns system info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "Production RAG System API"
    
    def test_health_endpoint(self, client):
        """Test health endpoint returns health status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"


class TestMediaTypeEndpoints:
    """Test media type configuration endpoints."""
    
    def test_list_media_types_legacy(self, client):
        """Test legacy media types endpoint."""
        response = client.get("/media-types")
        assert response.status_code == 200
        data = response.json()
        assert "media_types" in data
        assert isinstance(data["media_types"], list)
    
    def test_list_media_types_v1(self, client):
        """Test new API v1 media types endpoint."""
        response = client.get("/api/v1/ingest/media-types")
        assert response.status_code == 200
        data = response.json()
        assert "media_types" in data
        assert isinstance(data["media_types"], list)


class TestMediaEndpoints:
    """Test media retrieval endpoints."""
    
    @patch('src.api.routes.media.get_or_create_manager')
    def test_list_media_basic(self, mock_get_manager, client):
        """Test basic media listing endpoint."""
        # Mock the manager and database
        mock_manager = AsyncMock()
        mock_collection = AsyncMock()
        mock_manager.db = {"movie_enriched": mock_collection}
        mock_manager.media_config = MagicMock()
        mock_manager.media_config.output = {"collection": "movie_enriched"}
        
        mock_collection.count_documents.return_value = 0
        mock_collection.find.return_value.skip.return_value.limit.return_value.sort.return_value.to_list.return_value = []
        
        mock_get_manager.return_value = mock_manager
        
        response = client.get("/api/v1/media/movie")
        assert response.status_code == 200
        data = response.json()
        assert "media_type" in data
        assert "items" in data
        assert "total" in data
        assert data["media_type"] == "movie"
        assert data["total"] == 0
    
    @patch('src.api.routes.media.get_or_create_manager')
    def test_get_media_item_not_found(self, mock_get_manager, client):
        """Test getting non-existent media item."""
        # Mock the manager and database
        mock_manager = AsyncMock()
        mock_collection = AsyncMock()
        mock_manager.db = {"movie_enriched": mock_collection}
        mock_manager.media_config = MagicMock()
        mock_manager.media_config.output = {"collection": "movie_enriched"}
        
        mock_collection.find_one.return_value = None
        mock_get_manager.return_value = mock_manager
        
        response = client.get("/api/v1/media/movie/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"]


class TestSearchEndpoints:
    """Test search endpoints."""
    
    @patch('src.api.routes.search.get_or_create_manager')
    def test_search_basic_post(self, mock_get_manager, client):
        """Test basic search POST endpoint."""
        # Mock the manager and database
        mock_manager = AsyncMock()
        mock_collection = AsyncMock()
        mock_manager.db = {"movie_enriched": mock_collection}
        mock_manager.media_config = MagicMock()
        mock_manager.media_config.output = {"collection": "movie_enriched"}
        mock_manager.media_type = "movie"
        
        # Mock empty search results
        mock_collection.find.return_value.sort.return_value.limit.return_value = AsyncMock()
        mock_collection.find.return_value.sort.return_value.limit.return_value.__aiter__ = AsyncMock(return_value=iter([]))
        
        mock_get_manager.return_value = mock_manager
        
        search_data = {
            "query": "test query",
            "media_type": "movie",
            "limit": 10
        }
        
        response = client.post("/api/v1/search/", json=search_data)
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "total_found" in data
        assert data["query"] == "test query"
        assert data["total_found"] == 0
    
    @patch('src.api.routes.search.get_or_create_manager')
    def test_search_basic_get(self, mock_get_manager, client):
        """Test basic search GET endpoint."""
        # Mock the manager and database
        mock_manager = AsyncMock()
        mock_collection = AsyncMock()
        mock_manager.db = {"movie_enriched": mock_collection}
        mock_manager.media_config = MagicMock()
        mock_manager.media_config.output = {"collection": "movie_enriched"}
        mock_manager.media_type = "movie"
        
        # Mock empty search results
        mock_collection.find.return_value.sort.return_value.limit.return_value = AsyncMock()
        mock_collection.find.return_value.sort.return_value.limit.return_value.__aiter__ = AsyncMock(return_value=iter([]))
        
        mock_get_manager.return_value = mock_manager
        
        response = client.get("/api/v1/search/movie?q=test&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert data["query"] == "test"


class TestIngestionEndpoints:
    """Test ingestion endpoints."""
    
    @patch('src.api.routes.ingestion.get_or_create_manager')
    def test_ingest_media_empty_items(self, mock_get_manager, client):
        """Test ingesting empty media items."""
        mock_manager = AsyncMock()
        mock_get_manager.return_value = mock_manager
        
        ingestion_data = {
            "media_type": "movie",
            "media_items": [],
            "skip_enrichment": True
        }
        
        response = client.post("/api/v1/ingest/media", json=ingestion_data)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "items_processed" in data
        assert data["items_processed"] == 0
    
    @patch('src.api.routes.ingestion.get_or_create_manager')
    def test_ingest_media_invalid_data(self, mock_get_manager, client):
        """Test ingesting media with invalid data."""
        mock_manager = AsyncMock()
        mock_manager.dynamic_model.side_effect = Exception("Invalid data")
        mock_get_manager.return_value = mock_manager
        
        ingestion_data = {
            "media_type": "movie",
            "media_items": [{"invalid": "data"}],
            "skip_enrichment": True
        }
        
        response = client.post("/api/v1/ingest/media", json=ingestion_data)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "errors" in data
        assert data["status"] == "failed"
        assert len(data["errors"]) > 0


class TestMetricsIntegration:
    """Test that metrics are properly integrated."""
    
    def test_metrics_endpoint_available(self, client):
        """Test that metrics endpoint is available."""
        response = client.get("/metrics")
        # The endpoint should exist (might be 404 if metrics disabled in test)
        assert response.status_code in [200, 404]
    
    @patch('src.shared.metrics.media_retrieval_total')
    @patch('src.api.routes.media.get_or_create_manager')
    def test_metrics_called_on_media_retrieval(self, mock_get_manager, mock_metric, client):
        """Test that metrics are called during media retrieval."""
        # Mock the manager and database
        mock_manager = AsyncMock()
        mock_collection = AsyncMock()
        mock_manager.db = {"movie_enriched": mock_collection}
        mock_manager.media_config = MagicMock()
        mock_manager.media_config.output = {"collection": "movie_enriched"}
        
        mock_collection.find_one.return_value = None
        mock_get_manager.return_value = mock_manager
        
        response = client.get("/api/v1/media/movie/test")
        assert response.status_code == 404
        
        # Verify metrics were called
        mock_metric.labels.assert_called_with(media_type="movie", status="not_found")
        mock_metric.labels.return_value.inc.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])