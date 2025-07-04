"""
Test MongoDB integration components.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.data.models import Movie, MovieCreate, MovieUpdate
from src.data.mongo_client import MongoClient
from src.ingestion.jellyfin_connector import JellyfinConnector


class TestMongoClient:
    """Test MongoDB client functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = MagicMock()
        settings.mongodb_url = "mongodb://localhost:27017/test_db"
        settings.MONGODB_DATABASE = "test_db"
        return settings
    
    @pytest.fixture
    def mongo_client(self, mock_settings):
        """Create MongoDB client for testing."""
        with patch('src.data.mongo_client.get_settings', return_value=mock_settings):
            client = MongoClient()
            return client
    
    @pytest.fixture
    def sample_movie_data(self):
        """Sample movie data for testing."""
        return MovieCreate(
            title="Test Movie",
            year=2024,
            plot="A test movie plot",
            cast=["Actor 1", "Actor 2"],
            directors=["Director 1"],
            genres=["Action", "Drama"],
            rating="PG-13",
            jellyfin_id="test-123",
            jellyfin_path="/movies/test.mkv"
        )
    
    def test_movie_model_creation(self, sample_movie_data):
        """Test movie model creation."""
        movie = Movie(**sample_movie_data.dict())
        assert movie.title == "Test Movie"
        assert movie.year == 2024
        assert len(movie.cast) == 2
        assert len(movie.directors) == 1
        assert len(movie.genres) == 2
        assert movie.jellyfin_id == "test-123"
    
    def test_movie_create_model(self, sample_movie_data):
        """Test movie creation model."""
        assert sample_movie_data.title == "Test Movie"
        assert sample_movie_data.year == 2024
        assert sample_movie_data.jellyfin_id == "test-123"
    
    def test_movie_update_model(self):
        """Test movie update model."""
        update_data = MovieUpdate(
            title="Updated Title",
            year=2025,
            plot="Updated plot"
        )
        
        assert update_data.title == "Updated Title"
        assert update_data.year == 2025
        assert update_data.plot == "Updated plot"
        assert update_data.cast is None  # Should be None for fields not updated
    
    @pytest.mark.asyncio
    async def test_mongo_client_initialization(self, mongo_client):
        """Test MongoDB client initialization."""
        assert mongo_client.client is None
        assert mongo_client.database is None
        assert mongo_client.movies is None
    
    @pytest.mark.asyncio
    async def test_mongo_client_connect_success(self, mongo_client):
        """Test successful MongoDB connection."""
        # Mock the motor client
        mock_client = AsyncMock()
        mock_client.admin.command = AsyncMock(return_value={"ok": 1})
        mock_database = AsyncMock()
        mock_collection = AsyncMock()
        mock_collection.create_indexes = AsyncMock()
        
        mock_client.__getitem__ = MagicMock(return_value=mock_database)
        mock_database.__getattr__ = MagicMock(return_value=mock_collection)
        
        with patch('src.data.mongo_client.AsyncIOMotorClient', return_value=mock_client):
            await mongo_client.connect()
            
            assert mongo_client.client is not None
            assert mongo_client.database is not None
            assert mongo_client.movies is not None
            mock_client.admin.command.assert_called_once_with('ping')
    
    @pytest.mark.asyncio
    async def test_mongo_client_connect_failure(self, mongo_client):
        """Test MongoDB connection failure."""
        with patch('src.data.mongo_client.AsyncIOMotorClient', side_effect=Exception("Connection failed")):
            with pytest.raises(Exception, match="Connection failed"):
                await mongo_client.connect()
    
    @pytest.mark.asyncio
    async def test_mongo_client_disconnect(self, mongo_client):
        """Test MongoDB disconnection."""
        mock_client = AsyncMock()
        mongo_client.client = mock_client
        
        await mongo_client.disconnect()
        mock_client.close.assert_called_once()


class TestJellyfinConnector:
    """Test Jellyfin connector functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = MagicMock()
        settings.JELLYFIN_URL = "http://localhost:8096"
        settings.JELLYFIN_API_KEY = "test-api-key"
        settings.JELLYFIN_USER_ID = "test-user-id"
        return settings
    
    @pytest.fixture
    def jellyfin_connector(self, mock_settings):
        """Create Jellyfin connector for testing."""
        with patch('src.ingestion.jellyfin_connector.get_settings', return_value=mock_settings):
            connector = JellyfinConnector()
            return connector
    
    @pytest.fixture
    def sample_jellyfin_item(self):
        """Sample Jellyfin item data."""
        return {
            "Id": "jellyfin-123",
            "Name": "Test Movie",
            "ProductionYear": 2024,
            "Overview": "A test movie from Jellyfin",
            "Genres": ["Action", "Drama"],
            "People": [
                {"Name": "Actor 1", "Type": "Actor"},
                {"Name": "Actor 2", "Type": "Actor"},
                {"Name": "Director 1", "Type": "Director"}
            ],
            "OfficialRating": "PG-13",
            "Path": "/movies/test.mkv"
        }
    
    def test_jellyfin_connector_initialization(self, jellyfin_connector):
        """Test Jellyfin connector initialization."""
        assert jellyfin_connector.client is None
        assert jellyfin_connector.mongo_client is None
    
    @pytest.mark.asyncio
    async def test_jellyfin_connector_connect_success(self, jellyfin_connector):
        """Test successful Jellyfin connection."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "ServerName": "Test Server",
            "Version": "1.0.0"
        }
        mock_client.get.return_value = mock_response
        
        mock_mongo_client = AsyncMock()
        
        with patch('httpx.AsyncClient', return_value=mock_client), \
             patch('src.ingestion.jellyfin_connector.get_mongo_client', return_value=mock_mongo_client):
            
            await jellyfin_connector.connect()
            
            assert jellyfin_connector.client is not None
            assert jellyfin_connector.mongo_client is not None
            mock_client.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_jellyfin_connector_connect_failure(self, jellyfin_connector):
        """Test Jellyfin connection failure."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection failed")
        
        with patch('httpx.AsyncClient', return_value=mock_client), \
             patch('src.ingestion.jellyfin_connector.get_mongo_client'):
            
            with pytest.raises(Exception, match="Connection failed"):
                await jellyfin_connector.connect()
    
    def test_extract_movie_data(self, jellyfin_connector, sample_jellyfin_item):
        """Test movie data extraction from Jellyfin item."""
        movie_data = jellyfin_connector._extract_movie_data(sample_jellyfin_item)
        
        assert movie_data.title == "Test Movie"
        assert movie_data.year == 2024
        assert movie_data.plot == "A test movie from Jellyfin"
        assert len(movie_data.cast) == 2
        assert len(movie_data.directors) == 1
        assert len(movie_data.genres) == 2
        assert movie_data.rating == "PG-13"
        assert movie_data.jellyfin_id == "jellyfin-123"
        assert movie_data.jellyfin_path == "/movies/test.mkv"
    
    @pytest.mark.asyncio
    async def test_get_movies_success(self, jellyfin_connector):
        """Test successful movie retrieval from Jellyfin."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "Items": [{"Id": "1", "Name": "Movie 1"}, {"Id": "2", "Name": "Movie 2"}]
        }
        mock_client.get.return_value = mock_response
        jellyfin_connector.client = mock_client
        
        movies = await jellyfin_connector.get_movies(limit=2)
        
        assert len(movies) == 2
        assert movies[0]["Name"] == "Movie 1"
        assert movies[1]["Name"] == "Movie 2"
        mock_client.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ingest_movie_success(self, jellyfin_connector, sample_jellyfin_item):
        """Test successful movie ingestion."""
        mock_mongo_client = AsyncMock()
        mock_mongo_client.get_movie_by_jellyfin_id.return_value = None
        mock_movie = Movie(**{
            "title": "Test Movie",
            "year": 2024,
            "plot": "A test movie from Jellyfin",
            "cast": ["Actor 1", "Actor 2"],
            "directors": ["Director 1"],
            "genres": ["Action", "Drama"],
            "rating": "PG-13",
            "jellyfin_id": "jellyfin-123",
            "jellyfin_path": "/movies/test.mkv"
        })
        mock_mongo_client.create_movie.return_value = mock_movie
        
        jellyfin_connector.mongo_client = mock_mongo_client
        
        result = await jellyfin_connector.ingest_movie(sample_jellyfin_item)
        
        assert result is not None
        assert result.title == "Test Movie"
        mock_mongo_client.get_movie_by_jellyfin_id.assert_called_once_with("jellyfin-123")
        mock_mongo_client.create_movie.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ingest_movie_already_exists(self, jellyfin_connector, sample_jellyfin_item):
        """Test movie ingestion when movie already exists."""
        mock_mongo_client = AsyncMock()
        existing_movie = Movie(**{
            "title": "Test Movie",
            "year": 2024,
            "jellyfin_id": "jellyfin-123"
        })
        mock_mongo_client.get_movie_by_jellyfin_id.return_value = existing_movie
        
        jellyfin_connector.mongo_client = mock_mongo_client
        
        result = await jellyfin_connector.ingest_movie(sample_jellyfin_item)
        
        assert result is not None
        assert result.title == "Test Movie"
        mock_mongo_client.get_movie_by_jellyfin_id.assert_called_once_with("jellyfin-123")
        mock_mongo_client.create_movie.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])