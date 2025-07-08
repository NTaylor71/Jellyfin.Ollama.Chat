"""
MongoDB client with connection management for the RAG system.
Handles database connections and provides base database operations.
"""

import logging
from typing import Optional, Any, Dict
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class MongoDBClient:
    """
    MongoDB client wrapper with connection management.
    
    Provides both async (Motor) and sync (PyMongo) clients for different use cases.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._async_client: Optional[AsyncIOMotorClient] = None
        self._sync_client: Optional[MongoClient] = None
        self._async_db: Optional[AsyncIOMotorDatabase] = None
        
    @property
    def async_client(self) -> AsyncIOMotorClient:
        """Get or create async MongoDB client."""
        if self._async_client is None:
            self._async_client = AsyncIOMotorClient(
                self.settings.mongodb_url,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
        return self._async_client
    
    @property
    def sync_client(self) -> MongoClient:
        """Get or create sync MongoDB client."""
        if self._sync_client is None:
            self._sync_client = MongoClient(
                self.settings.mongodb_url,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
        return self._sync_client
    
    @property
    def async_db(self) -> AsyncIOMotorDatabase:
        """Get async database instance."""
        if self._async_db is None:
            self._async_db = self.async_client[self.settings.MONGODB_DATABASE]
        return self._async_db
    
    @property
    def sync_db(self):
        """Get sync database instance."""
        return self.sync_client[self.settings.MONGODB_DATABASE]
    
    async def test_connection(self) -> bool:
        """Test async database connection."""
        try:
            await self.async_client.admin.command('ping')
            logger.info("MongoDB async connection successful")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"MongoDB async connection failed: {e}")
            return False
    
    def test_sync_connection(self) -> bool:
        """Test sync database connection."""
        try:
            self.sync_client.admin.command('ping')
            logger.info("MongoDB sync connection successful")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"MongoDB sync connection failed: {e}")
            return False
    
    async def close(self):
        """Close all connections."""
        if self._async_client:
            self._async_client.close()
        if self._sync_client:
            self._sync_client.close()
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        try:
            db = self.sync_db
            if collection_name in db.list_collection_names():
                collection = db[collection_name]
                return {
                    "exists": True,
                    "document_count": collection.count_documents({}),
                    "indexes": list(collection.list_indexes())
                }
            else:
                return {"exists": False}
        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {e}")
            return {"exists": False, "error": str(e)}


# Global client instance
_mongodb_client: Optional[MongoDBClient] = None


def get_mongodb_client() -> MongoDBClient:
    """Get singleton MongoDB client instance."""
    global _mongodb_client
    if _mongodb_client is None:
        _mongodb_client = MongoDBClient()
    return _mongodb_client


async def get_async_db():
    """Get async database instance."""
    client = get_mongodb_client()
    return client.async_db


def get_sync_db():
    """Get sync database instance."""
    client = get_mongodb_client()
    return client.sync_db