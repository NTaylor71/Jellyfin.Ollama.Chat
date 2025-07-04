"""
Test suite for plugin MongoDB management system.
Tests plugin metadata storage, versioning, status tracking, and release workflow.
"""

import asyncio
import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Mock MongoDB client before importing the actual modules
class MockCollection:
    def __init__(self):
        self.documents = {}
        self.index_calls = []
    
    async def insert_one(self, document):
        from bson import ObjectId
        doc_id = ObjectId()
        document["_id"] = doc_id
        self.documents[str(doc_id)] = document
        result = MagicMock()
        result.inserted_id = doc_id
        return result
    
    async def find_one(self, filter_query):
        for doc in self.documents.values():
            if self._matches_filter(doc, filter_query):
                return doc
        return None
    
    async def update_one(self, filter_query, update_data):
        for doc_id, doc in self.documents.items():
            if self._matches_filter(doc, filter_query):
                if "$set" in update_data:
                    doc.update(update_data["$set"])
                result = MagicMock()
                result.matched_count = 1
                return result
        result = MagicMock()
        result.matched_count = 0
        return result
    
    def find(self, filter_query=None):
        docs = []
        for doc in self.documents.values():
            if filter_query is None or self._matches_filter(doc, filter_query):
                docs.append(doc)
        
        result = MagicMock()
        result.sort = MagicMock(return_value=result)
        result.to_list = AsyncMock(return_value=docs)
        return result
    
    def aggregate(self, pipeline):
        # Simple aggregation for statistics
        if any("$group" in stage for stage in pipeline):
            result = MagicMock()
            result.to_list = AsyncMock(return_value=[{
                "_id": None,
                "total_plugins": len(self.documents),
                "published_plugins": len([d for d in self.documents.values() if d.get("status") == "published"]),
                "enabled_plugins": len([d for d in self.documents.values() if d.get("is_enabled", True)]),
                "plugin_types": list(set(d.get("plugin_type", "general") for d in self.documents.values()))
            }])
            return result
        return MagicMock()
    
    async def create_index(self, index_spec, **kwargs):
        self.index_calls.append((index_spec, kwargs))
    
    def _matches_filter(self, doc, filter_query):
        for key, value in filter_query.items():
            if key not in doc or doc[key] != value:
                return False
        return True

class MockMongoClient:
    def __init__(self):
        self.collections = {}
    
    def get_collection(self, name):
        if name not in self.collections:
            self.collections[name] = MockCollection()
        return self.collections[name]

# Patch the mongo client import
with patch('src.plugins.mongo_manager.get_mongo_client', return_value=MockMongoClient()):
    from src.plugins.mongo_manager import (
        MongoPluginManager, PluginStatus, PluginDocument, 
        PluginVersion, PluginDependency, PluginDeployment
    )
    from src.plugins.base import PluginMetadata, PluginType, ExecutionPriority


class TestMongoPluginManager:
    """Test the MongoDB plugin manager."""
    
    @pytest.fixture
    async def plugin_manager(self):
        """Create a plugin manager instance for testing."""
        manager = MongoPluginManager()
        # Override with our mock client
        manager.mongo_client = MockMongoClient()
        await manager.initialize()
        return manager
    
    @pytest.fixture
    def sample_plugin_metadata(self):
        """Create sample plugin metadata for testing."""
        return PluginMetadata(
            name="TestPlugin",
            version="1.0.0",
            description="A test plugin for unit testing",
            author="Test Author",
            plugin_type=PluginType.QUERY_EMBELLISHER,
            tags=["test", "query"],
            dependencies=["base-plugin"],
            execution_priority=ExecutionPriority.NORMAL
        )
    
    @pytest.mark.asyncio
    async def test_plugin_registration(self, plugin_manager, sample_plugin_metadata):
        """Test plugin registration in MongoDB."""
        # Register a new plugin
        plugin_id = await plugin_manager.register_plugin(
            plugin_metadata=sample_plugin_metadata,
            file_path="/test/path/test_plugin.py",
            file_hash="abc123",
            file_size=1024
        )
        
        assert plugin_id is not None
        
        # Verify plugin was stored
        plugin_doc = await plugin_manager.get_plugin("TestPlugin")
        assert plugin_doc is not None
        assert plugin_doc.name == "TestPlugin"
        assert plugin_doc.current_version == "1.0.0"
        assert plugin_doc.status == PluginStatus.DRAFT
        assert plugin_doc.file_hash == "abc123"
        assert len(plugin_doc.versions) == 1
        assert plugin_doc.versions[0].version == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_plugin_update(self, plugin_manager, sample_plugin_metadata):
        """Test updating an existing plugin."""
        # Register initial plugin
        await plugin_manager.register_plugin(
            plugin_metadata=sample_plugin_metadata,
            file_path="/test/path/test_plugin.py",
            file_hash="abc123",
            file_size=1024
        )
        
        # Update plugin with new version
        updated_metadata = PluginMetadata(
            name="TestPlugin",
            version="1.1.0",
            description="Updated test plugin",
            author="Test Author",
            plugin_type=PluginType.QUERY_EMBELLISHER,
            tags=["test", "query", "updated"],
            dependencies=["base-plugin"],
            execution_priority=ExecutionPriority.HIGH
        )
        
        plugin_id = await plugin_manager.register_plugin(
            plugin_metadata=updated_metadata,
            file_path="/test/path/test_plugin.py",
            file_hash="def456",
            file_size=2048
        )
        
        assert plugin_id is not None
        
        # Verify update
        plugin_doc = await plugin_manager.get_plugin("TestPlugin")
        assert plugin_doc.current_version == "1.1.0"
        assert plugin_doc.description == "Updated test plugin"
        assert plugin_doc.file_hash == "def456"
        assert len(plugin_doc.versions) == 2
        assert plugin_doc.execution_priority == ExecutionPriority.HIGH
        
        # Check version history
        versions = [v.version for v in plugin_doc.versions]
        assert "1.0.0" in versions
        assert "1.1.0" in versions
    
    @pytest.mark.asyncio
    async def test_plugin_publishing(self, plugin_manager, sample_plugin_metadata):
        """Test plugin publishing workflow."""
        # Register plugin
        await plugin_manager.register_plugin(
            plugin_metadata=sample_plugin_metadata,
            file_path="/test/path/test_plugin.py",
            file_hash="abc123",
            file_size=1024
        )
        
        # Publish plugin
        success = await plugin_manager.publish_plugin("TestPlugin")
        assert success is True
        
        # Verify status change
        plugin_doc = await plugin_manager.get_plugin("TestPlugin")
        assert plugin_doc.status == PluginStatus.PUBLISHED
        assert plugin_doc.published_at is not None
    
    @pytest.mark.asyncio
    async def test_plugin_deprecation(self, plugin_manager, sample_plugin_metadata):
        """Test plugin deprecation."""
        # Register and publish plugin
        await plugin_manager.register_plugin(
            plugin_metadata=sample_plugin_metadata,
            file_path="/test/path/test_plugin.py",
            file_hash="abc123",
            file_size=1024
        )
        await plugin_manager.publish_plugin("TestPlugin")
        
        # Deprecate plugin
        success = await plugin_manager.deprecate_plugin("TestPlugin", "Outdated implementation")
        assert success is True
        
        # Verify deprecation
        plugin_doc = await plugin_manager.get_plugin("TestPlugin")
        assert plugin_doc.status == PluginStatus.DEPRECATED
    
    @pytest.mark.asyncio
    async def test_deployment_recording(self, plugin_manager, sample_plugin_metadata):
        """Test deployment recording."""
        # Register plugin
        await plugin_manager.register_plugin(
            plugin_metadata=sample_plugin_metadata,
            file_path="/test/path/test_plugin.py",
            file_hash="abc123",
            file_size=1024
        )
        
        # Record deployment
        deployment_id = await plugin_manager.record_deployment(
            plugin_name="TestPlugin",
            version="1.0.0",
            environment="production",
            deployed_by="admin_user",
            config_overrides={"timeout": 60}
        )
        
        assert deployment_id is not None
        
        # Get deployment history
        history = await plugin_manager.get_deployment_history("TestPlugin")
        assert len(history) == 1
        
        deployment = history[0]
        assert deployment.plugin_name == "TestPlugin"
        assert deployment.version == "1.0.0"
        assert deployment.environment == "production"
        assert deployment.deployed_by == "admin_user"
        assert deployment.config_overrides == {"timeout": 60}
    
    @pytest.mark.asyncio
    async def test_plugin_listing(self, plugin_manager):
        """Test plugin listing with filters."""
        # Register multiple plugins
        plugins_data = [
            ("Plugin1", PluginType.QUERY_EMBELLISHER, PluginStatus.PUBLISHED),
            ("Plugin2", PluginType.EMBED_DATA_EMBELLISHER, PluginStatus.DRAFT),
            ("Plugin3", PluginType.FAISS_CRUD, PluginStatus.PUBLISHED),
        ]
        
        for name, plugin_type, status in plugins_data:
            metadata = PluginMetadata(
                name=name,
                version="1.0.0",
                description=f"Test plugin {name}",
                author="Test Author",
                plugin_type=plugin_type
            )
            await plugin_manager.register_plugin(
                plugin_metadata=metadata,
                file_path=f"/test/{name}.py",
                file_hash=f"hash_{name}",
                file_size=1024
            )
            
            if status == PluginStatus.PUBLISHED:
                await plugin_manager.publish_plugin(name)
        
        # Test listing all plugins
        all_plugins = await plugin_manager.list_plugins()
        assert len(all_plugins) == 3
        
        # Test filtering by type
        query_plugins = await plugin_manager.list_plugins(plugin_type=PluginType.QUERY_EMBELLISHER)
        assert len(query_plugins) == 1
        assert query_plugins[0].name == "Plugin1"
        
        # Test filtering by status
        published_plugins = await plugin_manager.list_plugins(status=PluginStatus.PUBLISHED)
        assert len(published_plugins) == 2
        
        # Test enabled only filter
        enabled_plugins = await plugin_manager.list_plugins(enabled_only=True)
        assert len(enabled_plugins) == 3  # All should be enabled by default
    
    @pytest.mark.asyncio
    async def test_plugin_metrics_update(self, plugin_manager, sample_plugin_metadata):
        """Test plugin metrics updating."""
        # Register plugin
        await plugin_manager.register_plugin(
            plugin_metadata=sample_plugin_metadata,
            file_path="/test/path/test_plugin.py",
            file_hash="abc123",
            file_size=1024
        )
        
        # Update metrics
        metrics = {
            "execution_count": 100,
            "avg_execution_time_ms": 45.2,
            "success_rate": 0.98
        }
        
        success = await plugin_manager.update_plugin_metrics("TestPlugin", metrics)
        assert success is True
        
        # Verify metrics were stored
        plugin_doc = await plugin_manager.get_plugin("TestPlugin")
        assert plugin_doc.performance_metrics == metrics
        assert plugin_doc.last_used is not None
    
    @pytest.mark.asyncio
    async def test_plugin_statistics(self, plugin_manager):
        """Test plugin system statistics."""
        # Register some plugins with different types and statuses
        plugins_data = [
            ("Plugin1", PluginType.QUERY_EMBELLISHER, True),
            ("Plugin2", PluginType.EMBED_DATA_EMBELLISHER, False),
            ("Plugin3", PluginType.FAISS_CRUD, True),
        ]
        
        for name, plugin_type, is_enabled in plugins_data:
            metadata = PluginMetadata(
                name=name,
                version="1.0.0",
                description=f"Test plugin {name}",
                author="Test Author",
                plugin_type=plugin_type,
                is_enabled=is_enabled
            )
            await plugin_manager.register_plugin(
                plugin_metadata=metadata,
                file_path=f"/test/{name}.py",
                file_hash=f"hash_{name}",
                file_size=1024
            )
            
            # Publish first plugin
            if name == "Plugin1":
                await plugin_manager.publish_plugin(name)
        
        # Get statistics
        stats = await plugin_manager.get_plugin_statistics()
        
        assert stats["total_plugins"] == 3
        assert stats["published_plugins"] == 1
        assert stats["enabled_plugins"] == 2
        assert len(stats["plugin_types"]) == 3


class TestPluginDocumentModels:
    """Test the Pydantic models for plugin documents."""
    
    def test_plugin_version_model(self):
        """Test PluginVersion model."""
        version = PluginVersion(
            version="1.2.3",
            changelog="Added new features",
            is_stable=True
        )
        
        assert version.version == "1.2.3"
        assert version.changelog == "Added new features"
        assert version.is_stable is True
        assert version.release_date is not None
    
    def test_plugin_dependency_model(self):
        """Test PluginDependency model."""
        dependency = PluginDependency(
            name="base-plugin",
            version="^1.0.0",
            optional=False
        )
        
        assert dependency.name == "base-plugin"
        assert dependency.version == "^1.0.0"
        assert dependency.optional is False
    
    def test_plugin_document_model(self):
        """Test PluginDocument model."""
        doc = PluginDocument(
            name="TestPlugin",
            display_name="Test Plugin",
            description="A test plugin",
            author="Test Author",
            plugin_type=PluginType.QUERY_EMBELLISHER,
            current_version="1.0.0"
        )
        
        assert doc.name == "TestPlugin"
        assert doc.plugin_type == PluginType.QUERY_EMBELLISHER
        assert doc.status == PluginStatus.DRAFT  # default
        assert doc.is_enabled is True  # default
        assert doc.created_at is not None
    
    def test_plugin_deployment_model(self):
        """Test PluginDeployment model."""
        deployment = PluginDeployment(
            plugin_name="TestPlugin",
            version="1.0.0",
            environment="production",
            deployed_by="admin"
        )
        
        assert deployment.plugin_name == "TestPlugin"
        assert deployment.environment == "production"
        assert deployment.status == "deployed"  # default
        assert deployment.deployment_time is not None


def run_tests():
    """Run the test suite."""
    print("🧪 Running Plugin MongoDB Manager Tests")
    print("=" * 50)
    
    # Run tests
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return result.returncode == 0


if __name__ == "__main__":
    # Run a simple test without pytest if run directly
    async def simple_test():
        print("🧪 Running Simple Plugin Manager Test")
        
        # Mock the MongoDB client
        manager = MongoPluginManager()
        manager.mongo_client = MockMongoClient()
        manager._initialized = True  # Skip initialize to avoid calling get_mongo_client
        
        # Test plugin registration
        metadata = PluginMetadata(
            name="SimpleTestPlugin",
            version="1.0.0",
            description="Simple test plugin",
            author="Test Author",
            plugin_type=PluginType.QUERY_EMBELLISHER
        )
        
        plugin_id = await manager.register_plugin(
            plugin_metadata=metadata,
            file_path="/test/simple.py",
            file_hash="simple123",
            file_size=512
        )
        
        assert plugin_id is not None
        print("✅ Plugin registration successful")
        
        # Test plugin retrieval
        plugin_doc = await manager.get_plugin("SimpleTestPlugin")
        assert plugin_doc is not None
        assert plugin_doc.name == "SimpleTestPlugin"
        print("✅ Plugin retrieval successful")
        
        # Test plugin publishing
        success = await manager.publish_plugin("SimpleTestPlugin")
        assert success is True
        print("✅ Plugin publishing successful")
        
        print("🎉 All simple tests passed!")
    
    # Run the simple test
    asyncio.run(simple_test())