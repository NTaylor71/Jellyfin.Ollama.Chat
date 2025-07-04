"""
Simple test for admin API endpoints.
Tests the plugin management admin endpoints.
"""

import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock

# Mock the MongoDB and plugin systems
class MockPluginManager:
    def __init__(self):
        self.plugins = {}
    
    async def get_plugin(self, name):
        plugin_doc = MagicMock()
        plugin_doc.name = name
        plugin_doc.current_version = "1.0.0"
        plugin_doc.status = "draft"
        return plugin_doc if name in self.plugins else None
    
    async def publish_plugin(self, name, version=None):
        return name in self.plugins
    
    async def record_deployment(self, **kwargs):
        return "deployment_123"
    
    async def get_deployment_history(self, name, env=None):
        deployment = MagicMock()
        deployment.version = "1.0.0"
        deployment.rollback_version = "0.9.0"
        deployment.deployment_time.isoformat.return_value = "2024-01-01T00:00:00"
        return [deployment] if name in self.plugins else []

class MockPluginRegistry:
    def __init__(self):
        self.plugins = {}
    
    async def get_plugin(self, name):
        if name in self.plugins:
            plugin = MagicMock()
            plugin.health_check = AsyncMock(return_value={"status": "healthy"})
            return plugin
        return None

async def test_admin_api():
    """Test admin API functionality."""
    print("🧪 Testing Admin API Endpoints")
    
    # Mock the dependencies
    mock_plugin_manager = MockPluginManager()
    mock_plugin_registry = MockPluginRegistry()
    
    # Add a test plugin
    mock_plugin_manager.plugins["TestPlugin"] = True
    mock_plugin_registry.plugins["TestPlugin"] = True
    
    with patch('src.api.routes.admin.get_plugin_manager', return_value=mock_plugin_manager), \
         patch('src.api.routes.admin.get_plugin_registry', return_value=mock_plugin_registry):
        
        # Import after patching
        from src.api.routes.admin import (
            release_plugin, PluginReleaseRequest, update_plugin_status, 
            PluginStatusUpdate, _perform_health_check
        )
        from src.plugins.mongo_manager import PluginStatus
        
        print("✅ Admin routes imported successfully")
        
        # Test health check
        health_result = await _perform_health_check("TestPlugin", mock_plugin_registry)
        assert health_result["healthy"] is True
        print("✅ Health check working")
        
        # Test plugin release request
        release_request = PluginReleaseRequest(
            plugin_name="TestPlugin",
            environment="production",
            force=False
        )
        
        print(f"✅ Plugin release request created: {release_request.plugin_name}")
        
        # Test status update request
        status_request = PluginStatusUpdate(
            status=PluginStatus.PUBLISHED,
            reason="Test deployment"
        )
        
        print(f"✅ Status update request created: {status_request.status}")
        
        print("🎉 Admin API tests passed!")

async def test_configuration_system():
    """Test the enhanced configuration system."""
    print("🧪 Testing Configuration System")
    
    try:
        from src.plugins.config import PluginConfigManager, BasePluginConfig, ConfigSource
        from pathlib import Path
        import tempfile
        
        # Create a test config class
        class TestPluginConfig(BasePluginConfig):
            test_setting: str = "default_value"
            test_timeout: int = 30
        
        # Create temporary config directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            
            # Create config manager
            manager = PluginConfigManager(
                plugin_name="TestPlugin",
                config_class=TestPluginConfig,
                config_dir=config_dir
            )
            
            # Test loading default config
            config = manager.load_config()
            assert config.test_setting == "default_value"
            assert config.test_timeout == 30
            print("✅ Default configuration loading works")
            
            # Test runtime config update
            success = manager.update_config({
                "test_setting": "updated_value",
                "test_timeout": 60
            })
            assert success is True
            
            updated_config = manager.get_config()
            assert updated_config.test_setting == "updated_value"
            assert updated_config.test_timeout == 60
            print("✅ Runtime configuration updates work")
            
            # Test config summary
            summary = manager.get_config_summary()
            assert summary["plugin_name"] == "TestPlugin"
            assert "values" in summary
            print("✅ Configuration summary works")
            
        print("🎉 Configuration system tests passed!")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        raise

def run_tests():
    """Run all tests."""
    print("🧪 Running Admin API and Configuration Tests")
    print("=" * 50)
    
    async def run_all():
        await test_admin_api()
        await test_configuration_system()
        print("\n🎉 All tests completed successfully!")
    
    asyncio.run(run_all())

if __name__ == "__main__":
    run_tests()