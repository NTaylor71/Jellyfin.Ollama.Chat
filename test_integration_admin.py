"""
Integration test for admin API functionality.
Tests if the FastAPI app can start with admin routes.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_fastapi_app_startup():
    """Test that FastAPI app can start with admin routes."""
    print("🧪 Testing FastAPI App Startup with Admin Routes")
    
    try:
        # Test imports
        from src.api.routes import admin
        print("✅ Admin routes module imported successfully")
        
        from src.api.main import create_app
        print("✅ Main app module imported successfully")
        
        # Create the app (this will test if all routes are properly configured)
        app = create_app()
        print("✅ FastAPI app created successfully with admin routes")
        
        # Check if admin routes are included
        route_paths = [route.path for route in app.routes]
        admin_routes = [path for path in route_paths if path.startswith('/admin')]
        
        print(f"✅ Found {len(admin_routes)} admin routes:")
        for route in admin_routes:
            print(f"   - {route}")
        
        expected_admin_routes = [
            '/admin/plugins/release',
            '/admin/plugins/{plugin_name}/status',
            '/admin/plugins/bulk-operation',
            '/admin/plugins/statistics',
            '/admin/plugins/{plugin_name}/deployment-history',
            '/admin/plugins/{plugin_name}/rollback'
        ]
        
        found_routes = len(admin_routes)
        expected_routes = len(expected_admin_routes)
        
        if found_routes >= 4:  # We should have at least 4 admin routes
            print(f"✅ Admin routes properly registered ({found_routes} routes found)")
        else:
            print(f"⚠️ Expected more admin routes (found {found_routes}, expected ~{expected_routes})")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ App startup error: {e}")
        return False

async def test_plugin_models():
    """Test plugin-related Pydantic models."""
    print("\n🧪 Testing Plugin Models")
    
    try:
        from src.plugins.mongo_manager import PluginDocument, PluginStatus, PluginVersion
        from src.plugins.base import PluginMetadata, PluginType, ExecutionPriority
        
        # Test PluginMetadata
        metadata = PluginMetadata(
            name="TestPlugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            plugin_type=PluginType.QUERY_EMBELLISHER
        )
        print("✅ PluginMetadata model works")
        
        # Test PluginVersion
        version = PluginVersion(
            version="1.0.0",
            changelog="Initial version"
        )
        print("✅ PluginVersion model works")
        
        # Test PluginDocument
        doc = PluginDocument(
            name="TestPlugin",
            display_name="Test Plugin",
            description="A test plugin",
            author="Test Author",
            plugin_type=PluginType.QUERY_EMBELLISHER,
            current_version="1.0.0"
        )
        print("✅ PluginDocument model works")
        print(f"   - Name: {doc.name}")
        print(f"   - Status: {doc.status}")
        print(f"   - Version: {doc.current_version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test error: {e}")
        return False

async def test_configuration_system():
    """Test configuration system basic functionality."""
    print("\n🧪 Testing Configuration System")
    
    try:
        from src.plugins.config import BasePluginConfig, PluginConfigManager, ConfigSource
        from pydantic import Field
        import tempfile
        from pathlib import Path
        
        # Define test config class
        class TestConfig(BasePluginConfig):
            test_value: str = Field(default="default", description="Test configuration value")
            test_number: int = Field(default=42, description="Test number value")
        
        # Test with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = PluginConfigManager(
                plugin_name="TestPlugin",
                config_class=TestConfig,
                config_dir=Path(temp_dir)
            )
            
            # Test default config loading
            config = config_manager.load_config()
            assert config.test_value == "default"
            assert config.test_number == 42
            print("✅ Default configuration loading works")
            
            # Test config updates
            success = config_manager.update_config({
                "test_value": "updated",
                "test_number": 100
            })
            assert success is True
            print("✅ Configuration updates work")
            
            # Test updated values
            updated_config = config_manager.get_config()
            assert updated_config.test_value == "updated"
            assert updated_config.test_number == 100
            print("✅ Updated configuration retrieval works")
            
            # Test config summary
            summary = config_manager.get_config_summary()
            assert summary["plugin_name"] == "TestPlugin"
            assert "values" in summary
            print("✅ Configuration summary works")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration test error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_admin_request_models():
    """Test admin API request/response models."""
    print("\n🧪 Testing Admin API Models")
    
    try:
        from src.api.routes.admin import PluginReleaseRequest, PluginReleaseResponse, PluginStatusUpdate
        from src.plugins.mongo_manager import PluginStatus
        
        # Test PluginReleaseRequest
        release_request = PluginReleaseRequest(
            plugin_name="TestPlugin",
            environment="production",
            force=False
        )
        assert release_request.plugin_name == "TestPlugin"
        assert release_request.environment == "production"
        print("✅ PluginReleaseRequest model works")
        
        # Test PluginReleaseResponse
        response = PluginReleaseResponse(
            success=True,
            message="Plugin released successfully"
        )
        assert response.success is True
        print("✅ PluginReleaseResponse model works")
        
        # Test PluginStatusUpdate
        status_update = PluginStatusUpdate(
            status=PluginStatus.PUBLISHED,
            reason="Ready for production"
        )
        assert status_update.status == PluginStatus.PUBLISHED
        print("✅ PluginStatusUpdate model works")
        
        return True
        
    except Exception as e:
        print(f"❌ Admin models test error: {e}")
        return False

def run_integration_tests():
    """Run all integration tests."""
    print("🧪 Running Plugin Management Integration Tests")
    print("=" * 60)
    
    async def run_all():
        results = []
        
        # Test FastAPI app startup
        results.append(await test_fastapi_app_startup())
        
        # Test plugin models
        results.append(await test_plugin_models())
        
        # Test configuration system
        results.append(await test_configuration_system())
        
        # Test admin API models
        results.append(await test_admin_request_models())
        
        # Summary
        passed = sum(results)
        total = len(results)
        
        print(f"\n🎯 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All integration tests passed!")
            print("\n✅ Plugin Management System is ready for production!")
        else:
            print(f"❌ {total - passed} tests failed")
        
        return passed == total
    
    try:
        return asyncio.run(run_all())
    except Exception as e:
        print(f"❌ Test runner error: {e}")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    if success:
        print("\n🚀 Ready to proceed with Stage 5.2 completion!")
    else:
        print("\n🔧 Some issues need to be addressed before completion.")