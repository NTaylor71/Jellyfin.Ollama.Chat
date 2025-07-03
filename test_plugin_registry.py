"""
Comprehensive Plugin Registry Tests
Tests for plugin discovery, registration, execution, and management.
"""

import asyncio
import pytest
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock

from src.api.plugin_registry import PluginRegistry, RegisteredPlugin
from src.plugins.base import (
    BasePlugin, PluginType, PluginMetadata, ExecutionPriority,
    PluginExecutionContext, PluginExecutionResult, PluginResourceRequirements
)
from src.plugins.config import PluginConfigManager


class TestPlugin(BasePlugin):
    """Test plugin for registry testing."""
    
    def __init__(self, name: str = "TestPlugin", plugin_type: PluginType = PluginType.QUERY_EMBELLISHER):
        super().__init__()
        self._plugin_metadata = PluginMetadata(
            name=name,
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            plugin_type=plugin_type,
            execution_priority=ExecutionPriority.NORMAL
        )
        self._plugin_resource_requirements = PluginResourceRequirements(
            min_cpu_cores=1.0,
            preferred_cpu_cores=2.0,
            min_memory_mb=100.0
        )
        self.initialized = False
        self.cleanup_called = False
    
    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata for registration."""
        return self._plugin_metadata
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        """Resource requirements for this plugin."""
        return self._plugin_resource_requirements
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration."""
        self.initialized = True
        return True
        
    async def execute(self, data: Any, context: PluginExecutionContext) -> PluginExecutionResult:
        """Test execution that adds a prefix."""
        if isinstance(data, str):
            return PluginExecutionResult(
                success=True,
                data=f"processed_{data}",
                metadata={"test": "metadata"}
            )
        return PluginExecutionResult(success=True, data=data)
    
    async def cleanup(self) -> None:
        self.cleanup_called = True


class HighPriorityPlugin(TestPlugin):
    """High priority test plugin."""
    
    def __init__(self):
        super().__init__("HighPriorityPlugin")
        self._plugin_metadata.execution_priority = ExecutionPriority.HIGH


class LowPriorityPlugin(TestPlugin):
    """Low priority test plugin."""
    
    def __init__(self):
        super().__init__("LowPriorityPlugin")
        self._plugin_metadata.execution_priority = ExecutionPriority.LOW


class FailingPlugin(TestPlugin):
    """Plugin that fails during execution."""
    
    def __init__(self):
        super().__init__("FailingPlugin")
    
    async def execute(self, data: Any, context: PluginExecutionContext) -> PluginExecutionResult:
        raise Exception("Test failure")


class ResourceHungryPlugin(TestPlugin):
    """Plugin with high resource requirements."""
    
    def __init__(self):
        super().__init__("ResourceHungryPlugin")
        self._plugin_resource_requirements = PluginResourceRequirements(
            min_cpu_cores=100,  # Impossibly high
            min_memory_mb=1000000,  # 1TB
            requires_gpu=True
        )


@pytest.fixture
def mock_resource_limits():
    """Mock resource limits for testing."""
    return {
        "total_cpu_capacity": 8,
        "gpu_available": False,
        "local_memory_gb": 16
    }


@pytest.fixture
def temp_plugin_dir():
    """Create a temporary plugin directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plugin_dir = Path(temp_dir) / "test_plugins"
        plugin_dir.mkdir()
        yield plugin_dir


@pytest.fixture
def sample_plugin_file(temp_plugin_dir):
    """Create a sample plugin file for testing."""
    plugin_content = '''
from src.plugins.base import BasePlugin, PluginType, PluginMetadata, ExecutionPriority
from typing import Any

class SamplePlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="SamplePlugin",
            version="1.0.0",
            description="Sample plugin for testing",
            plugin_type=PluginType.QUERY_EMBELLISHER,
            execution_priority=ExecutionPriority.NORMAL
        )
    
    async def execute(self, data: Any, context) -> Any:
        return {"processed": data}
'''
    plugin_file = temp_plugin_dir / "sample_plugin.py"
    plugin_file.write_text(plugin_content)
    return plugin_file


class TestPluginRegistry:
    """Test suite for PluginRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create a test plugin registry."""
        return PluginRegistry(plugin_directories=[])
    
    @pytest.mark.asyncio
    async def test_registry_initialization(self, registry):
        """Test basic registry initialization."""
        assert registry.plugin_directories == []  # Empty because we passed []
        assert len(registry._plugins) == 0
        assert len(registry._plugins_by_type) == 0
    
    @pytest.mark.asyncio
    async def test_manual_plugin_registration(self, registry, mock_resource_limits):
        """Test manual plugin registration."""
        with patch('src.api.plugin_registry.get_resource_limits', return_value=mock_resource_limits):
            # Register plugin manually
            await registry._register_plugin_class(TestPlugin, "/fake/path", "test_module")
            
            # Check registration
            assert "TestPlugin" in registry._plugins
            assert registry._plugins["TestPlugin"].plugin_class == TestPlugin
            
            # Initialize the plugin
            await registry._initialize_plugin("TestPlugin", mock_resource_limits)
            
            # Check initialization
            registered_plugin = registry._plugins["TestPlugin"]
            assert registered_plugin.instance is not None
            assert registered_plugin.instance.initialized
            assert registered_plugin.initialization_error is None
    
    @pytest.mark.asyncio
    async def test_plugin_discovery_security(self, registry):
        """Test plugin discovery security measures."""
        # Test safe module name validation
        assert registry._is_safe_module_name("src.plugins.test_plugin")
        assert registry._is_safe_module_name("src.plugins.examples.test")
        
        # Test unsafe module names
        assert not registry._is_safe_module_name("../dangerous")
        assert not registry._is_safe_module_name("src.plugins.__init__")
        assert not registry._is_safe_module_name("os.system")
        assert not registry._is_safe_module_name("sys.modules")
        assert not registry._is_safe_module_name("subprocess.call")
        assert not registry._is_safe_module_name("eval")
        assert not registry._is_safe_module_name("exec")
    
    @pytest.mark.asyncio
    async def test_get_plugin(self, registry, mock_resource_limits):
        """Test getting plugin by name."""
        with patch('src.api.plugin_registry.get_resource_limits', return_value=mock_resource_limits):
            # Register and initialize plugin
            await registry._register_plugin_class(TestPlugin, "/fake/path", "test_module")
            await registry._initialize_plugin("TestPlugin", mock_resource_limits)
            
            # Get plugin
            plugin = await registry.get_plugin("TestPlugin")
            assert plugin is not None
            assert plugin.metadata.name == "TestPlugin"
            
            # Get non-existent plugin
            missing_plugin = await registry.get_plugin("NonExistentPlugin")
            assert missing_plugin is None
    
    @pytest.mark.asyncio
    async def test_get_plugins_by_type(self, registry, mock_resource_limits):
        """Test getting plugins by type."""
        with patch('src.api.plugin_registry.get_resource_limits', return_value=mock_resource_limits):
            # Register plugins of different types
            await registry._register_plugin_class(TestPlugin, "/fake/path1", "test_module1")
            await registry._register_plugin_class(HighPriorityPlugin, "/fake/path2", "test_module2")
            await registry._register_plugin_class(LowPriorityPlugin, "/fake/path3", "test_module3")
            
            # Initialize plugins
            await registry._initialize_plugin("TestPlugin", mock_resource_limits)
            await registry._initialize_plugin("HighPriorityPlugin", mock_resource_limits)
            await registry._initialize_plugin("LowPriorityPlugin", mock_resource_limits)
            
            # Get plugins by type
            query_plugins = await registry.get_plugins_by_type(PluginType.QUERY_EMBELLISHER)
            assert len(query_plugins) == 3
            
            # Check priority ordering (HIGH, NORMAL, LOW)
            priorities = [p.metadata.execution_priority for p in query_plugins]
            assert priorities == [ExecutionPriority.HIGH, ExecutionPriority.NORMAL, ExecutionPriority.LOW]
    
    @pytest.mark.asyncio
    async def test_plugin_execution(self, registry, mock_resource_limits):
        """Test plugin execution."""
        with patch('src.api.plugin_registry.get_resource_limits', return_value=mock_resource_limits):
            # Register and initialize plugins
            await registry._register_plugin_class(TestPlugin, "/fake/path", "test_module")
            await registry._initialize_plugin("TestPlugin", mock_resource_limits)
            
            # Execute plugins
            context = PluginExecutionContext(user_id="test_user", metadata={"query": "test query"})
            results = await registry.execute_plugins(PluginType.QUERY_EMBELLISHER, "test_data", context)
            
            assert len(results) == 1
            assert results[0].success
            assert results[0].data == "processed_test_data"
            assert results[0].metadata == {"test": "metadata"}
    
    @pytest.mark.asyncio
    async def test_plugin_execution_chain(self, registry, mock_resource_limits):
        """Test plugin execution chain where plugins modify data."""
        with patch('src.api.plugin_registry.get_resource_limits', return_value=mock_resource_limits):
            # Register multiple plugins
            await registry._register_plugin_class(TestPlugin, "/fake/path1", "test_module1")
            
            # Create second plugin with different name
            class SecondPlugin(TestPlugin):
                def __init__(self):
                    super().__init__("SecondPlugin")
                    
                async def execute(self, data: Any, context: PluginExecutionContext) -> PluginExecutionResult:
                    if isinstance(data, str):
                        return PluginExecutionResult(success=True, data=f"second_{data}")
                    return PluginExecutionResult(success=True, data=data)
            
            await registry._register_plugin_class(SecondPlugin, "/fake/path2", "test_module2")
            
            # Initialize plugins
            await registry._initialize_plugin("TestPlugin", mock_resource_limits)
            await registry._initialize_plugin("SecondPlugin", mock_resource_limits)
            
            # Execute plugins - data should be modified by each plugin
            context = PluginExecutionContext(user_id="test_user", metadata={"query": "test query"})
            results = await registry.execute_plugins(PluginType.QUERY_EMBELLISHER, "original", context)
            
            assert len(results) == 2
            assert all(r.success for r in results)
            # First plugin: "original" -> "processed_original"
            # Second plugin: "processed_original" -> "second_processed_original"
            assert results[1].data == "second_processed_original"
    
    @pytest.mark.asyncio
    async def test_plugin_execution_failure(self, registry, mock_resource_limits):
        """Test plugin execution with failures."""
        with patch('src.api.plugin_registry.get_resource_limits', return_value=mock_resource_limits):
            # Register failing plugin
            await registry._register_plugin_class(FailingPlugin, "/fake/path", "test_module")
            await registry._initialize_plugin("FailingPlugin", mock_resource_limits)
            
            # Execute plugin
            context = PluginExecutionContext(user_id="test_user", metadata={"query": "test query"})
            results = await registry.execute_plugins(PluginType.QUERY_EMBELLISHER, "test_data", context)
            
            assert len(results) == 1
            assert not results[0].success
            assert "Plugin execution failed" in results[0].error_message
    
    @pytest.mark.asyncio
    async def test_resource_requirements_check(self, registry, mock_resource_limits):
        """Test resource requirements validation."""
        with patch('src.api.plugin_registry.get_resource_limits', return_value=mock_resource_limits):
            # Register resource-hungry plugin
            await registry._register_plugin_class(ResourceHungryPlugin, "/fake/path", "test_module")
            await registry._initialize_plugin("ResourceHungryPlugin", mock_resource_limits)
            
            # Check that plugin wasn't initialized due to resource constraints
            registered_plugin = registry._plugins["ResourceHungryPlugin"]
            assert registered_plugin.instance is None
            assert "Insufficient resources" in registered_plugin.initialization_error
    
    @pytest.mark.asyncio
    async def test_enable_disable_plugin(self, registry, mock_resource_limits):
        """Test enabling and disabling plugins."""
        with patch('src.api.plugin_registry.get_resource_limits', return_value=mock_resource_limits):
            # Register plugin
            await registry._register_plugin_class(TestPlugin, "/fake/path", "test_module")
            await registry._initialize_plugin("TestPlugin", mock_resource_limits)
            
            # Plugin should be enabled by default
            assert registry._plugins["TestPlugin"].is_enabled
            
            # Disable plugin
            success = await registry.disable_plugin("TestPlugin")
            assert success
            assert not registry._plugins["TestPlugin"].is_enabled
            assert registry._plugins["TestPlugin"].instance is None
            
            # Enable plugin
            success = await registry.enable_plugin("TestPlugin")
            assert success
            assert registry._plugins["TestPlugin"].is_enabled
            assert registry._plugins["TestPlugin"].instance is not None
    
    @pytest.mark.asyncio
    async def test_plugin_status(self, registry, mock_resource_limits):
        """Test plugin status reporting."""
        with patch('src.api.plugin_registry.get_resource_limits', return_value=mock_resource_limits):
            # Register plugins
            await registry._register_plugin_class(TestPlugin, "/fake/path1", "test_module1")
            await registry._register_plugin_class(HighPriorityPlugin, "/fake/path2", "test_module2")
            
            # Initialize one plugin
            await registry._initialize_plugin("TestPlugin", mock_resource_limits)
            
            # Get status
            status = await registry.get_plugin_status()
            
            assert status["total_plugins"] == 2
            assert status["enabled_plugins"] == 2
            assert status["initialized_plugins"] == 1
            assert PluginType.QUERY_EMBELLISHER.value in status["plugins_by_type"]
            
            # Check individual plugin status
            assert "TestPlugin" in status["plugin_details"]
            assert "HighPriorityPlugin" in status["plugin_details"]
            
            test_plugin_status = status["plugin_details"]["TestPlugin"]
            assert test_plugin_status["enabled"]
            assert test_plugin_status["initialized"]
            assert "health" in test_plugin_status
    
    @pytest.mark.asyncio
    async def test_cleanup(self, registry, mock_resource_limits):
        """Test registry cleanup."""
        with patch('src.api.plugin_registry.get_resource_limits', return_value=mock_resource_limits):
            # Register and initialize plugin
            await registry._register_plugin_class(TestPlugin, "/fake/path", "test_module")
            await registry._initialize_plugin("TestPlugin", mock_resource_limits)
            
            # Verify plugin is initialized
            plugin = await registry.get_plugin("TestPlugin")
            assert plugin is not None
            assert not plugin.cleanup_called
            
            # Cleanup registry
            await registry.cleanup()
            
            # Verify cleanup was called and registry is empty
            assert plugin.cleanup_called
            assert len(registry._plugins) == 0
            assert len(registry._plugins_by_type) == 0
    
    @pytest.mark.asyncio
    async def test_reload_plugin(self, registry, mock_resource_limits):
        """Test plugin reloading."""
        with patch('src.api.plugin_registry.get_resource_limits', return_value=mock_resource_limits):
            # Register and initialize plugin
            await registry._register_plugin_class(TestPlugin, "/fake/path", "test_module")
            await registry._initialize_plugin("TestPlugin", mock_resource_limits)
            
            # Get original plugin
            original_plugin = await registry.get_plugin("TestPlugin")
            assert original_plugin is not None
            
            # Mock the plugin file loading to simulate reload
            with patch.object(registry, '_load_plugin_file') as mock_load:
                mock_load.return_value = None
                
                # Reload plugin
                success = await registry.reload_plugin("TestPlugin")
                assert success
                
                # Verify cleanup was called on original plugin
                assert original_plugin.cleanup_called
                
                # Verify reload time was updated
                registered_plugin = registry._plugins["TestPlugin"]
                assert registered_plugin.last_reload_time > 0
    
    @pytest.mark.asyncio
    async def test_reload_nonexistent_plugin(self, registry):
        """Test reloading non-existent plugin."""
        success = await registry.reload_plugin("NonExistentPlugin")
        assert not success
    
    @pytest.mark.asyncio
    async def test_plugin_discovery_with_subdirectories(self, registry, temp_plugin_dir):
        """Test plugin discovery in subdirectories."""
        # Create subdirectory structure
        subdir = temp_plugin_dir / "subdir"
        subdir.mkdir()
        
        # Create plugin in subdirectory
        plugin_content = '''
from src.plugins.base import BasePlugin, PluginType, PluginMetadata
from typing import Any

class SubdirPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="SubdirPlugin",
            version="1.0.0",
            description="Plugin in subdirectory",
            plugin_type=PluginType.QUERY_EMBELLISHER
        )
    
    async def execute(self, data: Any, context) -> Any:
        return data
'''
        plugin_file = subdir / "subdir_plugin.py"
        plugin_file.write_text(plugin_content)
        
        # Test directory scanning
        with patch('src.api.plugin_registry.importlib.import_module') as mock_import:
            mock_import.return_value = type('MockModule', (), {
                'SubdirPlugin': type('SubdirPlugin', (BasePlugin,), {
                    '__name__': 'SubdirPlugin',
                    '__init__': lambda self: None,
                    'metadata': PluginMetadata(
                        name="SubdirPlugin",
                        version="1.0.0", 
                        description="Plugin in subdirectory",
                        plugin_type=PluginType.QUERY_EMBELLISHER
                    )
                })
            })
            
            # Should find plugin in subdirectory
            await registry._scan_directory(temp_plugin_dir)
            
            # Verify import was called (plugin file was found)
            assert mock_import.called


if __name__ == "__main__":
    # Run tests
    print("Running Plugin Registry Tests...")
    
    async def run_tests():
        """Run all tests."""
        test_registry = TestPluginRegistry()
        
        # Test basic functionality
        registry = PluginRegistry(plugin_directories=[])
        await test_registry.test_registry_initialization(registry)
        print("✓ Registry initialization test passed")
        
        # Test security
        await test_registry.test_plugin_discovery_security(registry)
        print("✓ Plugin discovery security test passed")
        
        # Test plugin management
        mock_limits = {
            "total_cpu_capacity": 8,
            "gpu_available": False,
            "local_memory_gb": 16
        }
        
        with patch('src.api.plugin_registry.get_resource_limits', return_value=mock_limits):
            await test_registry.test_manual_plugin_registration(registry, mock_limits)
            print("✓ Manual plugin registration test passed")
            
            # Clean up for next test
            await registry.cleanup()
            registry = PluginRegistry(plugin_directories=[])
            
            await test_registry.test_get_plugin(registry, mock_limits)
            print("✓ Get plugin test passed")
            
            # Clean up for next test
            await registry.cleanup()
            registry = PluginRegistry(plugin_directories=[])
            
            await test_registry.test_get_plugins_by_type(registry, mock_limits)
            print("✓ Get plugins by type test passed")
            
            # Clean up for next test
            await registry.cleanup()
            registry = PluginRegistry(plugin_directories=[])
            
            await test_registry.test_plugin_execution(registry, mock_limits)
            print("✓ Plugin execution test passed")
            
            # Clean up for next test
            await registry.cleanup()
            registry = PluginRegistry(plugin_directories=[])
            
            await test_registry.test_plugin_execution_chain(registry, mock_limits)
            print("✓ Plugin execution chain test passed")
            
            # Clean up for next test
            await registry.cleanup()
            registry = PluginRegistry(plugin_directories=[])
            
            await test_registry.test_resource_requirements_check(registry, mock_limits)
            print("✓ Resource requirements check test passed")
            
            # Clean up for next test
            await registry.cleanup()
            registry = PluginRegistry(plugin_directories=[])
            
            await test_registry.test_enable_disable_plugin(registry, mock_limits)
            print("✓ Enable/disable plugin test passed")
            
            # Clean up for next test
            await registry.cleanup()
            registry = PluginRegistry(plugin_directories=[])
            
            await test_registry.test_plugin_status(registry, mock_limits)
            print("✓ Plugin status test passed")
            
            # Clean up for next test
            await registry.cleanup()
            registry = PluginRegistry(plugin_directories=[])
            
            await test_registry.test_cleanup(registry, mock_limits)
            print("✓ Cleanup test passed")
        
        print("\n🎉 All Plugin Registry tests passed!")
        print("✅ Plugin discovery and security validation")
        print("✅ Plugin registration and initialization")
        print("✅ Plugin execution and chaining")
        print("✅ Resource requirements validation")
        print("✅ Plugin management (enable/disable)")
        print("✅ Status reporting and monitoring")
        print("✅ Cleanup and resource management")
    
    # Run the tests
    asyncio.run(run_tests())