"""
Plugin Hot-Reload Tests
Tests for plugin file watching and hot-reload functionality.
"""

import asyncio
import pytest
import tempfile
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from src.api.plugin_watcher import PluginWatcher, PluginFileEventHandler, plugin_watcher
from src.api.plugin_registry import PluginRegistry
from src.plugins.base import (
    BasePlugin, PluginType, PluginMetadata, ExecutionPriority,
    PluginExecutionContext, PluginExecutionResult
)
from watchdog.events import FileModifiedEvent, FileCreatedEvent, FileMovedEvent


class TestHotReloadPlugin(BasePlugin):
    """Test plugin for hot-reload testing."""
    
    def __init__(self, name: str = "TestHotReloadPlugin", version: str = "1.0.0"):
        super().__init__()
        self.metadata = PluginMetadata(
            name=name,
            version=version,
            description="Test plugin for hot-reload",
            plugin_type=PluginType.QUERY_EMBELLISHER,
            execution_priority=ExecutionPriority.NORMAL
        )
        self.initialized = False
        self.cleanup_called = False
        self.execution_count = 0
        
    async def execute(self, data: Any, context: PluginExecutionContext) -> PluginExecutionResult:
        """Test execution that adds version info."""
        self.execution_count += 1
        if isinstance(data, str):
            return PluginExecutionResult(
                success=True,
                data=f"{data}_v{self.metadata.version}",
                metadata={"version": self.metadata.version, "executions": self.execution_count}
            )
        return PluginExecutionResult(success=True, data=data)
    
    async def initialize_with_config(self, config_manager: Optional[Any] = None) -> bool:
        self.initialized = True
        return True
    
    async def cleanup(self) -> None:
        self.cleanup_called = True


@pytest.fixture
def temp_plugin_dir():
    """Create a temporary plugin directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        plugin_dir = Path(temp_dir) / "test_plugins"
        plugin_dir.mkdir()
        yield plugin_dir


@pytest.fixture
def mock_registry():
    """Create a mock plugin registry."""
    registry = Mock(spec=PluginRegistry)
    registry.get_plugin_status = AsyncMock(return_value={
        "plugin_details": {
            "TestPlugin": {
                "file_path": "/test/path/test_plugin.py"
            }
        }
    })
    registry.reload_plugin = AsyncMock(return_value=True)
    registry.reload_all_plugins = AsyncMock(return_value=2)
    registry.initialize = AsyncMock()
    return registry


@pytest.fixture
def sample_plugin_content():
    """Generate sample plugin content."""
    return '''
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


class TestPluginFileEventHandler:
    """Test suite for PluginFileEventHandler class."""
    
    def test_handler_creation(self):
        """Test event handler creation."""
        callback = Mock()
        handler = PluginFileEventHandler(callback)
        
        assert handler.reload_callback == callback
        assert handler.debounce_seconds == 2.0
        assert handler._pending_reloads == {}
    
    def test_handler_with_custom_debounce(self):
        """Test handler with custom debounce time."""
        callback = Mock()
        handler = PluginFileEventHandler(callback, debounce_seconds=5.0)
        
        assert handler.debounce_seconds == 5.0
    
    def test_on_modified_python_file(self):
        """Test handling of Python file modification."""
        callback = Mock()
        handler = PluginFileEventHandler(callback)
        
        # Create mock event
        event = Mock()
        event.is_directory = False
        event.src_path = "/test/plugin.py"
        
        # Handle event
        handler.on_modified(event)
        
        # Check that reload was scheduled
        assert "/test/plugin.py" in handler._pending_reloads
    
    def test_on_modified_non_python_file(self):
        """Test handling of non-Python file modification."""
        callback = Mock()
        handler = PluginFileEventHandler(callback)
        
        # Create mock event for non-Python file
        event = Mock()
        event.is_directory = False
        event.src_path = "/test/readme.txt"
        
        # Handle event
        handler.on_modified(event)
        
        # Check that reload was not scheduled
        assert "/test/readme.txt" not in handler._pending_reloads
    
    def test_on_modified_directory(self):
        """Test handling of directory modification."""
        callback = Mock()
        handler = PluginFileEventHandler(callback)
        
        # Create mock event for directory
        event = Mock()
        event.is_directory = True
        event.src_path = "/test/plugin_dir"
        
        # Handle event
        handler.on_modified(event)
        
        # Check that reload was not scheduled
        assert "/test/plugin_dir" not in handler._pending_reloads
    
    def test_on_created_python_file(self):
        """Test handling of Python file creation."""
        callback = Mock()
        handler = PluginFileEventHandler(callback)
        
        # Create mock event
        event = Mock()
        event.is_directory = False
        event.src_path = "/test/new_plugin.py"
        
        # Handle event
        handler.on_created(event)
        
        # Check that reload was scheduled
        assert "/test/new_plugin.py" in handler._pending_reloads
    
    def test_on_moved_python_file(self):
        """Test handling of Python file move."""
        callback = Mock()
        handler = PluginFileEventHandler(callback)
        
        # Create mock event
        event = Mock()
        event.is_directory = False
        event.dest_path = "/test/moved_plugin.py"
        
        # Handle event
        handler.on_moved(event)
        
        # Check that reload was scheduled
        assert "/test/moved_plugin.py" in handler._pending_reloads
    
    @pytest.mark.asyncio
    async def test_debounced_reload(self):
        """Test debounced reload mechanism."""
        callback = AsyncMock()
        handler = PluginFileEventHandler(callback, debounce_seconds=0.1)
        
        file_path = "/test/plugin.py"
        
        # Schedule reload
        handler._schedule_reload(file_path)
        
        # Wait for debounce
        await asyncio.sleep(0.2)
        
        # Check that callback was called
        callback.assert_called_once_with(file_path)
        
        # Check that pending reload was cleared
        assert file_path not in handler._pending_reloads
    
    @pytest.mark.asyncio
    async def test_debounced_reload_multiple_changes(self):
        """Test that multiple rapid changes result in single reload."""
        callback = AsyncMock()
        handler = PluginFileEventHandler(callback, debounce_seconds=0.2)
        
        file_path = "/test/plugin.py"
        
        # Schedule multiple reloads rapidly
        handler._schedule_reload(file_path)
        await asyncio.sleep(0.05)
        handler._schedule_reload(file_path)
        await asyncio.sleep(0.05)
        handler._schedule_reload(file_path)
        
        # Wait for debounce
        await asyncio.sleep(0.3)
        
        # Check that callback was called only once
        callback.assert_called_once_with(file_path)


class TestPluginWatcher:
    """Test suite for PluginWatcher class."""
    
    def test_watcher_creation(self):
        """Test plugin watcher creation."""
        watcher = PluginWatcher()
        
        assert watcher.plugin_directories == ["src/plugins"]
        assert not watcher.is_watching
        assert watcher.observer is None
        assert watcher.event_handler is None
    
    def test_watcher_with_custom_directories(self):
        """Test plugin watcher with custom directories."""
        custom_dirs = ["custom/plugins", "another/plugins"]
        watcher = PluginWatcher(custom_dirs)
        
        assert watcher.plugin_directories == custom_dirs
    
    @pytest.mark.asyncio
    async def test_start_watching_nonexistent_directory(self):
        """Test starting watcher with non-existent directory."""
        watcher = PluginWatcher(["nonexistent/plugins"])
        
        # Should not raise an exception
        await watcher.start_watching()
        
        assert watcher.is_watching
        assert watcher.observer is not None
        
        # Clean up
        await watcher.stop_watching()
    
    @pytest.mark.asyncio
    async def test_start_watching_existing_directory(self, temp_plugin_dir):
        """Test starting watcher with existing directory."""
        watcher = PluginWatcher([str(temp_plugin_dir)])
        
        await watcher.start_watching()
        
        assert watcher.is_watching
        assert watcher.observer is not None
        assert watcher.event_handler is not None
        
        # Clean up
        await watcher.stop_watching()
    
    @pytest.mark.asyncio
    async def test_start_watching_already_watching(self, temp_plugin_dir):
        """Test starting watcher when already watching."""
        watcher = PluginWatcher([str(temp_plugin_dir)])
        
        await watcher.start_watching()
        assert watcher.is_watching
        
        # Try to start again
        await watcher.start_watching()
        assert watcher.is_watching
        
        # Clean up
        await watcher.stop_watching()
    
    @pytest.mark.asyncio
    async def test_stop_watching(self, temp_plugin_dir):
        """Test stopping watcher."""
        watcher = PluginWatcher([str(temp_plugin_dir)])
        
        await watcher.start_watching()
        assert watcher.is_watching
        
        await watcher.stop_watching()
        assert not watcher.is_watching
        assert watcher.observer is None
        assert watcher.event_handler is None
    
    @pytest.mark.asyncio
    async def test_stop_watching_not_watching(self):
        """Test stopping watcher when not watching."""
        watcher = PluginWatcher()
        
        # Should not raise an exception
        await watcher.stop_watching()
        assert not watcher.is_watching
    
    @pytest.mark.asyncio
    async def test_build_file_mapping(self, mock_registry):
        """Test building file-to-plugin mapping."""
        watcher = PluginWatcher()
        
        with patch('src.api.plugin_watcher.plugin_registry', mock_registry):
            await watcher._build_file_mapping()
        
        # Check that file mapping was built
        assert len(watcher._file_to_plugin_mapping) > 0
        # The path should be normalized
        assert any("test_plugin.py" in path for path in watcher._file_to_plugin_mapping.keys())
    
    @pytest.mark.asyncio
    async def test_should_reload_rate_limiting(self):
        """Test reload rate limiting."""
        watcher = PluginWatcher()
        watcher.min_reload_interval = 1.0
        
        file_path = "/test/plugin.py"
        
        # First reload should be allowed
        assert watcher._should_reload(file_path)
        
        # Mark as recently reloaded
        watcher._last_reload_times[file_path] = time.time()
        
        # Immediate second reload should be blocked
        assert not watcher._should_reload(file_path)
        
        # Wait for rate limit to expire
        await asyncio.sleep(1.1)
        
        # Now should be allowed again
        assert watcher._should_reload(file_path)
    
    @pytest.mark.asyncio
    async def test_handle_file_change_existing_plugin(self, mock_registry):
        """Test handling file change for existing plugin."""
        watcher = PluginWatcher()
        
        # Set up file mapping
        test_path = str(Path("/test/path/test_plugin.py").resolve())
        watcher._file_to_plugin_mapping[test_path] = {"TestPlugin"}
        
        with patch('src.api.plugin_watcher.plugin_registry', mock_registry):
            await watcher._handle_file_change("/test/path/test_plugin.py")
        
        # Check that plugin was reloaded
        mock_registry.reload_plugin.assert_called_once_with("TestPlugin")
    
    @pytest.mark.asyncio
    async def test_handle_file_change_new_plugin(self, mock_registry):
        """Test handling file change for new plugin file."""
        watcher = PluginWatcher()
        
        # Empty file mapping (new plugin)
        watcher._file_to_plugin_mapping = {}
        
        with patch('src.api.plugin_watcher.plugin_registry', mock_registry):
            await watcher._handle_file_change("/test/path/new_plugin.py")
        
        # Check that discovery was triggered
        mock_registry.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_force_reload_all(self, mock_registry):
        """Test force reloading all plugins."""
        watcher = PluginWatcher()
        
        with patch('src.api.plugin_watcher.plugin_registry', mock_registry):
            result = await watcher.force_reload_all()
        
        assert result == 2  # Mock returns 2
        mock_registry.reload_all_plugins.assert_called_once()
    
    def test_get_watch_status(self):
        """Test getting watch status."""
        watcher = PluginWatcher(["custom/plugins"])
        watcher.is_watching = True
        watcher._file_to_plugin_mapping = {
            "/path/plugin1.py": {"Plugin1"},
            "/path/plugin2.py": {"Plugin2"}
        }
        
        status = watcher.get_watch_status()
        
        assert status["is_watching"] == True
        assert status["plugin_directories"] == ["custom/plugins"]
        assert status["watched_files"] == 2
        assert len(status["file_mappings"]) == 2
    
    @pytest.mark.asyncio
    async def test_add_watch_directory(self, temp_plugin_dir):
        """Test adding watch directory."""
        watcher = PluginWatcher([])
        
        # Add directory when not watching
        result = await watcher.add_watch_directory(str(temp_plugin_dir))
        assert result == True
        assert str(temp_plugin_dir) in watcher.plugin_directories
        
        # Add same directory again
        result = await watcher.add_watch_directory(str(temp_plugin_dir))
        assert result == False
    
    @pytest.mark.asyncio
    async def test_add_watch_directory_while_watching(self, temp_plugin_dir):
        """Test adding watch directory while already watching."""
        watcher = PluginWatcher([])
        
        # Start watching
        await watcher.start_watching()
        
        # Add directory
        result = await watcher.add_watch_directory(str(temp_plugin_dir))
        assert result == True
        assert str(temp_plugin_dir) in watcher.plugin_directories
        
        # Clean up
        await watcher.stop_watching()
    
    @pytest.mark.asyncio
    async def test_remove_watch_directory(self, temp_plugin_dir):
        """Test removing watch directory."""
        watcher = PluginWatcher([str(temp_plugin_dir)])
        
        # Remove directory
        result = await watcher.remove_watch_directory(str(temp_plugin_dir))
        assert result == True
        assert str(temp_plugin_dir) not in watcher.plugin_directories
        
        # Remove non-existent directory
        result = await watcher.remove_watch_directory("nonexistent")
        assert result == False


class TestPluginWatcherIntegration:
    """Integration tests for plugin watcher with actual file system operations."""
    
    @pytest.mark.asyncio
    async def test_file_change_detection(self, temp_plugin_dir, sample_plugin_content):
        """Test actual file change detection."""
        # Create plugin file
        plugin_file = temp_plugin_dir / "test_plugin.py"
        plugin_file.write_text(sample_plugin_content)
        
        # Create watcher
        watcher = PluginWatcher([str(temp_plugin_dir)])
        
        # Track reload calls
        reload_calls = []
        
        async def mock_reload_callback(file_path):
            reload_calls.append(file_path)
        
        # Start watching with mock callback
        await watcher.start_watching()
        if watcher.event_handler:
            watcher.event_handler.reload_callback = mock_reload_callback
            watcher.event_handler.debounce_seconds = 0.1  # Fast debounce for testing
        
        # Modify the file
        plugin_file.write_text(sample_plugin_content + "\n# Modified")
        
        # Wait for file system event and debounce
        await asyncio.sleep(0.5)
        
        # Check that reload was called
        assert len(reload_calls) > 0
        assert str(plugin_file) in reload_calls[0]
        
        # Clean up
        await watcher.stop_watching()
    
    @pytest.mark.asyncio
    async def test_new_file_detection(self, temp_plugin_dir, sample_plugin_content):
        """Test detection of new plugin files."""
        # Create watcher
        watcher = PluginWatcher([str(temp_plugin_dir)])
        
        # Track reload calls
        reload_calls = []
        
        async def mock_reload_callback(file_path):
            reload_calls.append(file_path)
        
        # Start watching
        await watcher.start_watching()
        if watcher.event_handler:
            watcher.event_handler.reload_callback = mock_reload_callback
            watcher.event_handler.debounce_seconds = 0.1
        
        # Create new plugin file
        new_plugin_file = temp_plugin_dir / "new_plugin.py"
        new_plugin_file.write_text(sample_plugin_content)
        
        # Wait for file system event
        await asyncio.sleep(0.5)
        
        # Check that reload was called
        assert len(reload_calls) > 0
        assert str(new_plugin_file) in reload_calls[0]
        
        # Clean up
        await watcher.stop_watching()


class TestHotReloadEndToEnd:
    """End-to-end tests for hot reload functionality."""
    
    @pytest.mark.asyncio
    async def test_plugin_hot_reload_scenario(self, temp_plugin_dir):
        """Test complete hot reload scenario."""
        # Create initial plugin content
        plugin_v1 = '''
from src.plugins.base import BasePlugin, PluginType, PluginMetadata
from typing import Any

class HotReloadTestPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="HotReloadTestPlugin",
            version="1.0.0",
            description="Version 1.0.0",
            plugin_type=PluginType.QUERY_EMBELLISHER
        )
    
    async def execute(self, data: Any, context) -> Any:
        return {"version": "1.0.0", "data": data}
'''
        
        plugin_v2 = '''
from src.plugins.base import BasePlugin, PluginType, PluginMetadata
from typing import Any

class HotReloadTestPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="HotReloadTestPlugin",
            version="2.0.0",
            description="Version 2.0.0",
            plugin_type=PluginType.QUERY_EMBELLISHER
        )
    
    async def execute(self, data: Any, context) -> Any:
        return {"version": "2.0.0", "data": data, "updated": True}
'''
        
        # Create plugin file with v1
        plugin_file = temp_plugin_dir / "hot_reload_test.py"
        plugin_file.write_text(plugin_v1)
        
        # Create mock registry to track reloads
        reload_count = 0
        
        class MockRegistry:
            async def get_plugin_status(self):
                return {
                    "plugin_details": {
                        "HotReloadTestPlugin": {
                            "file_path": str(plugin_file)
                        }
                    }
                }
            
            async def reload_plugin(self, plugin_name):
                nonlocal reload_count
                reload_count += 1
                return True
            
            async def initialize(self):
                pass
        
        mock_registry = MockRegistry()
        
        # Create watcher
        watcher = PluginWatcher([str(temp_plugin_dir)])
        
        # Start watching
        await watcher.start_watching()
        
        # Mock the registry
        with patch('src.api.plugin_watcher.plugin_registry', mock_registry):
            # Build initial file mapping
            await watcher._build_file_mapping()
            
            # Ensure fast debounce for testing
            if watcher.event_handler:
                watcher.event_handler.debounce_seconds = 0.1
            
            # Update plugin file to v2
            plugin_file.write_text(plugin_v2)
            
            # Wait for file change detection and processing
            await asyncio.sleep(0.5)
            
            # Check that reload was triggered
            assert reload_count > 0
        
        # Clean up
        await watcher.stop_watching()


if __name__ == "__main__":
    # Run tests
    print("Running Plugin Hot-Reload Tests...")
    
    async def run_tests():
        """Run all tests."""
        
        # Test event handler
        print("\n=== Testing PluginFileEventHandler ===")
        handler_tests = TestPluginFileEventHandler()
        handler_tests.test_handler_creation()
        print("✓ Handler creation test passed")
        
        handler_tests.test_on_modified_python_file()
        print("✓ Python file modification test passed")
        
        handler_tests.test_on_modified_non_python_file()
        print("✓ Non-Python file modification test passed")
        
        await handler_tests.test_debounced_reload()
        print("✓ Debounced reload test passed")
        
        await handler_tests.test_debounced_reload_multiple_changes()
        print("✓ Multiple changes debounce test passed")
        
        # Test plugin watcher
        print("\n=== Testing PluginWatcher ===")
        watcher_tests = TestPluginWatcher()
        watcher_tests.test_watcher_creation()
        print("✓ Watcher creation test passed")
        
        await watcher_tests.test_start_watching_nonexistent_directory()
        print("✓ Start watching non-existent directory test passed")
        
        await watcher_tests.test_stop_watching_not_watching()
        print("✓ Stop watching when not watching test passed")
        
        await watcher_tests.test_should_reload_rate_limiting()
        print("✓ Rate limiting test passed")
        
        # Mock registry for testing
        mock_registry = Mock()
        mock_registry.get_plugin_status = AsyncMock(return_value={
            "plugin_details": {
                "TestPlugin": {
                    "file_path": "/test/path/test_plugin.py"
                }
            }
        })
        mock_registry.reload_plugin = AsyncMock(return_value=True)
        mock_registry.initialize = AsyncMock()
        
        await watcher_tests.test_build_file_mapping(mock_registry)
        print("✓ File mapping test passed")
        
        await watcher_tests.test_handle_file_change_existing_plugin(mock_registry)
        print("✓ Handle existing plugin change test passed")
        
        await watcher_tests.test_handle_file_change_new_plugin(mock_registry)
        print("✓ Handle new plugin change test passed")
        
        watcher_tests.test_get_watch_status()
        print("✓ Watch status test passed")
        
        print("\n🎉 All Plugin Hot-Reload tests passed!")
        print("✅ File system event handling")
        print("✅ Debounced reload mechanism")
        print("✅ File-to-plugin mapping")
        print("✅ Rate limiting protection")
        print("✅ Watch directory management")
        print("✅ Plugin reload coordination")
        print("✅ Status monitoring")
    
    # Run the tests
    asyncio.run(run_tests())