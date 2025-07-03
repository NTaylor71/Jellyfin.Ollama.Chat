"""
Plugin File Watcher
Monitors plugin files for changes and triggers hot-reload automatically.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Set, Callable, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from .plugin_registry import plugin_registry

logger = logging.getLogger(__name__)


class PluginFileEventHandler(FileSystemEventHandler):
    """Handle file system events for plugin files."""
    
    def __init__(self, reload_callback: Callable[[str], None], debounce_seconds: float = 2.0):
        super().__init__()
        self.reload_callback = reload_callback
        self.debounce_seconds = debounce_seconds
        self._pending_reloads: Dict[str, float] = {}
        self._reload_lock = asyncio.Lock()
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if not event.is_directory and event.src_path.endswith('.py'):
            self._schedule_reload(event.src_path)
    
    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if not event.is_directory and event.src_path.endswith('.py'):
            self._schedule_reload(event.src_path)
    
    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move events."""
        if not event.is_directory and event.dest_path.endswith('.py'):
            self._schedule_reload(event.dest_path)
    
    def _schedule_reload(self, file_path: str) -> None:
        """Schedule a reload with debouncing."""
        current_time = time.time()
        self._pending_reloads[file_path] = current_time
        
        # Use asyncio to handle the debounced reload
        asyncio.create_task(self._debounced_reload(file_path, current_time))
    
    async def _debounced_reload(self, file_path: str, scheduled_time: float) -> None:
        """Execute reload after debounce period."""
        await asyncio.sleep(self.debounce_seconds)
        
        # Check if this is still the latest reload request for this file
        if self._pending_reloads.get(file_path) == scheduled_time:
            async with self._reload_lock:
                try:
                    await self.reload_callback(file_path)
                    # Remove from pending reloads
                    self._pending_reloads.pop(file_path, None)
                except Exception as e:
                    logger.error(f"Error in reload callback for {file_path}: {e}")


class PluginWatcher:
    """Watches plugin directories for file changes and triggers hot-reload."""
    
    def __init__(self, plugin_directories: Optional[List[str]] = None):
        self.plugin_directories = plugin_directories or ["src/plugins"]
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[PluginFileEventHandler] = None
        self.is_watching = False
        self._file_to_plugin_mapping: Dict[str, Set[str]] = {}
        self._last_reload_times: Dict[str, float] = {}
        self.min_reload_interval = 1.0  # Minimum seconds between reloads
        
    async def start_watching(self) -> None:
        """Start watching plugin directories for changes."""
        if self.is_watching:
            logger.warning("Plugin watcher is already running")
            return
        
        logger.info("Starting plugin file watcher...")
        
        # Create event handler
        self.event_handler = PluginFileEventHandler(self._handle_file_change)
        
        # Create observer
        self.observer = Observer()
        
        # Watch each plugin directory
        for plugin_dir in self.plugin_directories:
            plugin_path = Path(plugin_dir)
            if plugin_path.exists():
                self.observer.schedule(
                    self.event_handler, 
                    str(plugin_path), 
                    recursive=True
                )
                logger.info(f"Watching plugin directory: {plugin_dir}")
            else:
                logger.warning(f"Plugin directory does not exist: {plugin_dir}")
        
        # Start the observer
        self.observer.start()
        self.is_watching = True
        
        # Build initial file-to-plugin mapping
        await self._build_file_mapping()
        
        logger.info("Plugin file watcher started successfully")
    
    async def stop_watching(self) -> None:
        """Stop watching plugin directories."""
        if not self.is_watching:
            return
        
        logger.info("Stopping plugin file watcher...")
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        
        self.event_handler = None
        self.is_watching = False
        self._file_to_plugin_mapping.clear()
        
        logger.info("Plugin file watcher stopped")
    
    async def _build_file_mapping(self) -> None:
        """Build mapping of files to plugin names."""
        self._file_to_plugin_mapping.clear()
        
        # Get plugin status from registry
        plugin_status = await plugin_registry.get_plugin_status()
        
        for plugin_name, details in plugin_status.get("plugin_details", {}).items():
            file_path = details.get("file_path")
            if file_path:
                # Normalize path for comparison
                normalized_path = str(Path(file_path).resolve())
                if normalized_path not in self._file_to_plugin_mapping:
                    self._file_to_plugin_mapping[normalized_path] = set()
                self._file_to_plugin_mapping[normalized_path].add(plugin_name)
    
    async def _handle_file_change(self, file_path: str) -> None:
        """Handle file change events and trigger plugin reloads."""
        try:
            # Normalize the file path
            normalized_path = str(Path(file_path).resolve())
            
            # Check if this file change should trigger a reload
            if not self._should_reload(normalized_path):
                return
            
            logger.info(f"Detected change in plugin file: {file_path}")
            
            # Update last reload time
            self._last_reload_times[normalized_path] = time.time()
            
            # Find plugins associated with this file
            plugins_to_reload = self._file_to_plugin_mapping.get(normalized_path, set())
            
            if plugins_to_reload:
                # Reload specific plugins
                for plugin_name in plugins_to_reload:
                    success = await plugin_registry.reload_plugin(plugin_name)
                    if success:
                        logger.info(f"Successfully reloaded plugin: {plugin_name}")
                    else:
                        logger.error(f"Failed to reload plugin: {plugin_name}")
            else:
                # File might be a new plugin file, trigger full discovery
                logger.info(f"New plugin file detected: {file_path}")
                await self._discover_new_plugins(file_path)
            
            # Rebuild file mapping after reload
            await self._build_file_mapping()
            
        except Exception as e:
            logger.error(f"Error handling file change {file_path}: {e}")
    
    def _should_reload(self, file_path: str) -> bool:
        """Check if a file should trigger a reload based on timing."""
        current_time = time.time()
        last_reload = self._last_reload_times.get(file_path, 0)
        
        if current_time - last_reload < self.min_reload_interval:
            logger.debug(f"Skipping reload for {file_path} due to rate limiting")
            return False
        
        return True
    
    async def _discover_new_plugins(self, file_path: str) -> None:
        """Discover and register new plugins from a file."""
        try:
            # Re-initialize the plugin registry to discover new plugins
            # This is a simplified approach - in production, you might want
            # to only scan the specific file
            await plugin_registry.initialize()
            logger.info(f"Completed plugin discovery after new file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error discovering new plugins from {file_path}: {e}")
    
    async def force_reload_all(self) -> int:
        """Force reload all plugins regardless of file changes."""
        logger.info("Force reloading all plugins...")
        
        try:
            success_count = await plugin_registry.reload_all_plugins()
            await self._build_file_mapping()
            
            logger.info(f"Force reload completed: {success_count} plugins reloaded")
            return success_count
            
        except Exception as e:
            logger.error(f"Error during force reload: {e}")
            return 0
    
    def get_watch_status(self) -> Dict[str, Any]:
        """Get current status of the plugin watcher."""
        return {
            "is_watching": self.is_watching,
            "plugin_directories": self.plugin_directories,
            "watched_files": len(self._file_to_plugin_mapping),
            "file_mappings": {
                path: list(plugins) for path, plugins in self._file_to_plugin_mapping.items()
            },
            "last_reload_times": self._last_reload_times.copy()
        }
    
    async def add_watch_directory(self, directory: str) -> bool:
        """Add a new directory to watch."""
        if directory in self.plugin_directories:
            return False
        
        self.plugin_directories.append(directory)
        
        # If already watching, add the new directory to observer
        if self.is_watching and self.observer and self.event_handler:
            plugin_path = Path(directory)
            if plugin_path.exists():
                self.observer.schedule(
                    self.event_handler,
                    str(plugin_path),
                    recursive=True
                )
                logger.info(f"Added watch directory: {directory}")
                await self._build_file_mapping()
                return True
            else:
                logger.warning(f"Directory does not exist: {directory}")
                return False
        
        return True
    
    async def remove_watch_directory(self, directory: str) -> bool:
        """Remove a directory from watching."""
        if directory not in self.plugin_directories:
            return False
        
        self.plugin_directories.remove(directory)
        
        # If watching, we need to restart the observer to remove the directory
        # This is a limitation of the watchdog library
        if self.is_watching:
            await self.stop_watching()
            await self.start_watching()
        
        logger.info(f"Removed watch directory: {directory}")
        return True


# Global plugin watcher instance
plugin_watcher = PluginWatcher()


async def get_plugin_watcher() -> PluginWatcher:
    """Get the global plugin watcher instance."""
    return plugin_watcher


async def start_plugin_watching() -> None:
    """Start the global plugin watcher."""
    await plugin_watcher.start_watching()


async def stop_plugin_watching() -> None:
    """Stop the global plugin watcher."""
    await plugin_watcher.stop_watching()