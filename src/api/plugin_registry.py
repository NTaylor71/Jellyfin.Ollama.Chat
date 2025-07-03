"""
Plugin Registry
Manages plugin registration, discovery, and execution with hot-reload capabilities.
"""

import asyncio
import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Set
from dataclasses import dataclass
from collections import defaultdict

from ..plugins.base import (
    BasePlugin, PluginType, PluginMetadata, ExecutionPriority, 
    PluginExecutionContext, PluginExecutionResult
)
from ..shared.hardware_config import get_resource_limits

logger = logging.getLogger(__name__)


@dataclass
class RegisteredPlugin:
    """Information about a registered plugin."""
    plugin_class: Type[BasePlugin]
    instance: Optional[BasePlugin] = None
    file_path: Optional[str] = None
    module_name: Optional[str] = None
    is_enabled: bool = True
    registration_time: float = 0.0
    last_reload_time: float = 0.0
    initialization_error: Optional[str] = None


class PluginRegistry:
    """Central registry for managing plugins with hot-reload capabilities."""
    
    def __init__(self, plugin_directories: Optional[List[str]] = None):
        self.plugin_directories = plugin_directories or ["src/plugins"]
        self._plugins: Dict[str, RegisteredPlugin] = {}
        self._plugins_by_type: Dict[PluginType, List[str]] = defaultdict(list)
        self._initialization_lock = asyncio.Lock()
        self._reload_lock = asyncio.Lock()
        self._loaded_modules: Set[str] = set()
        
    async def initialize(self) -> None:
        """Initialize the plugin registry and discover plugins."""
        async with self._initialization_lock:
            logger.info("Initializing plugin registry...")
            
            # Discover and register plugins
            await self._discover_plugins()
            
            # Initialize all enabled plugins
            await self._initialize_plugins()
            
            logger.info(f"Plugin registry initialized with {len(self._plugins)} plugins")
    
    async def _discover_plugins(self) -> None:
        """Discover plugin files and register plugin classes."""
        for plugin_dir in self.plugin_directories:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                logger.warning(f"Plugin directory does not exist: {plugin_dir}")
                continue
                
            logger.info(f"Discovering plugins in {plugin_dir}")
            await self._scan_directory(plugin_path)
    
    async def _scan_directory(self, directory: Path) -> None:
        """Recursively scan directory for plugin files."""
        for item in directory.iterdir():
            if item.is_file() and item.suffix == '.py' and not item.name.startswith('__'):
                await self._load_plugin_file(item)
            elif item.is_dir() and not item.name.startswith('__'):
                await self._scan_directory(item)
    
    async def _load_plugin_file(self, file_path: Path) -> None:
        """Load a plugin file and register any plugin classes found."""
        try:
            # Convert file path to module name
            relative_path = file_path.relative_to(Path.cwd())
            module_name = str(relative_path).replace(os.sep, '.').replace('.py', '')
            
            # Skip if already loaded
            if module_name in self._loaded_modules:
                return
            
            # Import the module
            if module_name in sys.modules:
                module = importlib.reload(sys.modules[module_name])
            else:
                module = importlib.import_module(module_name)
            
            self._loaded_modules.add(module_name)
            
            # Find plugin classes in the module
            plugin_classes = self._find_plugin_classes(module)
            
            # Register each plugin class
            for plugin_class in plugin_classes:
                await self._register_plugin_class(plugin_class, str(file_path), module_name)
                
        except Exception as e:
            logger.error(f"Failed to load plugin file {file_path}: {e}")
    
    def _find_plugin_classes(self, module) -> List[Type[BasePlugin]]:
        """Find all plugin classes in a module."""
        plugin_classes = []
        
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BasePlugin) and 
                obj != BasePlugin and
                not inspect.isabstract(obj)):
                plugin_classes.append(obj)
        
        return plugin_classes
    
    async def _register_plugin_class(self, plugin_class: Type[BasePlugin], 
                                   file_path: str, module_name: str) -> None:
        """Register a plugin class in the registry."""
        try:
            # Create temporary instance to get metadata
            temp_instance = plugin_class()
            metadata = temp_instance.metadata
            
            # Check if plugin with same name already exists
            if metadata.name in self._plugins:
                logger.warning(f"Plugin {metadata.name} already registered, skipping")
                return
            
            # Register the plugin
            registered_plugin = RegisteredPlugin(
                plugin_class=plugin_class,
                file_path=file_path,
                module_name=module_name,
                is_enabled=metadata.is_enabled,
                registration_time=asyncio.get_event_loop().time()
            )
            
            self._plugins[metadata.name] = registered_plugin
            self._plugins_by_type[metadata.plugin_type].append(metadata.name)
            
            logger.info(f"Registered plugin: {metadata.name} (type: {metadata.plugin_type})")
            
        except Exception as e:
            logger.error(f"Failed to register plugin class {plugin_class.__name__}: {e}")
    
    async def _initialize_plugins(self) -> None:
        """Initialize all enabled plugins."""
        resource_limits = await get_resource_limits()
        
        for name, registered_plugin in self._plugins.items():
            if registered_plugin.is_enabled:
                await self._initialize_plugin(name, resource_limits)
    
    async def _initialize_plugin(self, plugin_name: str, resource_limits: Dict[str, Any]) -> None:
        """Initialize a specific plugin."""
        try:
            registered_plugin = self._plugins[plugin_name]
            
            # Create plugin instance
            instance = registered_plugin.plugin_class()
            
            # Check resource requirements
            requirements = instance.resource_requirements
            if not self._check_resource_requirements(requirements, resource_limits):
                error_msg = f"Insufficient resources for plugin {plugin_name}"
                logger.warning(error_msg)
                registered_plugin.initialization_error = error_msg
                return
            
            # Initialize the plugin
            config = await self._get_plugin_config(plugin_name)
            success = await instance.initialize(config)
            
            if success:
                registered_plugin.instance = instance
                registered_plugin.initialization_error = None
                logger.info(f"Successfully initialized plugin: {plugin_name}")
            else:
                error_msg = f"Plugin {plugin_name} initialization returned False"
                logger.error(error_msg)
                registered_plugin.initialization_error = error_msg
                
        except Exception as e:
            error_msg = f"Failed to initialize plugin {plugin_name}: {str(e)}"
            logger.error(error_msg)
            self._plugins[plugin_name].initialization_error = error_msg
    
    def _check_resource_requirements(self, requirements, resource_limits: Dict[str, Any]) -> bool:
        """Check if resource requirements can be met."""
        # Check CPU requirements
        if requirements.min_cpu_cores > resource_limits.get("total_cpu_capacity", 0):
            return False
        
        # Check GPU requirements
        if requirements.requires_gpu and not resource_limits.get("gpu_available", False):
            return False
        
        # Check memory requirements
        available_memory_mb = resource_limits.get("local_memory_gb", 0) * 1024
        if requirements.min_memory_mb > available_memory_mb:
            return False
        
        return True
    
    async def _get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get configuration for a specific plugin."""
        # TODO: Implement plugin-specific configuration loading
        return {}
    
    async def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a plugin instance by name."""
        registered_plugin = self._plugins.get(plugin_name)
        if registered_plugin and registered_plugin.instance:
            return registered_plugin.instance
        return None
    
    async def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get all enabled plugins of a specific type."""
        plugins = []
        plugin_names = self._plugins_by_type.get(plugin_type, [])
        
        for name in plugin_names:
            registered_plugin = self._plugins.get(name)
            if (registered_plugin and 
                registered_plugin.is_enabled and 
                registered_plugin.instance):
                plugins.append(registered_plugin.instance)
        
        # Sort by execution priority
        plugins.sort(key=lambda p: self._get_priority_weight(p.metadata.execution_priority))
        return plugins
    
    def _get_priority_weight(self, priority: ExecutionPriority) -> int:
        """Get numeric weight for priority sorting."""
        weights = {
            ExecutionPriority.CRITICAL: 0,
            ExecutionPriority.HIGH: 1,
            ExecutionPriority.NORMAL: 2,
            ExecutionPriority.LOW: 3
        }
        return weights.get(priority, 2)
    
    async def execute_plugins(self, plugin_type: PluginType, data: Any, 
                            context: PluginExecutionContext) -> List[PluginExecutionResult]:
        """Execute all plugins of a specific type."""
        plugins = await self.get_plugins_by_type(plugin_type)
        results = []
        
        for plugin in plugins:
            try:
                result = await plugin.safe_execute(data, context)
                results.append(result)
                
                # If plugin succeeded and modified data, use the modified data for next plugin
                if result.success and result.data is not None:
                    data = result.data
                    
            except Exception as e:
                logger.error(f"Error executing plugin {plugin.metadata.name}: {e}")
                results.append(PluginExecutionResult(
                    success=False,
                    error_message=f"Plugin execution failed: {str(e)}"
                ))
        
        return results
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a specific plugin."""
        async with self._reload_lock:
            try:
                registered_plugin = self._plugins.get(plugin_name)
                if not registered_plugin:
                    logger.error(f"Plugin {plugin_name} not found")
                    return False
                
                # Cleanup existing instance
                if registered_plugin.instance:
                    await registered_plugin.instance.cleanup()
                
                # Reload module
                if registered_plugin.module_name in sys.modules:
                    importlib.reload(sys.modules[registered_plugin.module_name])
                
                # Re-register and initialize
                await self._load_plugin_file(Path(registered_plugin.file_path))
                resource_limits = await get_resource_limits()
                await self._initialize_plugin(plugin_name, resource_limits)
                
                registered_plugin.last_reload_time = asyncio.get_event_loop().time()
                logger.info(f"Successfully reloaded plugin: {plugin_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to reload plugin {plugin_name}: {e}")
                return False
    
    async def reload_all_plugins(self) -> int:
        """Reload all plugins. Returns number of successfully reloaded plugins."""
        success_count = 0
        plugin_names = list(self._plugins.keys())
        
        for plugin_name in plugin_names:
            if await self.reload_plugin(plugin_name):
                success_count += 1
        
        return success_count
    
    async def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        registered_plugin = self._plugins.get(plugin_name)
        if not registered_plugin:
            return False
        
        if not registered_plugin.is_enabled:
            registered_plugin.is_enabled = True
            if not registered_plugin.instance:
                resource_limits = await get_resource_limits()
                await self._initialize_plugin(plugin_name, resource_limits)
            return True
        
        return False
    
    async def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        registered_plugin = self._plugins.get(plugin_name)
        if not registered_plugin:
            return False
        
        if registered_plugin.is_enabled:
            registered_plugin.is_enabled = False
            if registered_plugin.instance:
                await registered_plugin.instance.cleanup()
                registered_plugin.instance = None
            return True
        
        return False
    
    async def get_plugin_status(self) -> Dict[str, Any]:
        """Get status of all plugins."""
        status = {
            "total_plugins": len(self._plugins),
            "enabled_plugins": sum(1 for p in self._plugins.values() if p.is_enabled),
            "initialized_plugins": sum(1 for p in self._plugins.values() if p.instance is not None),
            "plugins_by_type": {},
            "plugin_details": {}
        }
        
        # Count by type
        for plugin_type, plugin_names in self._plugins_by_type.items():
            status["plugins_by_type"][plugin_type.value] = len(plugin_names)
        
        # Individual plugin status
        for name, registered_plugin in self._plugins.items():
            plugin_status = {
                "enabled": registered_plugin.is_enabled,
                "initialized": registered_plugin.instance is not None,
                "file_path": registered_plugin.file_path,
                "registration_time": registered_plugin.registration_time,
                "last_reload_time": registered_plugin.last_reload_time,
                "initialization_error": registered_plugin.initialization_error
            }
            
            if registered_plugin.instance:
                plugin_status["health"] = await registered_plugin.instance.health_check()
            
            status["plugin_details"][name] = plugin_status
        
        return status
    
    async def cleanup(self) -> None:
        """Clean up all plugins."""
        logger.info("Cleaning up plugin registry...")
        
        for registered_plugin in self._plugins.values():
            if registered_plugin.instance:
                try:
                    await registered_plugin.instance.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up plugin {registered_plugin.instance.metadata.name}: {e}")
        
        self._plugins.clear()
        self._plugins_by_type.clear()
        self._loaded_modules.clear()


# Global plugin registry instance
plugin_registry = PluginRegistry()


async def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry instance."""
    return plugin_registry