"""
Plugin Loader - Direct plugin instantiation for the worker.

All plugins inherit from HTTPBasePlugin and handle their own routing.
"""

import logging
import importlib
from typing import Dict, Any, Set, Type, Optional
from pathlib import Path

from src.plugins.base import PluginExecutionContext, PluginExecutionResult
from src.plugins.http_base import HTTPBasePlugin
from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class PluginLoader:
    """Direct plugin loader that instantiates plugins without routing layers."""
    
    def __init__(self):
        self.settings = get_settings()
        self.plugins: Dict[str, HTTPBasePlugin] = {}
        self.plugin_classes: Dict[str, Type[HTTPBasePlugin]] = {}
        self.discovered_plugins: Set[str] = set()
    
    async def initialize(self) -> bool:
        """Initialize the plugin loader with direct plugin instantiation."""
        try:
            logger.info("Initializing Plugin Loader...")
            

            self._discover_plugin_classes()
            

            await self._instantiate_plugins()
            
            logger.info(f"✅ Plugin Loader initialized. {len(self.plugins)} plugins available.")
            return True
            
        except Exception as e:
            logger.error(f"❌ Plugin Loader initialization failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up plugins...")
        

        for plugin_name, plugin in self.plugins.items():
            try:
                await plugin.cleanup()
                logger.debug(f"Cleaned up plugin: {plugin_name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup plugin {plugin_name}: {e}")
        
        self.plugins.clear()
        self.plugin_classes.clear()
        logger.info("✅ Plugin Loader cleanup complete")
    
    def _discover_plugin_classes(self):
        """Discover all available plugin classes."""
        logger.info("Discovering plugin classes...")
        

        enrichment_dir = Path("src/plugins/enrichment")
        if not enrichment_dir.exists():
            logger.error(f"Plugin directory not found: {enrichment_dir}")
            return
            
        for plugin_file in enrichment_dir.glob("*_plugin.py"):
            class_name = self._extract_class_name_from_file(plugin_file)
            if class_name:
                self.discovered_plugins.add(class_name)

                plugin_class = self._load_plugin_class(plugin_file, class_name)
                if plugin_class:
                    self.plugin_classes[class_name] = plugin_class
        
        logger.info(f"Discovered {len(self.plugin_classes)} plugin classes")
    
    async def _instantiate_plugins(self):
        """Instantiate and initialize all discovered plugins."""
        logger.info("Instantiating plugins...")
        
        for class_name, plugin_class in self.plugin_classes.items():
            try:
                
                plugin_instance = plugin_class()
                
                
                if await plugin_instance.initialize({}):
                    self.plugins[class_name] = plugin_instance
                    logger.debug(f"Initialized plugin: {class_name}")
                else:
                    logger.warning(f"Failed to initialize plugin: {class_name}")
                    
            except Exception as e:
                logger.error(f"Failed to instantiate plugin {class_name}: {e}")
        
        logger.info(f"Successfully instantiated {len(self.plugins)} plugins")
    
    def _extract_class_name_from_file(self, plugin_file: Path) -> Optional[str]:
        """Extract actual class name by reading the plugin file."""
        try:
            with open(plugin_file, 'r') as f:
                content = f.read()

                import re
                match = re.search(r'class\s+(\w+Plugin)\s*\(HTTPBasePlugin\)', content)
                if match:
                    return match.group(1)
                else:
                    logger.warning(f"No HTTPBasePlugin class found in {plugin_file}")
                    return None
        except Exception as e:
            logger.warning(f"Failed to read class name from {plugin_file}: {e}")
            return None
    
    def _load_plugin_class(self, plugin_file: Path, class_name: str) -> Optional[Type[HTTPBasePlugin]]:
        """Load a plugin class from file."""
        try:

            module_path = str(plugin_file).replace('/', '.').replace('\\', '.').replace('.py', '')
            

            spec = importlib.util.spec_from_file_location(module_path, plugin_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            
            plugin_class = getattr(module, class_name)
            

            if issubclass(plugin_class, HTTPBasePlugin):
                logger.debug(f"Loaded plugin class: {class_name}")
                return plugin_class
            else:
                logger.warning(f"Class {class_name} is not a HTTPBasePlugin subclass")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load plugin class {class_name} from {plugin_file}: {e}")
            return None
    
    async def execute_plugin(
        self, 
        plugin_name: str, 
        plugin_type: str,
        data: Any, 
        context: PluginExecutionContext
    ) -> PluginExecutionResult:
        """Execute plugin directly without routing layers."""
        logger.info(f"🔍 EXECUTE_PLUGIN: Called with plugin_name={plugin_name}")
        logger.info(f"🔍 EXECUTE_PLUGIN: data type={type(data)}, data keys={list(data.keys()) if isinstance(data, dict) else 'not dict'}")
        
        if plugin_name not in self.plugins:
            logger.error(f"❌ EXECUTE_PLUGIN: Plugin {plugin_name} not found in loaded plugins")
            return PluginExecutionResult(
                success=False,
                error_message=f"Plugin {plugin_name} not found in loaded plugins"
            )
        
        try:
            
            plugin = self.plugins[plugin_name]
            logger.info(f"🔍 EXECUTE_PLUGIN: Found plugin instance: {plugin.__class__.__name__}")
            

            logger.info(f"🔍 EXECUTE_PLUGIN: About to call plugin.execute() with data type {type(data)}")
            result = await plugin.execute(data, context)
            
            logger.info(f"🔍 EXECUTE_PLUGIN: plugin.execute() returned: success={result.success}, exec_time={result.execution_time_ms}ms")
            return result
                
        except Exception as e:
            logger.error(f"Plugin execution error for {plugin_name}: {e}")
            return PluginExecutionResult(
                success=False,
                error_message=f"Plugin execution error: {str(e)}"
            )
    
    async def list_available_plugins(self) -> Dict[str, Any]:
        """List all available plugins."""
        plugins_info = {}
        for name, plugin in self.plugins.items():
            plugins_info[name] = {
                "class_name": plugin.__class__.__name__,
                "initialized": True,
                "service_url": await plugin.get_plugin_service_url()
            }
        
        return {
            "plugins": plugins_info,
            "total_count": len(self.plugins)
        }
    
    async def route_task_to_plugin(self, task_type: str, task_data: Dict[str, Any]) -> PluginExecutionResult:
        """Route task directly to plugin without intermediate layers."""
        logger.info(f"🔍 PLUGIN_LOADER: route_task_to_plugin called with task_type={task_type}")
        logger.info(f"🔍 PLUGIN_LOADER: task_data keys: {list(task_data.keys())}")
        

        inner_data = task_data.get("data", {})
        plugin_name = inner_data.get("plugin_name")
        plugin_type = inner_data.get("plugin_type")
        
        logger.info(f"🔍 PLUGIN_LOADER: extracted plugin_name={plugin_name}, plugin_type={plugin_type}")
        logger.info(f"🔍 PLUGIN_LOADER: available plugins: {list(self.plugins.keys())}")
        
        if not plugin_name or not plugin_type:
            logger.error(f"❌ PLUGIN_LOADER: Missing plugin_name or plugin_type")
            return PluginExecutionResult(
                success=False,
                error_message="Missing plugin_name or plugin_type"
            )
        
        
        context = PluginExecutionContext(
            user_id=inner_data.get("user_id"),
            session_id=inner_data.get("session_id"),
            request_id=inner_data.get("task_id"),
            execution_timeout=inner_data.get("timeout", 30.0)
        )
        

        logger.info(f"🔍 PLUGIN_LOADER: About to call execute_plugin with data keys: {list(inner_data.get('data', {}).keys())}")
        result = await self.execute_plugin(
            plugin_name=plugin_name,
            plugin_type=plugin_type,
            data=inner_data.get("data", {}),
            context=context
        )
        logger.info(f"🔍 PLUGIN_LOADER: execute_plugin returned success={result.success}")
        return result