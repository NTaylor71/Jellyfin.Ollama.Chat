"""
Plugin Loader - Dynamic plugin loading and management for the worker.

Provides lightweight plugin loading without importing heavy NLP dependencies
into the worker process.
"""

import asyncio
import logging
import importlib
import inspect
from typing import Dict, Any, Optional, List, Type, Set
from pathlib import Path
import httpx

from src.plugins.base import BasePlugin, PluginType, PluginExecutionContext, PluginExecutionResult
# ServiceRegistryPlugin will be imported dynamically
from src.worker.task_types import get_task_definition, validate_task_data, get_plugin_for_task
from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    Lightweight plugin loader for the worker.
    
    Features:
    - Dynamic plugin discovery and loading
    - Service-aware routing for heavy plugins
    - Health monitoring and fallback
    - No heavy NLP dependencies in worker
    """
    
    def __init__(self):
        self.loaded_plugins: Dict[str, BasePlugin] = {}
        self.plugin_service_mapping: Dict[str, str] = {}
        self.service_registry: Optional[ServiceRegistryPlugin] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.settings = get_settings()
        
        # Plugin categories (dynamically discovered)
        self.local_plugins: Set[str] = set()
        self.service_plugins: Set[str] = set()
        
        # Known local plugins (lightweight, always load locally)
        self.known_local_plugins = {
            "ServiceRegistryPlugin",
            "ServiceHealthMonitorPlugin"
        }
    
    async def initialize(self) -> bool:
        """Initialize the plugin loader."""
        try:
            logger.info("Initializing Plugin Loader...")
            
            # Initialize HTTP client for service communication
            self.http_client = httpx.AsyncClient(timeout=30.0)
            
            # Load service registry plugin first
            await self._load_service_registry()
            
            # Discover and categorize available plugins
            await self._discover_plugins()
            
            # Load lightweight local plugins only
            await self._load_local_plugins()
            
            logger.info(f"✅ Plugin Loader initialized. {len(self.loaded_plugins)} local plugins loaded.")
            return True
            
        except Exception as e:
            logger.error(f"❌ Plugin Loader initialization failed: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup plugin loader resources."""
        logger.info("Cleaning up Plugin Loader...")
        
        # Cleanup loaded plugins
        for plugin in self.loaded_plugins.values():
            try:
                await plugin.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up plugin {plugin.metadata.name}: {e}")
        
        # Cleanup service registry
        if self.service_registry:
            await self.service_registry.cleanup()
        
        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()
        
        logger.info("✅ Plugin Loader cleanup complete")
    
    async def _load_service_registry(self):
        """Load the service registry plugin."""
        try:
            module = importlib.import_module("src.plugins.system.service_registry_plugin")
            plugin_class = getattr(module, "ServiceRegistryPlugin")
            
            self.service_registry = plugin_class()
            if await self.service_registry.initialize_with_config():
                logger.info("✅ Service Registry Plugin loaded")
            else:
                logger.error("❌ Service Registry Plugin failed to initialize")
                
        except Exception as e:
            logger.error(f"❌ Failed to load Service Registry Plugin: {e}")
    
    async def _discover_plugins(self):
        """Discover available plugins and categorize them."""
        logger.info("Discovering available plugins...")
        
        # Add known local plugins
        self.local_plugins.update(self.known_local_plugins)
        
        # Discover HTTP-only plugins dynamically by scanning enrichment directory
        await self._discover_enrichment_plugins()
        
        # Map discovered service plugins to services
        for plugin_name in self.service_plugins:
            self.plugin_service_mapping[plugin_name] = self._get_service_for_plugin(plugin_name)
            logger.debug(f"Mapped service plugin: {plugin_name} -> {self.plugin_service_mapping[plugin_name]}")
        
        logger.info(f"Plugin discovery complete. Found {len(self.local_plugins)} local plugins, {len(self.service_plugins)} service plugins.")
    
    async def _discover_enrichment_plugins(self):
        """Dynamically discover HTTP-only plugins in enrichment directory."""
        try:
            from pathlib import Path
            import importlib.util
            
            enrichment_dir = Path("src/plugins/enrichment")
            if not enrichment_dir.exists():
                logger.warning("Field enrichment plugins directory not found")
                return
            
            for plugin_file in enrichment_dir.glob("*_plugin.py"):
                try:
                    # Extract plugin name from filename and convert to proper class name
                    module_name = plugin_file.stem
                    # Handle special cases like LLM, SUTime, HeidelTime
                    words = module_name.split('_')
                    class_name_parts = []
                    for word in words:
                        if word.lower() == 'llm':
                            class_name_parts.append('LLM')
                        elif word.lower() == 'sutime':
                            class_name_parts.append('SUTime')
                        elif word.lower() == 'heideltime':
                            class_name_parts.append('HeidelTime')
                        elif word.lower() == 'conceptnet':
                            class_name_parts.append('ConceptNet')
                        else:
                            class_name_parts.append(word.capitalize())
                    plugin_class_name = ''.join(class_name_parts)
                    
                    # Import module and get class
                    spec = importlib.util.spec_from_file_location(
                        f"src.plugins.enrichment.{module_name}", 
                        plugin_file
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Get plugin class
                    if hasattr(module, plugin_class_name):
                        plugin_class = getattr(module, plugin_class_name)
                        # Check if it's an HTTP-based plugin (inherits from HTTPBasePlugin)
                        if hasattr(plugin_class, '__bases__') and any(
                            'HTTPBasePlugin' in str(base) for base in plugin_class.__bases__
                        ):
                            self.service_plugins.add(plugin_class_name)
                            logger.debug(f"Discovered HTTP-only plugin: {plugin_class_name}")
                        
                except Exception as e:
                    logger.warning(f"Failed to inspect plugin file {plugin_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to discover field enrichment plugins: {e}")
            # Fallback to prevent system failure
            logger.info("Using fallback plugin discovery...")
            self._fallback_plugin_discovery()
    
    def _fallback_plugin_discovery(self):
        """Fallback plugin discovery if dynamic discovery fails."""
        logger.warning("Using fallback static plugin discovery")
        
        # Add known HTTP-only plugins as fallback
        fallback_plugins = {
            "ConceptNetKeywordPlugin",
            "LLMKeywordPlugin", 
            "GensimSimilarityPlugin",
            "SpacyTemporalPlugin",
            "HeidelTimeTemporalPlugin",
            "SUTimeTemporalPlugin",
            "LLMQuestionAnswerPlugin",
            "LLMTemporalIntelligencePlugin",
            "MergeKeywordsPlugin"
        }
        self.service_plugins.update(fallback_plugins)
    
    async def _load_local_plugins(self):
        """Load only lightweight local plugins."""
        for plugin_name in self.local_plugins:
            try:
                await self._load_plugin(plugin_name)
            except Exception as e:
                logger.warning(f"Failed to load local plugin {plugin_name}: {e}")
    
    async def _load_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Load a specific plugin."""
        if plugin_name in self.loaded_plugins:
            return self.loaded_plugins[plugin_name]
        
        try:
            # Map plugin name to module
            module_map = {
                "ServiceRegistryPlugin": "src.plugins.system.service_registry_plugin",
                "ServiceHealthMonitorPlugin": "src.plugins.system.service_health_monitor_plugin",
                "CacheManagerPlugin": "src.plugins.system.cache_manager_plugin",
                "MetricsPlugin": "src.plugins.system.metrics_plugin"
            }
            
            module_name = module_map.get(plugin_name)
            if not module_name:
                logger.warning(f"No module mapping for plugin: {plugin_name}")
                return None
            
            # Import and instantiate
            module = importlib.import_module(module_name)
            plugin_class = getattr(module, plugin_name)
            plugin = plugin_class()
            
            # Initialize plugin
            if await plugin.initialize_with_config():
                self.loaded_plugins[plugin_name] = plugin
                logger.info(f"✅ Loaded plugin: {plugin_name}")
                return plugin
            else:
                logger.error(f"❌ Failed to initialize plugin: {plugin_name}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load plugin {plugin_name}: {e}")
            return None
    
    def _get_service_for_plugin(self, plugin_name: str) -> str:
        """Determine which service should handle a plugin based on plugin name patterns."""
        # Use plugin name patterns to determine service routing
        # This avoids importing heavy plugins just to determine routing
        
        # LLM service patterns
        if any(pattern in plugin_name.lower() for pattern in ['llm', 'language_model', 'gpt']):
            return "llm_provider"
        
        # NLP service patterns (everything else including merge operations)
        # ConceptNet, Gensim, SpaCy, HeidelTime, SUTime, Merge plugins
        return "nlp_provider"
    
    async def execute_plugin(
        self, 
        plugin_name: str, 
        plugin_type: str,
        data: Any, 
        context: PluginExecutionContext
    ) -> PluginExecutionResult:
        """Execute a plugin either locally or via service."""
        
        # Check if it's a local plugin
        if plugin_name in self.loaded_plugins:
            plugin = self.loaded_plugins[plugin_name]
            return await plugin.safe_execute(data, context)
        
        # Check if it's a service plugin
        elif plugin_name in self.plugin_service_mapping:
            return await self._execute_via_service(plugin_name, plugin_type, data, context)
        
        # Try to load as local plugin
        else:
            plugin = await self._load_plugin(plugin_name)
            if plugin:
                return await plugin.safe_execute(data, context)
            else:
                return PluginExecutionResult(
                    success=False,
                    error_message=f"Plugin {plugin_name} not found or failed to load"
                )
    
    async def _execute_via_service(
        self,
        plugin_name: str,
        plugin_type: str, 
        data: Any,
        context: PluginExecutionContext
    ) -> PluginExecutionResult:
        """Execute plugin via appropriate service."""
        try:
            # Use plugin router service via config
            router_url = self.settings.router_service_url
            
            # Prepare request
            request_data = {
                "plugin_name": plugin_name,
                "plugin_type": plugin_type,
                "data": data if isinstance(data, dict) else {"input": data},
                "context": {
                    "user_id": context.user_id,
                    "session_id": context.session_id,
                    "request_id": context.request_id,
                    "execution_timeout": context.execution_timeout,
                    "priority": context.priority.value if hasattr(context.priority, 'value') else str(context.priority)
                }
            }
            
            # Execute via router
            response = await self.http_client.post(
                f"{router_url}/plugins/execute",
                json=request_data,
                timeout=context.execution_timeout
            )
            
            if response.status_code == 200:
                result_data = response.json()
                
                return PluginExecutionResult(
                    success=result_data.get("success", False),
                    data=result_data.get("result"),
                    error_message=result_data.get("error_message"),
                    execution_time_ms=result_data.get("execution_time_ms", 0),
                    metadata={
                        "service_used": result_data.get("service_used"),
                        "via_router": True,
                        **result_data.get("metadata", {})
                    }
                )
            else:
                return PluginExecutionResult(
                    success=False,
                    error_message=f"Service request failed: HTTP {response.status_code}"
                )
                
        except Exception as e:
            logger.error(f"Service execution failed for {plugin_name}: {e}")
            return PluginExecutionResult(
                success=False,
                error_message=f"Service execution error: {str(e)}"
            )
    
    async def get_plugin_health(self, plugin_name: str) -> Dict[str, Any]:
        """Get health status of a plugin."""
        if plugin_name in self.loaded_plugins:
            plugin = self.loaded_plugins[plugin_name]
            return await plugin.health_check()
        elif plugin_name in self.plugin_service_mapping:
            return await self._get_service_plugin_health(plugin_name)
        else:
            return {
                "status": "not_found",
                "error": f"Plugin {plugin_name} not loaded or configured"
            }
    
    async def _get_service_plugin_health(self, plugin_name: str) -> Dict[str, Any]:
        """Get health status of a service-based plugin."""
        try:
            service_name = self.plugin_service_mapping[plugin_name]
            
            if not self.service_registry:
                return {"status": "error", "error": "Service registry not available"}
            
            # Get service info from registry
            service = self.service_registry.get_service_by_capability("concept_expansion")
            if not service:
                return {"status": "error", "error": f"Service {service_name} not available"}
            
            # Check service health
            response = await self.http_client.get(f"{service.url}/health", timeout=5.0)
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "service": service_name,
                    "service_url": service.url,
                    "via_service": True
                }
            else:
                return {
                    "status": "unhealthy",
                    "service": service_name,
                    "http_status": response.status_code
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def list_available_plugins(self) -> Dict[str, Any]:
        """List all available plugins."""
        return {
            "local_plugins": {
                name: {
                    "loaded": True,
                    "metadata": plugin.metadata.dict() if hasattr(plugin.metadata, 'dict') else str(plugin.metadata)
                }
                for name, plugin in self.loaded_plugins.items()
            },
            "service_plugins": {
                name: {
                    "loaded": False,
                    "service": service,
                    "via_service": True
                }
                for name, service in self.plugin_service_mapping.items()
            }
        }
    
    async def route_task_to_plugin(self, task_type: str, task_data: Dict[str, Any]) -> PluginExecutionResult:
        """Route a task type to the appropriate plugin using task definitions."""
        
        # Validate task data first
        validation_data = task_data.get("data", {})
        is_valid, error_message = validate_task_data(task_type, validation_data)
        if not is_valid:
            return PluginExecutionResult(
                success=False,
                error_message=f"Task validation failed: {error_message}"
            )
        
        # Extract plugin info from nested task data structure
        inner_data = task_data.get("data", {})
        plugin_name = inner_data.get("plugin_name")
        plugin_type = inner_data.get("plugin_type")
        
        if not plugin_name:
            return PluginExecutionResult(
                success=False,
                error_message=f"No plugin_name provided in task data"
            )
        
        if not plugin_type:
            return PluginExecutionResult(
                success=False,
                error_message=f"No plugin_type provided in task data"
            )
        
        # Get task definition for timeout and other settings
        task_definition = get_task_definition(task_type)
        default_timeout = task_definition.execution_timeout if task_definition else 30.0
        
        # Create execution context from inner data
        context = PluginExecutionContext(
            user_id=inner_data.get("user_id"),
            session_id=inner_data.get("session_id"),
            request_id=inner_data.get("task_id"),
            execution_timeout=inner_data.get("timeout", default_timeout),
            metadata={
                "task_type": task_type,
                "requires_service": task_definition.requires_service if task_definition else False,
                "service_type": task_definition.service_type if task_definition else None
            }
        )
        
        # Execute plugin with the actual plugin data
        return await self.execute_plugin(
            plugin_name=plugin_name,
            plugin_type=plugin_type,  # Use actual plugin type, not task type
            data=inner_data.get("data", {}),  # Get the actual plugin data
            context=context
        )