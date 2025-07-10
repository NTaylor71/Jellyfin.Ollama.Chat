"""
Plugin Base Classes
Provides abstract base classes for all plugin types with hardware-aware resource management.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union, Callable, Type
from enum import Enum
import time
from pydantic import BaseModel, Field

from src.shared.hardware_config import get_resource_limits, ResourceType
from src.shared.media_types import MediaType

logger = logging.getLogger(__name__)


class PluginType(str, Enum):
    """Types of plugins supported by the system."""
    QUERY_EMBELLISHER = "query_embellisher"
    EMBED_DATA_EMBELLISHER = "embed_data_embellisher"
    FAISS_CRUD = "faiss_crud"
    GENERAL = "general"


class ExecutionPriority(str, Enum):
    """Plugin execution priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PluginResourceRequirements:
    """Defines resource requirements for a plugin."""
    min_cpu_cores: float = 1.0
    preferred_cpu_cores: float = 2.0
    min_memory_mb: float = 100.0
    preferred_memory_mb: float = 500.0
    requires_gpu: bool = False
    min_gpu_memory_mb: float = 0.0
    preferred_gpu_memory_mb: float = 0.0
    max_execution_time_seconds: float = 30.0
    can_use_distributed_resources: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_cpu_cores": self.min_cpu_cores,
            "preferred_cpu_cores": self.preferred_cpu_cores,
            "min_memory_mb": self.min_memory_mb,
            "preferred_memory_mb": self.preferred_memory_mb,
            "requires_gpu": self.requires_gpu,
            "min_gpu_memory_mb": self.min_gpu_memory_mb,
            "preferred_gpu_memory_mb": self.preferred_gpu_memory_mb,
            "max_execution_time_seconds": self.max_execution_time_seconds,
            "can_use_distributed_resources": self.can_use_distributed_resources
        }


@dataclass
class PluginExecutionContext:
    """Context information passed to plugins during execution."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    available_resources: Optional[Dict[str, Any]] = None
    execution_timeout: float = 30.0
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PluginExecutionResult:
    """Result of plugin execution."""
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    resources_used: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.resources_used is None:
            self.resources_used = {}
        if self.metadata is None:
            self.metadata = {}


class PluginMetadata(BaseModel):
    """Metadata for plugin registration."""
    name: str = Field(..., min_length=1, max_length=100)
    version: str = Field(..., pattern=r'^\d+\.\d+\.\d+$')
    description: str = Field(..., min_length=1, max_length=500)
    author: str = Field(..., min_length=1, max_length=100)
    plugin_type: PluginType
    tags: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    is_enabled: bool = Field(default=True)
    execution_priority: ExecutionPriority = Field(default=ExecutionPriority.NORMAL)


class BasePlugin(ABC):
    """Abstract base class for all plugins."""
    
    def __init__(self):
        self._metadata: Optional[PluginMetadata] = None
        self._resource_requirements: Optional[PluginResourceRequirements] = None
        self._is_initialized = False
        self._initialization_error: Optional[str] = None
        self._performance_metrics: Dict[str, Any] = {}
        self._logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
        self._config_manager: Optional[Any] = None  # Will be set during initialization
        self._current_config: Optional[Any] = None
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Plugin metadata for registration."""
        pass
    
    @property
    @abstractmethod
    def resource_requirements(self) -> PluginResourceRequirements:
        """Resource requirements for this plugin."""
        pass
    
    @property
    def config_class(self) -> Optional[Type]:
        """Configuration class for this plugin. Override in subclasses."""
        return None
    
    def get_config(self) -> Optional[Any]:
        """Get current plugin configuration."""
        return self._current_config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update plugin configuration at runtime."""
        if self._config_manager:
            return self._config_manager.update_config(updates)
        return False
    
    async def initialize_with_config(self, config_manager: Optional[Any] = None) -> bool:
        """Initialize the plugin with configuration manager."""
        try:
            # Set up configuration if config class is provided
            if self.config_class and config_manager:
                self._config_manager = config_manager
                self._current_config = config_manager.get_config()
                self._logger.info(f"Loaded configuration for plugin {self.metadata.name}")
            
            # Call plugin-specific initialization
            config_dict = self._current_config.model_dump() if self._current_config else {}
            success = await self.initialize(config_dict)
            
            if success:
                self._is_initialized = True
                self._initialization_error = None
            else:
                self._initialization_error = "Plugin initialization returned False"
            
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to initialize plugin {self.metadata.name}: {e}")
            self._initialization_error = str(e)
            return False
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration. Override in subclasses."""
        pass
    
    @abstractmethod
    async def execute(self, data: Any, context: PluginExecutionContext) -> PluginExecutionResult:
        """Execute the plugin with given data and context."""
        pass
    
    async def cleanup(self) -> None:
        """Clean up plugin resources (optional override)."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin health status."""
        health_info = {
            "status": "healthy" if self._is_initialized else "unhealthy",
            "initialized": self._is_initialized,
            "error": self._initialization_error,
            "metrics": self._performance_metrics,
            "resource_usage": await self._get_resource_usage(),
            "last_health_check": time.time()
        }
        
        # Add configuration information
        if self._config_manager:
            health_info["config"] = {
                "has_config": True,
                "config_summary": self._config_manager.get_config_summary()
            }
        else:
            health_info["config"] = {
                "has_config": False
            }
        
        return health_info
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Alias for health_check() method for compatibility."""
        return await self.health_check()
    
    async def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage for this plugin."""
        try:
            import psutil
            import os
            
            # Get process info
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                "memory_used_mb": round(memory_info.rss / 1024 / 1024, 2),
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "resource_requirements": self.resource_requirements.to_dict()
            }
        except Exception as e:
            self._logger.warning(f"Could not get resource usage: {e}")
            return {
                "memory_used_mb": 0,
                "cpu_percent": 0,
                "num_threads": 0,
                "open_files": 0,
                "resource_requirements": self.resource_requirements.to_dict(),
                "error": str(e)
            }
    
    def _update_performance_metrics(self, execution_time_ms: float, success: bool) -> None:
        """Update internal performance metrics."""
        if "executions" not in self._performance_metrics:
            self._performance_metrics["executions"] = 0
            self._performance_metrics["successful_executions"] = 0
            self._performance_metrics["failed_executions"] = 0
            self._performance_metrics["total_execution_time_ms"] = 0.0
            self._performance_metrics["avg_execution_time_ms"] = 0.0
        
        self._performance_metrics["executions"] += 1
        self._performance_metrics["total_execution_time_ms"] += execution_time_ms
        
        if success:
            self._performance_metrics["successful_executions"] += 1
        else:
            self._performance_metrics["failed_executions"] += 1
        
        self._performance_metrics["avg_execution_time_ms"] = (
            self._performance_metrics["total_execution_time_ms"] / 
            self._performance_metrics["executions"]
        )
        
        # Record metrics to Prometheus (if available)
        try:
            # Import here to avoid circular dependency
            from src.api.plugin_metrics import get_plugin_metrics
            metrics_collector = get_plugin_metrics()
            metrics_collector.record_plugin_execution(
                plugin_name=self.metadata.name,
                plugin_type=self.metadata.plugin_type.value,
                execution_time_ms=execution_time_ms,
                success=success
            )
        except ImportError:
            # Prometheus metrics not available (e.g., in worker service)
            pass
        except Exception as e:
            self._logger.warning(f"Failed to record Prometheus metrics: {e}")
    
    async def _check_resource_availability(self, context: PluginExecutionContext) -> bool:
        """Check if required resources are available."""
        try:
            if context.available_resources is None:
                context.available_resources = await get_resource_limits()
            
            requirements = self.resource_requirements
            available = context.available_resources
            
            # Check CPU requirements
            if requirements.min_cpu_cores > available.get("total_cpu_capacity", 0):
                self._logger.warning(f"Insufficient CPU cores: need {requirements.min_cpu_cores}, have {available.get('total_cpu_capacity', 0)}")
                return False
            
            # Check GPU requirements
            if requirements.requires_gpu and not available.get("gpu_available", False):
                self._logger.warning("GPU required but not available")
                return False
            
            if requirements.min_gpu_memory_mb > 0:
                available_gpu_memory = available.get("total_gpu_capacity", 0) * 1024  # Convert GB to MB
                if requirements.min_gpu_memory_mb > available_gpu_memory:
                    self._logger.warning(f"Insufficient GPU memory: need {requirements.min_gpu_memory_mb}MB, have {available_gpu_memory}MB")
                    return False
            
            # Check memory requirements
            available_memory_mb = available.get("local_memory_gb", 0) * 1024  # Convert GB to MB
            if requirements.min_memory_mb > available_memory_mb:
                self._logger.warning(f"Insufficient memory: need {requirements.min_memory_mb}MB, have {available_memory_mb}MB")
                return False
            
            return True
            
        except Exception as e:
            self._logger.error(f"Error checking resource availability: {e}")
            return False
    
    async def safe_execute(self, data: Any, context: PluginExecutionContext) -> PluginExecutionResult:
        """Execute plugin with safety checks and performance tracking."""
        start_time = time.time()
        
        try:
            # Check if plugin is initialized
            if not self._is_initialized:
                return PluginExecutionResult(
                    success=False,
                    error_message=f"Plugin {self.metadata.name} is not initialized"
                )
            
            # Check resource availability
            if not await self._check_resource_availability(context):
                return PluginExecutionResult(
                    success=False,
                    error_message="Insufficient resources available for plugin execution"
                )
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    self.execute(data, context),
                    timeout=context.execution_timeout
                )
                
                execution_time_ms = (time.time() - start_time) * 1000
                result.execution_time_ms = execution_time_ms
                
                # Update performance metrics
                self._update_performance_metrics(execution_time_ms, result.success)
                
                return result
                
            except asyncio.TimeoutError:
                execution_time_ms = (time.time() - start_time) * 1000
                self._update_performance_metrics(execution_time_ms, False)
                
                return PluginExecutionResult(
                    success=False,
                    error_message=f"Plugin execution timed out after {context.execution_timeout}s",
                    execution_time_ms=execution_time_ms
                )
        
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self._update_performance_metrics(execution_time_ms, False)
            
            self._logger.error(f"Error executing plugin {self.metadata.name}: {e}")
            return PluginExecutionResult(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )


class QueryEmbellisherPlugin(BasePlugin):
    """Base class for query embellishment plugins."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.__class__.__name__,
            version="1.0.0",
            description="Query embellishment plugin",
            author="System",
            plugin_type=PluginType.QUERY_EMBELLISHER
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        return PluginResourceRequirements()
    
    @abstractmethod
    async def embellish_query(self, query: str, context: PluginExecutionContext) -> str:
        """Embellish the input query and return enhanced version."""
        pass
    
    async def execute(self, data: Any, context: PluginExecutionContext) -> PluginExecutionResult:
        """Execute query embellishment."""
        try:
            if not isinstance(data, str):
                return PluginExecutionResult(
                    success=False,
                    error_message="Query embellisher expects string input"
                )
            
            enhanced_query = await self.embellish_query(data, context)
            
            return PluginExecutionResult(
                success=True,
                data=enhanced_query,
                metadata={"original_query": data, "enhanced_query": enhanced_query}
            )
            
        except Exception as e:
            return PluginExecutionResult(
                success=False,
                error_message=f"Query embellishment failed: {str(e)}"
            )


class EmbedDataEmbellisherPlugin(BasePlugin):
    """Base class for embed data embellishment plugins."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.__class__.__name__,
            version="1.0.0",
            description="Embed data embellishment plugin",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        return PluginResourceRequirements()
    
    @abstractmethod
    async def embellish_embed_data(self, data: Dict[str, Any], context: PluginExecutionContext) -> Dict[str, Any]:
        """Embellish data before embedding."""
        pass
    
    async def execute(self, data: Any, context: PluginExecutionContext) -> PluginExecutionResult:
        """Execute embed data embellishment with proper data merging."""
        try:
            if not isinstance(data, dict):
                return PluginExecutionResult(
                    success=False,
                    error_message="Embed data embellisher expects dict input"
                )
            
            # Get plugin-specific enhancements
            plugin_enhancements = await self.embellish_embed_data(data, context)
            
            # Create standardized enhanced data structure
            enhanced_data = data.copy()  # Preserve original data
            
            # Initialize plugin tracking structures if not present
            if 'plugin_enhancements' not in enhanced_data:
                enhanced_data['plugin_enhancements'] = {}
            
            if 'enhancement_metadata' not in enhanced_data:
                enhanced_data['enhancement_metadata'] = {
                    'processing_plugins': [],
                    'processing_timestamps': {},
                    'plugin_versions': {}
                }
            
            # Add this plugin's data with namespacing
            plugin_name = self.metadata.name
            enhanced_data['plugin_enhancements'][plugin_name] = plugin_enhancements
            
            # Update tracking metadata
            if plugin_name not in enhanced_data['enhancement_metadata']['processing_plugins']:
                enhanced_data['enhancement_metadata']['processing_plugins'].append(plugin_name)
            
            enhanced_data['enhancement_metadata']['processing_timestamps'][plugin_name] = asyncio.get_event_loop().time()
            enhanced_data['enhancement_metadata']['plugin_versions'][plugin_name] = self.metadata.version
            
            # Merge top-level fields from plugin (for backward compatibility)
            for key, value in plugin_enhancements.items():
                if key not in ['plugin_enhancements', 'enhancement_metadata']:
                    # Don't overwrite existing keys, use plugin-specific naming
                    if key in enhanced_data and key not in data:
                        # This key was added by a previous plugin, namespace it
                        enhanced_data[f"{plugin_name}_{key}"] = value
                    else:
                        enhanced_data[key] = value
            
            return PluginExecutionResult(
                success=True,
                data=enhanced_data,
                metadata={
                    "plugin_name": plugin_name,
                    "original_data_keys": list(data.keys()), 
                    "plugin_enhancement_keys": list(plugin_enhancements.keys()),
                    "final_data_keys": list(enhanced_data.keys())
                }
            )
            
        except Exception as e:
            return PluginExecutionResult(
                success=False,
                error_message=f"Embed data embellishment failed: {str(e)}"
            )


class FAISSCRUDPlugin(BasePlugin):
    """Base class for FAISS CRUD operation plugins."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.__class__.__name__,
            version="1.0.0",
            description="FAISS CRUD plugin",
            author="System",
            plugin_type=PluginType.FAISS_CRUD
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        return PluginResourceRequirements()
    
    @abstractmethod
    async def handle_faiss_operation(self, operation: str, data: Dict[str, Any], context: PluginExecutionContext) -> Any:
        """Handle FAISS CRUD operations."""
        pass
    
    async def execute(self, data: Any, context: PluginExecutionContext) -> PluginExecutionResult:
        """Execute FAISS CRUD operation."""
        try:
            if not isinstance(data, dict) or "operation" not in data:
                return PluginExecutionResult(
                    success=False,
                    error_message="FAISS CRUD plugin expects dict with 'operation' key"
                )
            
            operation = data["operation"]
            result = await self.handle_faiss_operation(operation, data, context)
            
            return PluginExecutionResult(
                success=True,
                data=result,
                metadata={"operation": operation}
            )
            
        except Exception as e:
            return PluginExecutionResult(
                success=False,
                error_message=f"FAISS operation failed: {str(e)}"
            )


class MediaTypePlugin(BasePlugin):
    """Plugin that adapts behavior based on media type"""
    
    @abstractmethod
    def get_supported_media_types(self) -> List[MediaType]:
        """Get list of media types this plugin supports."""
        pass
    
    @abstractmethod
    def analyze_for_media_type(self, text: str, media_type: MediaType) -> Dict[str, Any]:
        """Analyze text for a specific media type."""
        pass


def plugin_decorator(
    name: str,
    version: str,
    description: str,
    author: str,
    plugin_type: PluginType,
    resource_requirements: Optional[PluginResourceRequirements] = None,
    execution_priority: ExecutionPriority = ExecutionPriority.NORMAL,
    tags: Optional[List[str]] = None,
    dependencies: Optional[List[str]] = None
) -> Callable:
    """Decorator to register plugin metadata."""
    def decorator(cls):
        # Store metadata as class attributes
        cls._plugin_name = name
        cls._plugin_version = version
        cls._plugin_description = description
        cls._plugin_author = author
        cls._plugin_type = plugin_type
        cls._plugin_resource_requirements = resource_requirements or PluginResourceRequirements()
        cls._plugin_execution_priority = execution_priority
        cls._plugin_tags = tags or []
        cls._plugin_dependencies = dependencies or []
        
        # Override metadata property
        def get_metadata(self) -> PluginMetadata:
            return PluginMetadata(
                name=cls._plugin_name,
                version=cls._plugin_version,
                description=cls._plugin_description,
                author=cls._plugin_author,
                plugin_type=cls._plugin_type,
                tags=cls._plugin_tags,
                dependencies=cls._plugin_dependencies,
                execution_priority=cls._plugin_execution_priority
            )
        
        def get_resource_requirements(self) -> PluginResourceRequirements:
            return cls._plugin_resource_requirements
        
        cls.metadata = property(get_metadata)
        cls.resource_requirements = property(get_resource_requirements)
        
        return cls
    
    return decorator