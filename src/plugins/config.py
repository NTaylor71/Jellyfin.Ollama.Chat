"""
Plugin Configuration System
Provides configuration management for plugins with support for config files, environment variables, and runtime updates.
"""

import os
import json
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Type, Union, get_type_hints
from pathlib import Path
from enum import Enum
import logging
from datetime import datetime
from pydantic import BaseModel, ValidationError, Field

logger = logging.getLogger(__name__)


class ConfigSource(str, Enum):
    """Sources of configuration data."""
    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    RUNTIME = "runtime"
    MONGODB = "mongodb"  # For future use


@dataclass
class ConfigValue:
    """A configuration value with metadata about its source."""
    value: Any
    source: ConfigSource
    description: Optional[str] = None
    is_sensitive: bool = False
    last_updated: Optional[float] = None
    
    def __post_init__(self):
        if self.last_updated is None:
            import time
            self.last_updated = time.time()


class BasePluginConfig(BaseModel):
    """Base class for plugin configuration models."""
    
    # Common configuration fields
    enabled: bool = Field(default=True, description="Whether the plugin is enabled")
    debug: bool = Field(default=False, description="Enable debug logging for this plugin")
    timeout_seconds: float = Field(default=30.0, ge=0.1, le=300.0, description="Plugin execution timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum number of retry attempts")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow additional fields
        validate_assignment = True
        use_enum_values = True


class PluginConfigManager:
    """Manages configuration for a single plugin."""
    
    def __init__(self, plugin_name: str, config_class: Type[BasePluginConfig], config_dir: Optional[Path] = None):
        self.plugin_name = plugin_name
        self.config_class = config_class
        self.config_dir = config_dir or Path("config/plugins")
        self.config_values: Dict[str, ConfigValue] = {}
        self._current_config: Optional[BasePluginConfig] = None
        self._config_file_path: Optional[Path] = None
        self._env_prefix = f"PLUGIN_{plugin_name.upper().replace('-', '_')}_"
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_config_file_path(self) -> Path:
        """Get the path to the plugin's config file."""
        if self._config_file_path is None:
            # Try multiple formats
            for ext in ['.yml', '.yaml', '.json']:
                path = self.config_dir / f"{self.plugin_name}{ext}"
                if path.exists():
                    self._config_file_path = path
                    break
            else:
                # Default to yaml
                self._config_file_path = self.config_dir / f"{self.plugin_name}.yml"
        
        return self._config_file_path
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from the config class."""
        config_dict = {}
        
        # Get field defaults from Pydantic model (compatible with Pydantic 2.x)
        for field_name, field_info in self.config_class.model_fields.items():
            default_value = field_info.default
            if default_value is not ...:  # Ellipsis means no default
                config_dict[field_name] = default_value
                self.config_values[field_name] = ConfigValue(
                    value=default_value,
                    source=ConfigSource.DEFAULT,
                    description=field_info.description if hasattr(field_info, 'description') else None
                )
        
        return config_dict
    
    def _load_file_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        config_file = self._get_config_file_path()
        
        if not config_file.exists():
            return {}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    file_config = yaml.safe_load(f)
            
            # Update config values with file source
            for key, value in file_config.items():
                self.config_values[key] = ConfigValue(
                    value=value,
                    source=ConfigSource.FILE,
                    description=f"Loaded from {config_file.name}"
                )
            
            logger.info(f"Loaded config for {self.plugin_name} from {config_file}")
            return file_config
            
        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {e}")
            return {}
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Get type hints for proper type conversion
        type_hints = get_type_hints(self.config_class)
        
        for env_var, value in os.environ.items():
            if env_var.startswith(self._env_prefix):
                # Remove prefix and convert to lowercase
                config_key = env_var[len(self._env_prefix):].lower()
                
                # Convert value to appropriate type
                if config_key in type_hints:
                    try:
                        target_type = type_hints[config_key]
                        if target_type == bool:
                            converted_value = value.lower() in ('true', '1', 'yes', 'on')
                        elif target_type == int:
                            converted_value = int(value)
                        elif target_type == float:
                            converted_value = float(value)
                        else:
                            converted_value = value
                        
                        env_config[config_key] = converted_value
                        self.config_values[config_key] = ConfigValue(
                            value=converted_value,
                            source=ConfigSource.ENVIRONMENT,
                            description=f"From environment variable {env_var}",
                            is_sensitive="password" in config_key.lower() or "secret" in config_key.lower()
                        )
                        
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to convert env var {env_var}={value} to {target_type}: {e}")
                        continue
        
        if env_config:
            logger.info(f"Loaded {len(env_config)} environment variables for {self.plugin_name}")
        
        return env_config
    
    async def _load_mongodb_config(self) -> Dict[str, Any]:
        """Load configuration from MongoDB."""
        try:
            # Import here to avoid circular imports
            from ..plugins.mongo_manager import get_plugin_manager
            
            plugin_manager = await get_plugin_manager()
            plugin_doc = await plugin_manager.get_plugin(self.plugin_name)
            
            if plugin_doc and plugin_doc.default_config:
                # Update config values with MongoDB source
                for key, value in plugin_doc.default_config.items():
                    self.config_values[key] = ConfigValue(
                        value=value,
                        source=ConfigSource.MONGODB,
                        description="Loaded from MongoDB plugin registry"
                    )
                
                logger.debug(f"Loaded MongoDB config for plugin {self.plugin_name}")
                return plugin_doc.default_config
            
            return {}
            
        except Exception as e:
            logger.debug(f"Failed to load MongoDB config for plugin {self.plugin_name}: {e}")
            return {}
    
    async def load_config_async(self) -> BasePluginConfig:
        """Load configuration from all sources in priority order (async version)."""
        # Load in order: defaults -> file -> MongoDB -> environment -> runtime
        config_dict = {}
        
        # 1. Load defaults
        config_dict.update(self._load_default_config())
        
        # 2. Load from file (overrides defaults)
        config_dict.update(self._load_file_config())
        
        # 3. Load from MongoDB (overrides file)
        config_dict.update(await self._load_mongodb_config())
        
        # 4. Load from environment (overrides MongoDB)
        config_dict.update(self._load_env_config())
        
        # 5. Runtime values are handled separately via update_config()
        
        try:
            # Validate and create config instance
            self._current_config = self.config_class(**config_dict)
            logger.info(f"Successfully loaded configuration for plugin {self.plugin_name}")
            return self._current_config
            
        except ValidationError as e:
            logger.error(f"Configuration validation failed for plugin {self.plugin_name}: {e}")
            # Return config with just defaults
            default_config = self._load_default_config()
            self._current_config = self.config_class(**default_config)
            return self._current_config
    
    def load_config(self) -> BasePluginConfig:
        """Load configuration from all sources in priority order (sync version)."""
        # Load in order: defaults -> file -> environment -> runtime
        config_dict = {}
        
        # 1. Load defaults
        config_dict.update(self._load_default_config())
        
        # 2. Load from file (overrides defaults)
        config_dict.update(self._load_file_config())
        
        # 3. Load from environment (overrides file)
        config_dict.update(self._load_env_config())
        
        # 4. Runtime values are handled separately via update_config()
        
        try:
            # Validate and create config instance
            self._current_config = self.config_class(**config_dict)
            logger.info(f"Successfully loaded configuration for plugin {self.plugin_name}")
            return self._current_config
            
        except ValidationError as e:
            logger.error(f"Configuration validation failed for plugin {self.plugin_name}: {e}")
            # Return config with just defaults
            default_config = self._load_default_config()
            self._current_config = self.config_class(**default_config)
            return self._current_config
    
    def update_config(self, updates: Dict[str, Any], source: ConfigSource = ConfigSource.RUNTIME) -> bool:
        """Update configuration at runtime."""
        if self._current_config is None:
            logger.error("No current config to update")
            return False
        
        try:
            # Create updated config dict
            current_dict = self._current_config.model_dump()
            current_dict.update(updates)
            
            # Validate new config
            new_config = self.config_class(**current_dict)
            
            # Update config values tracking
            for key, value in updates.items():
                self.config_values[key] = ConfigValue(
                    value=value,
                    source=source,
                    description=f"Runtime update from {source.value}"
                )
            
            # Apply new config
            self._current_config = new_config
            logger.info(f"Updated configuration for plugin {self.plugin_name}: {list(updates.keys())}")
            return True
            
        except ValidationError as e:
            logger.error(f"Configuration update validation failed for plugin {self.plugin_name}: {e}")
            return False
    
    def get_config(self) -> BasePluginConfig:
        """Get current configuration."""
        if self._current_config is None:
            return self.load_config()
        return self._current_config
    
    def get_config_value(self, key: str) -> Optional[ConfigValue]:
        """Get detailed information about a configuration value."""
        return self.config_values.get(key)
    
    def save_config_to_file(self, config_dict: Optional[Dict[str, Any]] = None) -> bool:
        """Save current configuration to file."""
        if config_dict is None:
            if self._current_config is None:
                logger.error("No configuration to save")
                return False
            config_dict = self._current_config.model_dump()
        
        config_file = self._get_config_file_path()
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                if config_file.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                else:
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"Saved configuration for plugin {self.plugin_name} to {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config file {config_file}: {e}")
            return False
    
    async def save_config_to_mongodb(self, config_dict: Optional[Dict[str, Any]] = None) -> bool:
        """Save current configuration to MongoDB."""
        if config_dict is None:
            if self._current_config is None:
                logger.error("No configuration to save")
                return False
            config_dict = self._current_config.model_dump()
        
        try:
            from ..plugins.mongo_manager import get_plugin_manager
            
            plugin_manager = await get_plugin_manager()
            plugin_doc = await plugin_manager.get_plugin(self.plugin_name)
            
            if not plugin_doc:
                logger.error(f"Plugin {self.plugin_name} not found in MongoDB")
                return False
            
            # Update plugin document with new config
            from pymongo import UpdateOne
            update_doc = {
                "default_config": config_dict,
                "updated_at": datetime.utcnow()
            }
            
            plugins_coll = plugin_manager.mongo_client.get_collection(plugin_manager.plugins_collection)
            result = await plugins_coll.update_one(
                {"name": self.plugin_name},
                {"$set": update_doc}
            )
            
            if result.matched_count > 0:
                logger.info(f"Saved configuration for plugin {self.plugin_name} to MongoDB")
                return True
            else:
                logger.error(f"Failed to update MongoDB config for plugin {self.plugin_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to save config to MongoDB: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration including sources."""
        summary = {
            "plugin_name": self.plugin_name,
            "config_file": str(self._get_config_file_path()),
            "environment_prefix": self._env_prefix,
            "values": {}
        }
        
        for key, config_value in self.config_values.items():
            value_info = {
                "value": config_value.value if not config_value.is_sensitive else "***",
                "source": config_value.source.value,
                "description": config_value.description,
                "last_updated": config_value.last_updated
            }
            summary["values"][key] = value_info
        
        return summary


class GlobalPluginConfigManager:
    """Manages configuration for all plugins."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("config/plugins")
        self.plugin_managers: Dict[str, PluginConfigManager] = {}
    
    def register_plugin(self, plugin_name: str, config_class: Type[BasePluginConfig]) -> PluginConfigManager:
        """Register a plugin with its configuration class."""
        manager = PluginConfigManager(plugin_name, config_class, self.config_dir)
        self.plugin_managers[plugin_name] = manager
        return manager
    
    def get_plugin_config(self, plugin_name: str) -> Optional[BasePluginConfig]:
        """Get configuration for a plugin."""
        manager = self.plugin_managers.get(plugin_name)
        if manager:
            return manager.get_config()
        return None
    
    def update_plugin_config(self, plugin_name: str, updates: Dict[str, Any]) -> bool:
        """Update configuration for a plugin."""
        manager = self.plugin_managers.get(plugin_name)
        if manager:
            return manager.update_config(updates)
        return False
    
    def get_all_config_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration summaries for all plugins."""
        summaries = {}
        for plugin_name, manager in self.plugin_managers.items():
            summaries[plugin_name] = manager.get_config_summary()
        return summaries
    
    def reload_all_configs(self) -> Dict[str, bool]:
        """Reload all plugin configurations."""
        results = {}
        for plugin_name, manager in self.plugin_managers.items():
            try:
                manager.load_config()
                results[plugin_name] = True
            except Exception as e:
                logger.error(f"Failed to reload config for {plugin_name}: {e}")
                results[plugin_name] = False
        return results


# Global instance
_global_config_manager: Optional[GlobalPluginConfigManager] = None


def get_global_config_manager() -> GlobalPluginConfigManager:
    """Get the global plugin configuration manager."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = GlobalPluginConfigManager()
    return _global_config_manager


def register_plugin_config(plugin_name: str, config_class: Type[BasePluginConfig]) -> PluginConfigManager:
    """Register a plugin configuration class."""
    return get_global_config_manager().register_plugin(plugin_name, config_class)


def get_plugin_config(plugin_name: str) -> Optional[BasePluginConfig]:
    """Get configuration for a plugin."""
    return get_global_config_manager().get_plugin_config(plugin_name)


def update_plugin_config(plugin_name: str, updates: Dict[str, Any]) -> bool:
    """Update configuration for a plugin."""
    return get_global_config_manager().update_plugin_config(plugin_name, updates)