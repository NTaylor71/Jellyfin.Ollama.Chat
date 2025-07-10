"""
Hardware Configuration Loader

Loads hardware resource configuration from YAML files for queue management.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class HardwareConfigLoader:
    """Loads and manages hardware configuration from YAML files."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the hardware config loader.
        
        Args:
            config_dir: Directory containing hardware config files.
                       Defaults to config/hardware/
        """
        self.settings = get_settings()
        
        if config_dir is None:
            # Default to config/hardware/ relative to project root
            self.config_dir = Path(__file__).parent.parent.parent / "config" / "hardware"
        else:
            self.config_dir = Path(config_dir)
            
        self._config_cache: Dict[str, Any] = {}
        
    def load_config(self, config_name: str = "default") -> Dict[str, Any]:
        """
        Load hardware configuration from YAML file.
        
        Args:
            config_name: Name of the config file (without .yaml extension)
            
        Returns:
            Hardware configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        # Check cache first
        if config_name in self._config_cache:
            return self._config_cache[config_name]
            
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            logger.error(f"Hardware config file not found: {config_path}")
            raise FileNotFoundError(f"Hardware config '{config_name}' not found at {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            logger.info(f"Loaded hardware config from {config_path}")
            
            # Validate config structure
            self._validate_config(config)
            
            # Cache the config
            self._config_cache[config_name] = config
            
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse hardware config {config_path}: {e}")
            raise
            
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate hardware configuration structure.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If config is invalid
        """
        required_sections = ["hardware", "task_groups"]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in hardware config")
                
        # Validate hardware section
        hardware = config["hardware"]
        required_hardware = ["cpu", "gpu", "memory"]
        
        for key in required_hardware:
            if key not in hardware:
                raise ValueError(f"Missing required hardware config '{key}'")
                
        # Validate CPU config
        cpu_config = hardware["cpu"]
        if "cores" not in cpu_config or "threads" not in cpu_config:
            raise ValueError("CPU config must include 'cores' and 'threads'")
            
        # Validate task groups
        task_groups = config["task_groups"]
        if not task_groups:
            raise ValueError("At least one task group must be defined")
            
        for group_name, group_config in task_groups.items():
            required_fields = ["threads", "memory_mb", "cpu_cores"]
            for field in required_fields:
                if field not in group_config:
                    raise ValueError(f"Task group '{group_name}' missing required field '{field}'")
                    
    def get_resource_pool_config(self, config_name: str = "default") -> Dict[str, Any]:
        """
        Get resource pool configuration for worker initialization.
        
        Args:
            config_name: Name of the config to load
            
        Returns:
            Dictionary with resource pool parameters
        """
        config = self.load_config(config_name)
        hardware = config["hardware"]
        
        # Convert GB to MB for memory
        memory_gb = hardware["memory"]["total_gb"]
        memory_mb = int(memory_gb * 1024)
        
        # Apply memory overcommit if configured
        if "allocation" in config:
            memory_overcommit = config["allocation"].get("memory_overcommit", 1.0)
            memory_mb = int(memory_mb * memory_overcommit)
            
        return {
            "cpu_cores": hardware["cpu"]["cores"],
            "cpu_threads": hardware["cpu"]["threads"],
            "gpu_count": hardware["gpu"]["count"],
            "memory_mb": memory_mb
        }
        
    def get_task_group_config(self, group_name: str, config_name: str = "default") -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific task group.
        
        Args:
            group_name: Name of the task group
            config_name: Name of the config to load
            
        Returns:
            Task group configuration or None if not found
        """
        config = self.load_config(config_name)
        return config.get("task_groups", {}).get(group_name)
        
    def get_plugin_overrides(self, config_name: str = "default") -> Dict[str, Dict[str, Any]]:
        """
        Get plugin resource requirement overrides.
        
        Args:
            config_name: Name of the config to load
            
        Returns:
            Dictionary mapping plugin names to resource requirements
        """
        config = self.load_config(config_name)
        return config.get("plugin_overrides", {})
        
    def get_allocation_strategy(self, config_name: str = "default") -> Dict[str, Any]:
        """
        Get resource allocation strategy configuration.
        
        Args:
            config_name: Name of the config to load
            
        Returns:
            Allocation strategy configuration
        """
        config = self.load_config(config_name)
        return config.get("allocation", {
            "cpu_strategy": "balanced",
            "thread_oversubscription": 1.0,
            "memory_overcommit": 1.0,
            "priority_levels": {
                "high": 100,
                "medium": 50,
                "low": 10
            }
        })
        
    def list_available_configs(self) -> list[str]:
        """
        List all available hardware configuration files.
        
        Returns:
            List of config names (without .yaml extension)
        """
        if not self.config_dir.exists():
            return []
            
        configs = []
        for file in self.config_dir.glob("*.yaml"):
            configs.append(file.stem)
            
        return sorted(configs)


# Global instance for easy access
_hardware_config_loader = None


def get_hardware_config_loader() -> HardwareConfigLoader:
    """Get or create the global hardware config loader instance."""
    global _hardware_config_loader
    
    if _hardware_config_loader is None:
        _hardware_config_loader = HardwareConfigLoader()
        
    return _hardware_config_loader


def get_hardware_config(config_name: str = "default") -> Dict[str, Any]:
    """
    Convenience function to load hardware configuration.
    
    Args:
        config_name: Name of the config to load
        
    Returns:
        Hardware configuration dictionary
    """
    loader = get_hardware_config_loader()
    return loader.load_config(config_name)