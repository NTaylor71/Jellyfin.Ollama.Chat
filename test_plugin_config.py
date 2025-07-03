#!/usr/bin/env python3
"""
Test Plugin Configuration System
Validates that the plugin configuration system works correctly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.plugins.config import (
    BasePluginConfig, PluginConfigManager, GlobalPluginConfigManager,
    get_global_config_manager, register_plugin_config
)
from src.plugins.examples.adaptive_query_expander import AdaptiveQueryExpanderConfig
from pydantic import Field


class TestPluginConfig(BasePluginConfig):
    """Test configuration class."""
    test_setting: str = Field(default="default_value", description="A test setting")
    test_number: int = Field(default=42, description="A test number")
    test_flag: bool = Field(default=False, description="A test flag")


async def test_basic_config():
    """Test basic configuration loading."""
    print("=== Testing Basic Configuration ===")
    
    # Create config manager
    manager = PluginConfigManager("test_plugin", TestPluginConfig)
    
    # Load default config
    config = manager.load_config()
    print(f"Default config: {config.model_dump()}")
    
    # Test updating config
    success = manager.update_config({
        "test_setting": "updated_value",
        "test_number": 123
    })
    print(f"Update success: {success}")
    
    # Get updated config
    updated_config = manager.get_config()
    print(f"Updated config: {updated_config.model_dump()}")
    
    # Get config summary
    summary = manager.get_config_summary()
    print(f"Config summary: {summary}")
    
    return True


async def test_global_config_manager():
    """Test the global configuration manager."""
    print("\n=== Testing Global Configuration Manager ===")
    
    # Get global manager
    global_manager = get_global_config_manager()
    
    # Register a plugin
    plugin_manager = global_manager.register_plugin("adaptive_query_expander", AdaptiveQueryExpanderConfig)
    print("Registered adaptive query expander plugin")
    
    # Get config
    config = global_manager.get_plugin_config("adaptive_query_expander")
    print(f"Plugin config: {config.model_dump()}")
    
    # Update config
    success = global_manager.update_plugin_config("adaptive_query_expander", {
        "max_expansion_terms": 20,
        "enable_ollama": False
    })
    print(f"Update success: {success}")
    
    # Get all summaries
    summaries = global_manager.get_all_config_summaries()
    print(f"All summaries: {list(summaries.keys())}")
    
    return True


async def test_file_config():
    """Test file-based configuration."""
    print("\n=== Testing File Configuration ===")
    
    # Create a temporary config file
    config_dir = Path("config/plugins")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    test_config_file = config_dir / "test_plugin.yml"
    test_config_content = """
enabled: true
debug: true
timeout_seconds: 60.0
test_setting: "from_file"
test_number: 999
test_flag: true
"""
    
    with open(test_config_file, 'w') as f:
        f.write(test_config_content)
    
    # Create manager that will read the file
    manager = PluginConfigManager("test_plugin", TestPluginConfig, config_dir)
    config = manager.load_config()
    
    print(f"Config from file: {config.model_dump()}")
    
    # Check that values came from file
    assert config.test_setting == "from_file"
    assert config.test_number == 999
    assert config.test_flag == True
    
    # Clean up
    test_config_file.unlink()
    
    return True


async def test_env_config():
    """Test environment variable configuration."""
    print("\n=== Testing Environment Configuration ===")
    
    # Set environment variables
    os.environ["PLUGIN_TEST_PLUGIN_TEST_SETTING"] = "from_env"
    os.environ["PLUGIN_TEST_PLUGIN_TEST_NUMBER"] = "777"
    os.environ["PLUGIN_TEST_PLUGIN_ENABLED"] = "false"
    
    try:
        # Create manager
        manager = PluginConfigManager("test_plugin", TestPluginConfig)
        config = manager.load_config()
        
        print(f"Config from env: {config.model_dump()}")
        
        # Check that values came from environment
        assert config.test_setting == "from_env"
        assert config.test_number == 777
        assert config.enabled == False
        
    finally:
        # Clean up environment variables
        for key in ["PLUGIN_TEST_PLUGIN_TEST_SETTING", "PLUGIN_TEST_PLUGIN_TEST_NUMBER", "PLUGIN_TEST_PLUGIN_ENABLED"]:
            if key in os.environ:
                del os.environ[key]
    
    return True


async def main():
    """Run all tests."""
    print("Testing Plugin Configuration System")
    print("=" * 50)
    
    try:
        # Run tests
        await test_basic_config()
        await test_global_config_manager()
        await test_file_config()
        await test_env_config()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)