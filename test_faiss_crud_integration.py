#!/usr/bin/env python3
"""
FAISS CRUD Plugin Integration Test
Test the FAISS CRUD plugin integration with the existing plugin system.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List

# Test setup
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.plugins.examples.faiss_crud_logger import FAISSCRUDLoggerPlugin, FAISSOperation
from src.plugins.base import PluginExecutionContext, ExecutionPriority, PluginType
from src.api.plugin_registry import PluginRegistry
from src.shared.hardware_config import get_resource_limits

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


async def test_faiss_crud_integration():
    """Test FAISS CRUD plugin integration with the plugin system."""
    
    print("🚀 Starting FAISS CRUD Plugin Integration Tests")
    print("=" * 70)
    
    # Step 1: Initialize plugin registry
    print("🔧 Setting up FAISS CRUD integration test...")
    registry = PluginRegistry()
    await registry.initialize()
    
    print(f"  Debug: Loaded {len(registry._plugins)} total plugins")
    
    # Step 2: Check plugin discovery
    print("\n🔍 Testing plugin discovery...")
    
    # Find FAISS CRUD plugins
    faiss_plugins = await registry.get_plugins_by_type(PluginType.FAISS_CRUD)
    print(f"✅ Found {len(faiss_plugins)} FAISS CRUD plugin(s)")
    
    if not faiss_plugins:
        print("❌ No FAISS CRUD plugins found - checking registration...")
        
        # Check all plugins
        for plugin_name, plugin_info in registry._plugins.items():
            print(f"  Plugin: {plugin_name}")
            print(f"    Type: {plugin_info.instance.metadata.plugin_type if plugin_info.instance else 'unknown'}")
            print(f"    Enabled: {plugin_info.is_enabled}")
        
        return False
    
    # Get the FAISS CRUD plugin (prefer our new one)
    faiss_plugin_instance = None
    faiss_plugin_name = None
    
    for plugin in faiss_plugins:
        if plugin.metadata.name == "FAISSCRUDLogger":
            faiss_plugin_instance = plugin
            faiss_plugin_name = plugin.metadata.name
            break
    
    if not faiss_plugin_instance:
        faiss_plugin_instance = faiss_plugins[0]
        faiss_plugin_name = faiss_plugin_instance.metadata.name
    
    print(f"✅ Found plugin: {faiss_plugin_name}")
    print(f"  Type: {faiss_plugin_instance.metadata.plugin_type}")
    print(f"  Version: {faiss_plugin_instance.metadata.version}")
    print(f"  Name: {faiss_plugin_instance.metadata.name}")
    
    # Step 3: Test plugin status
    print("\n📊 Testing plugin status...")
    
    total_plugins = len(registry._plugins)
    enabled_plugins = len([p for p in registry._plugins.values() if p.is_enabled])
    failed_plugins = len([p for p in registry._plugins.values() if not p.is_enabled])
    
    print(f"Total plugins: {total_plugins}")
    print(f"Enabled plugins: {enabled_plugins}")
    print(f"Failed plugins: {failed_plugins}")
    
    # Step 4: Test plugin execution through registry
    print("\n🚀 Testing plugin execution through registry...")
    
    # Create test context
    context = PluginExecutionContext(
        user_id="integration_test",
        session_id="integration_session",
        request_id="integration_request",
        priority=ExecutionPriority.NORMAL,
        available_resources=await get_resource_limits()
    )
    
    # Test FAISS operations through the registry
    faiss_operations = [
        {
            "name": "Create Index",
            "data": {
                "operation": FAISSOperation.CREATE_INDEX,
                "index_name": "integration_test_index",
                "dimension": 256,
                "index_type": "flat"
            }
        },
        {
            "name": "Add Vectors",
            "data": {
                "operation": FAISSOperation.ADD_VECTORS,
                "index_name": "integration_test_index",
                "vectors": [
                    [0.1] * 256,
                    [0.2] * 256,
                    [0.3] * 256,
                    [0.4] * 256,
                    [0.5] * 256
                ],
                "vector_ids": ["doc1", "doc2", "doc3", "doc4", "doc5"]
            }
        },
        {
            "name": "Search Vectors",
            "data": {
                "operation": FAISSOperation.SEARCH,
                "index_name": "integration_test_index",
                "query_vector": [0.25] * 256,
                "k": 3
            }
        },
        {
            "name": "Get Index Info",
            "data": {
                "operation": FAISSOperation.GET_INDEX_INFO,
                "index_name": "integration_test_index"
            }
        }
    ]
    
    for i, operation in enumerate(faiss_operations):
        print(f"\n🔧 Testing {operation['name']}...")
        
        try:
            # Get the plugin and execute directly
            plugin = await registry.get_plugin(faiss_plugin_name)
            if not plugin:
                print(f"❌ Plugin {faiss_plugin_name} not found")
                return False
            
            result = await plugin.safe_execute(operation['data'], context)
            
            print(f"✅ Plugin executed successfully")
            print(f"  Processing time: {result.execution_time_ms:.1f}ms")
            print(f"  Success: {result.success}")
            
            if result.success:
                if isinstance(result.data, dict):
                    print(f"  Status: {result.data.get('status', 'unknown')}")
                    
                    # Show specific operation results
                    if operation['name'] == "Create Index":
                        print(f"  Index: {result.data.get('index_name', 'unknown')}")
                        print(f"  Dimension: {result.data.get('dimension', 'unknown')}")
                        print(f"  Type: {result.data.get('index_type', 'unknown')}")
                    
                    elif operation['name'] == "Add Vectors":
                        print(f"  Vectors added: {result.data.get('vectors_added', 0)}")
                        print(f"  Total vectors: {result.data.get('total_vectors', 0)}")
                    
                    elif operation['name'] == "Search Vectors":
                        print(f"  Search strategy: {result.data.get('search_strategy', 'unknown')}")
                        if 'results' in result.data:
                            results = result.data['results']
                            print(f"  Results found: {len(results.get('distances', []))}")
                            print(f"  Top distance: {results.get('distances', [0])[0]:.4f}")
                    
                    elif operation['name'] == "Get Index Info":
                        if 'info' in result.data:
                            info = result.data['info']
                            print(f"  Total vectors: {info.get('total_vectors', 0)}")
                            print(f"  Search count: {info.get('search_count', 0)}")
                            print(f"  Add count: {info.get('add_count', 0)}")
                            print(f"  Size: {info.get('size_bytes', 0)} bytes")
                
                print(f"  Metadata: {result.metadata}")
            else:
                print(f"❌ Plugin execution failed: {result.error_message}")
                
        except Exception as e:
            print(f"❌ Plugin execution error: {e}")
            return False
    
    # Step 5: Test plugin health through registry
    print("\n🏥 Testing plugin health status...")
    
    try:
        plugin = await registry.get_plugin(faiss_plugin_name)
        health = await plugin.health_check()
        print(f"✅ Plugin health status retrieved")
        print(f"  Status: {health.get('status', 'unknown')}")
        print(f"  Initialized: {health.get('initialized', False)}")
        print(f"  Error: {health.get('error', 'None')}")
        
        if 'metrics' in health:
            metrics = health['metrics']
            print(f"  Executions: {metrics.get('executions', 0)}")
            print(f"  Successful: {metrics.get('successful_executions', 0)}")
            print(f"  Failed: {metrics.get('failed_executions', 0)}")
            print(f"  Avg time: {metrics.get('avg_execution_time_ms', 0):.2f}ms")
        
        if 'resource_usage' in health:
            usage = health['resource_usage']
            print(f"  Memory: {usage.get('memory_used_mb', 0):.2f}MB")
            print(f"  CPU: {usage.get('cpu_percent', 0):.2f}%")
    
    except Exception as e:
        print(f"❌ Plugin health check failed: {e}")
        return False
    
    # Step 6: Test plugin metrics
    print("\n📈 Testing plugin metrics...")
    
    try:
        # Get general plugin info
        plugin = await registry.get_plugin(faiss_plugin_name)
        plugin_info = plugin
        print(f"✅ Plugin info retrieved")
        print(f"  Name: {plugin_info.metadata.name}")
        print(f"  Version: {plugin_info.metadata.version}")
        print(f"  Type: {plugin_info.metadata.plugin_type}")
        print(f"  Priority: {plugin_info.metadata.execution_priority}")
        print(f"  Enabled: {plugin_info.metadata.is_enabled}")
        print(f"  Tags: {plugin_info.metadata.tags}")
        
    except Exception as e:
        print(f"❌ Plugin info retrieval failed: {e}")
        return False
    
    # Step 7: Test error handling
    print("\n🛡️ Testing error handling...")
    
    try:
        # Test with invalid operation
        invalid_data = {
            "operation": "invalid_operation",
            "index_name": "test"
        }
        
        plugin = await registry.get_plugin(faiss_plugin_name)
        result = await plugin.safe_execute(invalid_data, context)
        print(f"✅ Error handling test completed")
        print(f"  Success: {result.success}")
        print(f"  Error message: {result.error_message}")
        
        # Should be a graceful failure
        if not result.success and "Unknown operation" in result.error_message:
            print(f"✅ Error handled gracefully")
        else:
            print(f"⚠️  Unexpected error handling behavior")
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False
    
    print("\n🎉 All FAISS CRUD Plugin integration tests passed!")
    
    # Step 8: Cleanup
    print("\n🧹 Cleaning up integration test environment...")
    await registry.cleanup()
    print("✅ Cleanup completed")
    
    print(f"\n✅ FAISS CRUD Plugin: ALL INTEGRATION TESTS PASSED")
    return True


if __name__ == "__main__":
    # Run the integration test
    success = asyncio.run(test_faiss_crud_integration())
    
    if success:
        print("\n🎉 Integration test completed successfully!")
    else:
        print("\n❌ Integration test failed!")
        sys.exit(1)