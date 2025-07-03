#!/usr/bin/env python3
"""
Test FAISS CRUD Plugin
Comprehensive tests for the FAISS CRUD Logger Plugin functionality.
"""

import asyncio
import json
import pytest
import time
from typing import Dict, Any, List

# Test imports
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.plugins.examples.faiss_crud_logger import FAISSCRUDLoggerPlugin, FAISSOperation
from src.plugins.base import (
    PluginExecutionContext, 
    PluginExecutionResult, 
    ExecutionPriority,
    PluginType
)


class TestFAISSCRUDPlugin:
    """Test suite for FAISS CRUD Logger Plugin."""
    
    @pytest.fixture
    async def plugin(self):
        """Create and initialize a FAISS CRUD plugin instance."""
        plugin = FAISSCRUDLoggerPlugin()
        await plugin.initialize({})
        return plugin
    
    @pytest.fixture
    def context(self):
        """Create a test execution context."""
        return PluginExecutionContext(
            user_id="test_user",
            session_id="test_session", 
            request_id="test_request",
            execution_timeout=10.0,
            priority=ExecutionPriority.NORMAL
        )
    
    def test_plugin_metadata(self):
        """Test plugin metadata properties."""
        plugin = FAISSCRUDLoggerPlugin()
        
        metadata = plugin.metadata
        assert metadata.name == "FAISSCRUDLogger"
        assert metadata.version == "1.0.0"
        assert metadata.plugin_type == PluginType.FAISS_CRUD
        assert metadata.author == "Production RAG System"
        assert "faiss" in metadata.tags
        assert "logging" in metadata.tags
        assert "monitoring" in metadata.tags
        
        # Test resource requirements
        requirements = plugin.resource_requirements
        assert requirements.min_cpu_cores == 1.0
        assert requirements.preferred_cpu_cores == 2.0
        assert requirements.min_memory_mb == 200.0
        assert requirements.preferred_memory_mb == 1000.0
        assert requirements.requires_gpu == False
        assert requirements.can_use_distributed_resources == True
    
    @pytest.mark.asyncio
    async def test_plugin_initialization(self):
        """Test plugin initialization."""
        plugin = FAISSCRUDLoggerPlugin()
        
        # Test initialization
        result = await plugin.initialize({})
        assert result == True
        assert plugin._is_initialized == True
        assert plugin._initialization_error is None
        
        # Test performance stats initialized
        assert len(plugin._performance_stats) > 0
        for op in FAISSOperation:
            assert op.value in plugin._performance_stats
    
    @pytest.mark.asyncio
    async def test_create_index_operation(self, plugin, context):
        """Test create index operation."""
        
        # Test basic index creation
        create_data = {
            "operation": FAISSOperation.CREATE_INDEX,
            "index_name": "test_index",
            "dimension": 128,
            "index_type": "flat"
        }
        
        result = await plugin.handle_faiss_operation(
            FAISSOperation.CREATE_INDEX, create_data, context
        )
        
        assert result["status"] == "success"
        assert result["index_name"] == "test_index"
        assert result["dimension"] == 128
        assert result["index_type"] == "flat"
        assert "real_faiss" in result
        
        # Verify index was created
        assert "test_index" in plugin._mock_indices
        index_info = plugin._mock_indices["test_index"]
        assert index_info.dimension == 128
        assert index_info.total_vectors == 0
        assert index_info.index_type == "flat"
    
    @pytest.mark.asyncio
    async def test_add_vectors_operation(self, plugin, context):
        """Test add vectors operation."""
        
        # First create an index
        create_data = {
            "operation": FAISSOperation.CREATE_INDEX,
            "index_name": "test_vectors",
            "dimension": 4,
            "index_type": "flat"
        }
        
        await plugin.handle_faiss_operation(
            FAISSOperation.CREATE_INDEX, create_data, context
        )
        
        # Now add vectors
        add_data = {
            "operation": FAISSOperation.ADD_VECTORS,
            "index_name": "test_vectors",
            "vectors": [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0]
            ],
            "vector_ids": ["vec1", "vec2", "vec3"]
        }
        
        result = await plugin.handle_faiss_operation(
            FAISSOperation.ADD_VECTORS, add_data, context
        )
        
        assert result["status"] == "success"
        assert result["index_name"] == "test_vectors"
        assert result["vectors_added"] == 3
        assert result["total_vectors"] == 3
        
        # Verify index stats updated
        index_info = plugin._mock_indices["test_vectors"]
        assert index_info.total_vectors == 3
        assert index_info.add_count == 3
        assert index_info.size_bytes > 0
    
    @pytest.mark.asyncio
    async def test_search_operation(self, plugin, context):
        """Test search operation with custom logic."""
        
        # Create index and add vectors first
        create_data = {
            "operation": FAISSOperation.CREATE_INDEX,
            "index_name": "search_test",
            "dimension": 4,
            "index_type": "flat"
        }
        
        await plugin.handle_faiss_operation(
            FAISSOperation.CREATE_INDEX, create_data, context
        )
        
        add_data = {
            "operation": FAISSOperation.ADD_VECTORS,
            "index_name": "search_test",
            "vectors": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        }
        
        await plugin.handle_faiss_operation(
            FAISSOperation.ADD_VECTORS, add_data, context
        )
        
        # Test search with different strategies
        search_tests = [
            {
                "k": 3,
                "expected_strategy": "exact_search"
            },
            {
                "k": 15,
                "expected_strategy": "approximate_search"
            },
            {
                "k": 25,
                "search_params": {"high_recall": True},
                "expected_strategy": "high_recall_search"
            },
            {
                "k": 30,
                "expected_strategy": "balanced_search"
            }
        ]
        
        for test_case in search_tests:
            search_data = {
                "operation": FAISSOperation.SEARCH,
                "index_name": "search_test",
                "query_vector": [1.0, 2.0, 3.0, 4.0],
                "k": test_case["k"]
            }
            
            if "search_params" in test_case:
                search_data["search_params"] = test_case["search_params"]
            
            result = await plugin.handle_faiss_operation(
                FAISSOperation.SEARCH, search_data, context
            )
            
            assert result["status"] == "success"
            assert result["search_strategy"] == test_case["expected_strategy"]
            assert "custom_logic" in result
            assert "results" in result
            assert len(result["results"]["distances"]) <= test_case["k"]
            assert len(result["results"]["indices"]) <= test_case["k"]
        
        # Verify search count updated
        index_info = plugin._mock_indices["search_test"]
        assert index_info.search_count == len(search_tests)
    
    @pytest.mark.asyncio
    async def test_get_index_info_operation(self, plugin, context):
        """Test get index info operation."""
        
        # Create and populate an index
        create_data = {
            "operation": FAISSOperation.CREATE_INDEX,
            "index_name": "info_test",
            "dimension": 8,
            "index_type": "ivf"
        }
        
        await plugin.handle_faiss_operation(
            FAISSOperation.CREATE_INDEX, create_data, context
        )
        
        add_data = {
            "operation": FAISSOperation.ADD_VECTORS,
            "index_name": "info_test",
            "vectors": [[1.0] * 8, [2.0] * 8, [3.0] * 8, [4.0] * 8, [5.0] * 8]
        }
        
        await plugin.handle_faiss_operation(
            FAISSOperation.ADD_VECTORS, add_data, context
        )
        
        # Search a few times to update stats
        for i in range(3):
            search_data = {
                "operation": FAISSOperation.SEARCH,
                "index_name": "info_test",
                "query_vector": [1.0] * 8,
                "k": 2
            }
            
            await plugin.handle_faiss_operation(
                FAISSOperation.SEARCH, search_data, context
            )
        
        # Get index info
        info_data = {
            "operation": FAISSOperation.GET_INDEX_INFO,
            "index_name": "info_test"
        }
        
        result = await plugin.handle_faiss_operation(
            FAISSOperation.GET_INDEX_INFO, info_data, context
        )
        
        assert result["status"] == "success"
        assert result["index_name"] == "info_test"
        
        info = result["info"]
        assert info["dimension"] == 8
        assert info["total_vectors"] == 5
        assert info["index_type"] == "ivf"
        assert info["search_count"] == 3
        assert info["add_count"] == 5
        assert info["size_bytes"] > 0
        assert info["created_timestamp"] > 0
        assert info["last_modified"] > 0
        
        # Check performance stats included
        assert "performance_stats" in result
    
    @pytest.mark.asyncio
    async def test_other_operations(self, plugin, context):
        """Test other CRUD operations (delete, update, save, load)."""
        
        operations = [
            {
                "operation": FAISSOperation.DELETE_VECTORS,
                "index_name": "test",
                "vector_ids": ["vec1", "vec2"]
            },
            {
                "operation": FAISSOperation.UPDATE_VECTORS,
                "index_name": "test",
                "vectors": [[1.0, 2.0]],
                "vector_ids": ["vec1"]
            },
            {
                "operation": FAISSOperation.SAVE_INDEX,
                "index_name": "test",
                "file_path": "/tmp/test_index.faiss"
            },
            {
                "operation": FAISSOperation.LOAD_INDEX,
                "index_name": "test",
                "file_path": "/tmp/test_index.faiss"
            }
        ]
        
        for op_data in operations:
            result = await plugin.handle_faiss_operation(
                op_data["operation"], op_data, context
            )
            
            assert result["status"] == "success"
            assert result["operation"] == op_data["operation"]
            assert "note" in result
    
    @pytest.mark.asyncio
    async def test_unknown_operation(self, plugin, context):
        """Test handling of unknown operations."""
        
        unknown_data = {
            "operation": "unknown_operation",
            "index_name": "test"
        }
        
        result = await plugin.handle_faiss_operation(
            "unknown_operation", unknown_data, context
        )
        
        assert result["status"] == "error"
        assert "Unknown operation" in result["error"]
        assert "supported_operations" in result
        assert len(result["supported_operations"]) == len(FAISSOperation)
    
    @pytest.mark.asyncio
    async def test_operation_logging(self, plugin, context):
        """Test operation logging functionality."""
        
        # Perform some operations
        operations = [
            {
                "operation": FAISSOperation.CREATE_INDEX,
                "index_name": "log_test",
                "dimension": 4
            },
            {
                "operation": FAISSOperation.ADD_VECTORS,
                "index_name": "log_test",
                "vectors": [[1.0, 2.0, 3.0, 4.0]]
            },
            {
                "operation": FAISSOperation.SEARCH,
                "index_name": "log_test",
                "query_vector": [1.0, 2.0, 3.0, 4.0],
                "k": 1
            }
        ]
        
        for op_data in operations:
            await plugin.handle_faiss_operation(
                op_data["operation"], op_data, context
            )
        
        # Check operation logs
        logs = await plugin.get_operation_logs(limit=10)
        
        # Should have start and complete logs for each operation
        assert len(logs) >= len(operations) * 2
        
        # Check log structure
        for log in logs:
            assert "operation_id" in log
            assert "operation" in log
            assert "timestamp" in log
            assert "phase" in log
            assert log["phase"] in ["start", "complete", "error"]
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, plugin, context):
        """Test performance tracking functionality."""
        
        # Perform operations to generate performance data
        for i in range(5):
            create_data = {
                "operation": FAISSOperation.CREATE_INDEX,
                "index_name": f"perf_test_{i}",
                "dimension": 4
            }
            
            await plugin.handle_faiss_operation(
                FAISSOperation.CREATE_INDEX, create_data, context
            )
        
        # Get performance summary
        summary = await plugin.get_performance_summary()
        
        assert summary["total_operations"] >= 5
        assert summary["total_time_ms"] > 0
        assert "operations_by_type" in summary
        assert FAISSOperation.CREATE_INDEX in summary["operations_by_type"]
        
        # Check operation stats
        create_stats = summary["operations_by_type"][FAISSOperation.CREATE_INDEX]
        assert create_stats["count"] >= 5
        assert create_stats["total_time"] > 0
        assert create_stats["avg_time"] > 0
        assert create_stats["success_count"] >= 5
        assert create_stats["error_count"] >= 0
        
        assert "indices_managed" in summary
        assert "faiss_available" in summary
    
    @pytest.mark.asyncio
    async def test_plugin_execute_method(self, plugin, context):
        """Test the main plugin execute method."""
        
        # Test with valid FAISS operation data
        operation_data = {
            "operation": FAISSOperation.CREATE_INDEX,
            "index_name": "execute_test",
            "dimension": 4,
            "index_type": "flat"
        }
        
        result = await plugin.execute(operation_data, context)
        
        assert isinstance(result, PluginExecutionResult)
        assert result.success == True
        assert result.data is not None
        assert result.data["status"] == "success"
        assert result.execution_time_ms > 0
        assert result.metadata["operation"] == FAISSOperation.CREATE_INDEX
        
        # Test with invalid data
        invalid_data = {"not_operation": "invalid"}
        
        result = await plugin.execute(invalid_data, context)
        
        assert isinstance(result, PluginExecutionResult)
        assert result.success == False
        assert "operation" in result.error_message
    
    @pytest.mark.asyncio
    async def test_plugin_health_check(self, plugin):
        """Test plugin health check functionality."""
        
        health = await plugin.health_check()
        
        assert health["status"] == "healthy"
        assert health["initialized"] == True
        assert health["error"] is None
        assert "metrics" in health
        assert "resource_usage" in health
        assert "last_health_check" in health
        
        # Test resource usage details
        resource_usage = health["resource_usage"]
        assert "memory_used_mb" in resource_usage
        assert "cpu_percent" in resource_usage
        assert "resource_requirements" in resource_usage
        
        requirements = resource_usage["resource_requirements"]
        assert requirements["min_cpu_cores"] == 1.0
        assert requirements["preferred_cpu_cores"] == 2.0
        assert requirements["min_memory_mb"] == 200.0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, plugin, context):
        """Test error handling in operations."""
        
        # Test operation with missing index
        search_data = {
            "operation": FAISSOperation.SEARCH,
            "index_name": "nonexistent_index",
            "query_vector": [1.0, 2.0, 3.0, 4.0],
            "k": 5
        }
        
        # This should still work but with mock data
        result = await plugin.handle_faiss_operation(
            FAISSOperation.SEARCH, search_data, context
        )
        
        assert result["status"] == "success"  # Mock operations don't fail
        assert "note" in result
        
        # Test get_index_info with nonexistent index
        info_data = {
            "operation": FAISSOperation.GET_INDEX_INFO,
            "index_name": "nonexistent_index"
        }
        
        result = await plugin.handle_faiss_operation(
            FAISSOperation.GET_INDEX_INFO, info_data, context
        )
        
        assert result["status"] == "error"
        assert "not found" in result["error"]


async def run_manual_tests():
    """Run manual tests for development."""
    
    print("🚀 Starting FAISS CRUD Plugin Tests")
    print("=" * 60)
    
    # Create and initialize plugin
    plugin = FAISSCRUDLoggerPlugin()
    await plugin.initialize({})
    
    context = PluginExecutionContext(
        user_id="manual_test",
        session_id="manual_session",
        request_id="manual_request"
    )
    
    print(f"✅ Plugin initialized: {plugin.metadata.name} v{plugin.metadata.version}")
    print(f"📊 Plugin type: {plugin.metadata.plugin_type}")
    
    # Test 1: Create Index
    print("\n🔧 Testing Index Creation...")
    create_result = await plugin.handle_faiss_operation(
        FAISSOperation.CREATE_INDEX,
        {
            "operation": FAISSOperation.CREATE_INDEX,
            "index_name": "movie_search",
            "dimension": 384,
            "index_type": "flat"
        },
        context
    )
    print(f"  Status: {create_result['status']}")
    print(f"  Index: {create_result['index_name']}")
    print(f"  Real FAISS: {create_result.get('real_faiss', False)}")
    
    # Test 2: Add Vectors
    print("\n📊 Testing Vector Addition...")
    add_result = await plugin.handle_faiss_operation(
        FAISSOperation.ADD_VECTORS,
        {
            "operation": FAISSOperation.ADD_VECTORS,
            "index_name": "movie_search",
            "vectors": [
                [0.1] * 384,  # Movie 1 embedding
                [0.2] * 384,  # Movie 2 embedding
                [0.3] * 384   # Movie 3 embedding
            ],
            "vector_ids": ["movie_1", "movie_2", "movie_3"]
        },
        context
    )
    print(f"  Status: {add_result['status']}")
    print(f"  Vectors added: {add_result['vectors_added']}")
    print(f"  Total vectors: {add_result['total_vectors']}")
    
    # Test 3: Search Operations
    print("\n🔍 Testing Search Operations...")
    search_strategies = [
        {"k": 3, "name": "Exact Search"},
        {"k": 15, "name": "Approximate Search"},
        {"k": 25, "search_params": {"high_recall": True}, "name": "High Recall Search"},
        {"k": 30, "name": "Balanced Search"}
    ]
    
    for strategy in search_strategies:
        search_data = {
            "operation": FAISSOperation.SEARCH,
            "index_name": "movie_search",
            "query_vector": [0.15] * 384,
            "k": strategy["k"]
        }
        
        if "search_params" in strategy:
            search_data["search_params"] = strategy["search_params"]
        
        search_result = await plugin.handle_faiss_operation(
            FAISSOperation.SEARCH, search_data, context
        )
        
        print(f"  {strategy['name']} (k={strategy['k']}): {search_result['status']}")
        print(f"    Strategy: {search_result['search_strategy']}")
        print(f"    Results: {len(search_result['results']['distances'])} items")
    
    # Test 4: Index Information
    print("\n📈 Testing Index Information...")
    info_result = await plugin.handle_faiss_operation(
        FAISSOperation.GET_INDEX_INFO,
        {
            "operation": FAISSOperation.GET_INDEX_INFO,
            "index_name": "movie_search"
        },
        context
    )
    
    if info_result["status"] == "success":
        info = info_result["info"]
        print(f"  Index: {info_result['index_name']}")
        print(f"  Dimension: {info['dimension']}")
        print(f"  Total vectors: {info['total_vectors']}")
        print(f"  Search count: {info['search_count']}")
        print(f"  Add count: {info['add_count']}")
        print(f"  Size: {info['size_bytes']} bytes")
    
    # Test 5: Performance Summary
    print("\n⚡ Testing Performance Summary...")
    perf_summary = await plugin.get_performance_summary()
    print(f"  Total operations: {perf_summary['total_operations']}")
    print(f"  Total time: {perf_summary['total_time_ms']:.2f}ms")
    print(f"  Indices managed: {perf_summary['indices_managed']}")
    print(f"  FAISS available: {perf_summary['faiss_available']}")
    
    # Test 6: Operation Logs
    print("\n📝 Testing Operation Logs...")
    logs = await plugin.get_operation_logs(limit=5)
    print(f"  Recent logs: {len(logs)} entries")
    for log in logs[-3:]:  # Show last 3 logs
        print(f"    {log['operation']} - {log['phase']} - {log.get('execution_time_ms', 0):.2f}ms")
    
    # Test 7: Plugin Health
    print("\n🏥 Testing Plugin Health...")
    health = await plugin.health_check()
    print(f"  Status: {health['status']}")
    print(f"  Initialized: {health['initialized']}")
    print(f"  Memory usage: {health['resource_usage']['memory_used_mb']:.2f}MB")
    print(f"  CPU usage: {health['resource_usage']['cpu_percent']:.2f}%")
    
    print("\n🎉 All FAISS CRUD Plugin tests completed successfully!")


if __name__ == "__main__":
    # Run manual tests
    asyncio.run(run_manual_tests())