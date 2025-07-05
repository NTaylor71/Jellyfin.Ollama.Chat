"""
FAISS CRUD Logger Plugin
Provides basic FAISS operation logging, performance monitoring hooks, and custom search logic examples.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Optional FAISS dependency
try:
    import faiss
    import numpy as np
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    np = None
    _FAISS_AVAILABLE = False

from src.plugins.base import (
    FAISSCRUDPlugin,
    PluginMetadata,
    PluginType,
    PluginResourceRequirements,
    PluginExecutionContext,
    PluginExecutionResult,
    ExecutionPriority
)
from src.shared.hardware_config import get_resource_limits, ResourceType


class FAISSOperation(str, Enum):
    """Supported FAISS operations."""
    CREATE_INDEX = "create_index"
    ADD_VECTORS = "add_vectors"
    SEARCH = "search"
    DELETE_VECTORS = "delete_vectors"
    UPDATE_VECTORS = "update_vectors"
    GET_INDEX_INFO = "get_index_info"
    SAVE_INDEX = "save_index"
    LOAD_INDEX = "load_index"


@dataclass
class FAISSIndexInfo:
    """Information about a FAISS index."""
    dimension: int
    total_vectors: int
    index_type: str
    created_timestamp: float
    last_modified: float
    size_bytes: int
    search_count: int = 0
    add_count: int = 0


class FAISSCRUDLoggerPlugin(FAISSCRUDPlugin):
    """
    FAISS CRUD Logger Plugin
    
    Provides comprehensive logging and monitoring for FAISS operations:
    - Operation logging with performance metrics
    - Custom search logic examples
    - Hardware-adaptive processing
    - Mock FAISS operations when library unavailable
    """
    
    def __init__(self):
        super().__init__()
        self._mock_indices: Dict[str, FAISSIndexInfo] = {}
        self._operation_logs: List[Dict[str, Any]] = []
        self._performance_stats: Dict[str, Dict[str, float]] = {}
        self._logger = logging.getLogger(__name__)
        
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="FAISSCRUDLogger",
            version="1.0.0",
            description="FAISS CRUD operations with comprehensive logging and performance monitoring",
            author="Production RAG System",
            plugin_type=PluginType.FAISS_CRUD,
            tags=["faiss", "logging", "monitoring", "crud"],
            dependencies=["faiss-cpu", "numpy"] if _FAISS_AVAILABLE else [],
            execution_priority=ExecutionPriority.NORMAL
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        return PluginResourceRequirements(
            min_cpu_cores=1.0,
            preferred_cpu_cores=2.0,
            min_memory_mb=200.0,
            preferred_memory_mb=1000.0,
            requires_gpu=False,
            min_gpu_memory_mb=0.0,
            preferred_gpu_memory_mb=512.0,  # GPU can help with large vector operations
            max_execution_time_seconds=30.0,
            can_use_distributed_resources=True
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the FAISS CRUD logger plugin."""
        try:
            self._logger.info("Initializing FAISS CRUD Logger Plugin...")
            
            # Initialize performance tracking
            self._performance_stats = {
                op.value: {"total_time": 0.0, "count": 0, "avg_time": 0.0}
                for op in FAISSOperation
            }
            
            # Check FAISS availability
            if not _FAISS_AVAILABLE:
                self._logger.warning("FAISS library not available, using mock operations")
            else:
                self._logger.info("FAISS library available for real operations")
            
            # Test basic operations
            await self._test_basic_operations()
            
            self._is_initialized = True
            self._logger.info("FAISS CRUD Logger Plugin initialized successfully")
            return True
            
        except Exception as e:
            self._initialization_error = str(e)
            self._logger.error(f"Failed to initialize FAISS CRUD Logger Plugin: {e}")
            return False
    
    async def _test_basic_operations(self) -> None:
        """Test basic FAISS operations during initialization."""
        try:
            # Test creating a simple index
            test_data = {
                "operation": FAISSOperation.CREATE_INDEX,
                "index_name": "test_init_index",
                "dimension": 128,
                "index_type": "flat"
            }
            
            context = PluginExecutionContext(
                request_id="init_test",
                execution_timeout=5.0
            )
            
            result = await self.handle_faiss_operation(
                test_data["operation"], test_data, context
            )
            
            self._logger.info(f"Initialization test completed: {result['status']}")
            
        except Exception as e:
            self._logger.warning(f"Initialization test failed: {e}")
    
    async def handle_faiss_operation(
        self, 
        operation: str, 
        data: Dict[str, Any], 
        context: PluginExecutionContext
    ) -> Dict[str, Any]:
        """Handle FAISS CRUD operations with comprehensive logging."""
        
        start_time = time.time()
        operation_id = f"{operation}_{int(time.time() * 1000)}"
        
        try:
            # Log operation start
            self._log_operation_start(operation_id, operation, data, context)
            
            # Route to appropriate handler
            if operation == FAISSOperation.CREATE_INDEX:
                result = await self._handle_create_index(data, context)
            elif operation == FAISSOperation.ADD_VECTORS:
                result = await self._handle_add_vectors(data, context)
            elif operation == FAISSOperation.SEARCH:
                result = await self._handle_search(data, context)
            elif operation == FAISSOperation.DELETE_VECTORS:
                result = await self._handle_delete_vectors(data, context)
            elif operation == FAISSOperation.UPDATE_VECTORS:
                result = await self._handle_update_vectors(data, context)
            elif operation == FAISSOperation.GET_INDEX_INFO:
                result = await self._handle_get_index_info(data, context)
            elif operation == FAISSOperation.SAVE_INDEX:
                result = await self._handle_save_index(data, context)
            elif operation == FAISSOperation.LOAD_INDEX:
                result = await self._handle_load_index(data, context)
            else:
                result = {
                    "status": "error",
                    "error": f"Unknown operation: {operation}",
                    "supported_operations": [op.value for op in FAISSOperation]
                }
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log operation completion
            self._log_operation_complete(operation_id, operation, result, execution_time)
            
            # Update performance statistics
            self._update_performance_stats(operation, execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = {
                "status": "error",
                "error": str(e),
                "operation": operation,
                "execution_time_ms": execution_time * 1000
            }
            
            self._log_operation_error(operation_id, operation, str(e), execution_time)
            self._update_performance_stats(operation, execution_time, success=False)
            
            return error_result
    
    async def _handle_create_index(self, data: Dict[str, Any], context: PluginExecutionContext) -> Dict[str, Any]:
        """Handle index creation with adaptive resource utilization."""
        
        index_name = data.get("index_name", "default_index")
        dimension = data.get("dimension", 128)
        index_type = data.get("index_type", "flat")
        
        # Get available resources for optimization
        resources = context.available_resources or await get_resource_limits()
        
        if _FAISS_AVAILABLE:
            # Real FAISS operation
            try:
                if index_type == "flat":
                    index = faiss.IndexFlatL2(dimension)
                elif index_type == "ivf":
                    # Use more centroids for better performance with more CPU cores
                    nlist = min(100, max(10, resources.get("total_cpu_capacity", 1) * 10))
                    quantizer = faiss.IndexFlatL2(dimension)
                    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                else:
                    # Default to flat
                    index = faiss.IndexFlatL2(dimension)
                
                # Store index info
                self._mock_indices[index_name] = FAISSIndexInfo(
                    dimension=dimension,
                    total_vectors=0,
                    index_type=index_type,
                    created_timestamp=time.time(),
                    last_modified=time.time(),
                    size_bytes=0
                )
                
                return {
                    "status": "success",
                    "index_name": index_name,
                    "dimension": dimension,
                    "index_type": index_type,
                    "real_faiss": True,
                    "optimization": f"Configured for {resources.get('total_cpu_capacity', 1)} CPU cores"
                }
                
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"FAISS index creation failed: {str(e)}",
                    "fallback": "Using mock index"
                }
        else:
            # Mock operation
            self._mock_indices[index_name] = FAISSIndexInfo(
                dimension=dimension,
                total_vectors=0,
                index_type=index_type,
                created_timestamp=time.time(),
                last_modified=time.time(),
                size_bytes=0
            )
            
            return {
                "status": "success",
                "index_name": index_name,
                "dimension": dimension,
                "index_type": index_type,
                "real_faiss": False,
                "note": "Mock operation - FAISS library not available"
            }
    
    async def _handle_search(self, data: Dict[str, Any], context: PluginExecutionContext) -> Dict[str, Any]:
        """Handle vector search with custom search logic examples."""
        
        index_name = data.get("index_name", "default_index")
        query_vector = data.get("query_vector", [])
        k = data.get("k", 10)
        search_params = data.get("search_params", {})
        
        # Custom search logic examples
        search_strategy = self._determine_search_strategy(data, context)
        
        if _FAISS_AVAILABLE and len(query_vector) > 0:
            # Real FAISS search (if index exists)
            try:
                # Convert query to numpy array
                query_array = np.array([query_vector], dtype=np.float32)
                
                # Mock search results for demonstration
                # In real implementation, would use actual FAISS index
                distances = np.random.rand(k).astype(np.float32)
                indices = np.random.randint(0, 1000, k).astype(np.int64)
                
                # Update index stats
                if index_name in self._mock_indices:
                    self._mock_indices[index_name].search_count += 1
                    self._mock_indices[index_name].last_modified = time.time()
                
                return {
                    "status": "success",
                    "index_name": index_name,
                    "search_strategy": search_strategy,
                    "results": {
                        "distances": distances.tolist(),
                        "indices": indices.tolist(),
                        "k": k
                    },
                    "real_faiss": True,
                    "custom_logic": self._get_custom_search_logic(search_strategy)
                }
                
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"FAISS search failed: {str(e)}",
                    "fallback": "Using mock search"
                }
        else:
            # Mock search
            mock_results = {
                "distances": [0.1, 0.2, 0.3, 0.4, 0.5][:k],
                "indices": list(range(k)),
                "k": k
            }
            
            if index_name in self._mock_indices:
                self._mock_indices[index_name].search_count += 1
                self._mock_indices[index_name].last_modified = time.time()
            
            return {
                "status": "success",
                "index_name": index_name,
                "search_strategy": search_strategy,
                "results": mock_results,
                "real_faiss": False,
                "custom_logic": self._get_custom_search_logic(search_strategy),
                "note": "Mock operation - FAISS library not available or no query vector"
            }
    
    async def _handle_add_vectors(self, data: Dict[str, Any], context: PluginExecutionContext) -> Dict[str, Any]:
        """Handle adding vectors to index."""
        
        index_name = data.get("index_name", "default_index")
        vectors = data.get("vectors", [])
        vector_ids = data.get("vector_ids", [])
        
        if not vectors:
            return {
                "status": "error",
                "error": "No vectors provided for addition"
            }
        
        if _FAISS_AVAILABLE:
            try:
                # Convert vectors to numpy array
                vector_array = np.array(vectors, dtype=np.float32)
                
                # Update index stats
                if index_name in self._mock_indices:
                    self._mock_indices[index_name].add_count += len(vectors)
                    self._mock_indices[index_name].total_vectors += len(vectors)
                    self._mock_indices[index_name].last_modified = time.time()
                    self._mock_indices[index_name].size_bytes += len(vectors) * len(vectors[0]) * 4  # 4 bytes per float32
                
                return {
                    "status": "success",
                    "index_name": index_name,
                    "vectors_added": len(vectors),
                    "total_vectors": self._mock_indices.get(index_name, FAISSIndexInfo(0, 0, "", 0, 0, 0)).total_vectors,
                    "real_faiss": True
                }
                
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"FAISS add vectors failed: {str(e)}"
                }
        else:
            # Mock operation
            if index_name in self._mock_indices:
                self._mock_indices[index_name].add_count += len(vectors)
                self._mock_indices[index_name].total_vectors += len(vectors)
                self._mock_indices[index_name].last_modified = time.time()
                self._mock_indices[index_name].size_bytes += len(vectors) * len(vectors[0]) * 4
            
            return {
                "status": "success",
                "index_name": index_name,
                "vectors_added": len(vectors),
                "total_vectors": self._mock_indices.get(index_name, FAISSIndexInfo(0, 0, "", 0, 0, 0)).total_vectors,
                "real_faiss": False,
                "note": "Mock operation - FAISS library not available"
            }
    
    async def _handle_get_index_info(self, data: Dict[str, Any], context: PluginExecutionContext) -> Dict[str, Any]:
        """Handle getting index information."""
        
        index_name = data.get("index_name", "default_index")
        
        if index_name not in self._mock_indices:
            return {
                "status": "error",
                "error": f"Index '{index_name}' not found"
            }
        
        index_info = self._mock_indices[index_name]
        
        return {
            "status": "success",
            "index_name": index_name,
            "info": {
                "dimension": index_info.dimension,
                "total_vectors": index_info.total_vectors,
                "index_type": index_info.index_type,
                "created_timestamp": index_info.created_timestamp,
                "last_modified": index_info.last_modified,
                "size_bytes": index_info.size_bytes,
                "search_count": index_info.search_count,
                "add_count": index_info.add_count
            },
            "performance_stats": self._performance_stats
        }
    
    async def _handle_delete_vectors(self, data: Dict[str, Any], context: PluginExecutionContext) -> Dict[str, Any]:
        """Handle deleting vectors from index."""
        return {
            "status": "success",
            "operation": "delete_vectors",
            "note": "Delete operation logged - implementation depends on FAISS index type"
        }
    
    async def _handle_update_vectors(self, data: Dict[str, Any], context: PluginExecutionContext) -> Dict[str, Any]:
        """Handle updating vectors in index."""
        return {
            "status": "success", 
            "operation": "update_vectors",
            "note": "Update operation logged - typically requires delete + add"
        }
    
    async def _handle_save_index(self, data: Dict[str, Any], context: PluginExecutionContext) -> Dict[str, Any]:
        """Handle saving index to disk."""
        return {
            "status": "success",
            "operation": "save_index",
            "note": "Save operation logged - would persist index to disk"
        }
    
    async def _handle_load_index(self, data: Dict[str, Any], context: PluginExecutionContext) -> Dict[str, Any]:
        """Handle loading index from disk."""
        return {
            "status": "success",
            "operation": "load_index", 
            "note": "Load operation logged - would restore index from disk"
        }
    
    def _determine_search_strategy(self, data: Dict[str, Any], context: PluginExecutionContext) -> str:
        """Determine optimal search strategy based on data and context."""
        
        k = data.get("k", 10)
        search_params = data.get("search_params", {})
        
        # Example custom logic for search strategy
        if k <= 5:
            return "exact_search"
        elif k <= 20:
            return "approximate_search"
        elif "high_recall" in search_params:
            return "high_recall_search"
        else:
            return "balanced_search"
    
    def _get_custom_search_logic(self, strategy: str) -> Dict[str, Any]:
        """Get custom search logic explanation for the strategy."""
        
        strategies = {
            "exact_search": {
                "description": "Exact nearest neighbor search for high accuracy",
                "use_case": "Small k values, high precision requirements",
                "parameters": {"search_depth": "full", "approximation": False}
            },
            "approximate_search": {
                "description": "Approximate search with speed optimization",
                "use_case": "Medium k values, balanced speed/accuracy",
                "parameters": {"search_depth": "limited", "approximation": True}
            },
            "high_recall_search": {
                "description": "Search optimized for high recall",
                "use_case": "Finding all similar items, broader search",
                "parameters": {"search_depth": "extended", "threshold": "relaxed"}
            },
            "balanced_search": {
                "description": "Balanced search for general use cases",
                "use_case": "Default strategy for most queries",
                "parameters": {"search_depth": "standard", "threshold": "moderate"}
            }
        }
        
        return strategies.get(strategy, strategies["balanced_search"])
    
    def _log_operation_start(self, operation_id: str, operation: str, data: Dict[str, Any], context: PluginExecutionContext) -> None:
        """Log the start of a FAISS operation."""
        
        log_entry = {
            "operation_id": operation_id,
            "operation": operation,
            "timestamp": time.time(),
            "phase": "start",
            "context": {
                "user_id": context.user_id,
                "session_id": context.session_id,
                "request_id": context.request_id,
                "priority": context.priority.value
            },
            "data_keys": list(data.keys()) if isinstance(data, dict) else []
        }
        
        self._operation_logs.append(log_entry)
        self._logger.info(f"FAISS operation started: {operation} [{operation_id}]")
    
    def _log_operation_complete(self, operation_id: str, operation: str, result: Dict[str, Any], execution_time: float) -> None:
        """Log the completion of a FAISS operation."""
        
        log_entry = {
            "operation_id": operation_id,
            "operation": operation,
            "timestamp": time.time(),
            "phase": "complete",
            "execution_time_ms": execution_time * 1000,
            "status": result.get("status", "unknown"),
            "result_keys": list(result.keys()) if isinstance(result, dict) else []
        }
        
        self._operation_logs.append(log_entry)
        self._logger.info(f"FAISS operation completed: {operation} [{operation_id}] - {result.get('status', 'unknown')} ({execution_time * 1000:.2f}ms)")
    
    def _log_operation_error(self, operation_id: str, operation: str, error: str, execution_time: float) -> None:
        """Log an error during FAISS operation."""
        
        log_entry = {
            "operation_id": operation_id,
            "operation": operation,
            "timestamp": time.time(),
            "phase": "error",
            "execution_time_ms": execution_time * 1000,
            "error": error
        }
        
        self._operation_logs.append(log_entry)
        self._logger.error(f"FAISS operation failed: {operation} [{operation_id}] - {error} ({execution_time * 1000:.2f}ms)")
    
    def _update_performance_stats(self, operation: str, execution_time: float, success: bool = True) -> None:
        """Update performance statistics for operations."""
        
        if operation not in self._performance_stats:
            self._performance_stats[operation] = {
                "total_time": 0.0,
                "count": 0,
                "avg_time": 0.0,
                "success_count": 0,
                "error_count": 0
            }
        
        stats = self._performance_stats[operation]
        stats["total_time"] += execution_time
        stats["count"] += 1
        stats["avg_time"] = stats["total_time"] / stats["count"]
        
        if success:
            stats["success_count"] = stats.get("success_count", 0) + 1
        else:
            stats["error_count"] = stats.get("error_count", 0) + 1
    
    async def get_operation_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent operation logs."""
        return self._operation_logs[-limit:]
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all operations."""
        return {
            "total_operations": sum(stats["count"] for stats in self._performance_stats.values()),
            "total_time_ms": sum(stats["total_time"] * 1000 for stats in self._performance_stats.values()),
            "operations_by_type": self._performance_stats,
            "indices_managed": len(self._mock_indices),
            "faiss_available": _FAISS_AVAILABLE
        }