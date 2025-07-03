"""
FAISS Logger Plugin - Starter Example
Demonstrates FAISS operation logging and performance monitoring.
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..base import (
    FAISSCRUDPlugin, PluginResourceRequirements, PluginExecutionContext,
    plugin_decorator, PluginType, ExecutionPriority
)


@plugin_decorator(
    name="FAISSLogger",
    version="1.0.0",
    description="Logs FAISS operations and provides performance monitoring hooks",
    author="System",
    plugin_type=PluginType.FAISS_CRUD,
    resource_requirements=PluginResourceRequirements(
        min_cpu_cores=0.5,
        preferred_cpu_cores=1.0,
        min_memory_mb=25.0,
        preferred_memory_mb=100.0,
        max_execution_time_seconds=2.0,
        can_use_distributed_resources=False
    ),
    execution_priority=ExecutionPriority.LOW,  # Run after main FAISS operations
    tags=["logging", "monitoring", "faiss", "observability"]
)
class FAISSLoggerPlugin(FAISSCRUDPlugin):
    """Example plugin that logs FAISS operations and monitors performance."""
    
    def __init__(self):
        super().__init__()
        self.operation_history: List[Dict[str, Any]] = []
        self.performance_stats: Dict[str, Any] = {}
        self.max_history_size = 1000
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the FAISS logger plugin."""
        try:
            self._logger.info("Initializing FAISSLogger plugin")
            
            # Configure from config
            self.max_history_size = config.get('max_history_size', 1000)
            
            # Initialize performance stats
            self.performance_stats = {
                'total_operations': 0,
                'operations_by_type': {},
                'average_response_times': {},
                'error_count': 0,
                'last_operation_time': None,
                'start_time': datetime.utcnow().isoformat()
            }
            
            self._is_initialized = True
            self._logger.info("FAISSLogger plugin initialized successfully")
            return True
            
        except Exception as e:
            self._initialization_error = str(e)
            self._logger.error(f"Failed to initialize FAISSLogger: {e}")
            return False
    
    async def handle_faiss_operation(self, operation: str, data: Dict[str, Any], 
                                   context: PluginExecutionContext) -> Dict[str, Any]:
        """Handle and log FAISS operations."""
        start_time = time.time()
        operation_id = f"{operation}_{int(start_time * 1000)}"
        
        try:
            # Log operation start
            self._logger.info(f"FAISS operation started: {operation} (ID: {operation_id})")
            
            # Process the operation
            result = await self._process_operation(operation, data, context, operation_id)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log successful operation
            await self._log_operation(operation, data, result, execution_time, context, operation_id, success=True)
            
            self._logger.info(f"FAISS operation completed: {operation} (ID: {operation_id}, Time: {execution_time:.3f}s)")
            
            return {
                'operation_id': operation_id,
                'operation': operation,
                'execution_time_ms': execution_time * 1000,
                'result': result,
                'logged': True
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log failed operation
            await self._log_operation(operation, data, None, execution_time, context, operation_id, 
                                    success=False, error=str(e))
            
            self._logger.error(f"FAISS operation failed: {operation} (ID: {operation_id}, Error: {e})")
            
            return {
                'operation_id': operation_id,
                'operation': operation,
                'execution_time_ms': execution_time * 1000,
                'error': str(e),
                'logged': True
            }
    
    async def _process_operation(self, operation: str, data: Dict[str, Any], 
                               context: PluginExecutionContext, operation_id: str) -> Any:
        """Process different types of FAISS operations."""
        # This is a logger plugin, so we simulate/mock FAISS operations
        # In a real implementation, this would interface with actual FAISS operations
        
        if operation == 'search':
            return await self._handle_search(data, context, operation_id)
        elif operation == 'add':
            return await self._handle_add(data, context, operation_id)
        elif operation == 'delete':
            return await self._handle_delete(data, context, operation_id)
        elif operation == 'update':
            return await self._handle_update(data, context, operation_id)
        elif operation == 'get_stats':
            return await self._handle_get_stats(data, context, operation_id)
        else:
            raise ValueError(f"Unsupported FAISS operation: {operation}")
    
    async def _handle_search(self, data: Dict[str, Any], context: PluginExecutionContext, 
                           operation_id: str) -> Dict[str, Any]:
        """Handle FAISS search operations."""
        query = data.get('query', '')
        k = data.get('k', 10)
        
        self._logger.debug(f"Search operation: query_length={len(query)}, k={k}")
        
        # Simulate search processing time based on query complexity
        if len(query) > 100:
            await self._simulate_processing_delay(0.1)  # Longer queries take more time
        else:
            await self._simulate_processing_delay(0.05)
        
        return {
            'query_processed': True,
            'requested_results': k,
            'query_length': len(query),
            'processing_notes': f"Processed search for query of length {len(query)}"
        }
    
    async def _handle_add(self, data: Dict[str, Any], context: PluginExecutionContext, 
                        operation_id: str) -> Dict[str, Any]:
        """Handle FAISS add operations."""
        vectors = data.get('vectors', [])
        metadata = data.get('metadata', {})
        
        vector_count = len(vectors) if isinstance(vectors, list) else 1
        self._logger.debug(f"Add operation: vector_count={vector_count}")
        
        # Simulate processing time based on vector count
        processing_time = max(0.01, vector_count * 0.001)
        await self._simulate_processing_delay(processing_time)
        
        return {
            'vectors_processed': vector_count,
            'metadata_fields': len(metadata) if isinstance(metadata, dict) else 0,
            'processing_notes': f"Processed {vector_count} vectors for addition"
        }
    
    async def _handle_delete(self, data: Dict[str, Any], context: PluginExecutionContext, 
                           operation_id: str) -> Dict[str, Any]:
        """Handle FAISS delete operations."""
        ids = data.get('ids', [])
        
        id_count = len(ids) if isinstance(ids, list) else 1
        self._logger.debug(f"Delete operation: id_count={id_count}")
        
        await self._simulate_processing_delay(0.02)
        
        return {
            'ids_processed': id_count,
            'processing_notes': f"Processed deletion of {id_count} items"
        }
    
    async def _handle_update(self, data: Dict[str, Any], context: PluginExecutionContext, 
                           operation_id: str) -> Dict[str, Any]:
        """Handle FAISS update operations."""
        ids = data.get('ids', [])
        vectors = data.get('vectors', [])
        
        update_count = max(len(ids) if isinstance(ids, list) else 0,
                          len(vectors) if isinstance(vectors, list) else 0)
        self._logger.debug(f"Update operation: update_count={update_count}")
        
        await self._simulate_processing_delay(0.05)
        
        return {
            'updates_processed': update_count,
            'processing_notes': f"Processed {update_count} updates"
        }
    
    async def _handle_get_stats(self, data: Dict[str, Any], context: PluginExecutionContext, 
                              operation_id: str) -> Dict[str, Any]:
        """Handle FAISS statistics requests."""
        self._logger.debug("Stats operation: returning current statistics")
        
        await self._simulate_processing_delay(0.01)
        
        return {
            'stats_retrieved': True,
            'performance_stats': self.performance_stats.copy(),
            'operation_history_size': len(self.operation_history),
            'processing_notes': "Retrieved FAISS statistics"
        }
    
    async def _simulate_processing_delay(self, delay_seconds: float) -> None:
        """Simulate processing delay for realistic logging."""
        import asyncio
        await asyncio.sleep(delay_seconds)
    
    async def _log_operation(self, operation: str, data: Dict[str, Any], result: Any, 
                           execution_time: float, context: PluginExecutionContext, 
                           operation_id: str, success: bool = True, error: Optional[str] = None) -> None:
        """Log the operation details."""
        
        # Create operation log entry
        log_entry = {
            'operation_id': operation_id,
            'operation': operation,
            'timestamp': datetime.utcnow().isoformat(),
            'execution_time_ms': execution_time * 1000,
            'success': success,
            'data_summary': self._summarize_data(data),
            'result_summary': self._summarize_result(result) if success else None,
            'error': error,
            'context': {
                'user_id': context.user_id,
                'session_id': context.session_id,
                'request_id': context.request_id,
                'priority': context.priority.value
            }
        }
        
        # Add to operation history
        self.operation_history.append(log_entry)
        
        # Maintain history size limit
        if len(self.operation_history) > self.max_history_size:
            self.operation_history = self.operation_history[-self.max_history_size:]
        
        # Update performance stats
        await self._update_performance_stats(operation, execution_time, success)
        
        # Log to structured logger
        self._logger.info(
            "FAISS operation logged",
            extra={
                'operation_id': operation_id,
                'operation': operation,
                'execution_time_ms': execution_time * 1000,
                'success': success,
                'error': error
            }
        )
    
    def _summarize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of input data for logging."""
        summary = {}
        
        for key, value in data.items():
            if isinstance(value, list):
                summary[key] = f"list[{len(value)}]"
            elif isinstance(value, dict):
                summary[key] = f"dict[{len(value)}]"
            elif isinstance(value, str):
                summary[key] = f"str[{len(value)}]"
            else:
                summary[key] = type(value).__name__
        
        return summary
    
    def _summarize_result(self, result: Any) -> Dict[str, Any]:
        """Create a summary of result data for logging."""
        if isinstance(result, dict):
            return {k: type(v).__name__ for k, v in result.items()}
        elif isinstance(result, list):
            return {'type': 'list', 'length': len(result)}
        else:
            return {'type': type(result).__name__}
    
    async def _update_performance_stats(self, operation: str, execution_time: float, success: bool) -> None:
        """Update performance statistics."""
        self.performance_stats['total_operations'] += 1
        self.performance_stats['last_operation_time'] = datetime.utcnow().isoformat()
        
        if not success:
            self.performance_stats['error_count'] += 1
        
        # Update operation type stats
        if operation not in self.performance_stats['operations_by_type']:
            self.performance_stats['operations_by_type'][operation] = {
                'count': 0,
                'total_time_ms': 0,
                'avg_time_ms': 0,
                'errors': 0
            }
        
        op_stats = self.performance_stats['operations_by_type'][operation]
        op_stats['count'] += 1
        
        if success:
            op_stats['total_time_ms'] += execution_time * 1000
            op_stats['avg_time_ms'] = op_stats['total_time_ms'] / op_stats['count']
        else:
            op_stats['errors'] += 1
        
        # Update average response times
        if operation not in self.performance_stats['average_response_times']:
            self.performance_stats['average_response_times'][operation] = []
        
        response_times = self.performance_stats['average_response_times'][operation]
        response_times.append(execution_time * 1000)
        
        # Keep only last 100 response times for rolling average
        if len(response_times) > 100:
            self.performance_stats['average_response_times'][operation] = response_times[-100:]
    
    def get_operation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get operation history with optional limit."""
        if limit:
            return self.operation_history[-limit:]
        return self.operation_history.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        summary = self.performance_stats.copy()
        
        # Calculate rolling averages
        for operation, times in summary.get('average_response_times', {}).items():
            if times:
                summary['average_response_times'][operation] = sum(times) / len(times)
        
        return summary
    
    def clear_history(self) -> int:
        """Clear operation history and return number of cleared entries."""
        cleared_count = len(self.operation_history)
        self.operation_history.clear()
        self._logger.info(f"Cleared {cleared_count} operation history entries")
        return cleared_count
    
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        self.operation_history.clear()
        self.performance_stats.clear()
        self._logger.info("FAISSLogger plugin cleaned up")