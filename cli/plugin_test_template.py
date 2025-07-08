#!/usr/bin/env python3
"""
Template CLI script for testing plugins via queue system.
Provides real-world testing using the actual Redis queue and worker system.
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Dict, Any, Optional

# Add project root to path
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.worker.resource_queue_manager import ResourceAwareQueueManager
from src.worker.resource_manager import create_resource_pool_from_config
from src.worker.task_types import TaskType, get_task_definition, validate_task_data
from src.shared.config import get_settings


class PluginTestCLI:
    """Base CLI class for testing plugins via queue system."""
    
    def __init__(self, plugin_name: str, task_type: str):
        self.plugin_name = plugin_name
        self.task_type = task_type
        self.settings = get_settings()
        
        # Create a minimal resource pool for CLI testing  
        resource_config = {
            "cpu_cores": 1,
            "gpu_count": 0,
            "memory_mb": 512
        }
        resource_pool = create_resource_pool_from_config(resource_config, worker_id="cli_test")
        self.queue_manager = ResourceAwareQueueManager(resource_pool)
        
    async def test_plugin(self, test_data: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        """
        Test a plugin by submitting task to queue and waiting for result.
        
        Args:
            test_data: Data to send to plugin
            timeout: Maximum time to wait for result
            
        Returns:
            Plugin result or error info
        """
        # Check Redis connection
        if not await self.queue_manager.health_check():
            return {"error": "Redis connection failed"}
        
        # Enqueue task - data will be wrapped by ResourceAwareQueueManager
        task_data = {
            "plugin_name": self.plugin_name,
            "plugin_type": "concept_expansion",  # ConceptNet is concept expansion
            "data": test_data  # The actual plugin data
        }
        
        # Validate task data  
        is_valid, error_msg = validate_task_data(self.task_type, task_data)
        if not is_valid:
            return {"error": f"Invalid task data: {error_msg}"}
        
        try:
            print(f"🔍 Debug: Sending task_data = {task_data}")
            task_id = await self.queue_manager.enqueue_task(
                task_type=self.task_type,
                data=task_data,
                plugin_name=self.plugin_name,
                priority=10  # High priority for testing
            )
            
            print(f"✓ Task enqueued with ID: {task_id}")
            print(f"✓ Plugin: {self.plugin_name}")
            print(f"✓ Task type: {self.task_type}")
            print(f"✓ Waiting for result (timeout: {timeout}s)...")
            
            # Wait for result
            result = await self._wait_for_result(task_id, timeout)
            
            if result:
                print(f"✓ Task completed successfully")
                return result
            else:
                return {"error": f"Task timed out after {timeout} seconds"}
                
        except Exception as e:
            return {"error": f"Failed to enqueue task: {str(e)}"}
    
    async def _wait_for_result(self, task_id: str, timeout: int) -> Optional[Dict[str, Any]]:
        """Wait for task result with polling."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check for result
            result_key = f"result:{task_id}"
            result_data = await self.queue_manager.redis_client.get(result_key)
            
            if result_data:
                try:
                    result = json.loads(result_data)
                    return result.get("result")
                except json.JSONDecodeError:
                    return {"error": "Invalid result format"}
            
            # Check queue stats
            stats = await self.queue_manager.get_queue_stats()
            total_pending = stats['queues']['total_pending']
            failed_tasks = stats['queues']['failed_tasks']
            
            # More informative status message
            if total_pending > 0:
                print(f"⏳ Waiting... (queued: {total_pending}, failed: {failed_tasks})")
            else:
                print(f"⏳ Processing... (worker busy, failed: {failed_tasks})")
            
            await asyncio.sleep(2)
        
        return None
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        return await self.queue_manager.get_queue_stats()
    
    def get_task_definition(self) -> Optional[Dict[str, Any]]:
        """Get task definition for this plugin."""
        definition = get_task_definition(self.task_type)
        if definition:
            return {
                "plugin_name": definition.plugin_name,
                "description": definition.description,
                "required_fields": definition.required_fields,
                "optional_fields": definition.optional_fields,
                "execution_timeout": definition.execution_timeout,
                "requires_service": definition.requires_service,
                "service_type": definition.service_type
            }
        return None


def create_base_parser(plugin_name: str, description: str) -> argparse.ArgumentParser:
    """Create base argument parser for plugin testing."""
    parser = argparse.ArgumentParser(
        description=f"Test {plugin_name} via queue system",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=60,
        help="Timeout in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show queue statistics only"
    )
    
    parser.add_argument(
        "--definition",
        action="store_true", 
        help="Show task definition only"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def format_result(result: Dict[str, Any], verbose: bool = False) -> str:
    """Format result for display."""
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    if verbose:
        return f"✅ Success:\n{json.dumps(result, indent=2)}"
    else:
        # Show key fields only - check nested data field first
        data = result.get("data", result)
        
        if "expanded_concepts" in data:
            concepts = data["expanded_concepts"][:5]  # Show first 5
            return f"✅ Success: {len(data['expanded_concepts'])} concepts: {concepts}"
        elif "expanded_keywords" in data:
            keywords = data["expanded_keywords"][:5]  # Show first 5
            return f"✅ Success: {len(data['expanded_keywords'])} keywords: {keywords}"
        elif "temporal_expressions" in data:
            return f"✅ Success: {len(data['temporal_expressions'])} temporal expressions found"
        elif "answer" in data:
            return f"✅ Success: {data['answer'][:100]}..."
        else:
            return f"✅ Success: {list(result.keys())}"


if __name__ == "__main__":
    # Example usage - this would be customized per plugin
    print("This is a template. Use specific plugin test scripts.")