#!/usr/bin/env python3
"""
Test Plugin Performance Monitoring
Tests the enhanced plugin performance monitoring capabilities.
"""

import asyncio
import json
import time
import unittest
from unittest.mock import Mock, patch
from prometheus_client import REGISTRY, CollectorRegistry
from prometheus_client.core import Sample

# Add src to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.api.plugin_metrics import (
    PluginMetricsCollector,
    plugin_executions_total,
    plugin_execution_duration_seconds,
    plugin_execution_phases,
    plugin_data_processing_bytes,
    plugin_concurrency_level,
    plugin_queue_size,
    plugin_health_status,
    plugin_memory_usage_bytes,
    plugin_cpu_usage_percent
)


class TestPluginPerformanceMonitoring(unittest.TestCase):
    """Test enhanced plugin performance monitoring."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a fresh metrics collector
        self.metrics = PluginMetricsCollector()
        
        # Clear metrics before each test
        self.metrics._last_collection_time = 0
        
        # Test data
        self.test_plugin_name = "TestPlugin"
        self.test_plugin_type = "query_embellisher"
        
    def test_basic_plugin_execution_recording(self):
        """Test basic plugin execution recording."""
        print("Testing basic plugin execution recording...")
        
        # Record a successful execution
        self.metrics.record_plugin_execution(
            plugin_name=self.test_plugin_name,
            plugin_type=self.test_plugin_type,
            execution_time_ms=250.5,
            success=True
        )
        
        # Check that counters were incremented
        samples = list(plugin_executions_total.collect())[0].samples
        success_samples = [s for s in samples if s.labels.get('status') == 'success']
        self.assertTrue(len(success_samples) > 0, "Should have success samples")
        
        # Check duration histogram
        duration_samples = list(plugin_execution_duration_seconds.collect())[0].samples
        self.assertTrue(len(duration_samples) > 0, "Should have duration samples")
        
        print("✓ Basic plugin execution recording works")
        
    def test_enhanced_plugin_execution_recording(self):
        """Test enhanced plugin execution recording with phase timings."""
        print("Testing enhanced plugin execution recording...")
        
        # Record execution with phase timings and data processing
        phase_timings = {
            "initialization": 50.0,
            "processing": 150.0,
            "cleanup": 25.0
        }
        
        self.metrics.record_plugin_execution(
            plugin_name=self.test_plugin_name,
            plugin_type=self.test_plugin_type,
            execution_time_ms=225.0,
            success=True,
            data_processed_bytes=1024000,  # 1MB
            phase_timings=phase_timings
        )
        
        # Check phase timing metrics
        phase_samples = list(plugin_execution_phases.collect())[0].samples
        self.assertTrue(len(phase_samples) > 0, "Should have phase timing samples")
        
        # Check data processing metrics
        data_samples = list(plugin_data_processing_bytes.collect())[0].samples
        self.assertTrue(len(data_samples) > 0, "Should have data processing samples")
        
        print("✓ Enhanced plugin execution recording works")
        
    def test_plugin_concurrency_tracking(self):
        """Test plugin concurrency level tracking."""
        print("Testing plugin concurrency tracking...")
        
        # Update concurrency level
        self.metrics.update_plugin_concurrency(
            plugin_name=self.test_plugin_name,
            plugin_type=self.test_plugin_type,
            concurrency_level=4
        )
        
        # Check concurrency metric
        concurrency_samples = list(plugin_concurrency_level.collect())[0].samples
        self.assertTrue(len(concurrency_samples) > 0, "Should have concurrency samples")
        
        print("✓ Plugin concurrency tracking works")
        
    def test_plugin_queue_size_tracking(self):
        """Test plugin queue size tracking."""
        print("Testing plugin queue size tracking...")
        
        # Update queue size
        self.metrics.update_plugin_queue_size(
            plugin_name=self.test_plugin_name,
            plugin_type=self.test_plugin_type,
            queue_size=12
        )
        
        # Check queue size metric
        queue_samples = list(plugin_queue_size.collect())[0].samples
        self.assertTrue(len(queue_samples) > 0, "Should have queue size samples")
        
        print("✓ Plugin queue size tracking works")
        
    def test_plugin_failure_recording(self):
        """Test plugin failure recording."""
        print("Testing plugin failure recording...")
        
        # Record a failed execution
        self.metrics.record_plugin_execution(
            plugin_name=self.test_plugin_name,
            plugin_type=self.test_plugin_type,
            execution_time_ms=500.0,
            success=False
        )
        
        # Check that failure counter was incremented
        samples = list(plugin_executions_total.collect())[0].samples
        failure_samples = [s for s in samples if s.labels.get('status') == 'failure']
        self.assertTrue(len(failure_samples) > 0, "Should have failure samples")
        
        print("✓ Plugin failure recording works")
        
    async def test_plugin_health_metrics_update(self):
        """Test plugin health metrics update."""
        print("Testing plugin health metrics update...")
        
        # Mock plugin registry
        mock_registry = Mock()
        mock_registry.get_plugin_status.return_value = {
            "total_plugins": 3,
            "enabled_plugins": 2,
            "initialized_plugins": 2,
            "plugin_details": {
                self.test_plugin_name: {
                    "initialized": True,
                    "health": {
                        "status": "healthy",
                        "resource_usage": {
                            "memory_used_mb": 128,
                            "cpu_percent": 15.5,
                            "num_threads": 4
                        }
                    }
                }
            }
        }
        
        # Mock plugin instance
        mock_plugin = Mock()
        mock_plugin.metadata.plugin_type.value = self.test_plugin_type
        mock_registry.get_plugin.return_value = mock_plugin
        
        # Update health metrics
        await self.metrics.update_plugin_health_metrics(mock_registry)
        
        # Check health status metric
        health_samples = list(plugin_health_status.collect())[0].samples
        self.assertTrue(len(health_samples) > 0, "Should have health status samples")
        
        # Check memory usage metric
        memory_samples = list(plugin_memory_usage_bytes.collect())[0].samples
        self.assertTrue(len(memory_samples) > 0, "Should have memory usage samples")
        
        # Check CPU usage metric
        cpu_samples = list(plugin_cpu_usage_percent.collect())[0].samples
        self.assertTrue(len(cpu_samples) > 0, "Should have CPU usage samples")
        
        print("✓ Plugin health metrics update works")
        
    def test_plugin_metrics_reset(self):
        """Test plugin metrics reset functionality."""
        print("Testing plugin metrics reset...")
        
        # First record some metrics
        self.metrics.record_plugin_execution(
            plugin_name=self.test_plugin_name,
            plugin_type=self.test_plugin_type,
            execution_time_ms=100.0,
            success=True
        )
        
        self.metrics.update_plugin_concurrency(
            plugin_name=self.test_plugin_name,
            plugin_type=self.test_plugin_type,
            concurrency_level=2
        )
        
        # Reset metrics
        self.metrics.reset_plugin_metrics(
            plugin_name=self.test_plugin_name,
            plugin_type=self.test_plugin_type
        )
        
        # Check that gauge metrics are reset to 0
        concurrency_samples = list(plugin_concurrency_level.collect())[0].samples
        queue_samples = list(plugin_queue_size.collect())[0].samples
        
        # Find samples for our test plugin
        test_concurrency_samples = [
            s for s in concurrency_samples 
            if s.labels.get('plugin_name') == self.test_plugin_name
        ]
        test_queue_samples = [
            s for s in queue_samples 
            if s.labels.get('plugin_name') == self.test_plugin_name
        ]
        
        if test_concurrency_samples:
            self.assertEqual(test_concurrency_samples[0].value, 0, "Concurrency should be reset to 0")
        if test_queue_samples:
            self.assertEqual(test_queue_samples[0].value, 0, "Queue size should be reset to 0")
        
        print("✓ Plugin metrics reset works")
        
    def test_performance_metrics_collection_rate_limiting(self):
        """Test that metrics collection is rate limited."""
        print("Testing performance metrics collection rate limiting...")
        
        # Set a very recent last collection time
        self.metrics._last_collection_time = time.time()
        
        # Mock plugin registry
        mock_registry = Mock()
        mock_registry.get_plugin_status.return_value = {
            "total_plugins": 1,
            "enabled_plugins": 1,
            "initialized_plugins": 1,
            "plugin_details": {}
        }
        
        # Try to update metrics - should be rate limited
        async def test_rate_limiting():
            await self.metrics.update_plugin_health_metrics(mock_registry)
            # Should not call get_plugin_status due to rate limiting
            mock_registry.get_plugin_status.assert_not_called()
        
        asyncio.run(test_rate_limiting())
        
        print("✓ Performance metrics collection rate limiting works")
        
    def test_multiple_plugin_metrics(self):
        """Test metrics collection for multiple plugins."""
        print("Testing metrics collection for multiple plugins...")
        
        plugins = [
            ("QueryExpander", "query_embellisher"),
            ("EmbedEnhancer", "embed_data_embellisher"),
            ("FAISLogger", "faiss_crud")
        ]
        
        # Record metrics for multiple plugins
        for plugin_name, plugin_type in plugins:
            self.metrics.record_plugin_execution(
                plugin_name=plugin_name,
                plugin_type=plugin_type,
                execution_time_ms=200 + hash(plugin_name) % 100,
                success=True,
                data_processed_bytes=1000 + hash(plugin_name) % 10000
            )
            
            self.metrics.update_plugin_concurrency(
                plugin_name=plugin_name,
                plugin_type=plugin_type,
                concurrency_level=hash(plugin_name) % 5 + 1
            )
        
        # Check that we have metrics for all plugins
        execution_samples = list(plugin_executions_total.collect())[0].samples
        concurrency_samples = list(plugin_concurrency_level.collect())[0].samples
        
        # Should have samples for all plugins
        plugin_names_in_execution = set(s.labels.get('plugin_name') for s in execution_samples)
        plugin_names_in_concurrency = set(s.labels.get('plugin_name') for s in concurrency_samples)
        
        for plugin_name, _ in plugins:
            self.assertIn(plugin_name, plugin_names_in_execution, f"Should have execution metrics for {plugin_name}")
            self.assertIn(plugin_name, plugin_names_in_concurrency, f"Should have concurrency metrics for {plugin_name}")
        
        print("✓ Multiple plugin metrics collection works")


def main():
    """Run all plugin performance monitoring tests."""
    print("=== Plugin Performance Monitoring Tests ===\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPluginPerformanceMonitoring)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n=== Test Results ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit(main())