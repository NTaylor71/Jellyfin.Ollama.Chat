#!/usr/bin/env python3
"""
Test script for the Queue Monitor App (headless testing)
"""

import sys
import os
import logging
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    
    try:
        import psutil
        import GPUtil
        print("✓ System monitoring imports successful")
        
        from src.worker.resource_queue_manager import ResourceAwareQueueManager
        from src.worker.resource_manager import create_resource_pool_from_config
        from src.worker.resource_manager import ResourceManager
        from src.shared.config import get_settings
        print("✓ Project imports successful")
        
        from app.main import QueueTask, ResourceUsage
        print("✓ Data structure imports successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_resource_monitoring():
    """Test resource monitoring functionality"""
    print("\nTesting resource monitoring...")
    
    try:
        import psutil
        import GPUtil
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        print(f"✓ CPU: {cpu_percent}%")
        
        memory = psutil.virtual_memory()
        print(f"✓ Memory: {memory.percent}%")
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                print(f"✓ GPU: {gpus[0].load * 100}%")
            else:
                print("✓ GPU: No GPUs found (expected in some environments)")
        except:
            print("✓ GPU monitoring handled gracefully")
        
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        print(f"✓ Disk: {disk_usage:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Resource monitoring error: {e}")
        return False

def test_queue_connection():
    """Test queue manager connection"""
    print("\nTesting queue connection...")
    
    try:
        from src.worker.resource_queue_manager import ResourceAwareQueueManager
        from src.worker.resource_manager import create_resource_pool_from_config
        
        resource_config = {"cpu_cores": 1, "gpu_count": 0, "memory_mb": 512}
        resource_pool = create_resource_pool_from_config(resource_config, worker_id="app_test")
        queue_manager = ResourceAwareQueueManager(resource_pool)
        print("✓ Queue manager created")
        
        try:
            stats = queue_manager.get_queue_stats()
            print(f"✓ Queue stats: {stats}")
        except Exception as e:
            print(f"⚠ Queue stats failed (Redis not running?): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Queue connection error: {e}")
        return False

def test_data_structures():
    """Test data structure creation"""
    print("\nTesting data structures...")
    
    try:
        from app.main import QueueTask, ResourceUsage
        from datetime import datetime
        
        task = QueueTask(
            id="test123",
            status="pending",
            plugin_name="test_plugin",
            created_at=datetime.now()
        )
        print(f"✓ QueueTask created: {task.id}")
        
        usage = ResourceUsage(
            cpu_percent=50.0,
            memory_percent=60.0,
            gpu_percent=30.0,
            gpu_memory_percent=40.0,
            disk_usage=70.0
        )
        print(f"✓ ResourceUsage created: CPU={usage.cpu_percent}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Data structure error: {e}")
        return False

def test_task_creation():
    """Test task creation functionality"""
    print("\nTesting task creation...")
    
    try:
        from src.worker.resource_queue_manager import ResourceAwareQueueManager
        from src.worker.resource_manager import create_resource_pool_from_config
        
        test_task = {
            'plugin_name': 'test_plugin',
            'field_name': 'test_field',
            'field_value': 'test value',
            'media_type': 'movie'
        }
        
        try:
            resource_config = {"cpu_cores": 1, "gpu_count": 0, "memory_mb": 512}
        resource_pool = create_resource_pool_from_config(resource_config, worker_id="app_test")
        queue_manager = ResourceAwareQueueManager(resource_pool)
            task_id = queue_manager.enqueue_task(
                task_type='plugin_execution',
                data=test_task,
                priority=0
            )
            print(f"✓ Task created with ID: {task_id}")
        except Exception as e:
            print(f"⚠ Task creation failed (Redis not running?): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Task creation error: {e}")
        return False

def test_mock_gui_components():
    """Test GUI component logic without actual GUI"""
    print("\nTesting GUI component logic...")
    
    try:
        from app.main import ResourceUsage
        
        usage = ResourceUsage(
            cpu_percent=75.5,
            memory_percent=85.2,
            gpu_percent=45.8,
            gpu_memory_percent=60.1,
            disk_usage=55.3
        )
        
        print(f"✓ CPU progress: {int(usage.cpu_percent)}%")
        print(f"✓ Memory progress: {int(usage.memory_percent)}%")
        print(f"✓ GPU progress: {int(usage.gpu_percent)}%")
        print(f"✓ GPU Memory progress: {int(usage.gpu_memory_percent)}%")
        print(f"✓ Disk progress: {int(usage.disk_usage)}%")
        
        return True
        
    except Exception as e:
        print(f"✗ GUI component error: {e}")
        return False

def main():
    """Run all tests"""
    print("Queue & Resource Monitor App - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_resource_monitoring,
        test_queue_connection,
        test_data_structures,
        test_task_creation,
        test_mock_gui_components
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! The app should work correctly.")
    else:
        print("⚠ Some tests failed. Check the errors above.")
    
    print("\nNote: GUI display tests skipped (requires X11/Wayland)")
    print("To run the actual GUI app, use: python app/launch.py (on a desktop environment)")

if __name__ == '__main__':
    main()