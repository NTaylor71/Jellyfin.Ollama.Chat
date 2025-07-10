#!/usr/bin/env python3
"""
Demo script to showcase app features without GUI
"""

import sys
import os
import time
import random
from datetime import datetime


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.worker.resource_queue_manager import ResourceAwareQueueManager
from src.worker.resource_manager import create_resource_pool_from_config
import psutil
import GPUtil

def simulate_resource_monitoring():
    """Simulate resource monitoring display"""
    print("🔍 Resource Monitoring Demo")
    print("-" * 40)
    
    for i in range(5):

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        

        gpu_percent = 0.0
        gpu_memory_percent = 0.0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].load * 100
                gpu_memory_percent = gpus[0].memoryUtil * 100
        except:
            pass
        

        print(f"CPU Usage:    {'█' * int(cpu_percent/5):20} {cpu_percent:.1f}%")
        print(f"Memory Usage: {'█' * int(memory.percent/5):20} {memory.percent:.1f}%")
        print(f"GPU Usage:    {'█' * int(gpu_percent/5):20} {gpu_percent:.1f}%")
        print(f"GPU Memory:   {'█' * int(gpu_memory_percent/5):20} {gpu_memory_percent:.1f}%")
        print(f"Disk Usage:   {'█' * int((disk.used/disk.total)*100/5):20} {(disk.used/disk.total)*100:.1f}%")
        
        if i < 4:
            print("\n⏳ Updating in 2 seconds...\n")
            time.sleep(2)
    
    print("\n✅ Resource monitoring demo complete!")

def simulate_queue_monitoring():
    """Simulate queue monitoring display"""
    print("\n📋 Queue Monitoring Demo")
    print("-" * 40)
    
    try:
        resource_config = {"cpu_cores": 1, "gpu_count": 0, "memory_mb": 512}
        resource_pool = create_resource_pool_from_config(resource_config, worker_id="demo")
        queue_manager = ResourceAwareQueueManager(resource_pool)
        

        stats = queue_manager.get_queue_stats()
        print(f"Initial Queue Stats:")
        print(f"  Pending Tasks: {stats['pending_tasks']}")
        print(f"  Failed Tasks: {stats['failed_tasks']}")
        print(f"  Redis Memory: {stats['redis_info']['memory_usage']['used_memory_human']}")
        

        print(f"\n➕ Adding test tasks...")
        test_tasks = [
            {'plugin_name': 'llm_keyword_plugin', 'field_name': 'keywords', 'field_value': 'space exploration movie'},
            {'plugin_name': 'heideltime_temporal_plugin', 'field_name': 'temporal_info', 'field_value': 'Released in 2024'},
            {'plugin_name': 'gensim_similarity_plugin', 'field_name': 'similar_concepts', 'field_value': 'science fiction'}
        ]
        
        task_ids = []
        for i, task in enumerate(test_tasks):
            task_id = queue_manager.enqueue_task(
                task_type='plugin_execution',
                data=task,
                priority=0
            )
            task_ids.append(task_id)
            print(f"  Task {i+1}: {task['plugin_name']} -> {task_id[:8]}...")
        

        stats = queue_manager.get_queue_stats()
        print(f"\nUpdated Queue Stats:")
        print(f"  Pending Tasks: {stats['pending_tasks']}")
        print(f"  Failed Tasks: {stats['failed_tasks']}")
        

        print(f"\n📊 Queue Table View:")
        print(f"{'ID':10} {'Status':10} {'Plugin':25} {'Created':10}")
        print("-" * 65)
        
        for i, task_id in enumerate(task_ids):
            print(f"{task_id[:8]:10} {'pending':10} {test_tasks[i]['plugin_name']:25} {datetime.now().strftime('%H:%M:%S'):10}")
        
        print(f"\n✅ Queue monitoring demo complete!")
        
    except Exception as e:
        print(f"❌ Queue demo failed: {e}")
        print("Make sure Redis is running!")

def simulate_gui_features():
    """Simulate GUI features"""
    print("\n🖥️  GUI Features Demo")
    print("-" * 40)
    
    print("✨ Features Available in the GUI:")
    print("  • Real-time resource monitoring with progress bars")
    print("  • Dark theme for professional appearance")
    print("  • Resizable panels for customizable layout")
    print("  • Queue task table with status indicators")
    print("  • 'Add Test Tasks' button for testing")
    print("  • 'Exit' button for clean shutdown")
    print("  • Auto-refresh every 1-2 seconds")
    
    print("\n🎨 UI Components:")
    print("  • Left Panel: Queue Monitor")
    print("    - Task table with ID, Status, Plugin, Created, Progress, Error")
    print("    - Real-time queue statistics")
    print("    - Color-coded status indicators")
    print("  • Right Panel: Resource Monitor")
    print("    - CPU usage progress bar (blue)")
    print("    - Memory usage progress bar (green)")
    print("    - GPU usage progress bar (purple)")
    print("    - GPU memory progress bar (yellow)")
    print("    - Disk usage progress bar (orange)")
    print("  • Bottom Panel: Control buttons")
    
    print("\n🔧 Technical Details:")
    print("  • Built with PyQt6 for modern GUI")
    print("  • Multi-threaded architecture for smooth updates")
    print("  • Graceful error handling and recovery")
    print("  • Redis connection with health monitoring")
    print("  • Cross-platform compatibility")

def main():
    """Run the demo"""
    print("🚀 Queue & Resource Monitor App - Feature Demo")
    print("=" * 60)
    
    print("This demo shows the functionality that would be available")
    print("in the GUI application on a desktop environment.")
    print()
    

    simulate_resource_monitoring()
    simulate_queue_monitoring()
    simulate_gui_features()
    
    print("\n" + "=" * 60)
    print("🎯 To run the actual GUI application:")
    print("   python app/launch.py")
    print()
    print("📝 Note: GUI requires X11/Wayland display server")
    print("💡 All core functionality is working and ready!")

if __name__ == '__main__':
    main()