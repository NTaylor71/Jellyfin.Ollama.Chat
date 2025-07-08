#!/usr/bin/env python3
"""
Queue and Resource Monitor App
A PyQt6 application for monitoring Redis queue and system resources
"""

import sys
import asyncio
import json
import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QProgressBar, QPushButton, QTextEdit, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QFrame, QGridLayout, QGroupBox, QScrollArea
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QPixmap, QPalette, QColor

# Import our existing Redis and resource monitoring code
import redis
import psutil
import GPUtil

from src.worker.resource_queue_manager import ResourceAwareQueueManager
from src.worker.resource_manager import ResourceManager, create_resource_pool_from_config
from src.shared.config import get_settings_for_environment


@dataclass
class QueueTask:
    """Represents a task in the queue"""
    id: str
    status: str
    plugin_name: str
    created_at: datetime
    started_at: datetime = None
    completed_at: datetime = None
    progress: float = 0.0
    error: str = None


@dataclass
class ResourceUsage:
    """Represents current resource usage"""
    cpu_percent: float
    memory_percent: float
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, int] = None


class ResourceMonitorThread(QThread):
    """Background thread for monitoring system resources"""
    
    resource_updated = pyqtSignal(ResourceUsage)
    
    def __init__(self):
        super().__init__()
        self.running = True
        
    def run(self):
        """Monitor resources every second"""
        while self.running:
            try:
                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                # GPU (if available)
                gpu_percent = 0.0
                gpu_memory_percent = 0.0
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_percent = gpus[0].load * 100
                        gpu_memory_percent = gpus[0].memoryUtil * 100
                except:
                    pass
                
                # Disk
                disk = psutil.disk_usage('/')
                disk_usage = (disk.used / disk.total) * 100
                
                # Network
                network = psutil.net_io_counters()
                
                usage = ResourceUsage(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    gpu_percent=gpu_percent,
                    gpu_memory_percent=gpu_memory_percent,
                    disk_usage=disk_usage,
                    network_io={'bytes_sent': network.bytes_sent, 'bytes_recv': network.bytes_recv}
                )
                
                self.resource_updated.emit(usage)
                
            except Exception as e:
                logging.error(f"Resource monitoring error: {e}")
                
            self.msleep(1000)  # Update every second
    
    def stop(self):
        """Stop the monitoring thread"""
        self.running = False
        self.wait()


class QueueMonitorThread(QThread):
    """Background thread for monitoring Redis queue"""
    
    queue_updated = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.queue_manager = None
        
    def run(self):
        """Monitor queue every 2 seconds"""
        try:
            config = {"cpu_cores": 4, "gpu_count": 1, "memory_mb": 8192}
            resource_pool = create_resource_pool_from_config(config)
            self.queue_manager = ResourceAwareQueueManager(resource_pool)
        except Exception as e:
            logging.error(f"Failed to initialize queue manager: {e}")
            return
            
        while self.running:
            try:
                # Get queue info
                tasks = self._get_queue_tasks()
                self.queue_updated.emit(tasks)
                
            except Exception as e:
                logging.error(f"Queue monitoring error: {e}")
                
            self.msleep(2000)  # Update every 2 seconds
    
    def _get_queue_tasks(self) -> List[QueueTask]:
        """Get current tasks from Redis queue"""
        tasks = []
        try:
            # Direct Redis connection to check queue sizes
            # Use docker environment settings to match the worker
            settings = get_settings_for_environment("docker")
            redis_client = redis.Redis(
                host='localhost', 
                port=6379, 
                decode_responses=True
            )
            
            # Check queue sizes (using ZCARD for sorted sets)
            # Use the same queue names as the Docker worker
            cpu_queue_size = redis_client.zcard("chat:queue:cpu")
            gpu_queue_size = redis_client.zcard("chat:queue:gpu") 
            dead_letter_size = redis_client.llen("rag:dead_letter")
            
            # Also check for completed tasks (check recent results)
            completed_count = 0
            try:
                # Look for result keys that were created recently (last 60 seconds)
                import time
                current_time = time.time()
                result_keys = redis_client.keys("result:*")
                for key in result_keys:
                    try:
                        ttl = redis_client.ttl(key)
                        if ttl > (24*3600 - 60):  # Created in last 60 seconds
                            completed_count += 1
                    except:
                        pass
            except:
                pass
            
            # Create representative tasks for display
            total_pending = cpu_queue_size + gpu_queue_size
            
            for i in range(min(total_pending, 15)):  # Show max 15 pending tasks
                queue_type = "CPU" if i < cpu_queue_size else "GPU"
                task = QueueTask(
                    id=f'{queue_type.lower()}_{i+1}',
                    status='pending',
                    plugin_name=f'various_plugins ({queue_type})',
                    created_at=datetime.now()
                )
                tasks.append(task)
            
            for i in range(min(dead_letter_size, 5)):  # Show max 5 failed tasks
                task = QueueTask(
                    id=f'failed_{i+1}',
                    status='error',
                    plugin_name='various_plugins',
                    created_at=datetime.now(),
                    error='Task failed'
                )
                tasks.append(task)
            
            # Add recently completed tasks for display
            for i in range(min(completed_count, 10)):  # Show max 10 completed tasks
                task = QueueTask(
                    id=f'completed_{i+1}',
                    status='completed',
                    plugin_name='various_plugins',
                    created_at=datetime.now(),
                    completed_at=datetime.now(),
                    progress=100.0
                )
                tasks.append(task)
                
        except Exception as e:
            logging.error(f"Error getting queue tasks: {e}")
            
        return tasks
    
    def stop(self):
        """Stop the monitoring thread"""
        self.running = False
        self.wait()


class QueueMonitorApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Queue & Resource Monitor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize monitoring threads
        self.resource_thread = ResourceMonitorThread()
        self.queue_thread = QueueMonitorThread()
        
        # Setup UI
        self.setup_ui()
        self.setup_dark_theme()
        
        # Connect signals
        self.resource_thread.resource_updated.connect(self.update_resource_display)
        self.queue_thread.queue_updated.connect(self.update_queue_display)
        
        # Start monitoring
        self.resource_thread.start()
        self.queue_thread.start()
    
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Queue Monitor
        queue_panel = self.create_queue_panel()
        splitter.addWidget(queue_panel)
        
        # Right panel - Resource Monitor
        resource_panel = self.create_resource_panel()
        splitter.addWidget(resource_panel)
        
        # Set splitter proportions
        splitter.setSizes([600, 600])
        
        main_layout.addWidget(splitter)
        
        # Bottom panel - Controls
        controls_layout = QHBoxLayout()
        
        # Add Tasks button
        self.add_tasks_btn = QPushButton("Add Test Tasks")
        self.add_tasks_btn.clicked.connect(self.add_test_tasks)
        controls_layout.addWidget(self.add_tasks_btn)
        
        # Stretch to push Exit button to right
        controls_layout.addStretch()
        
        # Exit button
        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)
        controls_layout.addWidget(self.exit_btn)
        
        main_layout.addLayout(controls_layout)
    
    def create_queue_panel(self) -> QWidget:
        """Create the queue monitoring panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Queue Monitor")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Queue table
        self.queue_table = QTableWidget()
        self.queue_table.setColumnCount(6)
        self.queue_table.setHorizontalHeaderLabels([
            "ID", "Status", "Plugin", "Created", "Progress", "Error"
        ])
        
        # Configure table
        header = self.queue_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        
        layout.addWidget(self.queue_table)
        
        # Queue stats
        self.queue_stats_label = QLabel("Queue Stats: Loading...")
        layout.addWidget(self.queue_stats_label)
        
        return panel
    
    def create_resource_panel(self) -> QWidget:
        """Create the resource monitoring panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Resource Monitor")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Resource widgets
        self.cpu_bar = self.create_resource_bar("CPU Usage", "rgb(52, 152, 219)")
        self.memory_bar = self.create_resource_bar("Memory Usage", "rgb(46, 204, 113)")
        self.gpu_bar = self.create_resource_bar("GPU Usage", "rgb(155, 89, 182)")
        self.gpu_memory_bar = self.create_resource_bar("GPU Memory", "rgb(241, 196, 15)")
        self.disk_bar = self.create_resource_bar("Disk Usage", "rgb(230, 126, 34)")
        
        layout.addWidget(self.cpu_bar)
        layout.addWidget(self.memory_bar)
        layout.addWidget(self.gpu_bar)
        layout.addWidget(self.gpu_memory_bar)
        layout.addWidget(self.disk_bar)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
    
    def create_resource_bar(self, name: str, color: str) -> QGroupBox:
        """Create a resource usage bar widget"""
        group = QGroupBox(name)
        layout = QVBoxLayout(group)
        
        # Progress bar
        progress_bar = QProgressBar()
        progress_bar.setMaximum(100)
        progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid #555;
                border-radius: 5px;
                text-align: center;
                background-color: #2b2b2b;
                color: white;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)
        
        layout.addWidget(progress_bar)
        
        # Store reference for updates
        setattr(self, f"{name.lower().replace(' ', '_')}_progress", progress_bar)
        
        return group
    
    def setup_dark_theme(self):
        """Apply dark theme to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QPushButton {
                background-color: #404040;
                color: #ffffff;
                border: 2px solid #555;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #353535;
            }
            QTableWidget {
                background-color: #353535;
                alternate-background-color: #404040;
                color: #ffffff;
                gridline-color: #555;
                selection-background-color: #4a4a4a;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #404040;
                color: #ffffff;
                padding: 8px;
                border: 1px solid #555;
                font-weight: bold;
            }
            QGroupBox {
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QSplitter::handle {
                background-color: #555;
            }
        """)
    
    def update_resource_display(self, usage: ResourceUsage):
        """Update the resource display with new data"""
        self.cpu_usage_progress.setValue(int(usage.cpu_percent))
        self.cpu_usage_progress.setFormat(f"{usage.cpu_percent:.1f}%")
        
        self.memory_usage_progress.setValue(int(usage.memory_percent))
        self.memory_usage_progress.setFormat(f"{usage.memory_percent:.1f}%")
        
        self.gpu_usage_progress.setValue(int(usage.gpu_percent))
        self.gpu_usage_progress.setFormat(f"{usage.gpu_percent:.1f}%")
        
        self.gpu_memory_progress.setValue(int(usage.gpu_memory_percent))
        self.gpu_memory_progress.setFormat(f"{usage.gpu_memory_percent:.1f}%")
        
        self.disk_usage_progress.setValue(int(usage.disk_usage))
        self.disk_usage_progress.setFormat(f"{usage.disk_usage:.1f}%")
    
    def update_queue_display(self, tasks: List[QueueTask]):
        """Update the queue display with new data"""
        # Clear existing rows
        self.queue_table.setRowCount(0)
        
        # Add tasks to table
        for i, task in enumerate(tasks):
            self.queue_table.insertRow(i)
            
            # ID
            self.queue_table.setItem(i, 0, QTableWidgetItem(task.id[:8]))
            
            # Status
            status_item = QTableWidgetItem(task.status)
            if task.status == 'running':
                status_item.setBackground(QColor(241, 196, 15))  # Yellow/Orange
            elif task.status == 'pending':
                status_item.setBackground(QColor(52, 152, 219))  # Blue
            elif task.status == 'completed':
                status_item.setBackground(QColor(46, 204, 113))  # Green
            elif task.status == 'error':
                status_item.setBackground(QColor(231, 76, 60))   # Red
            self.queue_table.setItem(i, 1, status_item)
            
            # Plugin
            self.queue_table.setItem(i, 2, QTableWidgetItem(task.plugin_name))
            
            # Created
            created_str = task.created_at.strftime("%H:%M:%S")
            self.queue_table.setItem(i, 3, QTableWidgetItem(created_str))
            
            # Progress
            progress_str = f"{task.progress:.1f}%" if task.progress > 0 else "-"
            self.queue_table.setItem(i, 4, QTableWidgetItem(progress_str))
            
            # Error
            error_str = task.error if task.error else "-"
            self.queue_table.setItem(i, 5, QTableWidgetItem(error_str))
        
        # Update stats
        pending_count = len([t for t in tasks if t.status == 'pending'])
        running_count = len([t for t in tasks if t.status == 'running'])
        completed_count = len([t for t in tasks if t.status == 'completed'])
        error_count = len([t for t in tasks if t.status == 'error'])
        total_count = len(tasks)
        
        self.queue_stats_label.setText(
            f"Total: {total_count} | Pending: {pending_count} | Running: {running_count} | Completed: {completed_count} | Errors: {error_count}"
        )
    
    def add_test_tasks(self):
        """Add fake test tasks to the queue"""
        try:
            import threading
            import time
            
            def add_tasks_sync():
                # Use a simple Redis connection instead of ResourceAwareQueueManager
                # to avoid Docker networking issues
                import redis
                import json
                import uuid
                
                try:
                    # Direct Redis connection using localhost
                    redis_client = redis.Redis(
                        host='localhost', 
                        port=6379, 
                        decode_responses=True
                    )
                    
                    # Create real-world CPU and GPU intensive test tasks
                    timestamp = int(time.time() * 1000)
                    test_tasks = [
                        # GPU-intensive LLM tasks
                        {
                            'task_id': str(uuid.uuid4()),
                            'task_type': 'plugin_execution',
                            'plugin_name': 'llm_keyword_plugin',
                            'field_name': 'keywords',
                            'field_value': f'A complex science fiction epic spanning multiple galaxies with intricate political systems, advanced alien civilizations, time travel paradoxes, and deep philosophical questions about consciousness and reality in the year 2157 {timestamp}',
                            'media_type': 'movie',
                            'priority': 1,
                            'resource_requirements': {
                                'cpu_cores': 1.0,
                                'gpu_count': 1,
                                'memory_mb': 2048
                            }
                        },
                        {
                            'task_id': str(uuid.uuid4()),
                            'task_type': 'plugin_execution',
                            'plugin_name': 'llm_temporal_intelligence_plugin',
                            'field_name': 'temporal_analysis',
                            'field_value': f'This epic story begins in the distant past of 1892, progresses through the industrial revolution, spans two world wars, covers the space age from 1960-2020, and concludes in a post-apocalyptic future of 2157 where humanity has colonized Mars and Europa {timestamp}',
                            'media_type': 'movie',
                            'priority': 1,
                            'resource_requirements': {
                                'cpu_cores': 1.0,
                                'gpu_count': 1,
                                'memory_mb': 2048
                            }
                        },
                        {
                            'task_id': str(uuid.uuid4()),
                            'task_type': 'plugin_execution', 
                            'plugin_name': 'llm_question_answer_plugin',
                            'field_name': 'plot_analysis',
                            'field_value': f'What are the main themes, character motivations, plot devices, narrative structure, cinematographic techniques, and underlying philosophical messages in this complex multi-generational saga about humanity\'s evolution and our place in the cosmos? {timestamp}',
                            'media_type': 'movie',
                            'priority': 1,
                            'resource_requirements': {
                                'cpu_cores': 1.0,
                                'gpu_count': 1,
                                'memory_mb': 2048
                            }
                        },
                        # CPU-intensive NLP tasks
                        {
                            'task_id': str(uuid.uuid4()),
                            'task_type': 'plugin_execution',
                            'plugin_name': 'heideltime_temporal_plugin',
                            'field_name': 'temporal_extraction',
                            'field_value': f'The film takes place across multiple time periods: starting in ancient Rome in 47 BCE, jumping to the Renaissance in 1503, then to the Victorian era in 1887, through both World Wars (1914-1918 and 1939-1945), the Cold War period from 1947-1991, and finally culminating in the near future of 2157 {timestamp}',
                            'media_type': 'movie',
                            'priority': 0,
                            'resource_requirements': {
                                'cpu_cores': 1.0,
                                'gpu_count': 0,
                                'memory_mb': 1024
                            }
                        },
                        {
                            'task_id': str(uuid.uuid4()),
                            'task_type': 'plugin_execution',
                            'plugin_name': 'gensim_similarity_plugin',
                            'field_name': 'semantic_analysis',
                            'field_value': f'science fiction space opera epic adventure galaxy exploration alien civilization first contact interstellar war political intrigue artificial intelligence consciousness philosophy existentialism time travel parallel universe quantum mechanics {timestamp}',
                            'media_type': 'movie',
                            'priority': 0,
                            'resource_requirements': {
                                'cpu_cores': 1.0,
                                'gpu_count': 0,
                                'memory_mb': 1024
                            }
                        },
                        {
                            'task_id': str(uuid.uuid4()),
                            'task_type': 'plugin_execution',
                            'plugin_name': 'spacy_temporal_plugin',
                            'field_name': 'linguistic_temporal',
                            'field_value': f'This narrative unfolds over centuries, beginning when our protagonists first meet during the signing of the Declaration of Independence in 1776, continuing through their involvement in the California Gold Rush of 1849, their separation during the American Civil War from 1861-1865, their reunion during the construction of the Transcontinental Railroad in 1869, and their final adventure during the Apollo 11 moon landing in 1969 {timestamp}',
                            'media_type': 'movie',
                            'priority': 0,
                            'resource_requirements': {
                                'cpu_cores': 1.0,
                                'gpu_count': 0,
                                'memory_mb': 1024
                            }
                        }
                    ]
                    
                    # Add tasks directly to the Redis queue that the worker monitors
                    for task in test_tasks:
                        # Route LLM tasks to GPU queue, others to CPU queue
                        if 'llm_' in task['plugin_name']:
                            queue_name = "chat:queue:gpu"
                            queue_type = "GPU"
                        else:
                            queue_name = "chat:queue:cpu"
                            queue_type = "CPU"
                        
                        task_json = json.dumps(task)
                        score = time.time()  # Use current timestamp as score
                        redis_client.zadd(queue_name, {task_json: score})
                        logging.info(f"Added {queue_type} task {task['task_id'][:8]}... ({task['plugin_name']}) to {queue_name}")
                    
                    logging.info(f"✅ Added {len(test_tasks)} test tasks to queue")
                        
                except Exception as e:
                    logging.error(f"Error in add_tasks_sync: {e}")
            
            # Run in separate thread to avoid blocking GUI
            thread = threading.Thread(target=add_tasks_sync)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            logging.error(f"Error adding test tasks: {e}")
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Stop monitoring threads
        if hasattr(self, 'resource_thread'):
            self.resource_thread.stop()
        if hasattr(self, 'queue_thread'):
            self.queue_thread.stop()
        
        event.accept()


def main():
    """Main application entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = QueueMonitorApp()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == '__main__':
    main()