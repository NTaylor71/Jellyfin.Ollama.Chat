# Queue & Resource Monitor App

A PyQt6-based GUI application for monitoring Redis queue status and system resource usage in real-time.

## Features

- **Real-time Queue Monitor**: View pending, running, and completed tasks
- **Resource Usage Monitor**: Track CPU, Memory, GPU, and Disk usage with progress bars
- **Dark Theme**: Professional dark UI theme
- **Test Task Injection**: Add fake test tasks to the queue for testing
- **Responsive Design**: Resizable panels and tables

## Installation

### Using uv (Recommended)

```bash
# Install GUI dependencies
uv add PyQt6>=6.4.0 GPUtil>=1.4.0

# Or install all local dependencies including GUI
uv add production-rag-system[local]
```

### Using pip

```bash
pip install PyQt6>=6.4.0 GPUtil>=1.4.0
```

## Usage

### Quick Start

```bash
cd app
python launch.py
```

### Direct Run

```bash
cd app
python main.py
```

### From Project Root

```bash
python -m app.main
```

## Application Components

### Queue Monitor Panel (Left)

- **Task Table**: Shows all queue tasks with:
  - Task ID (truncated for display)
  - Status (pending/running/completed/error)
  - Plugin name
  - Creation timestamp
  - Progress percentage
  - Error messages (if any)

- **Queue Statistics**: Real-time counts of total, pending, and running tasks

### Resource Monitor Panel (Right)

- **CPU Usage**: Real-time CPU utilization percentage
- **Memory Usage**: System memory usage percentage
- **GPU Usage**: GPU utilization (if available)
- **GPU Memory**: GPU memory usage (if available)
- **Disk Usage**: Root filesystem usage percentage

### Control Panel (Bottom)

- **Add Test Tasks**: Inject fake test tasks into the queue
- **Exit**: Close the application

## Configuration

The app automatically connects to:
- Redis server (configured in `src/shared/config.py`)
- System resources via `psutil` and `GPUtil`

## Dependencies

### Core Dependencies
- `PyQt6>=6.4.0`: GUI framework
- `psutil`: System resource monitoring
- `GPUtil>=1.4.0`: GPU monitoring
- `redis>=5.0.0`: Redis queue connection

### Project Dependencies
- `src.worker.queue_manager`: Queue management
- `src.worker.resource_manager`: Resource management
- `src.shared.config`: Configuration management

## Architecture

### Threading Model

The application uses a multi-threaded architecture:

1. **Main Thread**: UI updates and user interactions
2. **Resource Monitor Thread**: Polls system resources every 1 second
3. **Queue Monitor Thread**: Polls Redis queue every 2 seconds

### Data Flow

```
Redis Queue ← QueueMonitorThread ← Main UI
System Resources ← ResourceMonitorThread ← Main UI
```

## Troubleshooting

### Common Issues

1. **"No module named 'PyQt6'"**
   - Solution: Install PyQt6 using `uv add PyQt6>=6.4.0`

2. **"Failed to connect to Redis"**
   - Solution: Ensure Redis server is running and accessible
   - Check configuration in `src/shared/config.py`

3. **"GPU monitoring not available"**
   - Solution: Install `GPUtil` and ensure GPU drivers are installed
   - GPU monitoring will gracefully fall back to 0% if unavailable

4. **"Permission denied" errors**
   - Solution: Ensure the application has permission to access system resources

### Debug Mode

To enable debug logging, modify the logging level in `main.py`:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Development

### Adding New Features

1. **New Resource Monitors**: Add to `ResourceMonitorThread.run()`
2. **New Queue Information**: Modify `QueueMonitorThread._get_queue_tasks()`
3. **UI Enhancements**: Update `QueueMonitorApp.setup_ui()`

### Testing

```bash
# Test with fake tasks
python main.py
# Click "Add Test Tasks" button
```

## Performance

- **Memory Usage**: ~50-100MB depending on queue size
- **CPU Usage**: <1% when idle, <5% when actively monitoring
- **Update Frequency**: 
  - Resources: 1 second
  - Queue: 2 seconds

## License

Part of the Universal Media Framework project.