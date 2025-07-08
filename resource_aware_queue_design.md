# Resource-Aware Queue System Design

## Problem Statement
Current queue system treats all tasks equally, but reality requires:
- Home machine: 3 CPU tasks max, 1 GPU task max simultaneously
- Future clusters: Multiple CPUs/GPUs with different capacities
- No resource contention between CPU and GPU tasks
- Scalable from single machine to distributed clusters

## Current Architecture Issues

### 1. No Resource Awareness
```python
# Current: Simple priority queue
async def dequeue_task(self, timeout: int = 30):
    result = self.redis_client.bzpopmin(self.settings.REDIS_QUEUE, timeout=timeout)
    # No checking of available resources!
```

### 2. No Task Resource Requirements
```python
# Tasks don't declare resource needs
task_payload = {
    "task_id": task_id,
    "task_type": task_type,
    "data": data,
    # Missing: resource_requirements
}
```

## Proposed Solution: Resource-Aware Queue Manager

### 1. Resource Pool Architecture

```python
# src/redis_worker/resource_manager.py
from dataclasses import dataclass
from typing import Dict, List, Optional
import asyncio
from enum import Enum

class ResourceType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"

@dataclass
class ResourceRequirement:
    """What a task needs to run."""
    cpu_cores: float = 1.0  # Can be fractional
    gpu_count: int = 0      # 0 or 1 for now
    memory_mb: int = 512
    exclusive_gpu: bool = True  # GPU tasks run alone

@dataclass 
class ResourcePool:
    """Available resources on this worker."""
    total_cpu_cores: int
    total_gpus: int
    total_memory_mb: int
    
    # Current usage
    used_cpu_cores: float = 0.0
    used_gpus: int = 0
    used_memory_mb: int = 0
    
    # Running tasks
    running_tasks: Dict[str, ResourceRequirement] = None
    
    def __post_init__(self):
        if self.running_tasks is None:
            self.running_tasks = {}
    
    def can_schedule(self, req: ResourceRequirement) -> bool:
        """Check if we can run this task now."""
        # Check CPU
        if self.used_cpu_cores + req.cpu_cores > self.total_cpu_cores:
            return False
            
        # Check GPU
        if req.gpu_count > 0:
            if self.used_gpus + req.gpu_count > self.total_gpus:
                return False
            # Exclusive GPU mode - no other tasks when GPU in use
            if req.exclusive_gpu and len(self.running_tasks) > 0:
                return False
                
        # Check Memory
        if self.used_memory_mb + req.memory_mb > self.total_memory_mb:
            return False
            
        return True
    
    def allocate(self, task_id: str, req: ResourceRequirement):
        """Allocate resources for a task."""
        self.used_cpu_cores += req.cpu_cores
        self.used_gpus += req.gpu_count
        self.used_memory_mb += req.memory_mb
        self.running_tasks[task_id] = req
        
    def release(self, task_id: str):
        """Release resources after task completion."""
        if task_id in self.running_tasks:
            req = self.running_tasks[task_id]
            self.used_cpu_cores -= req.cpu_cores
            self.used_gpus -= req.gpu_count
            self.used_memory_mb -= req.memory_mb
            del self.running_tasks[task_id]
```

### 2. Enhanced Task Definition

```python
# Update task payload
task_payload = {
    "task_id": task_id,
    "task_type": task_type,
    "data": data,
    "resource_requirements": {
        "cpu_cores": 1.0,
        "gpu_count": 0,
        "memory_mb": 512,
        "exclusive_gpu": True
    },
    "priority": priority,
    # ... rest of fields
}
```

### 3. Resource-Aware Queue Manager

```python
# src/redis_worker/resource_queue_manager.py
import redis.asyncio as redis
from typing import Optional, Dict, Any, List

class ResourceAwareQueueManager(RedisQueueManager):
    """Queue manager that respects resource constraints."""
    
    def __init__(self, resource_pool: ResourcePool):
        super().__init__()
        self.resource_pool = resource_pool
        self.pending_queue = "queue:pending"
        self.gpu_queue = "queue:gpu"
        self.cpu_queue = "queue:cpu"
        
    async def enqueue_task(self, task_type: str, data: Dict[str, Any], 
                          resource_req: ResourceRequirement, priority: int = 0) -> str:
        """Enqueue with resource requirements."""
        task_id = str(uuid.uuid4())
        
        task_payload = {
            "task_id": task_id,
            "task_type": task_type,
            "data": data,
            "resource_requirements": {
                "cpu_cores": resource_req.cpu_cores,
                "gpu_count": resource_req.gpu_count,
                "memory_mb": resource_req.memory_mb,
                "exclusive_gpu": resource_req.exclusive_gpu
            },
            "priority": priority,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Route to appropriate queue
        if resource_req.gpu_count > 0:
            queue_name = self.gpu_queue
        else:
            queue_name = self.cpu_queue
            
        score = priority + (int(datetime.utcnow().timestamp()) / 1000000)
        
        await self.redis_client.zadd(
            queue_name,
            {json.dumps(task_payload): score}
        )
        
        return task_id
    
    async def dequeue_task(self, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Dequeue task that fits available resources."""
        end_time = time.time() + timeout
        
        while time.time() < end_time:
            # Try GPU queue first (usually higher priority)
            task = await self._try_dequeue_from_queue(self.gpu_queue)
            if task and self._can_run_task(task):
                return task
                
            # Try CPU queue
            task = await self._try_dequeue_from_queue(self.cpu_queue)
            if task and self._can_run_task(task):
                return task
                
            # No runnable tasks, wait a bit
            await asyncio.sleep(0.5)
            
        return None
    
    async def _try_dequeue_from_queue(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """Try to get a task from specific queue without blocking."""
        # Peek at top tasks without removing
        tasks = await self.redis_client.zrange(queue_name, 0, 10, withscores=False)
        
        for task_json in tasks:
            task = json.loads(task_json)
            req = ResourceRequirement(**task["resource_requirements"])
            
            if self.resource_pool.can_schedule(req):
                # Remove from queue
                await self.redis_client.zrem(queue_name, task_json)
                return task
                
        return None
    
    def _can_run_task(self, task: Dict[str, Any]) -> bool:
        """Check if task can run with current resources."""
        req = ResourceRequirement(**task["resource_requirements"])
        return self.resource_pool.can_schedule(req)
```

### 4. Worker Integration

```python
# src/redis_worker/main.py updates
class WorkerService:
    def __init__(self):
        # ... existing init ...
        
        # Initialize resource pool from config
        self.resource_pool = ResourcePool(
            total_cpu_cores=self.settings.WORKER_CPU_CORES,  # Default: 3
            total_gpus=self.settings.WORKER_GPU_COUNT,      # Default: 1
            total_memory_mb=self.settings.WORKER_MEMORY_MB  # Default: 8192
        )
        
        # Use resource-aware queue manager
        self.queue_manager = ResourceAwareQueueManager(self.resource_pool)
        
    async def process_task(self, task_data: Dict[str, Any]) -> None:
        """Process task with resource management."""
        task_id = task_data.get("task_id")
        
        # Allocate resources
        req = ResourceRequirement(**task_data.get("resource_requirements", {}))
        self.resource_pool.allocate(task_id, req)
        
        try:
            # ... existing task processing ...
            await self._process_task_internal(task_data)
            
        finally:
            # Always release resources
            self.resource_pool.release(task_id)
```

### 5. Plugin Resource Declaration

```python
# src/plugins/http_base.py
class HTTPBasePlugin(BasePlugin):
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        """Default for HTTP plugins - CPU only."""
        return PluginResourceRequirements(
            min_cpu_cores=0.1,
            preferred_cpu_cores=0.5,
            min_memory_mb=25.0,
            preferred_memory_mb=100.0,
            requires_gpu=False,
            max_execution_time_seconds=30.0
        )

# src/plugins/field_enrichment/llm_keyword_plugin.py  
class LLMKeywordPlugin(HTTPBasePlugin):
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        """LLM needs GPU."""
        return PluginResourceRequirements(
            min_cpu_cores=1.0,
            preferred_cpu_cores=2.0,
            min_memory_mb=2048.0,
            preferred_memory_mb=4096.0,
            requires_gpu=True,
            gpu_memory_mb=4096.0,
            exclusive_gpu=True,  # Don't share GPU
            max_execution_time_seconds=60.0
        )
```

### 6. Configuration for Different Environments

```yaml
# config/worker_profiles.yaml
profiles:
  home_machine:
    cpu_cores: 3
    gpu_count: 1
    memory_mb: 8192
    scheduling:
      max_cpu_tasks: 3
      max_gpu_tasks: 1
      gpu_exclusive: true
      
  small_cluster:
    workers:
      - name: cpu_worker_1
        cpu_cores: 8
        gpu_count: 0
        memory_mb: 16384
      - name: gpu_worker_1
        cpu_cores: 4
        gpu_count: 1
        memory_mb: 16384
        
  large_cluster:
    workers:
      - name: cpu_pool
        count: 10
        cpu_cores: 16
        gpu_count: 0
        memory_mb: 32768
      - name: gpu_pool
        count: 4
        cpu_cores: 8
        gpu_count: 2
        memory_mb: 65536
```

### 7. Dynamic Scaling Support

```python
# src/redis_worker/cluster_manager.py
class ClusterManager:
    """Manages multiple workers in a cluster."""
    
    async def register_worker(self, worker_id: str, resources: ResourcePool):
        """Register a worker with the cluster."""
        await self.redis_client.hset(
            "cluster:workers",
            worker_id,
            json.dumps({
                "resources": resources.__dict__,
                "last_heartbeat": datetime.utcnow().isoformat()
            })
        )
    
    async def get_cluster_resources(self) -> Dict[str, ResourcePool]:
        """Get all available resources in cluster."""
        workers = await self.redis_client.hgetall("cluster:workers")
        return {
            worker_id: ResourcePool(**json.loads(data)["resources"])
            for worker_id, data in workers.items()
        }
    
    async def route_task_to_worker(self, task: Dict[str, Any]) -> Optional[str]:
        """Find best worker for task."""
        req = ResourceRequirement(**task["resource_requirements"])
        cluster_resources = await self.get_cluster_resources()
        
        # Find workers that can run this task
        capable_workers = [
            worker_id for worker_id, pool in cluster_resources.items()
            if pool.can_schedule(req)
        ]
        
        if not capable_workers:
            return None
            
        # Simple strategy: least loaded worker
        return min(capable_workers, 
                  key=lambda w: cluster_resources[w].used_cpu_cores)
```

## Implementation Plan

### Phase 1: Single Machine (Your Home Setup)
1. Implement ResourcePool class
2. Update queue manager to be resource-aware
3. Add resource requirements to tasks
4. Test with 3 CPU / 1 GPU limits

### Phase 2: Multi-Worker Support
1. Add worker registration
2. Implement cluster manager
3. Add task routing between workers
4. Test with multiple worker processes

### Phase 3: Distributed Clusters
1. Add Kubernetes operator
2. Implement auto-scaling
3. Add resource monitoring
4. Support heterogeneous clusters

## Benefits

1. **No GPU contention** - Only 1 GPU task at a time
2. **Optimal CPU usage** - Run up to 3 CPU tasks in parallel
3. **Memory safety** - Prevent OOM by tracking memory
4. **Future-proof** - Scales from 1 machine to clusters
5. **Smart scheduling** - Tasks wait for resources instead of failing
6. **Observable** - Know exactly what's running where

## Example Usage

```python
# Home machine config
worker = WorkerService()
worker.resource_pool = ResourcePool(
    total_cpu_cores=3,
    total_gpus=1, 
    total_memory_mb=8192
)

# Submit CPU task
await queue_manager.enqueue_task(
    task_type="concept_expansion",
    data={"concept": "action movie"},
    resource_req=ResourceRequirement(cpu_cores=0.5, memory_mb=256)
)

# Submit GPU task  
await queue_manager.enqueue_task(
    task_type="llm_analysis",
    data={"prompt": "Analyze this plot"},
    resource_req=ResourceRequirement(cpu_cores=1, gpu_count=1, memory_mb=4096)
)

# Worker automatically schedules based on available resources
```

This design gives you exactly what you need:
- Respects your 3 CPU / 1 GPU limits
- Prevents resource conflicts
- Scales to larger systems without code changes
- Works with existing async architecture