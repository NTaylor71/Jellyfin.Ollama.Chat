"""
Resource Manager - Tracks and manages CPU/GPU/memory resources for queue scheduling.

Provides resource-aware task scheduling to prevent contention and optimize utilization.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be allocated."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"


@dataclass
class ResourceRequirement:
    """Resource requirements for a task."""
    cpu_cores: float = 1.0          # Can be fractional (0.5 = half core)
    gpu_count: int = 0              # Number of GPUs needed (0 or 1 for home setup)
    memory_mb: int = 512            # Memory in MB
    exclusive_gpu: bool = True      # GPU tasks run exclusively (no other tasks)
    estimated_runtime_seconds: int = 30  # For scheduling optimization


@dataclass
class TaskResourceAllocation:
    """Tracks resource allocation for a running task."""
    task_id: str
    requirement: ResourceRequirement
    allocated_at: datetime
    worker_id: str


@dataclass
class ResourcePool:
    """Available and used resources on a worker."""
    # Total available resources
    total_cpu_cores: int
    total_gpus: int
    total_memory_mb: int
    worker_id: str = "default"
    
    # Current usage
    used_cpu_cores: float = 0.0
    used_gpus: int = 0
    used_memory_mb: int = 0
    
    # Task tracking
    running_tasks: Dict[str, TaskResourceAllocation] = field(default_factory=dict)
    
    def can_schedule(self, req: ResourceRequirement) -> bool:
        """
        Check if we can schedule a task with given requirements.
        
        Args:
            req: Resource requirements for the task
            
        Returns:
            True if task can be scheduled immediately
        """
        # Check CPU capacity
        if self.used_cpu_cores + req.cpu_cores > self.total_cpu_cores:
            logger.debug(f"CPU check failed: {self.used_cpu_cores} + {req.cpu_cores} > {self.total_cpu_cores}")
            return False
            
        # Check if exclusive GPU task is running - blocks ALL other tasks
        gpu_tasks_running = any(
            alloc.requirement.gpu_count > 0 and alloc.requirement.exclusive_gpu
            for alloc in self.running_tasks.values()
        )
        if gpu_tasks_running:
            logger.debug("Exclusive GPU task running, cannot schedule any other task")
            return False
        
        # Check GPU capacity
        if req.gpu_count > 0:
            if self.used_gpus + req.gpu_count > self.total_gpus:
                logger.debug(f"GPU count check failed: {self.used_gpus} + {req.gpu_count} > {self.total_gpus}")
                return False
                
            # Exclusive GPU mode - if GPU is requested and exclusive, no other tasks can run
            if req.exclusive_gpu and len(self.running_tasks) > 0:
                logger.debug("GPU exclusive mode: other tasks running, cannot schedule GPU task")
                return False
                
        # Check memory capacity
        if self.used_memory_mb + req.memory_mb > self.total_memory_mb:
            logger.debug(f"Memory check failed: {self.used_memory_mb} + {req.memory_mb} > {self.total_memory_mb}")
            return False
            
        return True
    
    def allocate(self, task_id: str, req: ResourceRequirement) -> TaskResourceAllocation:
        """
        Allocate resources for a task.
        
        Args:
            task_id: Unique task identifier
            req: Resource requirements
            
        Returns:
            TaskResourceAllocation record
            
        Raises:
            ValueError: If resources cannot be allocated
        """
        if not self.can_schedule(req):
            raise ValueError(f"Cannot allocate resources for task {task_id}: insufficient resources")
        
        # Update usage counters
        self.used_cpu_cores += req.cpu_cores
        self.used_gpus += req.gpu_count
        self.used_memory_mb += req.memory_mb
        
        # Create allocation record
        allocation = TaskResourceAllocation(
            task_id=task_id,
            requirement=req,
            allocated_at=datetime.utcnow(),
            worker_id=self.worker_id
        )
        
        self.running_tasks[task_id] = allocation
        
        logger.info(f"Allocated resources for task {task_id}: CPU={req.cpu_cores}, GPU={req.gpu_count}, Mem={req.memory_mb}MB")
        logger.debug(f"Resource usage: CPU={self.used_cpu_cores}/{self.total_cpu_cores}, GPU={self.used_gpus}/{self.total_gpus}, Mem={self.used_memory_mb}/{self.total_memory_mb}")
        
        return allocation
        
    def release(self, task_id: str) -> bool:
        """
        Release resources after task completion.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if resources were released, False if task not found
        """
        if task_id not in self.running_tasks:
            logger.warning(f"Attempted to release resources for unknown task: {task_id}")
            return False
            
        allocation = self.running_tasks[task_id]
        req = allocation.requirement
        
        # Update usage counters
        self.used_cpu_cores -= req.cpu_cores
        self.used_gpus -= req.gpu_count
        self.used_memory_mb -= req.memory_mb
        
        # Remove allocation record
        del self.running_tasks[task_id]
        
        # Ensure no negative values due to floating point errors
        self.used_cpu_cores = max(0.0, self.used_cpu_cores)
        self.used_gpus = max(0, self.used_gpus)
        self.used_memory_mb = max(0, self.used_memory_mb)
        
        runtime_seconds = (datetime.utcnow() - allocation.allocated_at).total_seconds()
        
        logger.info(f"Released resources for task {task_id} after {runtime_seconds:.1f}s")
        logger.debug(f"Resource usage: CPU={self.used_cpu_cores}/{self.total_cpu_cores}, GPU={self.used_gpus}/{self.total_gpus}, Mem={self.used_memory_mb}/{self.total_memory_mb}")
        
        return True
    
    def get_utilization(self) -> Dict[str, float]:
        """Get current resource utilization percentages."""
        return {
            "cpu_utilization": (self.used_cpu_cores / self.total_cpu_cores) * 100 if self.total_cpu_cores > 0 else 0,
            "gpu_utilization": (self.used_gpus / self.total_gpus) * 100 if self.total_gpus > 0 else 0,
            "memory_utilization": (self.used_memory_mb / self.total_memory_mb) * 100 if self.total_memory_mb > 0 else 0,
            "active_tasks": len(self.running_tasks)
        }
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary."""
        utilization = self.get_utilization()
        
        gpu_tasks = [
            alloc for alloc in self.running_tasks.values()
            if alloc.requirement.gpu_count > 0
        ]
        
        cpu_tasks = [
            alloc for alloc in self.running_tasks.values()
            if alloc.requirement.gpu_count == 0
        ]
        
        return {
            "worker_id": self.worker_id,
            "total_resources": {
                "cpu_cores": self.total_cpu_cores,
                "gpus": self.total_gpus,
                "memory_mb": self.total_memory_mb
            },
            "current_usage": {
                "cpu_cores": round(self.used_cpu_cores, 2),
                "gpus": self.used_gpus,
                "memory_mb": self.used_memory_mb
            },
            "utilization": utilization,
            "running_tasks": {
                "total": len(self.running_tasks),
                "cpu_tasks": len(cpu_tasks),
                "gpu_tasks": len(gpu_tasks)
            },
            "task_details": [
                {
                    "task_id": alloc.task_id,
                    "cpu_cores": alloc.requirement.cpu_cores,
                    "gpu_count": alloc.requirement.gpu_count,
                    "memory_mb": alloc.requirement.memory_mb,
                    "runtime_seconds": (datetime.utcnow() - alloc.allocated_at).total_seconds()
                }
                for alloc in self.running_tasks.values()
            ]
        }


class ResourceManager:
    """Manages resource pools across multiple workers."""
    
    def __init__(self):
        self.pools: Dict[str, ResourcePool] = {}
        self._lock = asyncio.Lock()
    
    async def register_worker(self, worker_id: str, pool: ResourcePool):
        """Register a worker's resource pool."""
        async with self._lock:
            pool.worker_id = worker_id
            self.pools[worker_id] = pool
            logger.info(f"Registered worker {worker_id} with {pool.total_cpu_cores} CPU cores, {pool.total_gpus} GPUs, {pool.total_memory_mb}MB memory")
    
    async def unregister_worker(self, worker_id: str):
        """Unregister a worker."""
        async with self._lock:
            if worker_id in self.pools:
                del self.pools[worker_id]
                logger.info(f"Unregistered worker {worker_id}")
    
    async def find_capable_workers(self, req: ResourceRequirement) -> List[str]:
        """Find workers that can schedule the given requirement."""
        async with self._lock:
            capable = []
            for worker_id, pool in self.pools.items():
                if pool.can_schedule(req):
                    capable.append(worker_id)
            return capable
    
    async def get_best_worker(self, req: ResourceRequirement) -> Optional[str]:
        """
        Find the best worker to schedule a task.
        
        Simple strategy: least loaded worker by CPU utilization.
        """
        capable_workers = await self.find_capable_workers(req)
        
        if not capable_workers:
            return None
        
        # Choose worker with lowest CPU utilization
        best_worker = min(
            capable_workers,
            key=lambda w: self.pools[w].get_utilization()["cpu_utilization"]
        )
        
        return best_worker
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of all workers in the cluster."""
        async with self._lock:
            return {
                "total_workers": len(self.pools),
                "workers": {
                    worker_id: pool.get_status_summary()
                    for worker_id, pool in self.pools.items()
                }
            }


def create_resource_pool_from_config(config: Dict[str, Any], worker_id: str = "default") -> ResourcePool:
    """Create a ResourcePool from configuration."""
    return ResourcePool(
        total_cpu_cores=config.get("cpu_cores", 3),
        total_gpus=config.get("gpu_count", 1),
        total_memory_mb=config.get("memory_mb", 8192),
        worker_id=worker_id
    )


def get_plugin_resource_requirements(plugin_name: str) -> ResourceRequirement:
    """
    Get resource requirements for a plugin.
    
    This should eventually be loaded from plugin metadata,
    but for now we'll use reasonable defaults.
    """
    # Default requirements
    default_req = ResourceRequirement(
        cpu_cores=0.5,
        gpu_count=0,
        memory_mb=256,
        exclusive_gpu=False
    )
    
    # Plugin-specific requirements
    requirements_map = {
        # LLM plugins need GPU
        "LLMKeywordPlugin": ResourceRequirement(
            cpu_cores=1.0,
            gpu_count=1,
            memory_mb=2048,
            exclusive_gpu=True,
            estimated_runtime_seconds=45
        ),
        "LLMQuestionAnswerPlugin": ResourceRequirement(
            cpu_cores=1.0,
            gpu_count=1,
            memory_mb=2048,
            exclusive_gpu=True,
            estimated_runtime_seconds=60
        ),
        "LLMTemporalIntelligencePlugin": ResourceRequirement(
            cpu_cores=1.0,
            gpu_count=1,
            memory_mb=2048,
            exclusive_gpu=True,
            estimated_runtime_seconds=45
        ),
        
        # CPU-only plugins
        "ConceptNetKeywordPlugin": ResourceRequirement(
            cpu_cores=0.2,
            gpu_count=0,
            memory_mb=128,
            exclusive_gpu=False,
            estimated_runtime_seconds=10
        ),
        "GensimSimilarityPlugin": ResourceRequirement(
            cpu_cores=0.8,
            gpu_count=0,
            memory_mb=1024,
            exclusive_gpu=False,
            estimated_runtime_seconds=15
        ),
        "SpacyTemporalPlugin": ResourceRequirement(
            cpu_cores=0.6,
            gpu_count=0,
            memory_mb=512,
            exclusive_gpu=False,
            estimated_runtime_seconds=20
        ),
        "HeidelTimeTemporalPlugin": ResourceRequirement(
            cpu_cores=0.4,
            gpu_count=0,
            memory_mb=256,
            exclusive_gpu=False,
            estimated_runtime_seconds=15
        ),
        "SUTimeTemporalPlugin": ResourceRequirement(
            cpu_cores=0.4,
            gpu_count=0,
            memory_mb=256,
            exclusive_gpu=False,
            estimated_runtime_seconds=15
        ),
        "MergeKeywordsPlugin": ResourceRequirement(
            cpu_cores=0.1,
            gpu_count=0,
            memory_mb=64,
            exclusive_gpu=False,
            estimated_runtime_seconds=5
        )
    }
    
    return requirements_map.get(plugin_name, default_req)