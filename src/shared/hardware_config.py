"""
Hardware Configuration System
Allows admins to register and manage available hardware resources for plugin optimization.
Supports distributed resources with URL-based endpoints.
"""

import os
import platform
import psutil
import asyncio
import httpx
from dataclasses import dataclass
from typing import Dict, Optional, List, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator, HttpUrl
import json
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """Types of computational resources."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"


@dataclass
class ResourceEndpoint:
    """Represents a computational resource endpoint."""
    resource_type: ResourceType
    endpoint_url: Optional[str] = None
    capacity: float = 1.0
    model: Optional[str] = None
    health_check_path: Optional[str] = None
    is_available: bool = True
    response_time_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "resource_type": self.resource_type.value,
            "endpoint_url": self.endpoint_url,
            "capacity": self.capacity,
            "model": self.model,
            "health_check_path": self.health_check_path,
            "is_available": self.is_available,
            "response_time_ms": self.response_time_ms
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ResourceEndpoint":
        data["resource_type"] = ResourceType(data["resource_type"])
        return cls(**data)


@dataclass
class HardwareProfile:
    """Represents the available hardware resources across the cluster."""
    local_cpu_cores: int
    local_cpu_threads: int
    local_memory_gb: float
    local_storage_type: str = "hdd"
    

    cpu_endpoints: List[ResourceEndpoint] = None
    gpu_endpoints: List[ResourceEndpoint] = None
    memory_endpoints: List[ResourceEndpoint] = None
    
    is_auto_detected: bool = True
    
    def __post_init__(self):
        if self.cpu_endpoints is None:
            self.cpu_endpoints = []
        if self.gpu_endpoints is None:
            self.gpu_endpoints = []
        if self.memory_endpoints is None:
            self.memory_endpoints = []
    
    def to_dict(self) -> Dict:
        return {
            "local_cpu_cores": self.local_cpu_cores,
            "local_cpu_threads": self.local_cpu_threads,
            "local_memory_gb": self.local_memory_gb,
            "local_storage_type": self.local_storage_type,
            "cpu_endpoints": [ep.to_dict() for ep in self.cpu_endpoints],
            "gpu_endpoints": [ep.to_dict() for ep in self.gpu_endpoints],
            "memory_endpoints": [ep.to_dict() for ep in self.memory_endpoints],
            "is_auto_detected": self.is_auto_detected
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "HardwareProfile":

        data["cpu_endpoints"] = [ResourceEndpoint.from_dict(ep) for ep in data.get("cpu_endpoints", [])]
        data["gpu_endpoints"] = [ResourceEndpoint.from_dict(ep) for ep in data.get("gpu_endpoints", [])]
        data["memory_endpoints"] = [ResourceEndpoint.from_dict(ep) for ep in data.get("memory_endpoints", [])]
        return cls(**data)
    
    def get_total_cpu_capacity(self) -> float:
        """Get total CPU capacity across all endpoints."""
        local_capacity = self.local_cpu_cores
        remote_capacity = sum(ep.capacity for ep in self.cpu_endpoints if ep.is_available)
        return local_capacity + remote_capacity
    
    def get_total_gpu_capacity(self) -> float:
        """Get total GPU capacity across all endpoints."""
        return sum(ep.capacity for ep in self.gpu_endpoints if ep.is_available)
    
    def get_available_gpu_endpoints(self) -> List[ResourceEndpoint]:
        """Get list of available GPU endpoints."""
        return [ep for ep in self.gpu_endpoints if ep.is_available]


class ResourceEndpointModel(BaseModel):
    """Pydantic model for resource endpoint validation."""
    resource_type: ResourceType
    endpoint_url: Optional[str] = Field(None, description="URL endpoint for resource")
    capacity: float = Field(..., ge=0.1, le=1000, description="Resource capacity")
    model: Optional[str] = Field(None, max_length=100, description="Hardware model/name")
    health_check_path: Optional[str] = Field(None, description="Health check endpoint")
    is_available: bool = Field(True, description="Whether resource is available")
    response_time_ms: float = Field(0.0, ge=0.0, le=10000, description="Response time in ms")


class HardwareConfigModel(BaseModel):
    """Pydantic model for hardware configuration validation."""
    local_cpu_cores: int = Field(..., ge=1, le=128, description="Local CPU cores")
    local_cpu_threads: int = Field(..., ge=1, le=256, description="Local CPU threads")
    local_memory_gb: float = Field(..., ge=0.5, le=1024, description="Local RAM in GB")
    local_storage_type: str = Field("hdd", pattern="^(hdd|ssd|nvme)$", description="Local storage type")
    
    cpu_endpoints: List[ResourceEndpointModel] = Field(default_factory=list, description="CPU endpoints")
    gpu_endpoints: List[ResourceEndpointModel] = Field(default_factory=list, description="GPU endpoints")
    memory_endpoints: List[ResourceEndpointModel] = Field(default_factory=list, description="Memory endpoints")
    
    is_auto_detected: bool = Field(True, description="Whether config was auto-detected")
    
    @validator('local_cpu_threads')
    def threads_must_be_valid(cls, v, values):
        if 'local_cpu_cores' in values and v < values['local_cpu_cores']:
            raise ValueError('CPU threads must be >= CPU cores')
        return v


class HardwareDetector:
    """Automatically detects system hardware resources."""
    
    @staticmethod
    def detect_cpu() -> tuple[int, int]:
        """Detect CPU cores and threads."""
        try:
            cores = psutil.cpu_count(logical=False) or 1
            threads = psutil.cpu_count(logical=True) or cores
            return cores, threads
        except Exception as e:
            logger.warning(f"Failed to detect CPU: {e}")
            return 1, 1
    
    @staticmethod
    def detect_memory() -> float:
        """Detect system memory in GB."""
        try:
            memory_bytes = psutil.virtual_memory().total
            return round(memory_bytes / (1024**3), 2)
        except Exception as e:
            logger.warning(f"Failed to detect memory: {e}")
            return 1.0
    
    @staticmethod
    def detect_gpu() -> tuple[int, float, Optional[str]]:
        """Detect GPU information. Returns (count, memory_gb, model)."""
        try:

            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=count,memory.total,name', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_count = len(lines)
                if gpu_count > 0:
                    first_gpu = lines[0].split(', ')
                    memory_mb = int(first_gpu[1])
                    memory_gb = round(memory_mb / 1024, 2)
                    model = first_gpu[2]
                    return gpu_count, memory_gb, model
        except Exception as e:
            logger.debug(f"NVIDIA GPU detection failed: {e}")
        

        
        return 0, 0.0, None
    
    @staticmethod
    def detect_storage_type() -> str:
        """Detect primary storage type."""
        try:

            partitions = psutil.disk_partitions()
            for partition in partitions:
                if partition.mountpoint == "/" or partition.mountpoint == "C:\\":

                    return "ssd"
        except Exception as e:
            logger.debug(f"Storage detection failed: {e}")
        
        return "hdd"
    
    @classmethod
    def auto_detect(cls) -> HardwareProfile:
        """Auto-detect local hardware resources."""
        logger.info("Auto-detecting local hardware resources...")
        
        cores, threads = cls.detect_cpu()
        memory_gb = cls.detect_memory()
        storage_type = cls.detect_storage_type()
        gpu_count, gpu_memory_gb, gpu_model = cls.detect_gpu()
        
        profile = HardwareProfile(
            local_cpu_cores=cores,
            local_cpu_threads=threads,
            local_memory_gb=memory_gb,
            local_storage_type=storage_type,
            is_auto_detected=True
        )
        
        
        if gpu_count > 0:
            gpu_endpoint = ResourceEndpoint(
                resource_type=ResourceType.GPU,
                endpoint_url=None,
                capacity=gpu_memory_gb,
                model=gpu_model,
                is_available=True
            )
            profile.gpu_endpoints.append(gpu_endpoint)
            logger.info(f"Detected local GPU: {gpu_model} ({gpu_memory_gb}GB VRAM)")
        
        logger.info(f"Detected local hardware: {cores}C/{threads}T, {memory_gb}GB RAM, {storage_type} storage")
        
        return profile


class EndpointHealthChecker:
    """Handles health checking for resource endpoints."""
    
    @staticmethod
    async def check_endpoint_health(endpoint: ResourceEndpoint) -> tuple[bool, float]:
        """Check if an endpoint is healthy and measure response time."""
        if endpoint.endpoint_url is None:
            return True, 0.0
        
        try:
            import time
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                health_url = endpoint.endpoint_url
                if endpoint.health_check_path:
                    health_url = f"{endpoint.endpoint_url.rstrip('/')}/{endpoint.health_check_path.lstrip('/')}"
                
                response = await client.get(health_url)
                response_time = (time.time() - start_time) * 1000
                
                is_healthy = response.status_code == 200
                return is_healthy, response_time
                
        except Exception as e:
            logger.warning(f"Health check failed for {endpoint.endpoint_url}: {e}")
            return False, 10000.0
    
    @staticmethod
    async def update_endpoint_health(endpoint: ResourceEndpoint) -> ResourceEndpoint:
        """Update endpoint health status and response time."""
        is_healthy, response_time = await EndpointHealthChecker.check_endpoint_health(endpoint)
        endpoint.is_available = is_healthy
        endpoint.response_time_ms = response_time
        return endpoint


class HardwareConfigManager:
    """Manages hardware configuration storage and retrieval."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("hardware_config.json")
        self._current_profile: Optional[HardwareProfile] = None
        self._lock = asyncio.Lock()
        self.health_checker = EndpointHealthChecker()
    
    async def load_config(self) -> HardwareProfile:
        """Load hardware configuration from file or auto-detect."""
        async with self._lock:
            if self._current_profile is not None:
                return self._current_profile
            
            if self.config_path.exists():
                try:
                    with open(self.config_path, 'r') as f:
                        data = json.load(f)
                    

                    validated = HardwareConfigModel(**data)
                    self._current_profile = HardwareProfile.from_dict(validated.dict())
                    
                    logger.info(f"Loaded hardware config from {self.config_path}")
                    return self._current_profile
                    
                except Exception as e:
                    logger.error(f"Failed to load hardware config: {e}")
                    logger.info("Falling back to auto-detection")
            

            self._current_profile = HardwareDetector.auto_detect()

            await self._save_config_unlocked(self._current_profile)
            return self._current_profile
    
    async def _save_config_unlocked(self, profile: HardwareProfile) -> None:
        """Save hardware configuration to file without acquiring lock."""
        try:

            validated = HardwareConfigModel(**profile.to_dict())
            
            with open(self.config_path, 'w') as f:
                json.dump(validated.dict(), f, indent=2)
            
            self._current_profile = profile
            logger.info(f"Saved hardware config to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save hardware config: {e}")
            raise
    
    async def save_config(self, profile: HardwareProfile) -> None:
        """Save hardware configuration to file."""
        async with self._lock:
            await self._save_config_unlocked(profile)
    
    async def update_config(self, **kwargs) -> HardwareProfile:
        """Update specific hardware configuration values."""
        current = await self.load_config()
        
        
        update_data = current.to_dict()
        update_data.update(kwargs)
        update_data['is_auto_detected'] = False
        
        new_profile = HardwareProfile.from_dict(update_data)
        await self.save_config(new_profile)
        
        return new_profile
    
    async def reset_to_auto_detect(self) -> HardwareProfile:
        """Reset configuration to auto-detected values."""
        async with self._lock:
            self._current_profile = HardwareDetector.auto_detect()
            await self.save_config(self._current_profile)
            return self._current_profile
    
    async def add_endpoint(self, resource_type: ResourceType, endpoint_url: str, 
                          capacity: float, model: Optional[str] = None,
                          health_check_path: Optional[str] = None) -> None:
        """Add a new resource endpoint."""
        profile = await self.load_config()
        
        endpoint = ResourceEndpoint(
            resource_type=resource_type,
            endpoint_url=endpoint_url,
            capacity=capacity,
            model=model,
            health_check_path=health_check_path
        )
        
        
        endpoint = await self.health_checker.update_endpoint_health(endpoint)
        
        
        if resource_type == ResourceType.CPU:
            profile.cpu_endpoints.append(endpoint)
        elif resource_type == ResourceType.GPU:
            profile.gpu_endpoints.append(endpoint)
        elif resource_type == ResourceType.MEMORY:
            profile.memory_endpoints.append(endpoint)
        
        await self.save_config(profile)
        logger.info(f"Added {resource_type.value} endpoint: {endpoint_url}")
    
    async def remove_endpoint(self, resource_type: ResourceType, endpoint_url: str) -> None:
        """Remove a resource endpoint."""
        profile = await self.load_config()
        
        if resource_type == ResourceType.CPU:
            profile.cpu_endpoints = [ep for ep in profile.cpu_endpoints if ep.endpoint_url != endpoint_url]
        elif resource_type == ResourceType.GPU:
            profile.gpu_endpoints = [ep for ep in profile.gpu_endpoints if ep.endpoint_url != endpoint_url]
        elif resource_type == ResourceType.MEMORY:
            profile.memory_endpoints = [ep for ep in profile.memory_endpoints if ep.endpoint_url != endpoint_url]
        
        await self.save_config(profile)
        logger.info(f"Removed {resource_type.value} endpoint: {endpoint_url}")
    
    async def refresh_endpoint_health(self) -> None:
        """Refresh health status for all endpoints."""
        profile = await self.load_config()
        
        
        for endpoint_list in [profile.cpu_endpoints, profile.gpu_endpoints, profile.memory_endpoints]:
            for i, endpoint in enumerate(endpoint_list):
                endpoint_list[i] = await self.health_checker.update_endpoint_health(endpoint)
        
        await self.save_config(profile)
        logger.info("Refreshed endpoint health status")
    
    async def get_resource_limits(self) -> Dict[str, any]:
        """Get resource limits for plugin scheduling."""
        profile = await self.load_config()
        

        total_cpu_capacity = profile.get_total_cpu_capacity()
        total_gpu_capacity = profile.get_total_gpu_capacity()
        available_gpu_endpoints = profile.get_available_gpu_endpoints()
        
        return {
            "local_cpu_cores": profile.local_cpu_cores,
            "local_cpu_threads": profile.local_cpu_threads,
            "local_memory_gb": profile.local_memory_gb * 0.8,
            "total_cpu_capacity": total_cpu_capacity,
            "total_gpu_capacity": total_gpu_capacity,
            "gpu_endpoints": available_gpu_endpoints,
            "gpu_available": len(available_gpu_endpoints) > 0,
            "storage_type": profile.local_storage_type,
            "recommend_cpu_intensive": total_cpu_capacity >= 4,
            "recommend_gpu_intensive": total_gpu_capacity >= 4.0
        }



hardware_config = HardwareConfigManager()


async def get_hardware_profile() -> HardwareProfile:
    """Convenience function to get current hardware profile."""
    return await hardware_config.load_config()


async def get_resource_limits() -> Dict[str, any]:
    """Convenience function to get resource limits."""
    return await hardware_config.get_resource_limits()