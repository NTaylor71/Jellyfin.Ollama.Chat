"""
Base Concept Plugin - Foundation for all concept-related plugins.

Provides hardware-aware provider management, queue integration, and 
resource-based strategy selection for concept expansion plugins.
"""

import asyncio
import logging
from abc import abstractmethod
from typing import Dict, Any, Optional, List, Set, Tuple
from enum import Enum
from datetime import datetime
import json
import uuid

from src.plugins.base import (
    EmbedDataEmbellisherPlugin, PluginResourceRequirements, 
    PluginExecutionContext, PluginMetadata, ExecutionPriority
)
from src.redis_worker.queue_manager import RedisQueueManager
from src.shared.hardware_config import get_resource_limits
from src.concept_expansion.providers.base_provider import BaseProvider, ExpansionRequest
from src.data.cache_manager import get_cache_manager
from src.shared.plugin_contracts import PluginResult, create_field_expansion_result

logger = logging.getLogger(__name__)


class ProcessingStrategy(Enum):
    """Processing strategies based on available resources."""
    LOW_RESOURCE = "low_resource"      # Single provider, basic processing
    MEDIUM_RESOURCE = "medium_resource"  # Dual providers, some parallelism
    HIGH_RESOURCE = "high_resource"     # All providers, full parallelism
    QUEUE_ONLY = "queue_only"          # All processing via queue


class BaseConceptPlugin(EmbedDataEmbellisherPlugin):
    """
    Base class for concept-related plugins with provider management.
    
    Features:
    - Hardware-aware strategy selection
    - Queue integration for expensive operations
    - Provider lifecycle management
    - Result caching and fusion
    """
    
    def __init__(self):
        super().__init__()
        self.providers: Dict[str, BaseProvider] = {}
        self.queue_manager: Optional[RedisQueueManager] = None
        self.cache_manager = get_cache_manager()
        self._processing_strategy: Optional[ProcessingStrategy] = None
        self._hardware_limits: Optional[Dict[str, Any]] = None
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize plugin with providers and queue."""
        try:
            # Initialize base plugin
            self._logger.info(f"Initializing {self.metadata.name}...")
            
            # Initialize queue manager
            await self._initialize_queue()
            
            # Initialize hardware detection
            await self._initialize_hardware()
            
            # Initialize providers
            await self._initialize_providers()
            
            self._is_initialized = True
            self._logger.info(f"{self.metadata.name} initialized successfully")
            return True
            
        except Exception as e:
            self._initialization_error = str(e)
            self._logger.error(f"Failed to initialize {self.metadata.name}: {e}")
            return False
    
    async def _initialize_queue(self) -> None:
        """Initialize Redis queue manager for distributed processing."""
        try:
            self.queue_manager = RedisQueueManager()
            if self.queue_manager.health_check():
                self._logger.info("Queue manager initialized and healthy")
            else:
                self._logger.warning("Queue manager not available - will use direct processing")
                self.queue_manager = None
        except Exception as e:
            self._logger.warning(f"Could not initialize queue manager: {e}")
            self.queue_manager = None
    
    async def _initialize_hardware(self) -> None:
        """Initialize hardware detection for strategy selection."""
        try:
            self._hardware_limits = await get_resource_limits()
            cpu_cores = self._hardware_limits.get("total_cpu_capacity", 1)
            memory_gb = self._hardware_limits.get("local_memory_gb", 1)
            gpu_available = self._hardware_limits.get("gpu_available", False)
            
            self._logger.info(
                f"Hardware detected - CPU: {cpu_cores} cores, "
                f"Memory: {memory_gb}GB, GPU: {gpu_available}"
            )
        except Exception as e:
            self._logger.warning(f"Could not detect hardware: {e}")
            self._hardware_limits = {
                "total_cpu_capacity": 1,
                "local_memory_gb": 1,
                "gpu_available": False
            }
    
    @abstractmethod
    async def _initialize_providers(self) -> None:
        """Initialize specific providers for this plugin. Override in subclasses."""
        pass
    
    def _select_processing_strategy(self, context: PluginExecutionContext) -> ProcessingStrategy:
        """Select processing strategy based on available resources."""
        # Get current resource availability
        cpu_cores = self._hardware_limits.get("total_cpu_capacity", 1)
        memory_gb = self._hardware_limits.get("local_memory_gb", 1)
        gpu_available = self._hardware_limits.get("gpu_available", False)
        queue_available = self.queue_manager is not None
        
        # Override from context if provided
        if context.available_resources:
            cpu_cores = context.available_resources.get("total_cpu_capacity", cpu_cores)
            memory_gb = context.available_resources.get("local_memory_gb", memory_gb)
            gpu_available = context.available_resources.get("gpu_available", gpu_available)
        
        # Strategy selection logic
        if not queue_available and cpu_cores < 2 and memory_gb < 2:
            strategy = ProcessingStrategy.LOW_RESOURCE
        elif cpu_cores >= 4 and memory_gb >= 4 and (gpu_available or queue_available):
            strategy = ProcessingStrategy.HIGH_RESOURCE
        elif cpu_cores >= 2 or queue_available:
            strategy = ProcessingStrategy.MEDIUM_RESOURCE
        else:
            strategy = ProcessingStrategy.LOW_RESOURCE
        
        self._logger.debug(
            f"Selected strategy: {strategy.value} "
            f"(CPU: {cpu_cores}, Memory: {memory_gb}GB, GPU: {gpu_available}, Queue: {queue_available})"
        )
        
        return strategy
    
    async def _queue_task(
        self, 
        task_type: str, 
        data: Dict[str, Any], 
        priority: ExecutionPriority = ExecutionPriority.NORMAL
    ) -> str:
        """Queue a task for distributed processing."""
        if not self.queue_manager:
            raise RuntimeError("Queue manager not available")
        
        task_data = {
            "plugin_id": self.metadata.name,
            "plugin_version": self.metadata.version,
            "task_type": task_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Convert priority enum to numeric value
        priority_map = {
            ExecutionPriority.LOW: -1,
            ExecutionPriority.NORMAL: 0,
            ExecutionPriority.HIGH: 1,
            ExecutionPriority.CRITICAL: 2
        }
        
        task_id = self.queue_manager.enqueue_task(
            task_type=f"plugin_{task_type}",
            data=task_data,
            priority=priority_map.get(priority, 0)
        )
        
        self._logger.debug(f"Queued task {task_id} with type {task_type}")
        return task_id
    
    async def _collect_queue_results(
        self, 
        task_ids: List[str], 
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Collect results from queued tasks."""
        results = {}
        start_time = asyncio.get_event_loop().time()
        
        while len(results) < len(task_ids):
            if asyncio.get_event_loop().time() - start_time > timeout:
                self._logger.warning(f"Timeout collecting queue results. Got {len(results)}/{len(task_ids)}")
                break
            
            for task_id in task_ids:
                if task_id in results:
                    continue
                
                # Check for result (this is simplified - real implementation would use Redis)
                result_key = f"result:{task_id}"
                # In real implementation, check Redis for result
                # For now, we'll simulate with a placeholder
                
            await asyncio.sleep(0.1)  # Brief pause between checks
        
        return results
    
    async def _parallel_provider_execution(
        self,
        providers: List[str],
        concept: str,
        media_context: str,
        max_parallel: int = 3
    ) -> Dict[str, PluginResult]:
        """Execute multiple providers in parallel with concurrency limit."""
        results = {}
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def execute_provider(provider_name: str) -> Tuple[str, Optional[PluginResult]]:
            async with semaphore:
                provider = self.providers.get(provider_name)
                if not provider:
                    return provider_name, None
                
                try:
                    request = ExpansionRequest(
                        concept=concept,
                        media_context=media_context,
                        max_concepts=10
                    )
                    result = await provider.expand_concept(request)
                    return provider_name, result
                except Exception as e:
                    self._logger.error(f"Provider {provider_name} failed: {e}")
                    return provider_name, None
        
        # Execute all providers in parallel
        tasks = [execute_provider(p) for p in providers]
        provider_results = await asyncio.gather(*tasks)
        
        # Collect results
        for provider_name, result in provider_results:
            if result:
                results[provider_name] = result
        
        return results
    
    async def _fuse_provider_results(
        self,
        results: Dict[str, PluginResult],
        concept: str,
        field_name: str = "concept"
    ) -> PluginResult:
        """Fuse results from multiple providers intelligently."""
        # Collect all expanded concepts with confidence scores
        all_concepts = []
        confidence_scores = {}
        
        for provider_name, result in results.items():
            if result and result.success:
                concepts = result.enhanced_data.get("expanded_concepts", [])
                scores = result.confidence_score.per_item
                
                # Add provider weight based on type
                provider_weight = self._get_provider_weight(provider_name)
                
                for concept_item in concepts:
                    if concept_item not in all_concepts:
                        all_concepts.append(concept_item)
                    
                    # Combine confidence scores with provider weight
                    current_score = confidence_scores.get(concept_item, 0)
                    new_score = scores.get(concept_item, 0.5) * provider_weight
                    confidence_scores[concept_item] = max(current_score, new_score)
        
        # Sort by confidence and limit
        sorted_concepts = sorted(
            all_concepts,
            key=lambda c: confidence_scores.get(c, 0),
            reverse=True
        )[:15]
        
        # Create fused result
        return create_field_expansion_result(
            field_name=field_name,
            input_value=concept,
            expansion_result={
                "expanded_concepts": sorted_concepts,
                "original_concept": concept,
                "provider_count": len(results),
                "fusion_method": "confidence_weighted"
            },
            confidence_scores={c: confidence_scores.get(c, 0.5) for c in sorted_concepts},
            plugin_name=self.metadata.name,
            plugin_version=self.metadata.version,
            execution_time_ms=0,  # Will be set by caller
            media_context="movie"
        )
    
    def _get_provider_weight(self, provider_name: str) -> float:
        """Get weight for provider based on reliability/quality."""
        weights = {
            "llm": 1.0,        # Highest - context aware
            "conceptnet": 0.8,  # Good - linguistic relationships
            "gensim": 0.7,     # Statistical similarity
            "temporal": 0.9    # Domain-specific
        }
        return weights.get(provider_name.lower(), 0.5)
    
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        # Clean up providers
        for provider in self.providers.values():
            try:
                await provider.close()
            except Exception as e:
                self._logger.warning(f"Error closing provider: {e}")
        
        self.providers.clear()
        await super().cleanup()