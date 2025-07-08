"""
Concept Expansion Plugin - Orchestrates concept expansion using multiple providers.

Uses ConceptNet, LLM, and Gensim providers with hardware-aware strategy selection
and queue integration for expensive operations.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime

from src.plugins.base_concept import BaseConceptPlugin, ProcessingStrategy
from src.plugins.base import (
    PluginMetadata, PluginResourceRequirements, PluginExecutionContext,
    PluginExecutionResult, PluginType, ExecutionPriority
)
from src.concept_expansion.providers.conceptnet_provider import ConceptNetProvider
from src.concept_expansion.providers.llm.llm_provider import LLMProvider
from src.concept_expansion.providers.gensim_provider import GensimProvider
from src.concept_expansion.providers.base_provider import ExpansionRequest
from src.shared.plugin_contracts import PluginResult, CacheType
from src.shared.text_utils import extract_key_concepts
from src.data.cache_manager import CacheStrategy
from src.shared.media_field_config import get_media_field_config, detect_media_type_from_data

logger = logging.getLogger(__name__)


class ConceptExpansionPlugin(BaseConceptPlugin):
    """
    Plugin that orchestrates concept expansion using multiple providers.
    
    Features:
    - Uses ConceptNet for linguistic relationships (fast, cheap)
    - Uses LLM for semantic understanding (slow, expensive - queued)
    - Uses Gensim for statistical similarity (medium speed)
    - Hardware-aware strategy selection
    - Intelligent result fusion
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="ConceptExpansionPlugin",
            version="2.0.0",
            description="Orchestrates multi-provider concept expansion with queue integration",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["concept", "expansion", "nlp", "semantic"],
            execution_priority=ExecutionPriority.HIGH
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        return PluginResourceRequirements(
            min_cpu_cores=1.0,
            preferred_cpu_cores=4.0,
            min_memory_mb=200.0,
            preferred_memory_mb=1000.0,
            requires_gpu=False,  # Benefits from Ollama GPU if available
            max_execution_time_seconds=30.0,
            can_use_distributed_resources=True  # Can use queue system
        )
    
    async def _initialize_providers(self) -> None:
        """Initialize concept expansion providers."""
        # Initialize ConceptNet provider (lightweight)
        self.providers["conceptnet"] = ConceptNetProvider()
        await self.providers["conceptnet"].initialize()
        
        # Initialize LLM provider (heavyweight)
        self.providers["llm"] = LLMProvider()
        await self.providers["llm"].initialize()
        
        # Initialize Gensim provider (medium weight)
        self.providers["gensim"] = GensimProvider()
        await self.providers["gensim"].initialize()
        
        self._logger.info(f"Initialized {len(self.providers)} concept providers")
    
    async def embellish_embed_data(
        self, 
        data: Dict[str, Any], 
        context: PluginExecutionContext
    ) -> Dict[str, Any]:
        """
        Enhance data with expanded concepts using appropriate strategy.
        """
        start_time = datetime.now()
        
        try:
            # Select processing strategy based on resources
            strategy = self._select_processing_strategy(context)
            
            # Extract concepts from data
            concepts = await self._extract_concepts_from_data(data)
            if not concepts:
                self._logger.debug("No concepts found to expand")
                return data
            
            self._logger.info(
                f"Expanding {len(concepts)} concepts using {strategy.value} strategy"
            )
            
            # Execute expansion based on strategy
            if strategy == ProcessingStrategy.HIGH_RESOURCE:
                expanded_concepts = await self._high_resource_expansion(concepts, context)
            elif strategy == ProcessingStrategy.MEDIUM_RESOURCE:
                expanded_concepts = await self._medium_resource_expansion(concepts, context)
            elif strategy == ProcessingStrategy.LOW_RESOURCE:
                expanded_concepts = await self._low_resource_expansion(concepts, context)
            else:  # QUEUE_ONLY
                expanded_concepts = await self._queue_only_expansion(concepts, context)
            
            # Add expanded concepts to data
            if "enhanced_fields" not in data:
                data["enhanced_fields"] = {}
            
            data["enhanced_fields"]["expanded_concepts"] = expanded_concepts
            data["enhanced_fields"]["expansion_metadata"] = {
                "strategy": strategy.value,
                "concept_count": len(concepts),
                "expanded_count": sum(len(v) for v in expanded_concepts.values()),
                "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000
            }
            
            self._logger.info(
                f"Concept expansion complete: {len(concepts)} → "
                f"{sum(len(v) for v in expanded_concepts.values())} concepts"
            )
            
            return data
            
        except Exception as e:
            self._logger.error(f"Concept expansion failed: {e}")
            return data  # Return original data on error
    
    async def _extract_concepts_from_data(self, data: Dict[str, Any]) -> List[str]:
        """Extract key concepts from data using media-type-aware configuration."""
        try:
            # Detect media type from data
            media_type = detect_media_type_from_data(data)
            
            # Get field configuration manager
            config_manager = await get_media_field_config()
            
            # Extract concepts using configured rules
            concepts = config_manager.extract_concepts_from_data(data, media_type)
            
            self._logger.debug(f"Extracted {len(concepts)} concepts for {media_type.value}")
            return concepts
            
        except Exception as e:
            self._logger.error(f"Failed to extract concepts using media config: {e}")
            # Fallback to basic extraction
            return self._fallback_concept_extraction(data)
    
    def _fallback_concept_extraction(self, data: Dict[str, Any]) -> List[str]:
        """Fallback concept extraction when config system fails."""
        concepts = set()
        
        # Generic text fields that might exist in any media type
        for key, value in data.items():
            if isinstance(value, str) and len(value.strip()) > 0:
                if key.lower() in ['name', 'title', 'description', 'overview', 'summary']:
                    field_concepts = extract_key_concepts(value)
                    concepts.update(field_concepts[:3])
            elif isinstance(value, list):
                for item in value[:5]:
                    if isinstance(item, str):
                        concepts.add(item.lower().strip())
                    elif isinstance(item, dict):
                        # Try common name fields in dictionaries
                        for name_field in ['Name', 'name', 'title', 'Title', 'label', 'Label']:
                            if name_field in item and isinstance(item[name_field], str):
                                concepts.add(item[name_field].lower().strip())
                                break
        
        return list(concepts)[:10]
    
    async def _high_resource_expansion(
        self, 
        concepts: List[str], 
        context: PluginExecutionContext
    ) -> Dict[str, List[str]]:
        """
        High resource strategy: Use all providers with parallelism and queuing.
        """
        expanded_concepts = {}
        
        # Determine parallelism level
        cpu_cores = self._hardware_limits.get("total_cpu_capacity", 4)
        max_parallel = min(cpu_cores, 6)
        
        # Split concepts between direct and queued processing
        expensive_concepts = concepts[:5]  # LLM processing (expensive)
        cheap_concepts = concepts[:10]     # ConceptNet/Gensim (cheap)
        
        # Queue LLM expansions for expensive concepts
        llm_tasks = []
        if self.queue_manager and self.providers.get("llm"):
            for concept in expensive_concepts:
                task_data = {
                    "provider": "llm",
                    "concept": concept,
                    "media_context": "movie",
                    "max_concepts": 10
                }
                task_id = await self._queue_task(
                    "concept_expansion",
                    task_data,
                    ExecutionPriority.HIGH
                )
                llm_tasks.append((concept, task_id))
                self._logger.debug(f"Queued LLM expansion for '{concept}'")
        
        # Parallel direct processing for cheap providers
        direct_results = await self._parallel_provider_execution(
            ["conceptnet", "gensim"],
            cheap_concepts[0],  # Example concept
            "movie",
            max_parallel
        )
        
        # Process remaining concepts in batches
        for i in range(0, len(cheap_concepts), max_parallel):
            batch = cheap_concepts[i:i + max_parallel]
            batch_tasks = []
            
            for concept in batch:
                # ConceptNet expansion
                if "conceptnet" in self.providers:
                    batch_tasks.append(
                        self._expand_with_provider("conceptnet", concept, "movie")
                    )
                # Gensim expansion
                if "gensim" in self.providers:
                    batch_tasks.append(
                        self._expand_with_provider("gensim", concept, "movie")
                    )
            
            # Execute batch in parallel
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process batch results
            for j, concept in enumerate(batch):
                concept_expansions = []
                
                # Collect expansions from batch results
                for result in batch_results[j*2:(j+1)*2]:  # 2 providers per concept
                    if isinstance(result, Exception):
                        self._logger.warning(f"Provider failed for '{concept}': {result}")
                        continue
                    if result and result.success:
                        expansions = result.enhanced_data.get("expanded_concepts", [])
                        concept_expansions.extend(expansions)
                
                # Deduplicate and limit
                expanded_concepts[concept] = list(dict.fromkeys(concept_expansions))[:10]
        
        # Collect queued LLM results
        if llm_tasks:
            self._logger.info(f"Waiting for {len(llm_tasks)} queued LLM expansions...")
            # In real implementation, this would poll Redis for results
            # For now, we'll do direct LLM calls as fallback
            for concept, task_id in llm_tasks:
                try:
                    llm_result = await self._expand_with_provider("llm", concept, "movie")
                    if llm_result and llm_result.success:
                        expanded_concepts[concept] = llm_result.enhanced_data.get(
                            "expanded_concepts", []
                        )[:10]
                except Exception as e:
                    self._logger.warning(f"LLM expansion failed for '{concept}': {e}")
        
        return expanded_concepts
    
    async def _medium_resource_expansion(
        self, 
        concepts: List[str], 
        context: PluginExecutionContext
    ) -> Dict[str, List[str]]:
        """
        Medium resource strategy: ConceptNet + selective LLM, limited parallelism.
        """
        expanded_concepts = {}
        
        # Process with ConceptNet (cheap, fast)
        for concept in concepts[:10]:
            try:
                result = await self._expand_with_provider("conceptnet", concept, "movie")
                if result and result.success:
                    expanded_concepts[concept] = result.enhanced_data.get(
                        "expanded_concepts", []
                    )[:8]
            except Exception as e:
                self._logger.warning(f"ConceptNet failed for '{concept}': {e}")
        
        # Selective LLM for top concepts only
        top_concepts = concepts[:3]
        for concept in top_concepts:
            if self.queue_manager:
                # Queue if available
                task_data = {
                    "provider": "llm",
                    "concept": concept,
                    "media_context": "movie"
                }
                await self._queue_task("concept_expansion", task_data)
            else:
                # Direct call if no queue
                try:
                    result = await self._expand_with_provider("llm", concept, "movie")
                    if result and result.success:
                        # Merge with existing
                        existing = expanded_concepts.get(concept, [])
                        new_concepts = result.enhanced_data.get("expanded_concepts", [])
                        merged = list(dict.fromkeys(existing + new_concepts))[:10]
                        expanded_concepts[concept] = merged
                except Exception as e:
                    self._logger.warning(f"LLM failed for '{concept}': {e}")
        
        return expanded_concepts
    
    async def _low_resource_expansion(
        self, 
        concepts: List[str], 
        context: PluginExecutionContext
    ) -> Dict[str, List[str]]:
        """
        Low resource strategy: ConceptNet only, sequential processing.
        """
        expanded_concepts = {}
        
        # Limit concepts for low resource
        for concept in concepts[:8]:
            try:
                result = await self._expand_with_provider("conceptnet", concept, "movie")
                if result and result.success:
                    expanded_concepts[concept] = result.enhanced_data.get(
                        "expanded_concepts", []
                    )[:5]
            except Exception as e:
                self._logger.warning(f"ConceptNet failed for '{concept}': {e}")
        
        return expanded_concepts
    
    async def _queue_only_expansion(
        self, 
        concepts: List[str], 
        context: PluginExecutionContext
    ) -> Dict[str, List[str]]:
        """
        Queue-only strategy: All processing via queue (minimal local resources).
        """
        if not self.queue_manager:
            # Fallback to low resource if no queue
            return await self._low_resource_expansion(concepts, context)
        
        expanded_concepts = {}
        task_map = {}
        
        # Queue all concept expansions
        for concept in concepts[:10]:
            task_data = {
                "provider": "conceptnet",  # Use cheapest provider
                "concept": concept,
                "media_context": "movie"
            }
            task_id = await self._queue_task(
                "concept_expansion",
                task_data,
                ExecutionPriority.NORMAL
            )
            task_map[task_id] = concept
        
        # Wait for results (with timeout)
        # In real implementation, this would poll Redis
        self._logger.info(f"Queued {len(task_map)} concept expansions")
        
        # For now, return empty - real implementation would collect results
        return expanded_concepts
    
    async def _expand_with_provider(
        self,
        provider_name: str,
        concept: str,
        media_context: str
    ) -> Optional[PluginResult]:
        """Expand a concept using a specific provider."""
        provider = self.providers.get(provider_name)
        if not provider:
            self._logger.warning(f"Provider '{provider_name}' not available")
            return None
        
        try:
            request = ExpansionRequest(
                concept=concept,
                media_context=media_context,
                max_concepts=10
            )
            
            # Check cache first
            cache_key = self.cache_manager.generate_cache_key(
                cache_type=self._provider_to_cache_type(provider_name),
                field_name="concept",
                input_value=concept,
                media_context=media_context
            )
            
            # Use cache manager's get_or_compute
            result = await self.cache_manager.get_or_compute(
                cache_key=cache_key,
                compute_func=lambda: provider.expand_concept(request),
                strategy=CacheStrategy.CACHE_FIRST
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Provider '{provider_name}' failed for '{concept}': {e}")
            return None
    
    def _provider_to_cache_type(self, provider_name: str) -> CacheType:
        """Map provider name to cache type."""
        mapping = {
            "conceptnet": CacheType.CONCEPTNET,
            "llm": CacheType.LLM_CONCEPT,
            "gensim": CacheType.GENSIM_SIMILARITY
        }
        return mapping.get(provider_name, CacheType.CUSTOM)