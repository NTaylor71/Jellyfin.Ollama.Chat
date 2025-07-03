"""
Query Expander Plugin - Starter Example
Demonstrates adaptive query expansion that scales to available hardware.
"""

import asyncio
import re
from typing import Dict, Any, List

from ..base import (
    QueryEmbellisherPlugin, PluginResourceRequirements, PluginExecutionContext,
    plugin_decorator, PluginType, ExecutionPriority
)


@plugin_decorator(
    name="AdaptiveQueryExpander",
    version="1.0.0",
    description="Expands queries with synonyms and related terms, adapting to available CPU cores",
    author="System",
    plugin_type=PluginType.QUERY_EMBELLISHER,
    resource_requirements=PluginResourceRequirements(
        min_cpu_cores=1.0,
        preferred_cpu_cores=4.0,
        min_memory_mb=50.0,
        preferred_memory_mb=200.0,
        max_execution_time_seconds=5.0,
        can_use_distributed_resources=True
    ),
    execution_priority=ExecutionPriority.NORMAL,
    tags=["query", "expansion", "cpu-optimized"]
)
class AdaptiveQueryExpanderPlugin(QueryEmbellisherPlugin):
    """Example query expander that adapts to available CPU cores."""
    
    def __init__(self):
        super().__init__()
        self.synonym_cache: Dict[str, List[str]] = {}
        self.expansion_patterns = [
            # Common expansions
            (r'\bmovie\b', ['film', 'cinema', 'picture']),
            (r'\bshow\b', ['series', 'program', 'episode']),
            (r'\bmusic\b', ['song', 'audio', 'track']),
            (r'\bbook\b', ['novel', 'text', 'publication']),
            (r'\bgame\b', ['gaming', 'video game', 'play']),
        ]
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the query expander plugin."""
        try:
            self._logger.info("Initializing AdaptiveQueryExpander plugin")
            
            # Load additional patterns from config if provided
            additional_patterns = config.get('expansion_patterns', [])
            self.expansion_patterns.extend(additional_patterns)
            
            # Build synonym cache
            await self._build_synonym_cache()
            
            self._is_initialized = True
            self._logger.info("AdaptiveQueryExpander plugin initialized successfully")
            return True
            
        except Exception as e:
            self._initialization_error = str(e)
            self._logger.error(f"Failed to initialize AdaptiveQueryExpander: {e}")
            return False
    
    async def _build_synonym_cache(self) -> None:
        """Build synonym cache from expansion patterns."""
        for pattern, synonyms in self.expansion_patterns:
            # Extract the base word from regex pattern
            base_word = re.sub(r'[\\^$.*+?{}()|[\]]', '', pattern).replace('b', '')
            self.synonym_cache[base_word] = synonyms
    
    async def embellish_query(self, query: str, context: PluginExecutionContext) -> str:
        """Embellish query with adaptive expansion based on available resources."""
        try:
            # Get available CPU capacity
            available_resources = context.available_resources or {}
            cpu_capacity = available_resources.get('total_cpu_capacity', 1.0)
            
            # Adapt expansion strategy based on CPU capacity
            if cpu_capacity >= 8:
                # High-performance expansion
                return await self._advanced_expansion(query, context)
            elif cpu_capacity >= 4:
                # Medium expansion
                return await self._standard_expansion(query, context)
            else:
                # Basic expansion for limited resources
                return await self._basic_expansion(query, context)
                
        except Exception as e:
            self._logger.error(f"Error in query embellishment: {e}")
            return query  # Return original query on error
    
    async def _basic_expansion(self, query: str, context: PluginExecutionContext) -> str:
        """Basic query expansion for limited CPU resources."""
        expanded_terms = []
        words = query.lower().split()
        
        for word in words:
            expanded_terms.append(word)
            # Add one synonym if available
            if word in self.synonym_cache and self.synonym_cache[word]:
                expanded_terms.append(self.synonym_cache[word][0])
        
        expanded_query = ' '.join(expanded_terms)
        self._logger.info(f"Basic expansion: '{query}' -> '{expanded_query}'")
        return expanded_query
    
    async def _standard_expansion(self, query: str, context: PluginExecutionContext) -> str:
        """Standard query expansion for moderate CPU resources."""
        expanded_terms = []
        words = query.lower().split()
        
        # Use asyncio to process words concurrently
        tasks = [self._expand_word_standard(word) for word in words]
        expanded_words = await asyncio.gather(*tasks)
        
        for word_expansions in expanded_words:
            expanded_terms.extend(word_expansions)
        
        expanded_query = ' '.join(expanded_terms)
        self._logger.info(f"Standard expansion: '{query}' -> '{expanded_query}'")
        return expanded_query
    
    async def _expand_word_standard(self, word: str) -> List[str]:
        """Expand a single word with moderate complexity."""
        expansions = [word]
        
        # Add synonyms
        if word in self.synonym_cache:
            expansions.extend(self.synonym_cache[word][:2])  # Up to 2 synonyms
        
        # Add pattern-based expansions
        for pattern, synonyms in self.expansion_patterns:
            if re.search(pattern, word, re.IGNORECASE):
                expansions.extend(synonyms[:1])  # Add 1 pattern synonym
                break
        
        return expansions
    
    async def _advanced_expansion(self, query: str, context: PluginExecutionContext) -> str:
        """Advanced query expansion for high CPU resources."""
        expanded_terms = []
        words = query.lower().split()
        
        # Use more CPU cores for parallel processing
        cpu_capacity = context.available_resources.get('total_cpu_capacity', 1.0)
        max_workers = min(int(cpu_capacity), len(words))
        
        # Process words in parallel with more sophisticated expansion
        tasks = [self._expand_word_advanced(word) for word in words]
        expanded_words = await asyncio.gather(*tasks)
        
        for word_expansions in expanded_words:
            expanded_terms.extend(word_expansions)
        
        # Add contextual terms based on query analysis
        contextual_terms = await self._generate_contextual_terms(query)
        expanded_terms.extend(contextual_terms)
        
        expanded_query = ' '.join(expanded_terms)
        self._logger.info(f"Advanced expansion: '{query}' -> '{expanded_query}'")
        return expanded_query
    
    async def _expand_word_advanced(self, word: str) -> List[str]:
        """Expand a single word with high complexity."""
        expansions = [word]
        
        # Add all available synonyms
        if word in self.synonym_cache:
            expansions.extend(self.synonym_cache[word])
        
        # Add pattern-based expansions
        for pattern, synonyms in self.expansion_patterns:
            if re.search(pattern, word, re.IGNORECASE):
                expansions.extend(synonyms)
        
        # Add morphological variations (simplified)
        morphological = await self._generate_morphological_variants(word)
        expansions.extend(morphological)
        
        return expansions
    
    async def _generate_morphological_variants(self, word: str) -> List[str]:
        """Generate morphological variants of a word."""
        variants = []
        
        # Simple pluralization
        if not word.endswith('s'):
            variants.append(word + 's')
        
        # Simple past tense for verbs
        if len(word) > 3 and not word.endswith('ed'):
            variants.append(word + 'ed')
        
        # Simple present participle
        if not word.endswith('ing'):
            variants.append(word + 'ing')
        
        return variants
    
    async def _generate_contextual_terms(self, query: str) -> List[str]:
        """Generate contextual terms based on query analysis."""
        contextual = []
        
        # Media-related context
        media_terms = ['movie', 'show', 'film', 'series', 'episode']
        if any(term in query.lower() for term in media_terms):
            contextual.extend(['watch', 'stream', 'video', 'entertainment'])
        
        # Music-related context
        music_terms = ['music', 'song', 'album', 'artist']
        if any(term in query.lower() for term in music_terms):
            contextual.extend(['audio', 'listen', 'play', 'track'])
        
        return contextual[:3]  # Limit contextual terms
    
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        self.synonym_cache.clear()
        self._logger.info("AdaptiveQueryExpander plugin cleaned up")