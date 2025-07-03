"""
Adaptive Query Expander Plugin
Intelligently expands user queries based on movie data fields and available system resources.
"""

import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass

import httpx

from ..base import (
    QueryEmbellisherPlugin, PluginMetadata, PluginResourceRequirements, 
    PluginExecutionContext, PluginExecutionResult, PluginType, ExecutionPriority
)
from ...shared.config import get_settings
from ...shared.hardware_config import get_resource_limits, get_hardware_profile


@dataclass
class QueryExpansion:
    """Represents an expanded query with metadata."""
    original_query: str
    expanded_query: str
    added_terms: List[str]
    detected_entities: Dict[str, List[str]]
    confidence_score: float
    processing_method: str


class AdaptiveQueryExpanderPlugin(QueryEmbellisherPlugin):
    """
    Adaptive query expansion plugin that scales enhancement based on available resources.
    
    Expansion strategies:
    - Low resources: Basic keyword mapping
    - Medium resources: Synonym expansion + entity detection  
    - High resources: Full semantic expansion + parallel processing
    """
    
    def __init__(self):
        super().__init__()
        self._expansion_dictionaries = {}
        self._movie_entities = {}
        self._hardware_profile = None
        self._ollama_client = None
        self._ollama_gpu_available = False
        self._settings = None
        
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="AdaptiveQueryExpander",
            version="1.0.0",
            description="Intelligently expands user queries based on movie data fields and available system resources",
            author="RAG System",
            plugin_type=PluginType.QUERY_EMBELLISHER,
            tags=["query", "expansion", "movies", "search", "adaptive"],
            execution_priority=ExecutionPriority.HIGH
        )
    
    @property 
    def resource_requirements(self) -> PluginResourceRequirements:
        return PluginResourceRequirements(
            min_cpu_cores=1.0,
            preferred_cpu_cores=4.0,
            min_memory_mb=50.0,
            preferred_memory_mb=200.0,
            requires_gpu=False,
            max_execution_time_seconds=5.0,
            can_use_distributed_resources=True
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with expansion dictionaries and Ollama client."""
        try:
            self._logger.info("Initializing Adaptive Query Expander...")
            
            # Get settings
            self._settings = get_settings()
            
            # Initialize hardware profile
            await self._initialize_hardware_registry()
            
            # Initialize Ollama client
            await self._initialize_ollama_client()
            
            # Load expansion dictionaries
            await self._load_expansion_dictionaries()
            
            # Load movie-specific entities
            await self._load_movie_entities()
            
            self._is_initialized = True
            self._logger.info("Adaptive Query Expander initialized successfully")
            return True
            
        except Exception as e:
            self._initialization_error = str(e)
            self._logger.error(f"Failed to initialize query expander: {e}")
            return False
    
    async def _initialize_hardware_registry(self) -> None:
        """Initialize connection to hardware profile."""
        try:
            self._hardware_profile = await get_hardware_profile()
            resource_limits = await get_resource_limits()
            
            # Log available resources
            cpu_capacity = resource_limits.get("total_cpu_capacity", 0)
            memory_gb = resource_limits.get("local_memory_gb", 0)
            gpu_available = resource_limits.get("gpu_available", False)
            
            self._logger.info(f"Hardware registry initialized - CPU: {cpu_capacity} cores, Memory: {memory_gb}GB, GPU: {gpu_available}")
            
        except Exception as e:
            self._logger.warning(f"Could not initialize hardware profile: {e}")
            self._hardware_profile = None
    
    async def _initialize_ollama_client(self) -> None:
        """Initialize the Ollama HTTP client."""
        try:
            # Create HTTP client for Ollama
            self._ollama_client = httpx.AsyncClient(
                base_url=self._settings.OLLAMA_CHAT_BASE_URL,
                timeout=httpx.Timeout(30.0)
            )
            
            # Test Ollama connection and check GPU acceleration
            response = await self._ollama_client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                
                if self._settings.OLLAMA_CHAT_MODEL in model_names:
                    self._logger.info(f"✅ Ollama connected - model '{self._settings.OLLAMA_CHAT_MODEL}' available")
                    
                    # Check if Ollama is using GPU acceleration
                    await self._check_ollama_gpu_status()
                else:
                    self._logger.warning(f"⚠️ Model '{self._settings.OLLAMA_CHAT_MODEL}' not found in Ollama")
                    self._logger.info(f"Available models: {model_names}")
            else:
                self._logger.warning(f"⚠️ Ollama connection test failed: {response.status_code}")
                
        except Exception as e:
            self._logger.warning(f"⚠️ Could not connect to Ollama: {e}")
            self._ollama_client = None
    
    async def _check_ollama_gpu_status(self) -> None:
        """Check if Ollama is using GPU acceleration."""
        try:
            # Test with a small prompt to see response time and check for GPU indicators
            test_payload = {
                "model": self._settings.OLLAMA_CHAT_MODEL,
                "prompt": "Hello",
                "stream": False,
                "options": {"num_predict": 1}
            }
            
            import time
            start_time = time.time()
            response = await self._ollama_client.post("/api/generate", json=test_payload)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                # Fast response times typically indicate GPU acceleration
                if response_time < 1000:  # Less than 1 second
                    self._logger.info(f"🚀 Ollama appears to be GPU-accelerated (response: {response_time:.1f}ms)")
                    
                    # Store GPU status for strategy decisions
                    if hasattr(self, '_hardware_profile') and self._hardware_profile:
                        # Update our understanding that external GPU is available via Ollama
                        self._ollama_gpu_available = True
                else:
                    self._logger.info(f"💻 Ollama using CPU (response: {response_time:.1f}ms)")
                    self._ollama_gpu_available = False
            else:
                self._logger.warning(f"Could not test Ollama performance: {response.status_code}")
                self._ollama_gpu_available = False
                
        except Exception as e:
            self._logger.warning(f"Could not check Ollama GPU status: {e}")
            self._ollama_gpu_available = False
    
    async def _load_expansion_dictionaries(self) -> None:
        """Load synonym and expansion dictionaries for different categories."""
        
        # Genre expansions
        self._expansion_dictionaries["genres"] = {
            "comedy": ["funny", "humor", "humorous", "amusing", "hilarious", "laughs", "satire", "parody"],
            "drama": ["dramatic", "serious", "emotional", "intense", "compelling", "character-driven"],
            "horror": ["scary", "frightening", "terrifying", "spooky", "creepy", "thriller", "suspense"],
            "action": ["exciting", "adventure", "thrilling", "fast-paced", "explosive", "adrenaline"],
            "romance": ["romantic", "love", "relationship", "dating", "passion", "heartwarming"],
            "science fiction": ["sci-fi", "futuristic", "space", "alien", "technology", "cyberpunk"],
            "fantasy": ["magical", "supernatural", "mythical", "wizards", "dragons", "fairy tale"],
            "documentary": ["doc", "factual", "real", "educational", "informative", "non-fiction"],
            "animation": ["animated", "cartoon", "anime", "family", "kids", "pixar", "disney"],
            "western": ["cowboy", "frontier", "wild west", "gunslinger", "ranch", "outlaw"]
        }
        
        # Mood/tone expansions  
        self._expansion_dictionaries["moods"] = {
            "dark": ["gritty", "noir", "bleak", "sinister", "brooding", "atmospheric"],
            "light": ["upbeat", "cheerful", "feel-good", "optimistic", "bright", "positive"],
            "intense": ["gripping", "compelling", "powerful", "strong", "forceful"],
            "surreal": ["weird", "strange", "bizarre", "abstract", "dreamlike", "experimental"],
            "epic": ["grand", "spectacular", "monumental", "sweeping", "massive", "legendary"]
        }
        
        # Time period expansions
        self._expansion_dictionaries["time_periods"] = {
            "80s": ["1980s", "eighties", "retro", "vintage", "classic"],
            "90s": ["1990s", "nineties", "nostalgic"],
            "2000s": ["2000-2009", "millennium", "early 2000s"],
            "2010s": ["2010-2019", "modern", "recent", "contemporary"],
            "2020s": ["2020-2029", "current", "latest", "new", "fresh"]
        }
        
        # Cultural/geographic expansions
        self._expansion_dictionaries["locations"] = {
            "finland": ["finnish", "nordic", "scandinavian", "northern european"],
            "japan": ["japanese", "asian", "samurai", "ninja", "anime"],
            "usa": ["american", "hollywood", "united states"],
            "uk": ["british", "england", "english", "britain"],
            "france": ["french", "european", "art house"]
        }
        
        self._logger.debug(f"Loaded {len(self._expansion_dictionaries)} expansion dictionaries")
    
    async def _load_movie_entities(self) -> None:
        """Load common movie entities for detection."""
        
        # Common movie themes/tags that appear in real data
        self._movie_entities = {
            "martial_arts": ["samurai", "ninja", "kung fu", "karate", "fighting", "warrior", "chambara"],
            "crime": ["gangster", "mafia", "heist", "detective", "murder", "police"],
            "war": ["battlefield", "military", "soldier", "combat", "wwii", "vietnam"],
            "superhero": ["marvel", "dc", "batman", "superman", "powers", "cape"],
            "space": ["astronaut", "spaceship", "galaxy", "planet", "alien", "star wars"],
            "zombie": ["undead", "apocalypse", "survival", "outbreak", "walking dead"],
            "spy": ["espionage", "secret agent", "cia", "mission", "undercover"],
            "monster": ["creature", "beast", "kaiju", "giant", "mutant"]
        }
        
        self._logger.debug(f"Loaded {len(self._movie_entities)} movie entity categories")
    
    async def embellish_query(self, query: str, context: PluginExecutionContext) -> str:
        """Main query embellishment method that adapts to available resources."""
        try:
            # Get available resources from hardware profile (preferred) or context fallback
            if self._hardware_profile:
                resource_limits = await get_resource_limits()
                cpu_cores = resource_limits.get("total_cpu_capacity", 1)
                memory_gb = resource_limits.get("local_memory_gb", 1)
                gpu_available = resource_limits.get("gpu_available", False)
                
                # Check if system can handle parallel processing
                can_parallelize = cpu_cores >= 2 and memory_gb >= 2
                high_performance = cpu_cores >= 4 and memory_gb >= 4
                
                self._logger.debug(f"Hardware registry resources - CPU: {cpu_cores}, Memory: {memory_gb}GB, GPU: {gpu_available}")
            else:
                # Fallback to context resources
                resources = context.available_resources or {}
                cpu_cores = resources.get("total_cpu_capacity", 1)
                memory_gb = resources.get("local_memory_gb", 1)
                gpu_available = resources.get("gpu_available", False)
                can_parallelize = cpu_cores >= 2
                high_performance = cpu_cores >= 4
                
                self._logger.debug(f"Context resources - CPU: {cpu_cores}, Memory: {memory_gb}GB")
            
            # Determine processing strategy based on actual hardware capabilities
            ollama_available = self._ollama_client is not None
            ollama_gpu_accelerated = self._ollama_gpu_available
            
            # Enhanced strategy selection considering GPU acceleration
            if high_performance and ollama_available and ollama_gpu_accelerated:
                strategy = "ollama_gpu_enhanced"
                parallel_tasks = min(cpu_cores // 2, 6)  # More aggressive with GPU
            elif high_performance and ollama_available:
                strategy = "ollama_enhanced"
                parallel_tasks = min(cpu_cores // 2, 4)  # Standard Ollama
            elif high_performance:
                strategy = "high_resource"
                parallel_tasks = min(cpu_cores // 2, 4)
            elif can_parallelize:
                strategy = "medium_resource"
                parallel_tasks = 2
            else:
                strategy = "low_resource" 
                parallel_tasks = 1
            
            # Store parallelization info in context for use by expansion methods
            context.metadata = context.metadata or {}
            context.metadata["parallel_tasks"] = parallel_tasks
            context.metadata["gpu_available"] = gpu_available or ollama_gpu_accelerated
            context.metadata["ollama_gpu_accelerated"] = ollama_gpu_accelerated
            
            self._logger.debug(f"Strategy: {strategy} (CPU: {cpu_cores}, Memory: {memory_gb}GB, Local GPU: {gpu_available}, Ollama GPU: {ollama_gpu_accelerated}, Parallel: {parallel_tasks})")
            
            # Perform expansion based on strategy
            if strategy == "ollama_gpu_enhanced":
                expansion = await self._ollama_gpu_enhanced_expansion(query, context)
            elif strategy == "ollama_enhanced":
                expansion = await self._ollama_enhanced_expansion(query, context)
            elif strategy == "high_resource":
                expansion = await self._high_resource_expansion(query, context)
            elif strategy == "medium_resource":
                expansion = await self._medium_resource_expansion(query, context)
            else:
                expansion = await self._low_resource_expansion(query, context)
            
            # Log expansion details
            self._logger.info(f"Query expansion: '{expansion.original_query}' → '{expansion.expanded_query}'")
            self._logger.debug(f"Added terms: {expansion.added_terms}")
            self._logger.debug(f"Detected entities: {expansion.detected_entities}")
            self._logger.debug(f"Confidence: {expansion.confidence_score:.2f}")
            
            return expansion.expanded_query
            
        except Exception as e:
            self._logger.error(f"Query expansion failed: {e}")
            return query  # Return original query on failure
    
    async def _low_resource_expansion(self, query: str, context: PluginExecutionContext) -> QueryExpansion:
        """Basic expansion for low-resource environments."""
        added_terms = []
        detected_entities = {}
        
        # Simple keyword-based expansion
        query_lower = query.lower()
        expanded_parts = [query]
        
        # Check for exact genre matches
        for genre, synonyms in self._expansion_dictionaries["genres"].items():
            if genre in query_lower:
                # Add just the first synonym to keep it simple
                added_terms.append(synonyms[0])
                expanded_parts.append(synonyms[0])
                detected_entities["genre"] = [genre]
        
        # Check for mood keywords
        for mood, synonyms in self._expansion_dictionaries["moods"].items():
            if mood in query_lower:
                added_terms.append(synonyms[0])
                expanded_parts.append(synonyms[0])
                detected_entities["mood"] = [mood]
        
        expanded_query = " ".join(expanded_parts)
        
        return QueryExpansion(
            original_query=query,
            expanded_query=expanded_query,
            added_terms=added_terms,
            detected_entities=detected_entities,
            confidence_score=0.6,
            processing_method="low_resource"
        )
    
    async def _medium_resource_expansion(self, query: str, context: PluginExecutionContext) -> QueryExpansion:
        """Medium expansion with synonym detection and entity recognition."""
        added_terms = []
        detected_entities = {}
        
        query_lower = query.lower()
        expanded_parts = [query]
        
        # More thorough genre expansion
        for genre, synonyms in self._expansion_dictionaries["genres"].items():
            if genre in query_lower or any(syn in query_lower for syn in synonyms[:3]):
                # Add multiple synonyms
                selected_synonyms = synonyms[:2]  # Limit to 2 for medium resource
                added_terms.extend(selected_synonyms)
                expanded_parts.extend(selected_synonyms)
                detected_entities.setdefault("genre", []).append(genre)
        
        # Check movie entities
        for entity_type, keywords in self._movie_entities.items():
            if any(keyword in query_lower for keyword in keywords):
                # Add related terms
                related_terms = keywords[:2]  # Limit for medium resource
                added_terms.extend(related_terms)
                expanded_parts.extend(related_terms)
                detected_entities.setdefault("theme", []).append(entity_type)
        
        # Time period detection
        year_matches = re.findall(r'\b(19|20)\d{2}\b', query)
        for year in year_matches:
            decade = f"{year[:3]}0s"
            if decade in self._expansion_dictionaries["time_periods"]:
                period_terms = self._expansion_dictionaries["time_periods"][decade][:2]
                added_terms.extend(period_terms)
                expanded_parts.extend(period_terms)
                detected_entities.setdefault("time_period", []).append(decade)
        
        # Remove duplicates while preserving order
        unique_parts = []
        seen = set()
        for part in expanded_parts:
            if part.lower() not in seen:
                unique_parts.append(part)
                seen.add(part.lower())
        
        expanded_query = " ".join(unique_parts)
        
        return QueryExpansion(
            original_query=query,
            expanded_query=expanded_query,
            added_terms=added_terms,
            detected_entities=detected_entities,
            confidence_score=0.75,
            processing_method="medium_resource"
        )
    
    async def _high_resource_expansion(self, query: str, context: PluginExecutionContext) -> QueryExpansion:
        """Full expansion with parallel processing and comprehensive analysis."""
        
        # Get parallel task count from context metadata (set by hardware profile)
        parallel_tasks = context.metadata.get("parallel_tasks", 4) if context.metadata else 4
        gpu_available = context.metadata.get("gpu_available", False) if context.metadata else False
        
        self._logger.debug(f"High-resource expansion using {parallel_tasks} parallel tasks (GPU: {gpu_available})")
        
        # Use parallel processing for high-resource expansion
        tasks = [
            self._expand_genres_comprehensive(query),
            self._expand_entities_comprehensive(query),
            self._expand_temporal_comprehensive(query),
            self._expand_cultural_comprehensive(query)
        ]
        
        # Limit concurrent tasks based on available hardware
        if parallel_tasks < len(tasks):
            # Process in batches if we have limited parallel capacity
            results = []
            for i in range(0, len(tasks), parallel_tasks):
                batch = tasks[i:i + parallel_tasks]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                results.extend(batch_results)
        else:
            # Run all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        all_added_terms = []
        all_detected_entities = {}
        expanded_parts = [query]
        
        for result in results:
            if isinstance(result, Exception):
                self._logger.warning(f"Expansion task failed: {result}")
                continue
                
            added_terms, detected_entities, expansion_parts = result
            all_added_terms.extend(added_terms)
            expanded_parts.extend(expansion_parts)
            
            # Merge detected entities
            for key, values in detected_entities.items():
                all_detected_entities.setdefault(key, []).extend(values)
        
        # Remove duplicates and limit total expansion
        unique_parts = []
        seen = set()
        for part in expanded_parts:
            if part.lower() not in seen and len(unique_parts) < 20:  # Limit expansion
                unique_parts.append(part)
                seen.add(part.lower())
        
        expanded_query = " ".join(unique_parts)
        
        return QueryExpansion(
            original_query=query,
            expanded_query=expanded_query,
            added_terms=all_added_terms,
            detected_entities=all_detected_entities,
            confidence_score=0.9,
            processing_method="high_resource_parallel"
        )
    
    async def _expand_genres_comprehensive(self, query: str) -> tuple:
        """Comprehensive genre expansion."""
        query_lower = query.lower()
        added_terms = []
        detected_entities = {}
        expansion_parts = []
        
        for genre, synonyms in self._expansion_dictionaries["genres"].items():
            # Check both exact match and fuzzy match
            if genre in query_lower or any(syn in query_lower for syn in synonyms):
                # Add more synonyms for comprehensive expansion
                selected_synonyms = synonyms[:4]  # More synonyms for high resource
                added_terms.extend(selected_synonyms)
                expansion_parts.extend(selected_synonyms)
                detected_entities.setdefault("genre", []).append(genre)
        
        return added_terms, detected_entities, expansion_parts
    
    async def _expand_entities_comprehensive(self, query: str) -> tuple:
        """Comprehensive entity expansion."""
        query_lower = query.lower()
        added_terms = []
        detected_entities = {}
        expansion_parts = []
        
        for entity_type, keywords in self._movie_entities.items():
            matches = [kw for kw in keywords if kw in query_lower]
            if matches:
                # Add all related terms for comprehensive expansion
                related_terms = keywords[:5]  # More terms for high resource
                added_terms.extend(related_terms)
                expansion_parts.extend(related_terms)
                detected_entities.setdefault("theme", []).append(entity_type)
        
        return added_terms, detected_entities, expansion_parts
    
    async def _expand_temporal_comprehensive(self, query: str) -> tuple:
        """Comprehensive temporal expansion."""
        added_terms = []
        detected_entities = {}
        expansion_parts = []
        
        # Year detection
        year_matches = re.findall(r'\b(19|20)\d{2}\b', query)
        decade_matches = re.findall(r'\b(19|20)\d0s\b', query.lower())
        
        for year in year_matches:
            decade = f"{year[:3]}0s"
            if decade in self._expansion_dictionaries["time_periods"]:
                period_terms = self._expansion_dictionaries["time_periods"][decade]
                added_terms.extend(period_terms)
                expansion_parts.extend(period_terms)
                detected_entities.setdefault("time_period", []).append(decade)
        
        for decade in decade_matches:
            if decade in self._expansion_dictionaries["time_periods"]:
                period_terms = self._expansion_dictionaries["time_periods"][decade]
                added_terms.extend(period_terms)
                expansion_parts.extend(period_terms)
                detected_entities.setdefault("time_period", []).append(decade)
        
        return added_terms, detected_entities, expansion_parts
    
    async def _expand_cultural_comprehensive(self, query: str) -> tuple:
        """Comprehensive cultural/geographic expansion."""
        query_lower = query.lower()
        added_terms = []
        detected_entities = {}
        expansion_parts = []
        
        for location, descriptors in self._expansion_dictionaries["locations"].items():
            if location in query_lower or any(desc in query_lower for desc in descriptors):
                added_terms.extend(descriptors)
                expansion_parts.extend(descriptors)
                detected_entities.setdefault("location", []).append(location)
        
        return added_terms, detected_entities, expansion_parts
    
    async def _ollama_gpu_enhanced_expansion(self, query: str, context: PluginExecutionContext) -> QueryExpansion:
        """GPU-enhanced expansion using Ollama with more aggressive parameters."""
        try:
            self._logger.debug("Using GPU-enhanced Ollama expansion strategy")
            
            # Start with traditional expansion as fallback
            fallback_expansion = await self._high_resource_expansion(query, context)
            
            if not self._ollama_client:
                self._logger.warning("Ollama not available, using fallback expansion")
                return fallback_expansion
            
            # Create enhanced prompt for GPU-accelerated processing
            prompt = self._create_ollama_gpu_enhanced_prompt(query)
            
            # Call Ollama with GPU-optimized parameters
            ollama_result = await self._call_ollama_gpu_enhanced(prompt, context)
            
            if ollama_result:
                # Parse Ollama response
                ollama_expansion = await self._parse_ollama_response(query, ollama_result)
                
                # Combine with fallback expansion
                combined_expansion = await self._combine_expansions(fallback_expansion, ollama_expansion)
                combined_expansion.processing_method = "ollama_gpu_enhanced"
                combined_expansion.confidence_score = min(0.98, combined_expansion.confidence_score + 0.15)
                
                return combined_expansion
            else:
                self._logger.warning("GPU-enhanced Ollama expansion failed, using fallback")
                return fallback_expansion
                
        except Exception as e:
            self._logger.error(f"GPU-enhanced Ollama expansion error: {e}")
            # Return fallback expansion on error
            fallback_expansion = await self._high_resource_expansion(query, context)
            return fallback_expansion
    
    def _create_ollama_gpu_enhanced_prompt(self, query: str) -> str:
        """Create an enhanced prompt for GPU-accelerated Ollama processing."""
        return f"""You are an advanced movie search AI with deep knowledge of cinema. Provide comprehensive query expansion for this movie search.

Original query: "{query}"

Provide detailed expansion including:
- Primary genre synonyms and subgenres
- Thematic elements and narrative concepts  
- Cultural and historical context
- Mood and atmosphere descriptors
- Technical aspects (if relevant)
- Alternative phrasing and terminology

Respond with comprehensive JSON:
{{
    "expanded_terms": ["comprehensive", "list", "of", "expanded", "search", "terms"],
    "detected_categories": {{
        "genres": ["primary", "secondary", "subgenres"],
        "themes": ["major", "minor", "thematic", "elements"],
        "moods": ["atmosphere", "tone", "feeling"],
        "time_periods": ["era", "decade", "period"],
        "cultural_context": ["cultural", "geographic", "elements"],
        "technical_aspects": ["cinematography", "style", "elements"]
    }},
    "semantic_relationships": {{
        "similar_concepts": ["related", "ideas"],
        "broader_categories": ["parent", "categories"],
        "specific_examples": ["concrete", "examples"]
    }},
    "reasoning": "Detailed explanation of expansion strategy and cinema context"
}}

Leverage your deep movie knowledge for comprehensive, intelligent expansion."""
    
    async def _call_ollama_gpu_enhanced(self, prompt: str, context: PluginExecutionContext) -> Optional[str]:
        """Call Ollama with GPU-optimized parameters for enhanced processing."""
        try:
            # GPU-enhanced parameters for more comprehensive responses
            payload = {
                "model": self._settings.OLLAMA_CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,  # Higher creativity with GPU power
                    "top_p": 0.95,
                    "top_k": 60,
                    "num_predict": 300,  # Longer responses with GPU acceleration
                    "repeat_penalty": 1.1
                }
            }
            
            response = await self._ollama_client.post("/api/generate", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                self._logger.warning(f"GPU-enhanced Ollama API error: {response.status_code}")
                return None
                
        except asyncio.TimeoutError:
            self._logger.warning("GPU-enhanced Ollama request timed out")
            return None
        except Exception as e:
            self._logger.error(f"GPU-enhanced Ollama API call failed: {e}")
            return None
    
    async def _ollama_enhanced_expansion(self, query: str, context: PluginExecutionContext) -> QueryExpansion:
        """Enhanced expansion using Ollama for intelligent query understanding."""
        try:
            # Start with traditional expansion as fallback
            fallback_expansion = await self._high_resource_expansion(query, context)
            
            if not self._ollama_client:
                self._logger.warning("Ollama not available, using fallback expansion")
                return fallback_expansion
            
            # Create prompt for Ollama
            prompt = self._create_ollama_expansion_prompt(query)
            
            # Call Ollama for intelligent expansion
            ollama_result = await self._call_ollama(prompt, context)
            
            if ollama_result:
                # Parse Ollama response
                ollama_expansion = await self._parse_ollama_response(query, ollama_result)
                
                # Combine with fallback expansion
                combined_expansion = await self._combine_expansions(fallback_expansion, ollama_expansion)
                combined_expansion.processing_method = "ollama_enhanced"
                combined_expansion.confidence_score = min(0.95, combined_expansion.confidence_score + 0.1)
                
                return combined_expansion
            else:
                self._logger.warning("Ollama expansion failed, using fallback")
                return fallback_expansion
                
        except Exception as e:
            self._logger.error(f"Ollama expansion error: {e}")
            # Return fallback expansion on error
            fallback_expansion = await self._high_resource_expansion(query, context)
            return fallback_expansion
    
    def _create_ollama_expansion_prompt(self, query: str) -> str:
        """Create a prompt for Ollama to expand the movie search query."""
        return f"""You are a movie search expert. Help expand this movie search query with relevant synonyms and related terms.

Original query: "{query}"

Please provide expanded search terms that would help find movies related to this query. Consider:
- Genre synonyms (comedy → funny, humorous, amusing)
- Movie themes and elements
- Alternative words and phrases
- Related movie terminology

Respond with a JSON object containing:
{{
    "expanded_terms": ["list", "of", "expanded", "terms"],
    "detected_categories": {{
        "genres": ["detected", "genres"],
        "themes": ["detected", "themes"],
        "time_periods": ["detected", "time_periods"]
    }},
    "reasoning": "Brief explanation of the expansion"
}}

Focus on movie search terms only. Be concise but thorough."""
    
    async def _call_ollama(self, prompt: str, context: PluginExecutionContext = None) -> Optional[str]:
        """Call Ollama API for query expansion with hardware-aware parameters."""
        try:
            # Get hardware info for optimization
            parallel_tasks = 1
            if context and context.metadata:
                parallel_tasks = context.metadata.get("parallel_tasks", 1)
            
            # Adjust Ollama parameters based on available resources
            if parallel_tasks >= 4:
                # High-resource system: Use more creative parameters
                temperature = 0.4
                top_k = 50
                max_tokens = 150
            elif parallel_tasks >= 2:
                # Medium-resource system: Balanced parameters
                temperature = 0.3
                top_k = 40  
                max_tokens = 100
            else:
                # Low-resource system: Conservative parameters
                temperature = 0.2
                top_k = 30
                max_tokens = 75
            
            payload = {
                "model": self._settings.OLLAMA_CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": top_k,
                    "num_predict": max_tokens  # Limit response length based on resources
                }
            }
            
            response = await self._ollama_client.post("/api/generate", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                self._logger.warning(f"Ollama API error: {response.status_code}")
                return None
                
        except asyncio.TimeoutError:
            self._logger.warning("Ollama request timed out")
            return None
        except Exception as e:
            self._logger.error(f"Ollama API call failed: {e}")
            return None
    
    async def _parse_ollama_response(self, original_query: str, ollama_response: str) -> QueryExpansion:
        """Parse Ollama's JSON response into a QueryExpansion object."""
        try:
            # Try to extract JSON from the response
            json_start = ollama_response.find("{")
            json_end = ollama_response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = ollama_response[json_start:json_end]
                data = json.loads(json_str)
                
                expanded_terms = data.get("expanded_terms", [])
                detected_categories = data.get("detected_categories", {})
                reasoning = data.get("reasoning", "")
                
                # Build expanded query
                expanded_parts = [original_query] + expanded_terms
                expanded_query = " ".join(expanded_parts)
                
                self._logger.debug(f"Ollama reasoning: {reasoning}")
                
                return QueryExpansion(
                    original_query=original_query,
                    expanded_query=expanded_query,
                    added_terms=expanded_terms,
                    detected_entities=detected_categories,
                    confidence_score=0.85,
                    processing_method="ollama_parsed"
                )
            else:
                # Fallback: treat entire response as expansion terms
                words = ollama_response.split()
                # Filter out common words and duplicates
                expanded_terms = [word.strip('.,!?') for word in words 
                                if len(word) > 2 and word.lower() not in original_query.lower()]
                expanded_terms = list(dict.fromkeys(expanded_terms))[:10]  # Remove duplicates, limit to 10
                
                expanded_query = f"{original_query} {' '.join(expanded_terms)}"
                
                return QueryExpansion(
                    original_query=original_query,
                    expanded_query=expanded_query,
                    added_terms=expanded_terms,
                    detected_entities={},
                    confidence_score=0.7,
                    processing_method="ollama_fallback"
                )
                
        except json.JSONDecodeError:
            self._logger.warning("Could not parse Ollama JSON response")
            return QueryExpansion(
                original_query=original_query,
                expanded_query=original_query,
                added_terms=[],
                detected_entities={},
                confidence_score=0.5,
                processing_method="ollama_error"
            )
        except Exception as e:
            self._logger.error(f"Error parsing Ollama response: {e}")
            return QueryExpansion(
                original_query=original_query,
                expanded_query=original_query,
                added_terms=[],
                detected_entities={},
                confidence_score=0.5,
                processing_method="ollama_error"
            )
    
    async def _combine_expansions(self, fallback: QueryExpansion, ollama: QueryExpansion) -> QueryExpansion:
        """Combine fallback expansion with Ollama expansion."""
        # Combine terms from both expansions
        all_terms = list(dict.fromkeys(fallback.added_terms + ollama.added_terms))  # Remove duplicates
        
        # Combine detected entities
        combined_entities = fallback.detected_entities.copy()
        for key, values in ollama.detected_entities.items():
            combined_entities.setdefault(key, []).extend(values)
            # Remove duplicates from entity lists
            combined_entities[key] = list(dict.fromkeys(combined_entities[key]))
        
        # Create combined expanded query
        expanded_parts = [fallback.original_query] + all_terms
        expanded_query = " ".join(expanded_parts)
        
        # Calculate combined confidence
        combined_confidence = (fallback.confidence_score + ollama.confidence_score) / 2
        
        return QueryExpansion(
            original_query=fallback.original_query,
            expanded_query=expanded_query,
            added_terms=all_terms,
            detected_entities=combined_entities,
            confidence_score=combined_confidence,
            processing_method="combined_ollama_traditional"
        )
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        if self._ollama_client:
            await self._ollama_client.aclose()
            self._ollama_client = None
        
        self._expansion_dictionaries.clear()
        self._movie_entities.clear()
        self._logger.info("Adaptive Query Expander cleaned up")