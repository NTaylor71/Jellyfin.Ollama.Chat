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

# ── NLTK Setup (module level, run once) ──────────────────────────────────────
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Optional dependencies
try:
    from gensim.models import Word2Vec
    _GENSIM_AVAILABLE = True
except ImportError:
    Word2Vec = None
    _GENSIM_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _SKLEARN_AVAILABLE = True
except ImportError:
    TfidfVectorizer = None
    _SKLEARN_AVAILABLE = False

# Basic stopwords
_BASIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "for", "from", "has", "he", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "to", "was", "were",
    "will", "with", "movie", "film", "cinema", "watch", "see", "show", "find"
}

def ensure_nltk_corpora() -> None:
    """Ensure required NLTK corpora are downloaded."""
    for corpus in ("wordnet", "omw-1.4", "punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger"):
        try:
            if corpus == "punkt":
                nltk.data.find('tokenizers/punkt')
            elif corpus == "punkt_tab":
                nltk.data.find('tokenizers/punkt_tab')
            elif corpus == "wordnet":
                nltk.data.find('corpora/wordnet')
            elif corpus == "stopwords":
                nltk.data.find('corpora/stopwords')
            elif corpus == "averaged_perceptron_tagger":
                nltk.data.find('taggers/averaged_perceptron_tagger')
            elif corpus == "omw-1.4":
                nltk.data.find('corpora/omw-1.4')
        except LookupError:
            nltk.download(corpus, quiet=True)

# Initialize NLTK resources once at module load
ensure_nltk_corpora()

# Initialize global lemmatizer and stopwords
try:
    _LEMMATIZER = WordNetLemmatizer()
    _STOP_WORDS = set(stopwords.words('english')).union(_BASIC_STOPWORDS)
    _NLTK_AVAILABLE = True
except Exception:
    _LEMMATIZER = None
    _STOP_WORDS = _BASIC_STOPWORDS
    _NLTK_AVAILABLE = False

from ..base import (
    QueryEmbellisherPlugin, PluginMetadata, PluginResourceRequirements, 
    PluginExecutionContext, PluginExecutionResult, PluginType, ExecutionPriority
)
from ..config import BasePluginConfig
from ...shared.config import get_settings
from ...shared.hardware_config import get_resource_limits, get_hardware_profile
from typing import Type
from pydantic import Field


class AdaptiveQueryExpanderConfig(BasePluginConfig):
    """Configuration for Adaptive Query Expander Plugin."""
    
    # Expansion settings
    expansion_strategies: List[str] = Field(
        default=["synonyms", "related_terms", "context_expansion"],
        description="List of expansion strategies to use"
    )
    max_expansion_terms: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of terms to add during expansion"
    )
    
    # LLM settings
    enable_ollama: bool = Field(
        default=True,
        description="Enable Ollama LLM for advanced expansions"
    )
    ollama_model: str = Field(
        default="llama2",
        description="Ollama model to use for expansions"
    )
    llm_retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retry attempts for LLM calls"
    )
    llm_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Timeout for LLM API calls"
    )
    
    # Fallback settings
    fallback_strategy: str = Field(
        default="synonyms_only",
        description="Strategy to use when advanced methods fail"
    )
    
    # Hardware adaptation
    min_processing_strategy: str = Field(
        default="low",
        description="Minimum processing strategy (low/medium/high)"
    )
    preferred_processing_strategy: str = Field(
        default="high",
        description="Preferred processing strategy when resources allow"
    )
    enable_parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing when resources allow"
    )


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
    
    @property
    def config_class(self) -> Type[BasePluginConfig]:
        """Return the configuration class for this plugin."""
        return AdaptiveQueryExpanderConfig
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with expansion dictionaries and Ollama client."""
        try:
            self._logger.info("Initializing Adaptive Query Expander...")
            
            # Get plugin configuration
            plugin_config = self.get_config()
            if plugin_config:
                self._logger.info(f"Using plugin configuration: enabled={plugin_config.enabled}, max_terms={plugin_config.max_expansion_terms}")
                
                # Check if plugin is enabled
                if not plugin_config.enabled:
                    self._logger.info("Plugin is disabled in configuration")
                    return False
            
            # Get settings
            self._settings = get_settings()
            
            # Initialize hardware profile
            await self._initialize_hardware_registry()
            
            # Initialize Ollama client (if enabled in config)
            if not plugin_config or plugin_config.enable_ollama:
                await self._initialize_ollama_client()
            else:
                self._logger.info("Ollama integration disabled in configuration")
            
            # Load expansion dictionaries (minimal, as backup)
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
        """Basic expansion using WordNet synonyms for low-resource environments."""
        added_terms = []
        detected_entities = {}
        
        if not _NLTK_AVAILABLE:
            # Fallback to minimal expansion
            return QueryExpansion(
                original_query=query,
                expanded_query=query,
                added_terms=[],
                detected_entities={},
                confidence_score=0.5,
                processing_method="low_resource_nltk_unavailable"
            )
        
        try:
            # Tokenize and clean the query
            tokens = word_tokenize(query.lower())
            meaningful_tokens = [token for token in tokens if token.isalpha() and token not in _STOP_WORDS]
            
            # Get synonyms for key terms (limit to 1 synonym per word for low resource)
            for token in meaningful_tokens[:3]:  # Only process first 3 meaningful words
                synonyms = self._get_wordnet_synonyms(token, max_synonyms=1)
                if synonyms:
                    added_terms.extend(synonyms)
                    
                    # Try to detect entity types
                    entity_type = self._detect_entity_type(token)
                    if entity_type:
                        detected_entities.setdefault(entity_type, []).append(token)
            
            # Build expanded query (original + best synonyms)
            if added_terms:
                expanded_query = f"{query} {' '.join(added_terms)}"
            else:
                expanded_query = query
            
            return QueryExpansion(
                original_query=query,
                expanded_query=expanded_query,
                added_terms=added_terms,
                detected_entities=detected_entities,
                confidence_score=0.7,
                processing_method="low_resource_wordnet"
            )
            
        except Exception as e:
            self._logger.warning(f"Low resource expansion failed: {e}")
            return QueryExpansion(
                original_query=query,
                expanded_query=query,
                added_terms=[],
                detected_entities={},
                confidence_score=0.5,
                processing_method="low_resource_error"
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
        # Tokenize the query to extract keywords
        tokens = word_tokenize(query.lower()) if _NLTK_AVAILABLE else query.split()
        keywords = [token for token in tokens if token.isalpha() and len(token) > 2]
        
        prompt = (
            "Return a comprehensive expanded dict of movie search synonyms with cinema expertise - ONLY return JSON:\n"
            '{ "expanded": { "keyword": ["syn1","syn2","syn3","syn4","syn5"] }, '
            '"themes": ["theme1","theme2"], "genres": ["genre1","genre2"] }\n\n'
            f"Keywords: {json.dumps(keywords)}"
        )
        return prompt
    
    async def _call_ollama_gpu_enhanced(self, prompt: str, context: PluginExecutionContext) -> Optional[str]:
        """Call Ollama with GPU-optimized parameters and retry logic."""
        try:
            # Retry with error injection if JSON parsing fails
            for attempt in range(3):  # Max 3 attempts
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
                    
                    if response.status_code != 200:
                        self._logger.warning(f"GPU-enhanced Ollama API error: {response.status_code}")
                        continue
                    
                    result = response.json()
                    raw_response = result.get("response", "").strip("` \n")
                    
                    # Test if the response contains valid JSON
                    json_start = raw_response.find("{")
                    json_end = raw_response.rfind("}") + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = raw_response[json_start:json_end]
                        # Test parse to validate JSON
                        json.loads(json_str)
                        # If we get here, JSON is valid
                        return raw_response
                    
                    # JSON parsing failed - inject error into prompt for retry
                    if attempt < 2:
                        self._logger.warning(f"GPU-enhanced: Invalid JSON response on attempt {attempt + 1}, retrying with error correction")
                        prompt = (
                            f"Previous response was invalid JSON. Please fix and return ONLY valid JSON:\n"
                            f'{ "expanded": { "keyword": ["syn1","syn2","syn3","syn4","syn5"] }, "themes": ["theme1"], "genres": ["genre1"] }\n\n'
                            f"Original request: {prompt}\n"
                            f"Invalid response was: {raw_response[:100]}..."
                        )
                    else:
                        self._logger.warning("All GPU-enhanced JSON parsing attempts failed")
                        return None
                        
                except json.JSONDecodeError as e:
                    if attempt < 2:
                        self._logger.warning(f"GPU-enhanced JSON decode error on attempt {attempt + 1}: {e}")
                        prompt = f"Fix JSON syntax error: {e}\n\nReturn valid JSON only:\n{prompt}"
                    else:
                        self._logger.warning("GPU-enhanced JSON parsing failed after all retries")
                        return None
                except Exception as e:
                    self._logger.error(f"GPU-enhanced Ollama API call failed on attempt {attempt + 1}: {e}")
                    if attempt == 2:
                        return None
            
            return None
                
        except Exception as e:
            self._logger.error(f"GPU-enhanced Ollama expansion completely failed: {e}")
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
        # Tokenize the query to extract keywords
        tokens = word_tokenize(query.lower()) if _NLTK_AVAILABLE else query.split()
        keywords = [token for token in tokens if token.isalpha() and len(token) > 2]
        
        prompt = (
            "Return a verbose expanded dict of movie search synonyms based on the following keywords - ONLY return JSON:\n"
            '{ "expanded": { "keyword": ["syn1","syn2", ...] } }\n\n'
            f"Keywords: {json.dumps(keywords)}"
        )
        return prompt
    
    async def _call_ollama(self, prompt: str, context: PluginExecutionContext = None) -> Optional[str]:
        """Call Ollama API for query expansion with retry and error correction."""
        try:
            # Get hardware info for optimization
            parallel_tasks = 1
            if context and context.metadata:
                parallel_tasks = context.metadata.get("parallel_tasks", 1)
            
            # Adjust Ollama parameters based on available resources
            if parallel_tasks >= 4:
                temperature = 0.4
                max_tokens = 150
            elif parallel_tasks >= 2:
                temperature = 0.3
                max_tokens = 100
            else:
                temperature = 0.2
                max_tokens = 75
            
            # Retry with error injection if JSON parsing fails
            for attempt in range(3):  # Max 3 attempts
                try:
                    payload = {
                        "model": self._settings.OLLAMA_CHAT_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "top_p": 0.9,
                            "num_predict": max_tokens
                        }
                    }
                    
                    response = await self._ollama_client.post("/api/generate", json=payload)
                    
                    if response.status_code != 200:
                        self._logger.warning(f"Ollama API error: {response.status_code}")
                        continue
                    
                    result = response.json()
                    raw_response = result.get("response", "").strip("` \n")
                    
                    # Test if the response contains valid JSON
                    json_start = raw_response.find("{")
                    json_end = raw_response.rfind("}") + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = raw_response[json_start:json_end]
                        # Test parse to validate JSON
                        json.loads(json_str)
                        # If we get here, JSON is valid
                        return raw_response
                    
                    # JSON parsing failed - inject error into prompt for retry
                    if attempt < 2:
                        self._logger.warning(f"Invalid JSON response on attempt {attempt + 1}, retrying with error correction")
                        prompt = (
                            f"Previous response was invalid JSON. Please fix and return ONLY valid JSON:\n"
                            f'{ "expanded": { "keyword": ["syn1","syn2"] } }\n\n'
                            f"Original request: {prompt}\n"
                            f"Invalid response was: {raw_response[:100]}..."
                        )
                    else:
                        self._logger.warning("All JSON parsing attempts failed")
                        return None
                        
                except json.JSONDecodeError as e:
                    if attempt < 2:
                        self._logger.warning(f"JSON decode error on attempt {attempt + 1}: {e}")
                        prompt = f"Fix JSON syntax error: {e}\n\nReturn valid JSON only:\n{prompt}"
                    else:
                        self._logger.warning("JSON parsing failed after all retries")
                        return None
                except Exception as e:
                    self._logger.error(f"Ollama API call failed on attempt {attempt + 1}: {e}")
                    if attempt == 2:
                        return None
            
            return None
                
        except Exception as e:
            self._logger.error(f"Ollama expansion completely failed: {e}")
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
                
                # Parse the new format: {"expanded": {"keyword": ["syn1", "syn2"]}}
                expanded_data = data.get("expanded", {})
                themes = data.get("themes", [])
                genres = data.get("genres", [])
                
                # Collect all expansion terms
                expanded_terms = []
                detected_entities = {}
                
                # Add synonyms for each keyword
                for keyword, synonyms in expanded_data.items():
                    if isinstance(synonyms, list):
                        # Add best synonyms (limit to 2-3 per keyword)
                        best_synonyms = synonyms[:3]
                        expanded_terms.extend(best_synonyms)
                
                # Add themes and genres if provided
                if themes:
                    detected_entities["themes"] = themes
                    expanded_terms.extend(themes[:2])  # Add best themes
                    
                if genres:
                    detected_entities["genres"] = genres
                    expanded_terms.extend(genres[:2])  # Add best genres
                
                # Remove duplicates and filter out original query words
                original_words = set(original_query.lower().split())
                unique_terms = []
                seen = set()
                
                for term in expanded_terms:
                    term_clean = term.lower().strip()
                    if (term_clean not in original_words and 
                        term_clean not in seen and 
                        len(term_clean) > 2 and
                        term_clean.isalpha()):
                        unique_terms.append(term)
                        seen.add(term_clean)
                
                # Limit total expansion
                final_terms = unique_terms[:8]
                
                # Build expanded query
                if final_terms:
                    expanded_query = f"{original_query} {' '.join(final_terms)}"
                else:
                    expanded_query = original_query
                
                return QueryExpansion(
                    original_query=original_query,
                    expanded_query=expanded_query,
                    added_terms=final_terms,
                    detected_entities=detected_entities,
                    confidence_score=0.9,
                    processing_method="ollama_clean_json"
                )
            else:
                # No valid JSON found - fallback to word extraction
                self._logger.warning("No valid JSON found in Ollama response")
                return QueryExpansion(
                    original_query=original_query,
                    expanded_query=original_query,
                    added_terms=[],
                    detected_entities={},
                    confidence_score=0.5,
                    processing_method="ollama_no_json"
                )
                
        except json.JSONDecodeError as e:
            self._logger.warning(f"Could not parse Ollama JSON response: {e}")
            return QueryExpansion(
                original_query=original_query,
                expanded_query=original_query,
                added_terms=[],
                detected_entities={},
                confidence_score=0.5,
                processing_method="ollama_json_error"
            )
        except Exception as e:
            self._logger.error(f"Error parsing Ollama response: {e}")
            return QueryExpansion(
                original_query=original_query,
                expanded_query=original_query,
                added_terms=[],
                detected_entities={},
                confidence_score=0.5,
                processing_method="ollama_parse_error"
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
    
    def _get_wordnet_synonyms(self, word: str, max_synonyms: int = 3) -> List[str]:
        """Get WordNet synonyms for a word."""
        if not _NLTK_AVAILABLE:
            return []
        
        synonyms = set()
        
        try:
            # Get synonyms from WordNet
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ').lower()
                    if synonym != word and synonym.isalpha():
                        synonyms.add(synonym)
                        if len(synonyms) >= max_synonyms:
                            break
                if len(synonyms) >= max_synonyms:
                    break
            
            return list(synonyms)[:max_synonyms]
            
        except Exception as e:
            self._logger.debug(f"WordNet lookup failed for '{word}': {e}")
            return []
    
    def _detect_entity_type(self, word: str) -> Optional[str]:
        """Detect the type of entity (genre, mood, etc.) from a word."""
        # Simple entity type detection based on WordNet hypernyms
        try:
            synsets = wordnet.synsets(word)
            if not synsets:
                return None
            
            # Get the most common synset
            synset = synsets[0]
            
            # Check hypernyms to classify entity type
            hypernyms = [h.name() for h in synset.hypernyms()]
            
            # Movie genre indicators
            if any(h in hypernyms for h in ['genre.n.01', 'category.n.01', 'kind.n.01']):
                return 'genre'
            
            # Mood/emotion indicators
            if any(h in hypernyms for h in ['emotion.n.01', 'feeling.n.01', 'mood.n.01']):
                return 'mood'
            
            # Time indicators
            if any(h in hypernyms for h in ['time_period.n.01', 'decade.n.01']):
                return 'time_period'
            
            # Location indicators
            if any(h in hypernyms for h in ['location.n.01', 'place.n.01', 'country.n.01']):
                return 'location'
            
            return None
            
        except Exception:
            return None
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        if self._ollama_client:
            await self._ollama_client.aclose()
            self._ollama_client = None
        
        self._expansion_dictionaries.clear()
        self._movie_entities.clear()
        self._logger.info("Adaptive Query Expander cleaned up")