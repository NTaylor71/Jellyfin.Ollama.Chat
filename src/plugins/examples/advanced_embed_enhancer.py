"""
Advanced Embed Data Enhancer Plugin
Intelligently enhances document data before embedding with hardware-adaptive processing.
"""

import asyncio
import json
import re
import logging
from typing import Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import httpx

# ── NLTK Setup (module level, run once) ──────────────────────────────────────
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Optional dependencies
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    _SPACY_AVAILABLE = False

try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except ImportError:
    TextBlob = None
    _TEXTBLOB_AVAILABLE = False

# Basic stopwords
_BASIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "for", "from", "has", "he", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "to", "was", "were",
    "will", "with", "this", "these", "those", "they", "them"
}

def ensure_nltk_corpora() -> None:
    """Ensure required NLTK corpora are downloaded (with timeout protection)."""
    # Essential corpora only - avoid problematic ones
    essential_corpora = [
        ("wordnet", "corpora/wordnet"), ("punkt", "tokenizers/punkt"), ("stopwords", "corpora/stopwords")
    ]
    
    for corpus_name, corpus_path in essential_corpora:
        try:
            nltk.data.find(corpus_path)
        except LookupError:
            try:
                # Try to download with timeout protection
                print(f"Downloading NLTK corpus: {corpus_name}")
                nltk.download(corpus_name, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download NLTK corpus {corpus_name}: {e}")
                # Continue without this corpus - we'll handle missing corpora gracefully

# Initialize NLTK resources once at module load
try:
    ensure_nltk_corpora()
except Exception as e:
    print(f"Warning: NLTK setup failed: {e}")

# Initialize global resources
try:
    _LEMMATIZER = WordNetLemmatizer()
    try:
        _STOP_WORDS = set(stopwords.words('english')).union(_BASIC_STOPWORDS)
    except Exception:
        _STOP_WORDS = _BASIC_STOPWORDS
    _NLTK_AVAILABLE = True
except Exception:
    _LEMMATIZER = None
    _STOP_WORDS = _BASIC_STOPWORDS
    _NLTK_AVAILABLE = False

from src.plugins.base import (
    EmbedDataEmbellisherPlugin, PluginMetadata, PluginResourceRequirements, 
    PluginExecutionContext, PluginExecutionResult, PluginType, ExecutionPriority
)
from src.shared.config import get_settings
from src.shared.hardware_config import get_resource_limits, get_hardware_profile


@dataclass
class ProcessingStrategy:
    """Represents a processing strategy with resource requirements."""
    name: str
    min_cpu_cores: float
    min_memory_mb: float
    max_execution_time: float
    features: List[str]
    

@dataclass
class EnhancementResult:
    """Represents the result of data enhancement."""
    original_data: Dict[str, Any]
    enhanced_data: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    confidence_score: float
    processing_strategy: str


class AdvancedEmbedDataEnhancerPlugin(EmbedDataEmbellisherPlugin):
    """
    Advanced embed data enhancer that scales processing based on available resources.
    
    Processing strategies:
    - Low resources: Basic text cleaning and simple entity extraction
    - Medium resources: NLTK processing with parallel text operations
    - High resources: Full NLP pipeline with multiprocessing
    - Ollama Enhanced: LLM-based semantic understanding
    """
    
    def __init__(self):
        super().__init__()
        self._processing_strategies = {}
        self._hardware_profile = None
        self._ollama_client = None
        self._ollama_gpu_available = False
        self._settings = None
        self._spacy_model = None
        self._movie_patterns = {}
        
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="AdvancedEmbedDataEnhancer",
            version="1.0.0",
            description="Intelligently enhances document data before embedding with hardware-adaptive processing",
            author="RAG System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["embedding", "preprocessing", "nlp", "movies", "adaptive"],
            execution_priority=ExecutionPriority.HIGH
        )
    
    @property 
    def resource_requirements(self) -> PluginResourceRequirements:
        return PluginResourceRequirements(
            min_cpu_cores=1.0,
            preferred_cpu_cores=4.0,
            min_memory_mb=100.0,
            preferred_memory_mb=500.0,
            requires_gpu=False,
            max_execution_time_seconds=15.0,
            can_use_distributed_resources=True
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with processing strategies and NLP resources."""
        try:
            self._logger.info("Initializing Advanced Embed Data Enhancer...")
            
            # Get settings
            self._settings = get_settings()
            
            # Initialize hardware profile
            await self._initialize_hardware_registry()
            
            # Initialize processing strategies
            self._initialize_processing_strategies()
            
            # Initialize movie-specific patterns
            self._initialize_movie_patterns()
            
            # Initialize Ollama client
            await self._initialize_ollama_client()
            
            # Initialize spaCy model if available
            await self._initialize_spacy_model()
            
            self._is_initialized = True
            self._logger.info("Advanced Embed Data Enhancer initialized successfully")
            return True
            
        except Exception as e:
            self._initialization_error = str(e)
            self._logger.error(f"Failed to initialize Advanced Embed Data Enhancer: {e}")
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
    
    def _initialize_processing_strategies(self) -> None:
        """Initialize processing strategies based on resource requirements."""
        self._processing_strategies = {
            "low_resource": ProcessingStrategy(
                name="Low Resource",
                min_cpu_cores=1.0,
                min_memory_mb=100.0,
                max_execution_time=5.0,
                features=["basic_cleaning", "simple_entities", "basic_metadata"]
            ),
            "medium_resource": ProcessingStrategy(
                name="Medium Resource",
                min_cpu_cores=2.0,
                min_memory_mb=200.0,
                max_execution_time=10.0,
                features=["nltk_processing", "enhanced_entities", "parallel_processing", "sentiment_analysis"]
            ),
            "high_resource": ProcessingStrategy(
                name="High Resource",
                min_cpu_cores=4.0,
                min_memory_mb=500.0,
                max_execution_time=15.0,
                features=["full_nlp_pipeline", "multiprocessing", "advanced_metadata", "content_scoring"]
            ),
            "ollama_enhanced": ProcessingStrategy(
                name="Ollama Enhanced",
                min_cpu_cores=2.0,
                min_memory_mb=300.0,
                max_execution_time=20.0,
                features=["llm_understanding", "semantic_metadata", "advanced_content_analysis"]
            )
        }
    
    def _initialize_movie_patterns(self) -> None:
        """Initialize movie-specific patterns and dictionaries."""
        self._movie_patterns = {
            "genres": [
                "action", "adventure", "animation", "comedy", "crime", "documentary",
                "drama", "family", "fantasy", "horror", "mystery", "romance",
                "sci-fi", "science fiction", "thriller", "war", "western", "musical",
                "biographical", "historical", "psychological", "superhero", "zombie"
            ],
            "movie_keywords": [
                "film", "movie", "cinema", "picture", "flick", "feature",
                "director", "actor", "actress", "cast", "crew", "producer",
                "screenplay", "script", "plot", "story", "narrative",
                "rating", "review", "critique", "oscar", "award", "nomination"
            ],
            "quality_indicators": [
                "excellent", "outstanding", "masterpiece", "brilliant", "amazing",
                "terrible", "awful", "disappointing", "mediocre", "average",
                "classic", "cult", "iconic", "legendary", "groundbreaking"
            ]
        }
    
    async def _initialize_ollama_client(self) -> None:
        """Initialize Ollama client for LLM-enhanced processing."""
        try:
            # Try different settings attributes for Ollama URL
            ollama_url = getattr(self._settings, 'ollama_url', None) or getattr(self._settings, 'OLLAMA_CHAT_BASE_URL', None)
            if not ollama_url:
                self._logger.info("No Ollama URL configured, skipping LLM integration")
                return
            
            self._ollama_client = httpx.AsyncClient(
                base_url=ollama_url,
                timeout=30.0
            )
            
            # Test Ollama connection
            response = await self._ollama_client.get("/api/tags")
            if response.status_code == 200:
                self._ollama_gpu_available = True
                self._logger.info("Ollama client initialized successfully")
            else:
                self._logger.warning(f"Ollama connection test failed: {response.status_code}")
                
        except Exception as e:
            self._logger.warning(f"Could not initialize Ollama client: {e}")
            self._ollama_client = None
    
    async def _initialize_spacy_model(self) -> None:
        """Initialize spaCy model if available."""
        if not _SPACY_AVAILABLE:
            self._logger.info("spaCy not available, using NLTK for NLP tasks")
            return
            
        try:
            # Try to load English model
            self._spacy_model = spacy.load("en_core_web_sm")
            self._logger.info("spaCy model loaded successfully")
        except Exception as e:
            self._logger.warning(f"Could not load spaCy model: {e}")
            self._spacy_model = None
    
    async def embellish_embed_data(self, data: Dict[str, Any], context: PluginExecutionContext) -> Dict[str, Any]:
        """Enhance embed data with adaptive processing based on available resources."""
        try:
            # Determine optimal processing strategy
            strategy = self._select_processing_strategy(context)
            
            self._logger.info(f"Using {strategy.name} processing strategy")
            
            # Extract text content
            text_content = self._extract_text_content(data)
            if not text_content:
                self._logger.warning("No text content found for enhancement")
                return data
            
            # Process data based on strategy
            if strategy.name == "Low Resource":
                enhanced_data = await self._process_low_resource(data, text_content, context)
            elif strategy.name == "Medium Resource":
                enhanced_data = await self._process_medium_resource(data, text_content, context)
            elif strategy.name == "High Resource":
                enhanced_data = await self._process_high_resource(data, text_content, context)
            elif strategy.name == "Ollama Enhanced":
                enhanced_data = await self._process_ollama_enhanced(data, text_content, context)
            else:
                enhanced_data = await self._process_low_resource(data, text_content, context)
            
            # Add processing metadata
            enhanced_data['enhancement_metadata'] = {
                'processing_strategy': strategy.name,
                'processing_timestamp': datetime.utcnow().isoformat(),
                'plugin_version': self.metadata.version,
                'available_cpu_cores': context.available_resources.get('cpu_cores', 1.0),
                'available_memory_mb': context.available_resources.get('memory_mb', 100.0)
            }
            
            self._logger.info(f"Enhanced data with {len(enhanced_data) - len(data)} new fields")
            return enhanced_data
            
        except Exception as e:
            self._logger.error(f"Error enhancing embed data: {e}")
            return data  # Return original data on error
    
    def _select_processing_strategy(self, context: PluginExecutionContext) -> ProcessingStrategy:
        """Select the optimal processing strategy based on available resources."""
        # Handle different resource field names for compatibility
        available_cpu = context.available_resources.get('total_cpu_capacity', 
                          context.available_resources.get('cpu_cores', 1.0))
        
        # Handle memory in both MB and GB formats
        available_memory_mb = context.available_resources.get('memory_mb', None)
        if available_memory_mb is None:
            # Convert from GB to MB
            available_memory_gb = context.available_resources.get('local_memory_gb', 0.1)
            available_memory_mb = available_memory_gb * 1024
        
        available_memory = available_memory_mb
        
        # Check for Ollama availability
        if self._ollama_client and self._ollama_gpu_available:
            ollama_strategy = self._processing_strategies["ollama_enhanced"]
            if (available_cpu >= ollama_strategy.min_cpu_cores and 
                available_memory >= ollama_strategy.min_memory_mb):
                return ollama_strategy
        
        # Check for high resource strategy
        high_strategy = self._processing_strategies["high_resource"]
        if (available_cpu >= high_strategy.min_cpu_cores and 
            available_memory >= high_strategy.min_memory_mb):
            return high_strategy
        
        # Check for medium resource strategy
        medium_strategy = self._processing_strategies["medium_resource"]
        if (available_cpu >= medium_strategy.min_cpu_cores and 
            available_memory >= medium_strategy.min_memory_mb):
            return medium_strategy
        
        # Default to low resource strategy
        return self._processing_strategies["low_resource"]
    
    def _extract_text_content(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract text content from various data fields including Jellyfin format."""
        text_parts = []
        
        # Primary content fields (plot/description)
        primary_fields = ['text', 'content', 'description', 'summary', 'title', 'body', 'plot', 'overview', 'Overview']
        for field in primary_fields:
            if field in data and isinstance(data[field], str) and data[field].strip():
                text_parts.append(data[field].strip())
        
        # Jellyfin-specific fields
        jellyfin_fields = {
            'Name': 'title',  # Movie title
            'OriginalTitle': 'original_title',
            'ProductionYear': 'year',
            'Taglines': 'taglines',  # Marketing taglines - great for search!
            'Genres': 'genres',
            'Tags': 'tags',  # Curated descriptive keywords
            'OfficialRating': 'rating',
            'Language': 'language'  # Primary language
        }
        
        for field, description in jellyfin_fields.items():
            if field in data:
                value = data[field]
                if isinstance(value, str) and value.strip():
                    text_parts.append(value.strip())
                elif isinstance(value, list):
                    # Handle arrays like Genres, Tags, Taglines
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            text_parts.append(item.strip())
                        elif isinstance(item, dict):
                            # Handle complex objects in arrays
                            if 'Name' in item:
                                text_parts.append(str(item['Name']))
                elif isinstance(value, (int, float)):
                    # Handle years, ratings
                    text_parts.append(str(value))
        
        # Handle People array (cast/crew)
        if 'People' in data and isinstance(data['People'], list):
            for person in data['People']:
                if isinstance(person, dict):
                    name = person.get('Name', '')
                    role = person.get('Role', '')
                    person_type = person.get('Type', '')
                    if name:
                        text_parts.append(f"{name} {role} {person_type}".strip())
        
        # Extract languages from MediaStreams
        if 'MediaStreams' in data and isinstance(data['MediaStreams'], list):
            languages = set()
            for stream in data['MediaStreams']:
                if isinstance(stream, dict) and 'Language' in stream:
                    lang = stream['Language']
                    if lang and lang != 'Unknown':
                        languages.add(lang)
            if languages:
                text_parts.extend(list(languages))
        
        # Fallback for simple metadata fields
        metadata_fields = ['genre', 'year', 'director', 'cast', 'rating', 'author', 'keywords']
        for field in metadata_fields:
            if field in data:
                value = data[field]
                if isinstance(value, str) and value.strip():
                    text_parts.append(value.strip())
                elif isinstance(value, list):
                    text_parts.extend([str(item) for item in value if str(item).strip()])
        
        return ' '.join(text_parts) if text_parts else None
    
    async def _process_low_resource(self, data: Dict[str, Any], text: str, context: PluginExecutionContext) -> Dict[str, Any]:
        """Process data with minimal resource usage."""
        enhanced_data = data.copy()
        
        # Basic text cleaning
        cleaned_text = self._clean_text_basic(text)
        enhanced_data['cleaned_text'] = cleaned_text
        
        # Simple entity extraction
        entities = self._extract_entities_basic(cleaned_text)
        enhanced_data['extracted_entities'] = entities
        
        # Basic metadata
        metadata = self._extract_metadata_basic(cleaned_text)
        enhanced_data['basic_metadata'] = metadata
        
        return enhanced_data
    
    async def _process_medium_resource(self, data: Dict[str, Any], text: str, context: PluginExecutionContext) -> Dict[str, Any]:
        """Process data with medium resource usage and NLTK."""
        enhanced_data = data.copy()
        
        # Enhanced text cleaning
        cleaned_text = await self._clean_text_nltk(text)
        enhanced_data['cleaned_text'] = cleaned_text
        
        # NLTK-based entity extraction
        entities = await self._extract_entities_nltk(cleaned_text)
        enhanced_data['extracted_entities'] = entities
        
        # Sentiment analysis
        if _TEXTBLOB_AVAILABLE:
            sentiment = self._analyze_sentiment(cleaned_text)
            enhanced_data['sentiment_analysis'] = sentiment
        
        # Enhanced metadata
        try:
            metadata = await self._extract_metadata_enhanced(cleaned_text, data)
            enhanced_data['enhanced_metadata'] = metadata
        except Exception as e:
            self._logger.warning(f"Enhanced metadata extraction failed: {e}")
            # Fallback to basic metadata
            basic_metadata = self._extract_metadata_basic(cleaned_text)
            enhanced_data['enhanced_metadata'] = basic_metadata
        
        return enhanced_data
    
    async def _process_high_resource(self, data: Dict[str, Any], text: str, context: PluginExecutionContext) -> Dict[str, Any]:
        """Process data with high resource usage and multiprocessing."""
        enhanced_data = data.copy()
        
        # Use multiprocessing for intensive operations
        available_cores = min(int(context.available_resources.get('cpu_cores', 2.0)), 4)
        
        # Create tasks for parallel processing
        tasks = []
        
        # Text cleaning task
        tasks.append(self._clean_text_advanced(text))
        
        # Entity extraction task
        tasks.append(self._extract_entities_advanced(text))
        
        # Metadata extraction task
        tasks.append(self._extract_metadata_advanced(text, data))
        
        # Content analysis task
        tasks.append(self._analyze_content_advanced(text))
        
        # Execute tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._logger.warning(f"Task {i} failed: {result}")
                continue
                
            if i == 0:  # Text cleaning
                enhanced_data['cleaned_text'] = result
            elif i == 1:  # Entity extraction
                enhanced_data['extracted_entities'] = result
            elif i == 2:  # Metadata extraction
                enhanced_data['advanced_metadata'] = result
            elif i == 3:  # Content analysis
                enhanced_data['content_analysis'] = result
        
        return enhanced_data
    
    async def _process_ollama_enhanced(self, data: Dict[str, Any], text: str, context: PluginExecutionContext) -> Dict[str, Any]:
        """Process data with Ollama LLM enhancement."""
        enhanced_data = await self._process_high_resource(data, text, context)
        
        # Add LLM-based enhancements
        try:
            llm_analysis = await self._analyze_with_ollama(text)
            enhanced_data['llm_analysis'] = llm_analysis
        except Exception as e:
            self._logger.warning(f"Ollama analysis failed: {e}")
        
        return enhanced_data
    
    def _clean_text_basic(self, text: str) -> str:
        """Basic text cleaning with minimal resource usage."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    async def _clean_text_nltk(self, text: str) -> str:
        """Enhanced text cleaning using NLTK."""
        if not _NLTK_AVAILABLE:
            return self._clean_text_basic(text)
        
        # Start with basic cleaning
        cleaned = self._clean_text_basic(text)
        
        # Tokenize and clean
        try:
            tokens = word_tokenize(cleaned)
            
            # Remove stopwords and short tokens
            filtered_tokens = [
                token.lower() for token in tokens
                if token.lower() not in _STOP_WORDS and len(token) > 2
            ]
            
            # Lemmatize
            if _LEMMATIZER:
                lemmatized_tokens = [_LEMMATIZER.lemmatize(token) for token in filtered_tokens]
            else:
                lemmatized_tokens = filtered_tokens
            
            return ' '.join(lemmatized_tokens)
            
        except Exception as e:
            self._logger.warning(f"NLTK cleaning failed: {e}")
            return cleaned
    
    async def _clean_text_advanced(self, text: str) -> str:
        """Advanced text cleaning with spaCy if available."""
        if self._spacy_model:
            try:
                doc = self._spacy_model(text)
                
                # Extract lemmatized tokens, excluding stop words and punctuation
                cleaned_tokens = [
                    token.lemma_.lower() for token in doc
                    if not token.is_stop and not token.is_punct and len(token.text) > 2
                ]
                
                return ' '.join(cleaned_tokens)
                
            except Exception as e:
                self._logger.warning(f"spaCy cleaning failed: {e}")
                return await self._clean_text_nltk(text)
        
        return await self._clean_text_nltk(text)
    
    def _extract_entities_basic(self, text: str) -> Dict[str, List[str]]:
        """Basic entity extraction using regex patterns."""
        entities = {
            'years': re.findall(r'\b(19\d\d|20\d\d)\b', text),
            'titles': re.findall(r'"([^"]+)"', text),
            'persons': re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', text),
            'genres': []
        }
        
        # Extract genres
        text_lower = text.lower()
        for genre in self._movie_patterns['genres']:
            if genre in text_lower:
                entities['genres'].append(genre)
        
        # Also check for partial matches for compound genres
        if 'science' in text_lower and 'fiction' in text_lower:
            entities['genres'].append('science fiction')
        if 'sci' in text_lower and 'fi' in text_lower:
            entities['genres'].append('sci-fi')
        
        return entities
    
    async def _extract_entities_nltk(self, text: str) -> Dict[str, List[str]]:
        """Enhanced entity extraction using NLTK."""
        if not _NLTK_AVAILABLE:
            return self._extract_entities_basic(text)
        
        entities = self._extract_entities_basic(text)
        
        try:
            # Use NLTK tokenization if available
            tokens = word_tokenize(text)
            
            # Try POS tagging if available
            try:
                pos_tags = pos_tag(tokens)
                
                # Try named entity chunking if available (optional)
                try:
                    named_entities = ne_chunk(pos_tags)
                    
                    # Extract named entities
                    nltk_entities = []
                    for chunk in named_entities:
                        if hasattr(chunk, 'label'):
                            entity_name = ' '.join([token for token, pos in chunk.leaves()])
                            nltk_entities.append((entity_name, chunk.label()))
                    
                    entities['nltk_entities'] = nltk_entities
                    
                except Exception as e:
                    self._logger.debug(f"NLTK NE chunking not available: {e}")
                    # Continue without NE chunking
                    
            except Exception as e:
                self._logger.debug(f"NLTK POS tagging not available: {e}")
                # Continue without POS tagging
                
        except Exception as e:
            self._logger.warning(f"NLTK entity extraction failed: {e}")
        
        return entities
    
    async def _extract_entities_advanced(self, text: str) -> Dict[str, List[str]]:
        """Advanced entity extraction using spaCy."""
        if self._spacy_model:
            try:
                doc = self._spacy_model(text)
                
                entities = {
                    'persons': [ent.text for ent in doc.ents if ent.label_ == 'PERSON'],
                    'organizations': [ent.text for ent in doc.ents if ent.label_ == 'ORG'],
                    'dates': [ent.text for ent in doc.ents if ent.label_ == 'DATE'],
                    'locations': [ent.text for ent in doc.ents if ent.label_ in ('GPE', 'LOC')],
                    'works_of_art': [ent.text for ent in doc.ents if ent.label_ == 'WORK_OF_ART'],
                    'events': [ent.text for ent in doc.ents if ent.label_ == 'EVENT']
                }
                
                # Add movie-specific entities
                text_lower = text.lower()
                entities['genres'] = [genre for genre in self._movie_patterns['genres'] if genre in text_lower]
                entities['movie_keywords'] = [kw for kw in self._movie_patterns['movie_keywords'] if kw in text_lower]
                
                return entities
                
            except Exception as e:
                self._logger.warning(f"spaCy entity extraction failed: {e}")
        
        return await self._extract_entities_nltk(text)
    
    def _extract_metadata_basic(self, text: str) -> Dict[str, Any]:
        """Basic metadata extraction."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
        }
    
    async def _extract_metadata_enhanced(self, text: str, original_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced metadata extraction with movie-specific analysis."""
        basic_metadata = self._extract_metadata_basic(text)
        
        # Movie-specific metadata
        text_lower = text.lower()
        
        # Content type detection
        content_type = 'general'
        if any(word in text_lower for word in ['plot', 'story', 'synopsis', 'overview']):
            content_type = 'plot'
        elif any(word in text_lower for word in ['review', 'rating', 'critic']):
            content_type = 'review'
        elif any(word in text_lower for word in ['cast', 'actor', 'director']):
            content_type = 'cast_info'
        
        # Quality indicators
        quality_indicators = [indicator for indicator in self._movie_patterns['quality_indicators'] if indicator in text_lower]
        
        # Extract Jellyfin-specific metadata
        jellyfin_metadata = {}
        if 'Tags' in original_data:
            jellyfin_metadata['jellyfin_tags'] = original_data['Tags']
        if 'Taglines' in original_data:
            jellyfin_metadata['jellyfin_taglines'] = original_data['Taglines']
        if 'Genres' in original_data:
            jellyfin_metadata['jellyfin_genres'] = original_data['Genres']
        if 'ProductionYear' in original_data:
            jellyfin_metadata['production_year'] = original_data['ProductionYear']
        if 'Language' in original_data:
            jellyfin_metadata['primary_language'] = original_data['Language']
        
        # Extract languages from MediaStreams
        if 'MediaStreams' in original_data and isinstance(original_data['MediaStreams'], list):
            audio_languages = []
            subtitle_languages = []
            for stream in original_data['MediaStreams']:
                if isinstance(stream, dict) and 'Language' in stream:
                    lang = stream['Language']
                    if lang and lang != 'Unknown':
                        stream_type = stream.get('Type', '')
                        if stream_type == 'Audio':
                            audio_languages.append(lang)
                        elif stream_type == 'Subtitle':
                            subtitle_languages.append(lang)
            
            if audio_languages:
                jellyfin_metadata['audio_languages'] = list(set(audio_languages))
            if subtitle_languages:
                jellyfin_metadata['subtitle_languages'] = list(set(subtitle_languages))
        
        enhanced_metadata = {
            **basic_metadata,
            'content_type': content_type,
            'quality_indicators': quality_indicators,
            'movie_relevance_score': self._calculate_movie_relevance(text_lower),
            **jellyfin_metadata
        }
        
        return enhanced_metadata
    
    async def _extract_metadata_advanced(self, text: str, original_data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced metadata extraction with full analysis."""
        enhanced_metadata = await self._extract_metadata_enhanced(text, original_data)
        
        # Add advanced analysis
        if self._spacy_model:
            try:
                doc = self._spacy_model(text)
                
                # Linguistic features
                linguistic_features = {
                    'noun_phrases': [chunk.text for chunk in doc.noun_chunks],
                    'dependency_relations': [(token.text, token.dep_, token.head.text) for token in doc],
                    'pos_distribution': self._get_pos_distribution(doc)
                }
                
                enhanced_metadata['linguistic_features'] = linguistic_features
                
            except Exception as e:
                self._logger.warning(f"Advanced metadata extraction failed: {e}")
        
        return enhanced_metadata
    
    async def _analyze_content_advanced(self, text: str) -> Dict[str, Any]:
        """Advanced content analysis."""
        analysis = {}
        
        # Readability analysis
        try:
            sentences = sent_tokenize(text) if _NLTK_AVAILABLE else text.split('.')
        except Exception:
            sentences = text.split('.')
            
        words = text.split()
        
        if sentences and words:
            avg_sentence_length = len(words) / len(sentences)
            analysis['readability'] = {
                'avg_sentence_length': avg_sentence_length,
                'complexity_score': min(avg_sentence_length / 15.0, 1.0)  # Normalized complexity
            }
        
        # Movie-specific analysis
        analysis['movie_analysis'] = {
            'genre_mentions': [genre for genre in self._movie_patterns['genres'] if genre in text.lower()],
            'movie_keywords_count': sum(1 for kw in self._movie_patterns['movie_keywords'] if kw in text.lower()),
            'quality_sentiment': self._analyze_quality_sentiment(text)
        }
        
        return analysis
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using TextBlob."""
        if not _TEXTBLOB_AVAILABLE:
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'confidence': abs(polarity)
            }
            
        except Exception as e:
            self._logger.warning(f"Sentiment analysis failed: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0}
    
    def _calculate_movie_relevance(self, text_lower: str) -> float:
        """Calculate how relevant the text is to movies."""
        movie_keywords = self._movie_patterns['movie_keywords']
        genre_keywords = self._movie_patterns['genres']
        
        keyword_count = sum(1 for kw in movie_keywords if kw in text_lower)
        genre_count = sum(1 for genre in genre_keywords if genre in text_lower)
        
        total_words = len(text_lower.split())
        relevance_score = (keyword_count + genre_count * 2) / max(total_words, 1)
        
        return min(relevance_score, 1.0)
    
    def _analyze_quality_sentiment(self, text: str) -> str:
        """Analyze quality sentiment (positive/negative/neutral)."""
        text_lower = text.lower()
        
        positive_indicators = ['excellent', 'outstanding', 'brilliant', 'amazing', 'great', 'fantastic']
        negative_indicators = ['terrible', 'awful', 'horrible', 'disappointing', 'bad', 'worst']
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _get_pos_distribution(self, doc) -> Dict[str, int]:
        """Get part-of-speech distribution."""
        pos_counts = {}
        for token in doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        return pos_counts
    
    async def _analyze_with_ollama(self, text: str) -> Dict[str, Any]:
        """Analyze text using Ollama LLM."""
        if not self._ollama_client:
            return {}
        
        try:
            # Create prompt for content analysis
            prompt = f"""Analyze the following text and provide a structured analysis in JSON format:

Text: {text[:1000]}...

Please provide:
1. content_type: (plot, review, cast_info, technical, general)
2. main_themes: list of main themes
3. key_entities: important people, places, works mentioned
4. sentiment: overall sentiment (positive, negative, neutral)
5. quality_assessment: assessment of content quality
6. movie_relevance: how relevant this is to movies (0-1 scale)

Respond with valid JSON only."""

            response = await self._ollama_client.post(
                "/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.8,
                        "num_predict": 300
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                # Try to parse JSON response
                try:
                    analysis = json.loads(generated_text)
                    return analysis
                except json.JSONDecodeError:
                    # If JSON parsing fails, extract key information
                    return {
                        'raw_analysis': generated_text,
                        'extraction_method': 'ollama_raw'
                    }
            
        except Exception as e:
            self._logger.warning(f"Ollama analysis failed: {e}")
        
        return {}
    
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        if self._ollama_client:
            await self._ollama_client.aclose()
            self._ollama_client = None
        
        self._processing_strategies.clear()
        self._movie_patterns.clear()
        
        self._logger.info("Advanced Embed Data Enhancer plugin cleaned up")