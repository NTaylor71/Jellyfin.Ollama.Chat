# COMPREHENSIVE REFACTORING PLAN: JELLYFIN FIELD ADAPTER & MEDIA-AGNOSTIC ARCHITECTURE

## Overview
Transform the current Jellyfin-specific, movie-centric system into a media-agnostic, adapter-driven architecture that can handle any media source (Jellyfin, Plex, Emby, TMDB, etc.) and any media type (movies, TV, books, music, comics, audiobooks).

## Phase 1: Jellyfin Field Adapter System (IMMEDIATE - 1 week)

### Goal
Eliminate hard-coded Jellyfin field assumptions and create a JSON config-driven field mapping system.

### 1.1: Create Field Mapping Configuration System

**Files to Create:**
```
config/adapters/
├── jellyfin_movies.json          # Current Jellyfin movie fields
├── jellyfin_tv_shows.json        # Future Jellyfin TV fields  
├── plex_movies.json              # Future Plex movie fields
├── tmdb_movies.json              # Future TMDB fields
└── adapter_schema.json           # Validation schema
```

**Jellyfin Movies Config Structure:**
```json
{
  "adapter_name": "jellyfin_movies",
  "media_type": "movie",
  "version": "1.0",
  "field_mappings": {
    "core_fields": {
      "title": ["name", "title"],
      "description": ["overview", "plot", "synopsis"],  
      "release_date": ["premiere_date", "production_year"],
      "genres": ["genres"],
      "cast": ["people.actor", "people.director"],
      "runtime": ["run_time_ticks"],
      "rating": ["community_rating", "critic_rating"]
    },
    "metadata_fields": {
      "external_ids": ["provider_ids"],
      "images": ["image_tags", "backdrop_image_tags"],
      "technical": ["container", "has_subtitles", "is_hd"],
      "production": ["studios", "production_locations"]
    },
    "search_fields": {
      "primary": ["name", "overview", "taglines"],
      "secondary": ["people.name", "genres", "studios"],
      "keywords": ["tags", "keywords"]
    }
  },
  "data_transformations": {
    "runtime_ticks_to_minutes": "run_time_ticks / 600000000",
    "people_by_type": "group people by type (Actor, Director, etc)",
    "genre_normalization": "map to standard genre taxonomy"
  }
}
```

### 1.2: Create Base Adapter Classes

**File: `src/adapters/base.py`**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class MediaAdapter(ABC):
    """Base class for all media source adapters"""
    
    @abstractmethod
    def normalize_fields(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert source-specific fields to standard format"""
        pass
    
    @abstractmethod
    def extract_search_content(self, normalized_data: Dict[str, Any]) -> str:
        """Extract searchable text content"""
        pass
    
    @abstractmethod
    def get_media_type(self) -> str:
        """Return media type (movie, tv_show, book, etc)"""
        pass

class ConfigDrivenAdapter(MediaAdapter):
    """JSON config-driven adapter implementation"""
    
    def __init__(self, config_file: str):
        self.config = self._load_config(config_file)
        self.field_mappings = self.config["field_mappings"]
        self.transformations = self.config["data_transformations"]
```

### 1.3: Implement Jellyfin Movie Adapter

**File: `src/adapters/jellyfin_movies.py`**
```python
class JellyfinMovieAdapter(ConfigDrivenAdapter):
    """Jellyfin-specific movie data adapter"""
    
    def __init__(self):
        super().__init__("config/adapters/jellyfin_movies.json")
    
    def normalize_fields(self, jellyfin_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Jellyfin movie data to standard movie format"""
        normalized = {}
        
        # Map core fields using config
        for standard_field, jellyfin_fields in self.field_mappings["core_fields"].items():
            normalized[standard_field] = self._extract_field_value(jellyfin_data, jellyfin_fields)
        
        # Apply transformations
        normalized = self._apply_transformations(normalized, jellyfin_data)
        
        # Preserve original for debugging
        normalized["_source"] = {
            "adapter": "jellyfin_movies",
            "original_data": jellyfin_data,
            "mapping_version": self.config["version"]
        }
        
        return normalized
```

### 1.4: Update Data Models for Adapter Support

**File: `src/data/models.py` - Enhanced**
```python
class StandardMovie(BaseModel):
    """Standard movie format - adapter-agnostic"""
    
    # Core fields (mapped from any source)
    title: str
    description: Optional[str] = None
    release_date: Optional[datetime] = None
    genres: List[str] = []
    cast: List[Person] = []
    runtime_minutes: Optional[int] = None
    rating: Optional[float] = None
    
    # Enhanced processing fields
    enhanced_fields: Dict[str, Any] = {}
    
    # Source tracking
    source_adapter: str
    source_data: Dict[str, Any] = {}
    
    # Search optimization
    search_content: str = ""
    search_tags: List[str] = []

class MovieAdapter(BaseModel):
    """Adapter configuration model"""
    adapter_name: str
    media_type: str
    version: str
    field_mappings: Dict[str, Any]
    data_transformations: Dict[str, str]
```

### 1.5: Update Ingestion Pipeline

**File: `src/ingestion/jellyfin_connector.py` - Refactored**
```python
class MediaIngestionPipeline:
    """Adapter-driven media ingestion pipeline"""
    
    def __init__(self):
        self.adapters = self._load_adapters()
        
    def _load_adapters(self) -> Dict[str, MediaAdapter]:
        """Load all available adapters"""
        return {
            "jellyfin_movies": JellyfinMovieAdapter(),
            # Future: "plex_movies": PlexMovieAdapter(),
            # Future: "tmdb_movies": TMDBMovieAdapter(),
        }
    
    async def ingest_media(self, raw_data: Dict[str, Any], adapter_name: str) -> StandardMovie:
        """Ingest media using specified adapter"""
        adapter = self.adapters[adapter_name]
        
        # Normalize fields using adapter
        normalized_data = adapter.normalize_fields(raw_data)
        
        # Create standard movie object
        movie = StandardMovie(**normalized_data)
        
        # Extract search content
        movie.search_content = adapter.extract_search_content(normalized_data)
        
        # Run enhancement plugins
        movie = await self._run_enhancement_plugins(movie)
        
        return movie
```

### 1.6: Files Requiring Updates

**Files to Modify:**
1. `src/data/mongo_client.py` - Update for StandardMovie model
2. `src/plugins/examples/movie_summary_enhancer.py` - Use adapter-agnostic fields
3. `src/search/text_processor.py` - Use configured search fields
4. `test_mongodb_integration.py` - Update for new adapter system

**Hard-Coded Field References to Replace:**
```python
# OLD (Jellyfin-specific)
data.get("name")  # Hard-coded Jellyfin field
data.get("overview")  # Hard-coded Jellyfin field
data.get("people", [])  # Hard-coded Jellyfin structure

# NEW (Adapter-agnostic)  
normalized_data.get("title")  # Standard field
normalized_data.get("description")  # Standard field
normalized_data.get("cast", [])  # Standard structure
```

## Phase 2: Media Type Abstraction (2 weeks)

### 2.1: Create Media Type System
```python
class MediaType(Enum):
    MOVIE = "movie"
    TV_SHOW = "tv_show" 
    BOOK = "book"
    MUSIC_ALBUM = "music_album"
    PODCAST = "podcast"
    COMIC = "comic"
    AUDIOBOOK = "audiobook"

class MediaTypeAdapter(ABC):
    """Base for media type-specific processing"""
    
    @abstractmethod
    def get_search_fields(self) -> List[str]:
        """Get fields relevant for search"""
        pass
    
    @abstractmethod
    def get_temporal_fields(self) -> List[str]:
        """Get fields containing temporal information"""
        pass
```

### 2.2: Update Plugin System
- Make all plugins media-type aware
- Add media type configuration to plugin configs
- Update plugin discovery to filter by media type

### 2.3: Search System Updates
- Media-type specific search strategies
- Configurable field weights per media type
- Genre taxonomies per media type

## Phase 3: Configuration-Driven Patterns (1 week)

### 3.1: Replace Hard-Coded Patterns
```yaml
# config/media_patterns/movies.yml
genre_patterns:
  action: ["action", "adventure", "thriller"]
  drama: ["drama", "biographical", "historical"]
  
temporal_patterns:
  decade_indicators: ["90s", "eighties", "millennium"]
  release_patterns: ["released in", "from", "circa"]

search_weights:
  title: 3.0
  description: 2.0
  cast: 2.5
```

### 3.2: Plugin Configuration System
- Media-type specific plugin configs
- Dynamic pattern loading
- Runtime pattern updates

## Phase 4: Future Extensions (ongoing)

### 4.1: Additional Adapters
- Plex Movie Adapter
- TMDB Adapter  
- Goodreads Book Adapter
- Spotify Music Adapter

### 4.2: Additional Media Types
- TV Shows (episodes, seasons, series)
- Books (authors, publishers, series)
- Music (albums, artists, tracks)
- Podcasts (episodes, series)

## Implementation Strategy

### Week 1: Jellyfin Adapter Foundation
- [ ] Create adapter configuration system
- [ ] Implement JellyfinMovieAdapter
- [ ] Update StandardMovie model
- [ ] Test with existing Jellyfin data

### Week 2: Pipeline Integration  
- [ ] Update ingestion pipeline
- [ ] Modify MongoDB client
- [ ] Update enhancement plugins
- [ ] Integration testing

### Week 3: Search System Updates
- [ ] Update text processor for adapter fields
- [ ] Modify search strategies
- [ ] Test search functionality
- [ ] Performance optimization

### Week 4: Documentation & Testing
- [ ] Complete test coverage
- [ ] Performance benchmarking
- [ ] Documentation updates
- [ ] Production readiness

## Success Criteria

1. **Zero Hard-Coded Jellyfin Fields**: All field references go through adapter
2. **Config-Driven**: All media source differences handled via JSON config
3. **Backward Compatibility**: Existing Jellyfin movie ingestion works unchanged
4. **Extensibility**: New media sources can be added with just config files
5. **Performance**: No regression in ingestion or search performance
6. **Test Coverage**: 95%+ coverage of adapter system

## Benefits After Refactoring

1. **Media Source Agnostic**: Easy to add Plex, TMDB, etc.
2. **Media Type Ready**: Foundation for books, music, TV shows
3. **Maintainable**: Changes to field mappings don't require code changes
4. **Testable**: Adapter logic is isolated and easily tested
5. **Scalable**: New media sources/types via configuration
6. **Future-Proof**: Architecture supports any media ecosystem

## Ubuntu Reboot Benefits

With full Ubuntu access, you'll be able to:
- **Run tests automatically** during refactoring
- **Validate data transformations** with real data
- **Performance test** adapter overhead
- **Integration test** the full pipeline
- **Debug issues** in real-time
- **Ensure quality** with continuous testing

This refactoring will transform the system from Jellyfin-specific to truly media-agnostic while maintaining all existing functionality!