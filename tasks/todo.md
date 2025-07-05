# Complete Redo: Elegant Media Intelligence System for (in future) prompt based search

- building a hybrid llm/nlp/literal media search engine that only offers suggestions from its real-world-data db

## Core Philosophy

**NO HARD-CODING ANYWHERE** - Everything procedural, intelligent, and data-driven.

**Two Types of Intelligence:**
1. **Ingestion Plugins** - Enrich raw media data (movies, future: TV/books/music) using NLP/LLM
2. **Query Plugins** - Intelligently understand user search intent using same NLP/LLM tools
3. **Dual-Use Plugins** - Same intelligence for both ingestion and queries

**Two Types of Caching:**
1. **Media Entity Cache** - Enhanced metadata stored within media documents (initially movie data from jellyfin - see data/example data/example_movie_data.py) the point being : jellyfin fields should be enough input to enrich from, whether movie fields or tv fields or book fields
2. **Concept Expansion Cache** - Reusable NLP results (ConceptNet expansions, LLM responses) to avoid duplicate API calls

## What to Preserve/Rebuild ✅

These are the core architectural patterns and infrastructure that work well and should be recreated:

- **Plugin Architecture** - Dynamic loading, hot-reload, hardware-adaptive plugin system
- **Redis Queue System** - Hardware-aware task distribution and queue management
- **Hardware Config** - Resource management and hardware detection
- **Environment Setup** - Proper dependency management, environment variables, development setup
- **MongoDB Integration** - Document storage with rich metadata capabilities
- **NLP Tools Integration** - Pattern of integrating sklearn, Duckling, Spacey, sutime, Faiss, ConceptNet, Wordnet, Ollama, NLTK, Gensim, Heideltime within plugins
- **NLP Tools should centrally initialize** - all download packs (things like ("wordnet", "corpora/wordnet"), ("punkt", "tokenizers/punkt"), ("stopwords", "corpora/stopwords"), )
- **Procedural Enhancement Pattern** - Plugins that generate intelligent enhancements rather than using hard-coded rules
- **Metrics are sent to Prometheus/Grafana for as much as possible : plugins, api, searches etc etc etc**

## Plugin Architecture Details

### Core Plugin System Design
**Base Plugin Classes:**
- `BasePlugin` - Abstract base with resource requirements, health checks, safe execution
- `QueryEmbellisherPlugin` - For query enhancement/expansion 
- `EmbedDataEmbellisherPlugin` - For data enrichment during ingestion
- `FAISSCRUDPlugin` - For FAISS operations (future use)

**Key Plugin Features:**
- **Hardware-Adaptive** - Plugins declare resource needs (CPU, memory, GPU)
- **Hot-Reload** - File watcher detects changes, reloads without restart
- **Health Monitoring** - Each plugin reports health status and metrics
- **Safe Execution** - Timeout protection, error handling, resource checks
- **Configuration Support** - Plugin-specific config files (YAML/JSON)

**Plugin Execution Flow:**
1. Plugin declares resource requirements
2. System checks available hardware via hardware_config
3. Plugin executes with timeout and resource monitoring
4. Results include execution time, success status, metrics
5. Failures handled gracefully with fallback behavior

**Plugin Registry Pattern:**
- Automatic discovery of plugins in directory
- Registration with metadata (name, version, type)
- Dependency resolution between plugins
- Enable/disable plugins at runtime
- Performance metrics collection

**Example Plugin Structure:**
```python
class MyEnhancerPlugin(EmbedDataEmbellisherPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="MyEnhancer",
            version="1.0.0",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        return PluginResourceRequirements(
            min_cpu_cores=1.0,
            preferred_memory_mb=500.0,
            requires_gpu=False
        )
    
    async def embellish_embed_data(self, data: Dict, context: PluginExecutionContext) -> Dict:
        # Procedural enhancement logic here
        # Check cache, call NLP tools, return enriched data
```

## Stage 1: Clean Foundation
**Goal: Build core system architecture without any hard-coded patterns from the start**

### 1.1: Core Infrastructure Setup
- [ ] **Explain your implementation plan for stage 1.1**
- [ ] **Initialize project structure** - Basic Python project with proper package structure
- [ ] **Set up environment management** - pyproject.toml, .env support, dev_setup scripts, PEP-621 "single source of truth", see dev_setup.sh, .env & src/shared/config.py
- [ ] **Install base dependencies** - MongoDB driver, Redis, FastAPI, essential tools - see pyproject.toml & docker-compose.dev.yml
- [ ] **Create config system** - Environment variable handling, no hard-coded values, see config.py, .env, docker-compose.dev.yml
- [ ] **Verify basic connectivity** - MongoDB, Redis, development environment working
- [ ] **teach me what you did**

### 1.2: Establish Data Flow Contracts
- [ ] **Define MediaEntity interface** - What fields every media type (movie/TV/book/music) must have
- [ ] **Define EnhancementResult interface** - Standard format for plugin outputs
- [ ] **Define CacheKey interface** - How we identify and retrieve cached NLP results
- [ ] **Create test data** - Real Jellyfin movie data samples for testing

## Stage 2: Concept Expansion Cache
**Goal: Never call ConceptNet/LLM twice for same input**

### 2.1: Cache Infrastructure
- [ ] **Create ConceptCache collection** in MongoDB
  ```javascript
  {
    "_id": ObjectId,
    "cache_key": "concept:action:movie", // Type:Term:MediaType
    "input_term": "action",
    "media_type": "movie", 
    "expansion_type": "conceptnet", // conceptnet|llm|gensim|nltk
    "expanded_terms": ["fight", "combat", "battle", "intense"],
    "confidence_scores": {"fight": 0.9, "combat": 0.85},
    "source_metadata": {"api": "conceptnet", "endpoint": "/related"},
    "created_at": ISODate,
    "expires_at": ISODate // TTL for cache refresh
  }
  ```

### 2.2: Cache Management Service
- [ ] **CacheManager class** - Check cache before calling external APIs
- [ ] **Cache key generation** - Consistent hashing for lookups
- [ ] **TTL management** - Configurable expiration for different cache types
- [ ] **Cache warming** - Pre-populate common terms

## Stage 3: Procedural Concept Expansion
**Goal: avoid all hard-coded genre/keyword lists with intelligent expansion**

### 3.1: ConceptNet Expansion Service
- [ ] **ConceptExpander class**
  - Input example: "action" + "movie" context
  - Check cache first
  - If miss: Call ConceptNet API and or llms if plugins suggest this behaviour - with rate limiting
  - Store results in cache
  - Return expanded concepts: ["fight", "combat", "battle", "intense", "fast-paced"]

### 3.2: LLM Concept Understanding
- [ ] **LLMConceptAnalyzer class**
  - Input: User query "psychological thriller movies"
  - Use Ollama to understand intent and extract concepts - a plugin might then enrich those concepts further
  - Cache the analysis results
  - Return structured concept analysis

### 3.3: Multi-Source Concept Fusion
- [ ] **ConceptFusion class**
  - Combine ConceptNet + LLM + Gensim results + other plugins
  - Weight and rank concepts by confidence
  - Handle conflicts between sources
  - Return unified concept expansion
  - everything stored as pure asci : unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")

## Stage 4: Intelligent Media Analysis
**Goal: Analyze real movie data to understand what concepts actually mean**

### 4.1: Movie Content Analyzer
- [ ] **Analyze real Jellyfin movie data**
  - example data/data/example_movie_data.py
  - Take actual field names and use the same as the source in any cache/mongodb. Initial fields for movies : 'Name', 'OriginalTitle', ProductionYear', 'Taglines', 'Genres', 'Tags', 'OfficialRating', 'Language'
  - never assume the fields contain simple strings, thiough they might be simple strings, preserve list types
  - Use NLP to identify patterns: "What words appear in action movie descriptions?"
  - Build statistical models of concept relationships
  - Store insights in MongoDB for reuse

### 4.2: Contextual Concept Learning
- [ ] **Learn from actual usage**
  - "Movies tagged 'Action' actually contain these words: [intense, fast, combat]"
  - "Movies users search for with 'thriller' have these common elements"
  - Build concept-to-content mappings from real data

## Stage 5: Media-Agnostic Intelligence
**Goal: Same intelligence works for movies, TV, books, music, comics**

### 5.1: Media Type Detection and Adaptation
- [ ] **MediaTypeDetector**
  - Analyze content to determine media type
  - Adapt concept expansion based on media type
  - "action" in movies ≠ "action" in books ≠ "action" in music

### 5.2: Cross-Media Concept Transfer
- [ ] **Learn concept patterns per media type**
  - How "dark" manifests in movies vs books vs music
  - Transfer learning between media types
  - Elegant abstraction without hard-coding

## Stage 6: Intelligent Query Processing
**Goal: Understand user intent without pattern matching**

### 6.1: Query Intent Understanding
- [ ] **QueryAnalyzer using LLM**
  - "fast-paced psychological thriller" → Extract: tempo=fast, genre=psychological+thriller
  - Use cached concept expansions
  - No hard-coded query patterns

### 6.2: Concept-to-Search Translation
- [ ] **Search query generation from concepts**
  - User concepts → Expanded concepts → MongoDB query
  - Use actual movie data patterns, not hard-coded rules
  - Dynamic query weighting based on concept confidence

## Stage 7: End-to-End Intelligence
**Goal: Complete flow from user query to intelligent results**

### 7.1: Ingestion Pipeline
- [ ] **Movie ingestion with procedural enhancement**
  - Raw Jellyfin data → NLP analysis via plugins → Enhanced metadata via plugins → MongoDB
  - All enhancement results cached for reuse
  - No hard-coded enhancement rules

### 7.2: Query Pipeline  
- [ ] **User query to intelligent results**
  - User query → LLM intent analysis via plugins → Concept expansion via plugins → Search + enrichment realtime plugins → Results
  - All steps use cached intelligence
  - No hard-coded query processing

## Success Criteria

### ✅ Procedural Intelligence
- [ ] Zero hard-coded genre/keyword/pattern lists anywhere
- [ ] All concept understanding comes from NLP/LLM analysis
- [ ] System learns from actual data, not programmer assumptions

### ✅ Elegant Caching
- [ ] Never call same ConceptNet/LLM/NLP API twice for same input
- [ ] Intelligent cache invalidation and refresh
- [ ] Performance: <200ms for cached lookups, graceful for cache misses

### ✅ Media Agnostic
- [ ] Same intelligence works for movies, TV, books, music
- [ ] Easy to add new media types without code changes
- [ ] Cross-media concept understanding

### ✅ Real-World Intelligence
- [ ] Understands "psychological thriller" without hard-coding what that means
- [ ] Learns from actual movie content what "action" really looks like
- [ ] Adapts to user language and search patterns

## Implementation Principles

1. **Cache First** - Always check cache before external API calls
2. **Learn from Data** - Use actual movie content to understand concepts
3. **Procedural Everything** - No lists, all generated from intelligence
4. **Media Agnostic** - Design works for any content type
5. **Graceful Degradation** - System works even when NLP services are down
6. **Performance Aware** - Cache and optimize for real-world usage

## Key Files to Focus On

**Keep and Enhance:**
- Plugin architecture (`src/plugins/`)
- Hardware config (`src/shared/hardware_config.py`)
- MongoDB integration (`src/data/`)
- Environment setup (`pyproject.toml`, `config.py`)

**Replace with Intelligence:**
- Any file with hard-coded patterns
- Query analyzers with pattern matching
- Genre/keyword lists anywhere

**Build New:**
- Concept expansion cache
- Media-agnostic intelligence layer
- Procedural concept generation
