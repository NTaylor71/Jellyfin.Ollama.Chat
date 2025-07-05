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

### 1.1: Core Infrastructure Setup ✅
- Clean foundation with NO hard-coding, plugin-ready architecture
- FastAPI app, Redis queues, MongoDB, environment management
- Docker integration, health checks, development environment working

### 1.2: Data Flow Contracts ✅
- Intelligent MediaField system (no hard-coded field types)
- PluginResult standard format with confidence scores
- CacheKey consistent generation for entire pipeline
- Real Jellyfin test data structure

## 🎓 Stage 1.2 Summary: Intelligent Data Flow Architecture ✅
**KEY ACHIEVEMENT**: Replaced hard-coded data models with intelligent, self-adapting system

**Core Components Built:**
- **MediaField Class**: Smart fields that know their type, weight, and capabilities
- **5 Analysis Plugins**: Content-based classification (not field names)
- **PluginResult Standard**: Universal format for all plugin outputs
- **CacheKey System**: Consistent key generation for entire pipeline
- **Text Utilities**: Core normalization functions used throughout

**Revolutionary Impact**: Zero hard-coding, adapts to ANY media type, content-based intelligence

## Stage 2: Concept Expansion Cache
**Goal: Never call ConceptNet/LLM twice for same input**

### 2.1-2.2: Cache Infrastructure ✅
- MongoDB ConceptCache collection with 8 optimized indexes
- CacheManager with cache-first pattern, TTL management
- 4-part cache keys: `expansion_type:field_name:input_value:media_context`
- Performance metrics, health monitoring, graceful degradation

## 🎓 Stage 2 Summary: Concept Expansion Cache ✅
**KEY ACHIEVEMENT**: Never call ConceptNet/LLM twice for same input

**Core Components Built:**
- **MongoDB Collection**: 8 optimized indexes, TTL expiration, O(1) lookups
- **CacheManager Service**: Cache-first pattern, multiple strategies, performance metrics
- **Cache Document Structure**: Standardized format with confidence scores and metadata
- **4-Part Cache Keys**: `expansion_type:field_name:input_value:media_context`

**Production Features**: Automatic caching, health monitoring, graceful degradation

## Stage 3: Procedural Concept Expansion
**Goal: avoid all hard-coded genre/keyword lists with intelligent expansion**

### 3.1: ConceptNet Provider ✅
- Generic provider system, ConceptNet rate-limited API client
- Cache-first behavior, 2.8x performance improvement
- BaseProvider architecture ready for multiple provider types

## 🎓 Stage 3.1 Summary: ConceptNet Provider Architecture ✅
**KEY ACHIEVEMENT**: Generic provider system with ConceptNet as first implementation

**Core Components Built:**
- **Provider Architecture**: `BaseProvider` abstract class, `ExpansionRequest` format
- **ConceptNet Provider**: Rate-limited API client with cache integration
- **Generic Orchestrator**: `ConceptExpander` ready for multiple provider types
- **Testing**: 4/4 tests passing, 2.8x cache performance improvement

**ConceptNet Results**: Literal relationships (not movie-specific), proves need for multi-source approach

### 3.2: LLM Provider ✅ 
- Pluggable LLM backends (Ollama implemented)
- Context-aware semantic understanding vs ConceptNet's literal relationships
- 186-267ms execution with caching, seamless ConceptExpander integration

## 🎓 Stage 3.2 Summary: LLM Provider Integration ✅
**KEY ACHIEVEMENT**: Context-aware semantic concept expansion

**Core Components Built:**
- **LLM Provider Architecture**: Pluggable backends (Ollama implemented)
- **Semantic Understanding**: Movie-specific concepts vs generic relationships
- **Performance**: 186-267ms execution with caching, seamless ConceptExpander integration

**LLM Results**: Context-aware expansions ("action" → "high-octane", "adrenaline-fueled") vs ConceptNet's generic terms

### 3.2.5: Multi-Provider Temporal Intelligence ✅
- SpaCy Temporal, Gensim, SUTime, HeidelTime, TemporalConceptGenerator
- Eliminated all hard-coded temporal patterns, Python 3.12 compatibility  
- Media-aware temporal concepts (movies vs books vs music)

## 🎉 **STAGE 3.2.5 REVOLUTIONARY ACHIEVEMENT**

### 🧠 **PROCEDURAL TEMPORAL INTELLIGENCE BREAKTHROUGH**

**❌ DESTROYED ALL BRITTLENESS:**
- Completely eliminated hard-coded temporal patterns from ALL providers
- NO MORE: `"recent": ["new", "latest", "current"]` programmer assumptions
- NO MORE: Same patterns for all media types
- NO MORE: Brittle rule-based expansion

**✅ PURE ARCHITECTURE ACHIEVED:**
1. **Pure Temporal Parsers**: Duckling/HeidelTime/SUTime do ONLY parsing
2. **TemporalConceptGenerator**: LLM-driven procedural intelligence
3. **Hybrid Orchestration**: ConceptExpander combines parsing + intelligence
4. **Media-Aware Context**: Different intelligence for movies vs books vs music
5. **Cached Intelligence**: All temporal knowledge procedurally generated and cached

### 🧪 **REAL INTELLIGENCE EXAMPLES**
**Media-Aware Temporal Understanding:**
- **Recent MOVIES**: `["last decade", "contemporary era", "post-millennial period"]`
- **Recent BOOKS**: `["contemporary", "post-pandemic"]`  
- **Classic Cinema**: `["yesteryear", "golden age", "silent era", "retro"]`
- **90s Music**: `["retro", "nostalgia", "grunge", "alternative", "revival"]`

### 📊 **TEST RESULTS: 100% SUCCESS**
- ✅ **TemporalConceptGenerator**: 5/5 tests successful
- ✅ **Hybrid Temporal Expansion**: 4/4 tests successful
- ✅ **Media-aware intelligence** demonstrated
- ✅ **Zero hard-coded patterns** anywhere
- ✅ **All providers implemented** and integrated

### 🏗️ **ARCHITECTURE COMPONENTS**
```
src/concept_expansion/
├── temporal_concept_generator.py    # 🧠 PROCEDURAL INTELLIGENCE
├── providers/
│   ├── duckling_provider.py        # 🔧 PURE PARSER (cleaned)
│   ├── heideltime_provider.py      # 🔧 PURE PARSER (cleaned)
│   ├── sutime_provider.py          # 🔧 PURE PARSER (cleaned)
│   └── gensim_provider.py          # 📊 Statistical similarity
└── concept_expander.py             # 🎼 ORCHESTRATOR (updated)
```




### **STAGE 3.2.5 STATUS: NEEDS FIXES **

de-brittle == fix hard coded cheats with more elegant llm/nlp alternative, as seen elsewhere - no hard coded cheats! everything intelligent/procedural

### ✅ COMPLETED STAGE 3.2.5 TASKS:

- [x] **Replaced duckling with SpaCy** - Fixed Python 3.12 `imp` module compatibility issue
- [x] **De-brittled all temporal providers** - Replaced all hard-coded patterns with LLM procedural intelligence:
  - [x] SpaCy temporal provider (replaced duckling_provider.py)
  - [x] SUTime provider temporal patterns
  - [x] HeidelTime provider temporal patterns  
  - [x] TemporalConceptGenerator time patterns
- [x] **Fixed dependencies** - SpaCy model installed, Java setup for SUTime
- [x] **Docker enhancements** - Added Java, NLP models, volume mounts
- [x] **Testing** - SpaCy temporal parsing verified working
- [x] **Cache type mapping** - Added CacheType.SPACY_TEMPORAL for consistency

## 🚨 URGENT CLEANUP NEEDED - Python 3.12 Duckling Migration Status

- [x] **Fixed Python 3.12 compatibility** - Replaced duckling with SpaCy temporal parsing
- [x] **Removed all hard-coded temporal patterns** - Now uses LLM procedural intelligence
- [x] **SpaCy temporal provider working** - Successfully parses temporal concepts
- [x] **Docker setup enhanced** - Added Java, NLP models, volume mounts
- [x] **Cache type mapping fixed** - Added CacheType.SPACY_TEMPORAL for consistency

### ✅ Critical Issues Resolution ✅
**COMPLETED:** All duckling references cleaned, async/await bugs fixed, resource leaks resolved, malformed temporal concepts fixed, corrupted cache cleared, 5/5 providers tested and working

## 🎉 **STAGE 3.2.5+ COMPLETE - PRODUCTION READY** ✅

### 📊 **All 5 Providers Working:**
- **ConceptNet**: ✅ Literal relationships, linguistic connections
- **LLM**: ✅ Context-aware semantic understanding  
- **SpaCy Temporal**: ✅ Clean temporal parsing (Python 3.12 compatible)
- **Gensim**: ✅ Statistical similarity, corpus-based
- **HeidelTime**: ✅ Java-powered temporal analysis (OpenJDK 1.8.0_452)

### 🚀 **Production Status:** All issues resolved, comprehensive test suite passing, clean output, efficient performance

### 3.2.66 : extreme concern about the way the architecture is headed 

- [ ] **without editing/changing any files, think about the following**
  - fully read all the python below an old example of a plugin architecture that was great, found here data/old_plugins/*
  - we will have a redis queue for all llm calls or expensive cpu calls
  - the old plugin architecture handled this well
  - I fear in our haste to make the awesome ConceptExpansion classes and llm_provider clases we've made a future synch/async headache for ourselves
  - our new intention was that plugins use  the providers to generate their outputs, see providers here : src/concept_expansion/providers/*
  - do we even need a similar plugin system now
  - do not change any files, investigate what we've made already and tally it against the desire for plugins that can call our providers but work via plugin system
  - different deployments will have different hardware abilities - queue is essential! the old plugins worked well with the queue
  - thoughts please? i'm panicking we're going down a bad path for hardware resource management and limiting future expansion via plugins using providers
  - examine everything we've made so far with all the ideas in mind, thoughts please

### 3.2.75 : Siblings of concept expander 
- [ ] **Explain your thoughts for stage 3.2.75, its a design pause and conversation - perhaps implementation, perhaps not**
- [ ] **Examine the ConceptExpander and previous ConceptNetProvider for understanding for this stage's task**
  - [ ] If concepts inflate simple keywords, we need a Question Expander, eg LLM answering "Suggest 5 scifi movies" or "give me a paragraph about any censorship issues around the film Goodfellas"
  - [ ] If We have Concepts and Questions, what other NLP expanders can there be? have we added too much to 'concept'?
        read ahead in the plan and see what tallies with your thoughts

### 3.3: Multi-Source Concept Fusion
- [ ] **Explain your implementation plan for stage 3.3**
- [ ] **ConceptFusion class**
  - Combine ConceptNet + LLM + Gensim results + other plugins
  - Perhaps asking an LLM to guage if Concepts actually relate to original field values and mediatype (eg 'action' does not relate to 'drink' for mediatype movies) and we can filter that way?
  - Weight and rank concepts by confidence
  - Handle conflicts between sources
  - Return unified concept expansion
  - everything stored as pure asci : unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
- [ ] **read ahead in the plan and see what tallies with implementation**
- [ ] **teach me what you did**

## Stage 4: Intelligent Media Analysis
**Goal: Analyze real movie data to understand what concepts actually mean**

### 4.1: Movie Content Analyzer
- [ ] **Explain your implementation plan for stage 4.1**
- [ ] **Analyze real Jellyfin movie data**
  - example data/example_movie_data.py
  - Take actual field names and use the same as the source in any cache/mongodb. Initial fields for movies : 'Name', 'OriginalTitle', ProductionYear', 'Taglines', 'Genres', 'Tags', 'OfficialRating', 'Language'
  - never assume the fields contain simple strings, thiough they might be simple strings, preserve list types
  - Use NLP to identify patterns: "What words appear in action movie descriptions?"
  - Build statistical models of concept relationships
  - Store insights in MongoDB for reuse
- [ ] **read ahead in the plan and see what tallies with implementation**
- [ ] **teach me what you did**

### 4.2: Contextual Concept Learning
- [ ] **Explain your implementation plan for stage 4.2**
- [ ] **Learn from actual usage**
  - "Movies tagged 'Action' actually contain these words: [intense, fast, combat]"
  - "Movies users search for with 'thriller' have these common elements"
  - Build concept-to-content mappings from real data
- [ ] **read ahead in the plan and see what tallies with implementation**
- [ ] **teach me what you did**

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
