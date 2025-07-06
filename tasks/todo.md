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

## Stage 1: Clean Foundation ✅
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

## 🎓 Stage 1 Summary: Intelligent Data Flow Architecture ✅
**KEY ACHIEVEMENT**: Replaced hard-coded data models with intelligent, self-adapting system

**Core Components Built:**
- **MediaField Class**: Smart fields that know their type, weight, and capabilities
- **5 Analysis Plugins**: Content-based classification (not field names)
- **PluginResult Standard**: Universal format for all plugin outputs
- **CacheKey System**: Consistent key generation for entire pipeline
- **Text Utilities**: Core normalization functions used throughout

**Revolutionary Impact**: Zero hard-coding, adapts to ANY media type, content-based intelligence

## Stage 2: Concept Expansion Cache ✅
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

## Stage 3: Procedural Concept Expansion ✅
**Goal: avoid all hard-coded genre/keyword lists with intelligent expansion**

### 3.1: ConceptNet Provider ✅
- Generic provider system, ConceptNet rate-limited API client
- Cache-first behavior, 2.8x performance improvement
- BaseProvider architecture ready for multiple provider types

### 3.2: LLM Provider ✅ 
- Pluggable LLM backends (Ollama implemented)
- Context-aware semantic understanding vs ConceptNet's literal relationships
- 186-267ms execution with caching, seamless ConceptExpander integration

### 3.2.5: Multi-Provider Temporal Intelligence ✅
- SpaCy Temporal, Gensim, SUTime, HeidelTime, TemporalConceptGenerator
- Eliminated all hard-coded temporal patterns, Python 3.12 compatibility  
- Media-aware temporal concepts (movies vs books vs music)

### 3.2.66: Plugin Architecture Recovery ✅
- BaseConceptPlugin foundation with queue/hardware integration
- ConceptExpansionPlugin multi-provider orchestration  
- TemporalAnalysisPlugin temporal parsing + intelligence
- QuestionExpansionPlugin natural language understanding

### 3.2.69: Integration Testing ✅
- Docker Stack Integration - Redis, MongoDB, API, Worker all healthy and communicating
- Plugin Queue Integration - ConceptExpansionPlugin successfully using Redis queue for task distribution
- Hardware Detection - RTX 4090 GPU (24GB VRAM) properly detected and integrated
- Provider Integration - All 5 providers (ConceptNet, LLM, Gensim, SpaCy, HeidelTime) working through plugin system

### 3.2.95: CLI Tools - Model Management ✅
- Enhanced manage_models.py CLI with comprehensive management features
- Docker entrypoint enhancement with automatic model management on container start
- Docker Stack Integration with updated container configurations

### 3.3: Multi-Source Concept Fusion ⭐ NEXT
**Status**: Foundation built in ConceptExpansionPlugin, needs dedicated fusion logic

- [ ] **ConceptFusionPlugin** (builds on existing _fuse_provider_results)
  - [x] Basic fusion implemented in ConceptExpansionPlugin._fuse_provider_results
  - [ ] Advanced LLM-based relevance filtering ("action" ≠ "drink" for movies)  
  - [ ] Confidence scoring and conflict resolution
  - [ ] Media-type aware fusion strategies
  - [ ] ASCII normalization: unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")

**Implementation approach**: Extract and enhance the fusion logic from ConceptExpansionPlugin into dedicated ConceptFusionPlugin that can be used across all concept plugins

## 🎓 Stage 4.2 Summary: Honest Testing - NO MORE FALLBACK CHEATS ✅
**KEY ACHIEVEMENT**: All 8 test files now FAIL FAST with clear error messages

**Files Fixed:**
- **test_dependencies.py** (NEW): Comprehensive dependency validator - run this FIRST
- **test_stage_3_providers.py**: No more try/except hiding provider failures
- **test_temporal_intelligence.py**: Upfront SpaCy/Ollama validation, no LLM fallbacks  
- **test_integration.py**: No more try/except pass patterns masking service failures
- **test_simple_providers.py**: Fail fast on broken metadata/capabilities
- **test_working_providers.py**: No success counting, fail immediately on expansion failures
- **test_heideltime.py**: Upfront Java/HeidelTime validation
- **test_model_manager_integration.py**: Fail fast on Docker/ModelManager issues

**Benefits**: Faster debugging, honest results, clear setup instructions, production readiness

## Stage 4: Fix Fundamental Testing and Installation Issues
**Goal: STOP THE CHEATING - Fix all the fallback BS that masks real problems**

### 4.1: Fix dev_setup.sh pip behavior ✅
- [x] **Remove pip recommendations bloat** - Match Docker's controlled installs with `--no-cache-dir`
- [x] **Explicit package groups only** - No more `.[local]` that pulls everything
- [x] **Test with actual venv-specific imports** - Use `import ollama, gensim, spacy` not system-wide garbage
- [x] **FAIL FAST approach** - No fallbacks that mask real dependency conflicts

### 4.2: Rewrite All Tests to Fail Fast ✅

- [x] **Remove ALL fallback logic from tests - No more "✅ PASSED" when core components are missing**
- [x] **SpaCy model missing → FAIL IMMEDIATELY - No LLM fallback BS**
- [x] **MongoDB connection fails → FAIL IMMEDIATELY - No cache workaround BS**
- [x] **Java dependencies missing → FAIL IMMEDIATELY - No SUTime fallback BS**
- [x] **Remove provider import fallbacks - SpaCy/HeidelTime/Gensim AVAILABLE = False lies**
- [x] **Remove provider method fallbacks - No TemporalConceptGenerator/LLM/regex fallback chains**
- [x] **Provider initialization must fail-fast - No degraded mode, fully functional or unavailable**
- [x] **Create test_dependencies.py - Validate ALL required packages, models, and external deps**
- [x] **Rewrite test_integration.py - Remove all "try/except pass" patterns that mask failures**
- [x] **Rewrite test_stage_3_providers.py - FAIL immediately if any provider can't initialize**
- [x] **Rewrite test_temporal_intelligence.py - FAIL if SpaCy models missing, Java deps missing**
- [x] **Rewrite test_simple_providers.py - FAIL immediately if provider metadata/capabilities broken**
- [x] **Rewrite test_working_providers.py - FAIL immediately if expansion fails, no success counting BS**
- [x] **Rewrite test_heideltime.py - FAIL immediately if Java/HeidelTime missing**
- [x] **Rewrite test_model_manager_integration.py - FAIL immediately if Docker/ModelManager broken**

### 4.2.5: IMMEDIATE TEST VALIDATION ✅ COMPLETE

**Goal**: Test the new fail-fast approach to ensure all fallback cheats are eliminated

- [x] **Test Dependency Validator FIRST** - `python test_dependencies.py`
  - [x] Run with missing SpaCy models (should fail with clear instructions)
  - [x] Run with Java missing (should fail with clear instructions)
  - [x] Run with full environment (should pass completely)
  
**COMPLETED FIXES:**
- [x] Fixed dependency validator to use proper model manager instead of manual installs
- [x] Fixed NLTK wordnet ZIP extraction issue in model manager  
- [x] Fixed Ollama model detection bug (was using wrong API response format)
- [x] Added proper JAVA_HOME handling to config system with environment-aware detection
- [x] Updated environment validation to use shared/config.py instead of raw env vars
- [x] Validated clean rebuild workflow: purge → fail on missing → download via model manager → pass

- [x] **Test Individual Provider Tests** - Each should fail fast on missing deps
  - [x] `python test_simple_providers.py` (basic provider functionality) - ✅ WORKING
  - [x] `python test_working_providers.py` (ConceptNet + LLM only) - ✅ WORKING 
  - [x] `python test_heideltime.py` (HeidelTime with Java validation) - ✅ CORRECTLY FAILS FAST
  - [x] `python test_stage_3_providers.py` (all 5 providers) - ✅ CORRECTLY FAILS FAST
  - [x] `python test_temporal_intelligence.py` (temporal intelligence) - ✅ WORKING (but slow)

- [x] **Test Integration Tests** - Should fail fast on service issues
  - [x] `python test_integration.py` (Redis, MongoDB, plugins) - ✅ CORRECTLY FAILS FAST
  - [x] `python test_model_manager_integration.py` (Docker, ModelManager) - ✅ WORKING

- [x] **Validate Error Messages** - Each test failure should provide:
  - [x] Clear description of what's broken - ✅ ALL TESTS NOW HAVE CLEAR ERROR MESSAGES
  - [x] Specific installation/setup instructions - ✅ "Run: pip install py-heideltime", etc.
  - [x] No more "⚠️ Some tests failed" vague messages - ✅ ELIMINATED
  - [x] No more "✅ PASSED" when components are missing - ✅ FIXED

- [x] Check all test_*.py tests get all env vars via the shared/config.py (a switch between local dev .env and docker's own env [see the compose for the environment var ENV=docker etc]) - **PARTIALLY DONE** - dependency validator fixed, but need to fix remaining 6 test files


### 4.3: Service-Oriented Plugin Architecture with Clean Nomenclature

**Goal**: Transform monolithic worker into lightweight orchestrator + specialized services while maintaining clean plugin patterns

#### 4.3.1: Create Provider Service Architecture
- [ ] **NLPProviderService** - FastAPI service hosting SpaCy, HeidelTime, Gensim providers
- [ ] **LLMProviderService** - Dedicated service for Ollama/LLM operations  
- [ ] **PluginRouterService** - Routes plugin requests to appropriate services
- [ ] **ServiceRegistryPlugin** - New plugin type for service discovery

#### 4.3.2: Implement Plugin Task Dispatcher  
- [ ] **Complete worker/main.py TODO** - Route task_types to plugins:
  - `plugin_execution` → Load plugin by name and execute
  - `concept_expansion` → ConceptExpansionPlugin
  - `temporal_analysis` → TemporalAnalysisPlugin
  - `question_expansion` → QuestionExpansionPlugin
- [ ] **Add PluginLoader** - Dynamic plugin loading from task data
- [ ] **Implement plugin health checks** - Verify plugins can reach their services

#### 4.3.3: Create Service Client Plugins
- [ ] **HTTPProviderPlugin** - Base class for plugins that call HTTP services
- [ ] **RemoteConceptExpansionPlugin** - Extends ConceptExpansionPlugin to use services
- [ ] **RemoteTemporalAnalysisPlugin** - Temporal analysis via HTTP
- [ ] **ServiceHealthMonitorPlugin** - Monitor service availability

#### 4.3.4: Expand Plugin Types
- [ ] **Add SERVICE_PROVIDER to PluginType enum** - New type for service-backed plugins
- [ ] **Create ServiceProviderPlugin base class** - Common HTTP client functionality
- [ ] **Add service discovery to PluginExecutionContext** - Available services info
- [ ] **Implement circuit breaker pattern** - Graceful degradation when services down

#### 4.3.5: Test Microservices Architecture
- [ ] **Lightweight worker verification** - Confirm <500MB without NLP deps
- [ ] **Plugin routing test** - Queue → Worker → Plugin → Service → Result
- [ ] **Service scaling test** - 3 workers + 1 NLP service configuration
- [ ] **Performance benchmarks** - Compare monolithic vs service latency

**Clean Nomenclature Maintained**:
- Plugins remain `*Plugin` (execution logic)
- Services become `*Service` (HTTP endpoints)  
- Providers stay `*Provider` (concept expansion logic)
- Clear separation: Plugin orchestrates, Service provides, Provider implements

### 4.4: Proper Error Reporting

- [ ] Every test failure explains EXACTLY what's broken - No more mysterious failures
- [ ] No more "✅ PASSED" when core components missing - Tests should be honest
- [ ] Clear error messages for each dependency failure - Make debugging possible
- [ ] Provider-specific error messages - SpaCy: "Run: pip install spacy && python -m spacy download en_core_web_sm"
- [ ] HeidelTime error messages - "py-heideltime not installed. Requires Java 17+. Run: pip install py-heideltime"
- [ ] Gensim error messages - "Gensim not installed. Run: pip install gensim
- [ ] Environment validation reports - Pre-test environment checker with explicit pass/fail
- [ ] Ollama connectivity validation - Actual API call test, not just URL ping

### 4.5: Remove Insane Fallback Logic
- [ ] NLP tool requires NLP components - period - Remove fallbacks that mask problems
- [ ] Make system honest about actual capabilities - No lying about what works
- [ ] Remove fallbacks that mask real problems - Let things fail when they should fail
- [ ] SpacyTemporalProvider fallback removal - Remove _fallback_temporal_detection() method (lines 240-270)
- [ ] SpacyTemporalProvider TemporalConceptGenerator fallback - Remove fallback (lines 223-233)
- [ ] SpacyTemporalProvider import fallback - Remove SPACY_AVAILABLE = False pattern (lines 22-28)
- [ ] HeidelTimeProvider LLM context fallbacks - Remove document context fallbacks (lines 405-407, 439-444)
- [ ] HeidelTimeProvider regex fallback - Remove _has_temporal_patterns fallback (lines 470-475)
- [ ] HeidelTimeProvider import fallback - Remove HEIDELTIME_AVAILABLE = False pattern (lines 22-31)
- [ ] GensimProvider word fallback - Remove "try different common words" logic (lines 290-297)
- [ ] GensimProvider import fallback - Remove GENSIM_AVAILABLE = False pattern (lines 22-29)
- [ ] Plugin system fallback removal - Remove plugin fallback chains that mask missing providers
- [ ] ConceptExpander fallback removal - No more "if provider fails, try different provider" logic

### 4.75: code comments and docstring audit

- [ ] review all code comments : they should never be 'the what', they should be 'the why', but:
  - Only add 'the why' if the code isnt clear, 95% of the time the why is obvious by just reading the code
  - key goal : reduce the size of our code by removing comments as much as possible
  - look for comments in all files and clean house!
- [ ] review all method docstrings and format for googledocs style docstrings
  - bonus points for helpful info about the method or class or file 


## Stage 5: Intelligent Media Analysis (FUTURE)
**Goal: Analyze real movie data to understand what concepts actually mean**

### 5.1: Movie Content Analyzer
- [x] **Jellyfin data extraction** - Already implemented in ConceptExpansionPlugin._extract_concepts_from_data
- [ ] **Dedicated ContentAnalysisPlugin** - Extract logic from existing plugins
- [ ] **Statistical pattern analysis** - "What words appear in action movie descriptions?"
- [ ] **Concept-relationship modeling** - Store insights in MongoDB

### 5.2: Contextual Concept Learning
- [ ] **Usage-based learning plugin**
  - Build on existing provider results: "Movies tagged 'Action' contain [intense, fast, combat]"
  - Learn from search patterns and user behavior
  - Create concept-to-content mappings from real data
  - Feed insights back into fusion algorithms

## Stage 6: Media-Agnostic Intelligence (FUTURE)
**Goal: Same intelligence works for movies, TV, books, music, comics**

### 6.1: Media Type Detection and Adaptation
- [x] **Basic media context detection** - Implemented in TemporalAnalysisPlugin._detect_media_context
- [x] **Media-aware processing** - All plugins support media_context parameter
- [ ] **Enhanced MediaTypeDetector** - Dedicated plugin for automatic detection
- [ ] **Cross-media adaptation** - "action" in movies ≠ "action" in books ≠ "action" in music

### 6.2: Cross-Media Concept Transfer
- [ ] **Media-specific concept learning** - How "dark" manifests across media types
- [ ] **Transfer learning framework** - Share insights between media types
- [ ] **Abstraction without hard-coding** - Procedural cross-media understanding

## Stage 7: Intelligent Query Processing (FUTURE)
**Goal: Understand user intent without pattern matching**

### 7.1: Query Intent Understanding  
- [x] **QuestionExpansionPlugin** - Handles "fast-paced psychological thriller" type queries
- [x] **Intent classification** - Recommendation, search, information, etc.
- [x] **LLM-based understanding** - No hard-coded patterns
- [ ] **Enhanced intent extraction** - More sophisticated parameter extraction

### 7.2: Concept-to-Search Translation
- [ ] **SearchTranslationPlugin** - Convert expanded concepts to MongoDB queries
- [ ] **Dynamic query weighting** - Based on concept confidence scores
- [ ] **Real data patterns** - Use actual movie data, not hard-coded rules

## Stage 8: End-to-End Intelligence (FUTURE)
**Goal: Complete flow from user query to intelligent results**

### 8.1: Ingestion Pipeline
- [x] **Plugin-based enhancement** - ConceptExpansionPlugin, TemporalAnalysisPlugin ready
- [x] **Cached results** - All providers use cache-first strategy
- [x] **No hard-coded rules** - All enhancement procedural
- [ ] **Pipeline orchestration** - Chain plugins for complete ingestion

### 8.2: Query Pipeline  
- [x] **Question analysis** - QuestionExpansionPlugin handles user queries
- [x] **Concept expansion** - Multi-provider expansion ready
- [x] **Cached intelligence** - All steps use cached results
- [ ] **Search integration** - Connect to actual search and ranking
- [ ] **Real-time enrichment** - Plugin-based result enhancement

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
7. **FAIL FAST** - No fallbacks that mask real problems

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

---

## 🎯 **CURRENT STATUS: STAGE 4 - FIX THE TESTING CHAOS**

**Next Priority**: Stop all the fallback BS and make tests honest about what actually works vs what's broken.