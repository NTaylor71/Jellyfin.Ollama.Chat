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

## ✅ COMPLETED STAGES

### Stage 1: Clean Foundation ✅
- **Core Infrastructure**: FastAPI, Redis, MongoDB, Docker, environment management
- **Intelligent Data Flow**: MediaField system, PluginResult format, CacheKey generation
- **Plugin Architecture**: Hardware-adaptive, hot-reload, health monitoring, safe execution

### Stage 2: Concept Expansion Cache ✅
- **MongoDB Collection**: 8 optimized indexes, TTL expiration, O(1) lookups
- **CacheManager**: Cache-first pattern, 4-part keys, performance metrics
- **Impact**: Never call ConceptNet/LLM twice for same input

### Stage 3: Procedural Concept Expansion ✅
- **5 Providers**: ConceptNet, Ollama LLM, Gensim, SpaCy, HeidelTime
- **3 Core Plugins**: ConceptExpansion, TemporalAnalysis, QuestionExpansion  
- **Integration**: Docker stack, Redis queues, GPU detection, model management
- **Performance**: 186-267ms execution with caching, 2.8x cache improvement
- **Impact**: Eliminated all hard-coded genre/keyword lists

### Stage 4.1-4.2: Testing & Installation Fixes ✅
- **Honest Testing**: 8 test files now FAIL FAST with clear error messages
- **No Fallback Cheats**: Eliminated all "✅ PASSED" when components missing
- **dev_setup.sh**: Fixed pip behavior, explicit package groups, fail-fast approach
- **Validation**: test_dependencies.py comprehensive validator, clear setup instructions
- **Impact**: Faster debugging, honest results, production readiness


### 4.3: Service-Oriented Plugin Architecture with Clean Nomenclature

**Goal**: Transform monolithic worker into lightweight orchestrator + specialized services while maintaining clean plugin patterns

#### 4.3.1: Ollama Containerization with GPU Support 🎯 FOUNDATION PRIORITY
**Goal**: Replace host-based Ollama with containerized service for better portability and GPU acceleration

##### 4.3.1.1: Add Ollama Service to Docker Stack
- [x] **GPU-enabled Ollama service** - Add `ollama/ollama:latest` with NVIDIA runtime support
- [x] **Shared model storage** - Use `model-data:/root/.ollama` for unified model storage across all services  
- [x] **Service health checks** - Ollama health endpoint with model validation
- [ ] **GPU memory configuration** - Set CUDA memory limits and device mappings
- [x] **Fix volume architecture** - Use shared `model-data` volume instead of separate `./docker/ollama` directory

##### 4.3.1.2: Update Configuration Management
- [ ] **Fix shared/config.py** - Update Docker URLs from `localhost:11434` to `ollama:11434`
- [ ] **Update .env configs** - Fix Docker environment variables to point to containerized service
- [ ] **Remove host networking** - Eliminate all `host.docker.internal` and `extra_hosts` configurations
- [ ] **Update OllamaBackendClient** - Modify `src/concept_expansion/providers/llm/ollama_backend_client.py` for container URLs
- [ ] **Add GPU configuration** - Add GPU detection and optimization settings to config

##### 4.3.1.3: Service Integration and Dependencies  
- [ ] **Update API service config** - Change from `host.docker.internal:11434` to `ollama:11434`
- [ ] **Update Worker service config** - Change from `host.docker.internal:11434` to `ollama:11434`
- [ ] **Add Ollama dependencies** - All LLM-using services depend on `ollama: condition: service_healthy`
- [ ] **Integrate Ollama with LLMProviderService** - Update `src/services/llm_provider_service.py` to use containerized Ollama
- [ ] **Update ServiceRegistry** - Add Ollama service monitoring to `src/plugins/service_registry_plugin.py`

##### 4.3.1.4: Model Management Endpoints
- [ ] **Create OllamaService** - Implement `src/services/ollama_service.py` with FastAPI endpoints
- [ ] **Add /api/models/pull endpoint** - Trigger model downloads via HTTP API in OllamaService
- [ ] **Add /api/models/status endpoint** - Check model availability and download progress in OllamaService
- [ ] **Model initialization script** - Auto-download required models on first startup
- [ ] **Update model_manager.py** - Integrate with containerized Ollama in `src/shared/model_manager.py`
- [ ] **Dependency group updates** - Consider adding model management utils to `pyproject.toml`

##### 4.3.1.5: Testing and Validation
- [ ] **GPU vs CPU performance testing** - Validate GPU acceleration works correctly
- [ ] **Cross-platform compatibility** - Test on different operating systems  
- [ ] **Model persistence testing** - Verify models survive container restarts
- [ ] **Provider integration testing** - Test `src/concept_expansion/providers/llm/ollama_backend_client.py` with containerized Ollama
- [ ] **Service integration testing** - Verify LLMProviderService and OllamaService work together
- [ ] **End-to-end testing** - Ensure all 16 Python files work with containerized Ollama

#### 4.3.2: Test Provider Service Architecture 🚧 PARTIALLY COMPLETE
**Status**: Code exists but needs containerized Ollama foundation first
- [x] **NLPProviderService** - FastAPI service created (`src/services/minimal_nlp_service.py`)
- [x] **LLMProviderService** - FastAPI service created (`src/services/minimal_llm_service.py`)  
- [x] **PluginRouterService** - Service router created (`src/services/plugin_router_service.py`)
- [x] **ServiceRegistryPlugin** - Service discovery created (`src/plugins/service_registry_plugin.py`)
- [x] **ServiceRunner** - CLI utility created (`src/services/service_runner.py`)
- [x] **Environment-aware URLs** - All services use `shared/config.py` (Rule 14 compliance)
- [ ] **Update with containerized Ollama** - Apply 4.3.1 changes to all existing services
- [ ] **END-TO-END TESTING** - Verify all services work with containerized Ollama

#### 4.3.3: Test Plugin Task Dispatcher 🚧 CODE EXISTS, NEEDS TESTING
**Status**: Worker architecture created but needs containerized foundation
- [x] **WorkerService structure** - Main worker created (`src/redis_worker/main.py`) with Prometheus metrics
- [x] **PluginLoader** - Plugin loading system created (`src/redis_worker/plugin_loader.py`)  
- [x] **HealthMonitor** - Health checking system created (`src/redis_worker/health_monitor.py`)
- [x] **TaskTypes** - Task validation definitions created (`src/redis_worker/task_types.py`)
- [x] **QueueManager** - Redis queue interface created (`src/redis_worker/queue_manager.py`)
- [ ] **Update with containerized Ollama** - Apply 4.3.1 changes to worker configuration
- [ ] **Integration testing** - Test complete queue → worker → plugin → service → result flow
- [ ] **Plugin execution paths** - Verify ConceptExpansion, TemporalAnalysis, QuestionExpansion plugins work
- [ ] **Service-aware execution** - Test worker correctly routes to NLP/LLM services
- [ ] **Docker build stages** - Ensure services use multi-stage builds for efficiency

#### 4.3.4: Create Service Client Plugins 📅 PENDING
- [ ] **HTTPProviderPlugin** - Base class for plugins that call HTTP services
- [ ] **RemoteConceptExpansionPlugin** - Extends ConceptExpansionPlugin to use services
- [ ] **RemoteTemporalAnalysisPlugin** - Temporal analysis via HTTP
- [ ] **ServiceHealthMonitorPlugin** - Monitor service availability

#### 4.3.5: Expand Plugin Types 📅 PENDING
- [ ] **Add SERVICE_PROVIDER to PluginType enum** - New type for service-backed plugins
- [ ] **Create ServiceProviderPlugin base class** - Common HTTP client functionality
- [ ] **Add service discovery to PluginExecutionContext** - Available services info
- [ ] **Implement circuit breaker pattern** - Graceful degradation when services down

#### 4.3.6: Test Microservices Architecture 📅 PENDING
- [ ] **Lightweight worker verification** - Confirm <500MB without NLP deps
- [ ] **Plugin routing test** - Queue → Worker → Plugin → Service → Result
- [ ] **Service scaling test** - 3 workers + 1 NLP service configuration
- [ ] **Performance benchmarks** - Compare monolithic vs service latency


#### 4.4: Virtual Environment Path Resolution During NLP Service Model Downloads ✅ RESOLVED

**Final Status**: ✅ ALL MODELS WORKING
- ✅ SpaCy downloads work (en_core_web_sm model available)
- ✅ NLTK downloads work (punkt, stopwords, wordnet models available)
- ✅ Gensim downloads work (word2vec-google-news-300 model available)  
- ✅ Service runs as nlp user with proper virtual environment access
- ✅ Health checks report unhealthy until all models ready
- ✅ Container dependencies work correctly (router waits for nlp-service health)

**Commands Used for Final Validation**:
```bash
# Clean test - ALL SUCCESSFUL
docker compose -f docker-compose.dev.yml down -v
docker compose -f docker-compose.dev.yml build nlp-service  
docker compose -f docker-compose.dev.yml up nlp-service mongodb redis
curl http://localhost:8001/health  # Returns: {"status":"healthy"...}
curl http://localhost:8001/providers  # Shows: gensim and spacy providers ready
```

**Success Criteria**: ✅ **ACHIEVED** - All models (NLTK punkt/stopwords/wordnet, Gensim word2vec, SpaCy en_core_web_sm) download automatically and nlp-service starts healthy.

### 4.5: code comments and docstring audit

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

## 🎯 **CURRENT STATUS: STAGE 4.3 - SERVICE-ORIENTED PLUGIN ARCHITECTURE + OLLAMA CONTAINERIZATION**

**FOUNDATION WORK (4.3.2 & 4.3.3)**: 🚧 Code structure exists but needs containerized foundation first
- **FastAPI Services**: NLP, LLM, Router services created with basic functionality  
- **Worker Infrastructure**: Redis queue manager, plugin loader, health monitoring systems
- **Plugin System**: ConceptExpansion, TemporalAnalysis, QuestionExpansion plugins ready
- **Configuration**: Environment-aware URLs using shared/config.py ✅
- **Monitoring**: Prometheus metrics integration in worker and services ✅

**IMMEDIATE PRIORITY (4.3.1)**: 🎯 Ollama Containerization with GPU Support - FOUNDATION FIRST  
- **Current Issue**: Services use `host.docker.internal:11434` - not portable across platforms
- **Why Foundation**: All services depend on Ollama - containerize first, then test everything
- **Goal**: GPU-enabled Ollama container with persistent model storage
- **Impact**: Better performance, cross-platform compatibility, easier deployment

**LOGICAL FLOW**:
1. **4.3.1: Containerize Ollama** - Foundation infrastructure with GPU support
2. **4.3.2: Test services** - Verify existing services work with containerized Ollama  
3. **4.3.3: Test worker** - Validate complete plugin → service → Ollama flow
4. **4.3.4-4.3.6: Complete architecture** - Finish remaining microservices features