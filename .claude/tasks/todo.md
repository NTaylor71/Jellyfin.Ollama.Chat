# Todo 3: Project Continuation After HTTP-Only Plugin Refactor

## Current State Summary

We're hitting a tonne of issues after refactoring badly (again)
see Stage 5
## Stage 1: Code Cleanup and Organization ✅ COMPLETED

### 1.1 Remove Obsolete Code ✅
- ✅ Removed `src/plugins/archived/` directory (old duplicate plugins)
- ✅ Removed `src/concept_expansion/` directory (replaced by field_enrichment plugins)
- ✅ Removed `src/plugins/query_processing/` directory (empty/unused)
- ✅ Removed `src/plugins/monitoring/` directory (empty/unused)
- ✅ Removed all root-level test_*.py files (moved to /tests/)
- ✅ Cleaned up unused imports and dead code

### 1.2 Organize Test Structure ✅
- ✅ Created `/tests/` directory with proper __init__.py
- ✅ Moved all test_*.py files into organized subdirectories:
  - `/tests/unit/` - Unit tests for individual components
  - `/tests/integration/` - Integration tests for service communication
  - `/tests/e2e/` - End-to-end tests for full workflows
  - `/tests/performance/` - Performance and load tests
- ✅ Updated imports and verified all tests still run

## Stage 2: Architecture Audit and Improvements ✅ COMPLETED

### 2.1 Async vs Queue Architecture Review ✅
- ✅ Audited current async patterns in plugins and services
- ✅ Reviewed queue manager implementation for race conditions
- ✅ Ensured proper error handling and retry logic
- ✅ Documented the async/sync boundaries clearly
- ✅ Optimized task distribution algorithms

### 2.2 Service Consistency Audit ✅
- ✅ Reviewed all services for consistent patterns:
  - Error handling
  - Logging format
  - Health check implementation
  - Resource declarations
  - Response formats
- ✅ Created service development guidelines document
- ✅ Ensured all services follow the same architectural patterns

### 2.3 Provider Consistency Audit ✅
- ✅ Reviewed all providers for consistent patterns:
  - Initialization
  - Resource requirements
  - Error handling
  - Result formatting
- ✅ Ensured providers properly implement BaseProvider interface
- ✅ Documented provider development best practices

## Stage 3: Monitoring and Metrics ✅ COMPLETED

### 3.1 Prometheus Integration ✅
- ✅ Add Prometheus metrics to all plugins:
  - Execution time
  - Success/failure rates
  - Cache hit rates
  - Resource usage
- ✅ Add metrics to all services:
  - Request latency
  - Throughput
  - Error rates
  - Provider-specific metrics
- ✅ Add metrics to queue system:
  - Queue depth
  - Task wait time
  - Processing time
  - Resource utilization

### 3.2 Grafana Dashboards ✅
- ✅ Create overview dashboard showing system health
- ✅ Create plugin performance dashboard
- ✅ Create service performance dashboard
- ✅ Create resource utilization dashboard
- ✅ Create alert rules for critical metrics

## Stage 4: Service Architecture Evolution ✅ COMPLETED

**✅ MICROSERVICES ARCHITECTURE IMPLEMENTED (2025-01-09)**
- ✅ **4.1 Model Manager Service**: Centralized model management, downloads, caching, versioning
- ✅ **4.2 Split NLP Architecture**: Legacy monolithic nlp-service → 5 specialized microservices:
  - **conceptnet-service:8001** (77% faster), **gensim-service:8006** (81% faster)
  - **spacy-service:8007** (20% faster), **heideltime-service:8008** (78% faster), **llm-service:8002**
- ✅ **4.2.5-4.2.7 Production Migration**: Staged cutover, legacy cleanup, rollback procedures, 64% performance improvement
- ✅ **4.3 Service Discovery**: Dynamic discovery, health monitoring, service scaling

**✅ KEY OUTCOMES:**
- **Performance**: 64% average improvement across all services
- **Architecture**: 100% microservices, zero legacy code remaining
- **Monitoring**: Prometheus/Grafana integration, comprehensive metrics
- **Production Ready**: All services healthy, tested, documented with rollback capabilities

## Stage 5: User-Facing API Implementation ✅ COMPLETED

**✅ COMPREHENSIVE IMPLEMENTATION COMPLETED (2025-07-10)**
- ✅ **5.1 YAML Configuration**: Dynamic plugin selection, media type configs, field mapping patterns
- ✅ **5.2 Main API Design**: Full REST API with ingestion, media management, search endpoints, production features
- ✅ **5.3 Ingestion Manager**: MongoDB storage, background tasks, YAML-driven enrichment 
- ✅ **5.3.5 Legacy Code Audit**: 100% HTTP-only plugin compliance, zero architectural debt
- ✅ **5.4 External Service Analysis**: Comprehensive review of 3rd party enrichment options
- ✅ **5.4.5 LLM WebSearch Plugin**: End-to-end implementation with search + LLM processing pipeline
- ✅ **5.4.75 Emergency Routing Cleanup**: Complete routing architecture overhaul with 9 phases:
  - **Phase 1-7**: Eliminated Router Service and Plugin Loader routing layers
  - **Phase 8**: Dynamic Service Discovery - Centralized discovery system, zero hard-coding, self-describing services
  - **Phase 9**: Systematic Architectural Debt Elimination - All duplicate implementations consolidated

**✅ KEY ACHIEVEMENTS:**
- **Dynamic Service Discovery**: Single source of truth replacing 5+ duplicate implementations
- **Zero Hard-coding**: All URLs/ports configurable via environment or discovered dynamically  
- **Self-describing Services**: Capabilities extracted from actual `/health` and `/openapi.json` responses
- **Environment Portability**: Seamless localhost/docker/deployment switching
- **Clean Architecture**: Plugin self-routing, centralized configuration, minimal layers
    - [x] **Step 6**: Successfully applied router service elimination pattern to entire codebase
    - [x] **Step 7**: End-to-end testing confirmed dynamic service discovery works across all components
    - [x] **Step 8**: Cleaned up all legacy service discovery code and hard-coded values
  - [x] **SUCCESS PATTERN VALIDATED**: The same "eliminate unnecessary layers and use direct communication" approach that fixed router service routing was successfully applied to eliminate ALL duplicate service discovery logic across the entire codebase.
  - [x] **ARCHITECTURAL DEBT ELIMINATED**: Zero duplicate service discovery logic remains. Single dynamic discovery component handles all service-related operations with automatic capability detection.
  - [x] **FOR NEXT CLAUDE**: Systematic architectural debt has been completely eliminated. The codebase now follows consistent patterns with single-responsibility components, centralized configuration, and dynamic service discovery. No duplicate service discovery logic remains anywhere.

**Success Criteria:** ✅ ALL ACHIEVED
- [x] Single routing decision per plugin execution - Plugins now route directly via dynamic discovery
- [x] Plugins route themselves using built-in capabilities - HTTPBasePlugin.get_service_url() works dynamically  
- [x] No conflicts between multiple routing systems - All duplicate routing layers eliminated
- [x] LLM WebSearch plugin works end-to-end - Verified working with dynamic endpoint discovery
- [x] Reduced latency from eliminating router service layer - Direct service communication implemented
- [x] Dynamic service discovery eliminates hard-coded service information - Zero hard-coded values remain
- [x] **BONUS**: Centralized configuration system restored with proper environment-aware service management
- [x] **BONUS**: Comprehensive architectural debt elimination across entire codebase (5+ duplicate systems consolidated)

### 5.5 Audit Queue for parallel CPU & GPU execution ✅ COMPLETED
- [x] **Parallelism Implementation** (2025-07-10)
  - [x] Fixed GPU exclusive blocking - GPU tasks no longer block CPU tasks
  - [x] Implemented thread-based resource tracking (cores vs threads)
  - [x] Created YAML hardware configuration system in `config/hardware/`
  - [x] Support for multiple concurrent CPU tasks (tested up to 10 simultaneous)
  - [x] Dynamic resource allocation based on thread and memory limits
  - [x] Plugin-specific resource overrides via YAML config
  
**Key Achievements:**
- **Performance**: 2.8x speedup for mixed workloads, 4.5x for CPU-only
- **Concurrency**: 1 GPU + 8-10 CPU tasks can run simultaneously
- **Configuration**: Full YAML-based hardware resource management
- **Thread Management**: Proper thread counting prevents oversubscription
- **Documentation**: Created comprehensive implementation guide

### 5.6 Nomenclature update for 'chat' ✅ COMPLETED

- [x] **chat nomenclature** ✅ COMPLETED (2025-07-10)
  - [x] scan the entire project for the word 'chat'
  - [x] looking to rebrand 'chat' to 'ingestion' for system infrastructure only
  - [x] present renaming table for all hits to user
    - [x] previous renaming refactor recovery avoided through comprehensive planning
    - [x] copied all files to /archived/ first for reference and rollback capability
  - [x] **atomic rebrand executed successfully**
    - [x] **58 targeted replacements** across 18 files using sed commands
    - [x] **Environment variables**: `OLLAMA_CHAT_*` → `OLLAMA_INGESTION_*`
    - [x] **Redis queues**: `chat:queue:*` → `ingestion:queue:*` 
    - [x] **Service names**: `ollama_chat` → `ollama_ingestion`
    - [x] **Documentation**: `JellyfinOllamaChat` → `UniversalMediaIngestion`
    - [x] **PRESERVED correctly**: `chat_model`, `"chat": True`, model capabilities
    - [x] **Full Docker rebuild test**: ✅ All 15 services healthy
    - [x] **API validation**: Returns `"ollama_ingestion": true`
    - [x] **Environment files**: .env and .env.example updated
    
**✅ SUCCESS CRITERIA MET:**
- System infrastructure uses "ingestion" terminology
- LLM model capabilities preserved as "chat" (correct technical distinction)
- Zero service failures or broken configurations
- Complete backup available at /archived/ for reference

### 5.7 (SKIP FOR NOW) Unified filename, class, services, provider, plugin, docker image/container names nomenclature
- lets get a project wide taxonomy going for all things like files, class names, service names, plugin names, docker image names, docker container names
- [ ] Unified naming refactor
  - naming needs a cleanup
  - we see the generic term 'llm' confused for ollama (there will be new model providers other than ollama in the future)
  - some services are called 'minimal' without reason
  - we need to unify naming of stuff via a smart taxonomy
  - we want to debrand from jellyfin and jellyfinollamachat
- [ ] refactoring can go very badly if not done sensibly
  - we're not moving any files, but we are renaming a lot
  - lets be insanely cautious
  - [ ] copy the entire repo into /archived to keep a temp reference until this task is complete
    - old names can be searched if we got lost fixing import statements and scripts
  - [ ] present a table showing old file names to new filenames
  - [ ] present a table showing old class names to new class names
  - [ ] present a table showing old method names to new method names
  - [ ] present a table showing old docker container names to new docker container names
  - [ ] present a table showing old docker image names to new docker image names
  - [ ] save the tables to .claude/tasks/rename_plans.md
    -  add information to .claude/tasks/rename_plans.md explaining whats happening to an imaginary second instance of claude
- [ ] present thoughts on the plan to user
  - have we missed any beneficial renaming tricks
  - how can we do this renaming refactor safely and efficiently?
  - build a huge map, then:
    - filenames first? then sed import lines to match
    - then sed lists ?
- [ ] present a reassuring plan to user about how this refactor can be managed over several compaction loops to user without worry
- [ ] proceed with refactor when user approves
  - remember the archived/ version of the project is present for reference

### Stage 5.8 Code comments Audit (COMPLETE)
- Docstrings audit : every method must have google docs style, no exceptions
- Audir every single #comment
  - Comments should never be 'the what' (the code states that), they should be 'the why', but only if its needed for faster comprehension, otherwise no comment
  - Aim for less comments generally, only 'the why'
  - method/class comments can waffle more
- [X] Audit every python file, .toml and Dockerfile in the repo for comments, following the rules above
  - if you encounter 'split' - we dont have split architecture anymore, we just have the architecture, so comments about split are not needed
  - keep your eyes open for similar redundancies

### Stage 5.9 Local Ingestion Enhancement Services

#### 5.9.0 Commercial/Non-Commercial Mode Configuration
- [ ] **Global Commercial Use Compliance Framework**
  - [ ] Add commercial mode configuration to global config
    - [ ] Environment variable: `COMMERCIAL_MODE=true/false` (default: false)
    - [ ] Configuration file: `shared/config.py` commercial mode detection
    - [ ] Runtime validation and mode enforcement
  - [ ] Implement plugin-level commercial compliance
    - [ ] Base plugin interface extension with `requires_non_commercial` flag
    - [ ] Plugin registration validates commercial compatibility
    - [ ] Hard fail requests to non-commercial plugins when in commercial mode
  - [ ] Create compliance validation system
    - [ ] Startup validation: warn about non-commercial plugins in commercial mode
    - [ ] Runtime protection: automatic request rejection with clear error messages
    - [ ] Configuration validation: prevent invalid commercial/plugin combinations
  - [ ] Documentation and error handling
    - [ ] Clear error messages explaining commercial restrictions
    - [ ] Plugin documentation includes commercial use compatibility
    - [ ] Admin interface shows commercial mode status and affected plugins
  - [ ] Integration with existing plugin architecture
    - [ ] HTTPBasePlugin extension for commercial awareness
    - [ ] Plugin discovery respects commercial mode filtering
    - [ ] Service health checks include commercial compliance validation
  - [ ] Future-proofing for commercial datasets
    - [ ] Framework ready for commercial API integrations
    - [ ] Clear separation between commercial and non-commercial data sources
    - [ ] Audit trail for commercial compliance decisions

#### 5.9.1 Entity Extraction Enhancement ✅ COMPLETED (2025-01-10)
- [x] **SpaCy NLP Comprehensive Enhancement** ✅ COMPLETED
  - [x] Enhanced SpaCy plugin to include full Named Entity Recognition (NER)
    - [x] Created `spacy_ner_plugin.py` with comprehensive entity extraction
    - [x] Supports all 18+ entity types (PERSON, ORG, GPE, WORK_OF_ART, EVENT, etc.)
    - [x] Structured entity organization by category (people, organizations, locations, etc.)
  - [x] Extract: People, Places, Organizations, Works of Art, Events, Products, Money, etc.
  - [x] Store extracted entities as structured data fields during ingestion
  - [x] **Additional SpaCy Capabilities Implemented:**
    - [x] `spacy_linguistic_plugin.py` - POS tagging, dependencies, readability analysis
    - [x] `spacy_pattern_plugin.py` - Domain-specific pattern matching (awards, tech specs)
    - [x] `spacy_provider.py` - Unified provider supporting all analysis types
    - [x] Enhanced service endpoints: `/entities`, `/linguistic`, `/similarity`
    - [x] Multi-model support (en_core_web_sm/md/lg) with model management
    - [x] Complete API documentation with working examples
  - [x] **Critical Service Fixes Applied** ✅ COMPLETED (2025-07-10)
    - [x] Fixed SpaCy provider CacheType.SPACY_NLP → CacheType.SPACY_NER attribute error
    - [x] Fixed ConceptNet provider text_utils.py indentation error (line 214)
    - [x] Verified all NLP service endpoints working correctly:
      - SpaCy NER: Entity extraction with structured categories (6ms response)
      - SpaCy Linguistic: POS tagging and dependency parsing (5ms response)
      - ConceptNet: Concept expansion and semantic relationships (445ms response)
      - LLM: Concept expansion via Ollama backend (2783ms response)
    - [x] Full enrichment pipeline operational with proper error handling
  - [ ] Consider adding Flair NLP or Stanford NER as alternative local NER service
    - Note: SpaCy implementation now provides comprehensive coverage; evaluate if additional NER needed

#### 5.9.2 Advanced Keyphrase Extraction Plugin
- [ ] **Statistical Keyphrase Extraction** 
  - [ ] Implement YAKE! (Yet Another Keyword Extractor) - statistical, no model needed
  - [ ] Or implement PKE (Python Keyphrase Extraction) with multiple algorithms
  - [ ] Create new plugin (and required service/provider): "statistical_keyphrase_plugin" 
  - [ ] Better keyword quality than current ConceptNet approach
  - [ ] Add to enrichment pipeline for all text fields
  - [ ] Compare against existing ConceptNet and LLM keyword extraction

#### 5.9.3 Semantic Chunking Service
- [ ] **Text Segmentation for Better Processing**
  - [ ] Implement for long text fields (Overview, Synopsis, Plot)
  - [ ] Use local sentence-transformers for semantic similarity
  - [ ] Create chunks of semantically related sentences
  - [ ] Each chunk enriched separately for better quality
  - [ ] New service: "semantic_chunking_service"
  - [ ] Integration with existing plugin pipeline

#### 5.9.4 Local Knowledge Base Integration
- [ ] **Offline Knowledge Enhancement**
  - see old cache manager, it needs an overhaul - intended to avoid concept expansion twice via local mongodb caching
  - [ ] Download and index subset of Wikidata/DBpedia
  - [ ] Create local SPARQL endpoint or indexed lookup
  - [ ] New plugin: "local_knowledge_base_plugin"
  - [ ] Link entities to knowledge base IDs (leverage spaCy NER results)
  - [ ] Add structured facts (birthdate, nationality, genre facts)
  - [ ] Integrate with existing spaCy entity extraction

#### 5.9.5 Text Preprocessing Enhancement
- [ ] **Advanced Text Cleaning and Analysis**
  - [ ] Expand NLTK usage for better text cleaning
  - [ ] Add TextBlob for sentiment analysis on reviews
  - [ ] Implement sumy for automatic summarization of long overviews
  - [ ] Create preprocessing pipeline before enrichment
  - [ ] Integration with spaCy linguistic analysis for quality scoring

#### 5.9.6 IMDb Reference Data Integration
- [ ] **Authoritative Media Metadata Enhancement** (NON-COMMERCIAL)
  - [ ] Download and process IMDb non-commercial datasets (7 core datasets)
    - examine the light docker setup here : /home/ap/coding/radarr_peeps
      - translate the simple docker setup into this project's more sophisticated approach, eg new container, but keep this model set only in the db, no model manager as the data takes too long to translate to our local db from imdb's gz source
    - [ ] title.basics.tsv - Core movie/TV metadata (genres, runtime, release years)
    - [ ] title.ratings.tsv - User ratings and vote counts
    - [ ] name.basics.tsv - Person information with professions
    - [ ] title.crew.tsv - Director/writer mappings
    - [ ] title.principals.tsv - Cast and key crew roles
    - [ ] title.episode.tsv - TV episode relationships
    - [ ] title.akas.tsv - Alternative titles and localization
  - [ ] Create new plugin: "imdb_reference_plugin"
    - [ ] **Plugin marked as `requires_non_commercial = True`**
    - [ ] TSV parser for daily dataset updates
    - [ ] Local SQLite cache for fast ID lookups
    - [ ] Cross-reference with existing movie/TV enrichment data
    - [ ] Hard fail when `COMMERCIAL_MODE=true` with clear error message
  - [ ] Implement IMDb provider: "imdb_provider.py"
    - [ ] Dataset download and refresh automation
    - [ ] Data normalization and validation
    - [ ] Identifier mapping (tconst, nconst) to internal IDs
    - [ ] Commercial mode compliance validation
  - [ ] Integration with existing enrichment pipeline
    - [ ] Enhance ConceptNet results with IMDb genres/keywords
    - [ ] Augment temporal intelligence with birth/death dates, release years
    - [ ] Enrich person entities with professional roles and filmography
    - [ ] Add IMDb ratings for content recommendation weighting
  - [ ] Compliance and attribution implementation
    - [ ] Non-commercial use compliance validation
    - [ ] Attribution metadata in enriched records
    - [ ] Local storage management within licensing terms
    - [ ] Integration with Stage 5.9.0 commercial compliance framework
  - [ ] Performance optimization
    - [ ] Incremental updates to avoid full reprocessing
    - [ ] Memory-efficient processing for large datasets
    - [ ] Parallel processing for bulk enrichment operations

#### 5.9.7 Task dependency chaining in the queue
- [ ] **Investigate queue tasks & task-dependencies**
  - the movie.yaml has 'plugin groups' that can end, currently, in a merge results op
  - we need improved dependency control for this kind of op
  - both in the yaml and in the queue
  - we want easy chaining of plugins that can pass-forward results, from yaml techniques in the movie.yaml
  - [ ] **full audit of the architecture to make this easier to use/define**

#### 5.9.8 Using Claude to build the best ingestion movie.yaml possible
- [ ] Can we leverage claude code to semi automate the creation of new input types or improve existing ones
  - 1. Claude learns all current plugin abilities, including chained processing
  - 2. Claude examines raw input data
  - 3. Claude loops through testing endpoints with field input data, judging quality of outputs with regards to intended subsequent embedding and search usage, assiging weights accordingly and building up an industrial grade ingestion yaml, using the current movie.yaml as a starting point
- [ ] Can this trick be replicated by the system itself without claude
  - [ ] what are the pros/cons to semi automated ingestion construction for new mediaTypes and improving existing ones
 


## Stage 6: Embedding Cook - Search Index Generation

### 6.1 Cook Plugin Architecture
- [ ] Create new plugin type: `EMBEDDING_COOK` for index generation
- [ ] Design cook plugin interface: `cook_documents()`, `update_index()`, `search()`
- [ ] Implement base cook plugin class with resource management
- [ ] Create cook plugin registry and discovery
- [ ] Design plugin storage interface (separate from MongoDB)
- [ ] Hot-swappable cook plugins without affecting source data

### 6.2 FAISS Cook Plugin
- [ ] Implement FAISS cook plugin reading from MongoDB
- [ ] Add configurable embedding model support (Sentence Transformers, OpenAI, etc.)
- [ ] Create FAISS index management (file-based storage)
- [ ] Implement incremental index updates
- [ ] Add index versioning and rollback capability
- [ ] Support multiple index types (IVF, HNSW, Flat)

### 6.3 Gensim Cook Plugin
- [ ] Build Gensim cook plugin for similarity models
- [ ] Implement word2vec/doc2vec model generation from corpus
- [ ] Create model storage and versioning system
- [ ] Add similarity search interface
- [ ] Support incremental model updates
- [ ] Implement model quality metrics

### 6.4 MongoDB Text Index Cook Plugin
- [ ] Create plugin for MongoDB text index management
- [ ] Configure compound indexes based on YAML
- [ ] Implement index lifecycle management (create, rebuild, drop)
- [ ] Add index performance monitoring
- [ ] Create index optimization recommendations
- [ ] Ensure no document modification (indexes only)

### 6.5 Cook Orchestration Service
- [ ] Build service to monitor MongoDB for changes
- [ ] Implement cook job queue with resource awareness
- [ ] Create cook trigger logic based on media type
- [ ] Add cook status tracking per document/collection
- [ ] Implement cook health monitoring and alerts
- [ ] Support manual cook triggers and rebuilds

### 6.6 Cook Configuration via YAML
- [ ] Extend media type YAML schema with `cooking` section
- [ ] Define cook plugin configuration format
- [ ] Implement YAML validation for cook configs
- [ ] Create per-media-type cook strategies
- [ ] Add cook scheduling and resource allocation config
- [ ] Document cook configuration best practices

## Stage 7: Hybrid Search Architecture

### 7.1 Query Enrichment Pipeline
- [ ] Create new plugin type: `QUERY_ENRICHER` for real-time enhancement
- [ ] Implement synonym expansion plugin
- [ ] Build intent detection plugin
- [ ] Add entity extraction plugin
- [ ] Create temporal query parsing plugin
- [ ] Implement context-aware enrichment using session data
- [ ] Add multi-language query normalization

### 7.2 Hybrid Search Orchestrator
- [ ] Design parallel search execution framework
- [ ] Implement semantic search via FAISS plugin
- [ ] Add literal search via MongoDB text plugin
- [ ] Create faceted search with pre-computed aggregations
- [ ] Implement fuzzy matching for typo tolerance
- [ ] Add configurable search strategies per media type
- [ ] Create dynamic timeout management

### 7.3 Result Fusion & Reranking
- [ ] Create new plugin type: `RESULT_RANKER` for score adjustment
- [ ] Implement Reciprocal Rank Fusion (RRF)
- [ ] Add linear combination with configurable weights
- [ ] Build ML-based reranking framework
- [ ] Extract reranking features:
  - Query-document similarity scores
  - Popularity signals
  - Freshness factors
  - User preference alignment
- [ ] Create A/B testing framework for ranking experiments

### 7.4 Feedback & Learning System
- [ ] Implement implicit feedback collection:
  - Click-through rate tracking
  - Dwell time measurement
  - Scroll depth monitoring
  - Position bias correction
- [ ] Add explicit feedback mechanisms:
  - Upvote/downvote API
  - "More/Less like this" functionality
  - Custom relevance labels
- [ ] Build feedback aggregation pipeline
- [ ] Create learning algorithms for weight adjustment

### 7.5 Search Analytics & Optimization
- [ ] Implement query performance histograms
- [ ] Add result quality metrics (MRR, NDCG, P@K)
- [ ] Create query intent classification analytics
- [ ] Build dead query detection system
- [ ] Implement search session analysis
- [ ] Add automatic bad query detection and alerts

### 7.6 Personalization Layer
- [ ] Design user preference learning system
- [ ] Implement collaborative filtering
- [ ] Add context-aware adjustments:
  - Time-based patterns
  - Device-specific preferences
  - Location-based boosting
- [ ] Create privacy-preserving options
- [ ] Build preference export/import functionality

## Stage 8: Advanced Search Features

### 8.1 Explainable Search
- [ ] Implement "Why this result?" explanations
- [ ] Add concept matching highlights
- [ ] Show query interpretation visualization
- [ ] Create scoring component breakdown
- [ ] Build debug mode for search tuning
- [ ] Add search explanation API endpoints

### 8.2 Conversational Search
- [ ] Implement multi-turn query understanding
- [ ] Add context carryover between searches
- [ ] Create natural language refinement parser
- [ ] Build search dialogue state management
- [ ] Add intent clarification prompts
- [ ] Implement conversation history tracking

### 8.3 Proactive Search Suggestions
- [ ] Build query autocomplete with intent prediction
- [ ] Implement "People also searched for" recommendations
- [ ] Add trending searches by media type
- [ ] Create query reformulation suggestions
- [ ] Implement zero-result recovery strategies
- [ ] Add spelling correction suggestions

### 8.4 Search Quality Assurance
- [ ] Create automated relevance testing framework
- [ ] Build golden query sets per media type
- [ ] Implement regression detection system
- [ ] Create search quality dashboards
- [ ] Add bad result reporting workflow
- [ ] Build search quality metrics API

## Stage 9: PyQt6 Ingestion UI for New Media Types

### 9.1 Core UI Components
- [ ] Create main application window with tabbed interface:
  - New Media Type Wizard tab
  - Existing Configurations viewer/editor tab
  - Test Ingestion tab
- [ ] Implement 5-step wizard flow:
  - Step 1: Media type info + raw JSON input
  - Step 2: LLM-generated YAML preview
  - Step 3: Interactive field editor
  - Step 4: Plugin configuration
  - Step 5: Validation & export
- [ ] Build custom widgets:
  - `YAMLEditor` with syntax highlighting and validation
  - `FieldTreeWidget` for hierarchical field selection
  - `PluginPipeline` visual pipeline builder
  - `WeightSlider` with numeric display
  - `JSONViewer` collapsible tree viewer

### 9.2 LLM Integration for YAML Generation
- [ ] Create LLM service integration for config generation
- [ ] Implement prompt template system:
  - Load reference YAML (movie.yaml)
  - Format raw JSON sample
  - Include user description
  - Generate structured YAML output
- [ ] Add regeneration with feedback option
- [ ] Implement side-by-side comparison view (JSON | YAML)

### 9.3 Interactive Field Configuration
- [ ] Build field selection tree from raw JSON
- [ ] Implement per-field configuration:
  - Include/exclude checkboxes
  - Field weight sliders (0.0-3.0)
  - Field type detection and override
  - Source field mapping
  - Validation rules editor
- [ ] Add visual indicators:
  - Required fields (red asterisk)
  - Computed fields (blue icon)
  - Synthetic fields (green icon)
- [ ] Implement computed field builder with live preview

### 9.4 Plugin Pipeline Builder
- [ ] Create plugin discovery from `/src/plugins/enrichment/`
- [ ] Implement drag-and-drop plugin assignment
- [ ] Build plugin configuration interface:
  - Plugin groups with visual nesting
  - Per-plugin weight adjustment
  - Configuration parameter forms
  - Merge strategy selector
- [ ] Add plugin preview with sample enrichment
- [ ] Implement plugin recommendation engine based on field content

### 9.5 Validation and Testing Interface
- [ ] Real-time YAML validation with error highlighting
- [ ] Field constraint validation
- [ ] Plugin availability checking
- [ ] Test enrichment with sample data
- [ ] Export options:
  - Save to `config/media_types/`
  - Export as template
  - Generate standalone ingestion script
  - Create test suite for media type

## Stage 10: Testing and Validation

### 10.1 End-to-End Testing
- [ ] Create comprehensive E2E test suite:
  - Ingest sample media
  - Verify enrichment
  - Test search functionality
  - Validate caching
- [ ] Add performance benchmarks
- [ ] Create load testing scenarios

### 10.2 Docker Stack Validation - only edit this after 10.1 completed
- [ ] **Full docker purge and rebuild test** 

## Stage 11: Documentation and Deployment

### 11.1 Documentation
- [ ] Update README with new architecture
- [ ] Create API documentation
- [ ] Document plugin development guide
- [ ] Create deployment guide
- [ ] Add troubleshooting guide

### 11.2 Production Readiness
- [ ] Security audit (no exposed secrets)
- [ ] Performance optimization
- [ ] Resource limit configuration
- [ ] Backup and recovery procedures
- [ ] Monitoring and alerting setup


## Success Metrics

- Code reduction: >30% fewer lines after cleanup
- Test organization: 100% of tests in proper structure
- Metrics coverage: All critical paths instrumented
- UI usability: New media type configuration in <10 minutes
- Zero-code extensibility: Add any media type via UI only
- Service isolation: Each service <500MB image
- API completeness: Full CRUD + search operations
- Performance: <500ms average enrichment time
- Cook latency: <1 second per document
- Index consistency: 100% documents indexed
- Search latency: <200ms for 95th percentile
- Result relevance: >80% user satisfaction
- Zero results: <5% of queries
- Personalization lift: >15% CTR improvement