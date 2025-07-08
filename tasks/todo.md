# Todo 3: Project Continuation After HTTP-Only Plugin Refactor

## Current State Summary

We've successfully completed a major architectural refactor (Issue #1) that transformed the system from a monolithic plugin architecture to a clean HTTP-only microservices pattern. All plugins now communicate via HTTP to specialized services that own the actual NLP/LLM providers.

### What's Working Now:
- ✅ **HTTP-Only Plugin Architecture**: All plugins inherit from HTTPBasePlugin
- ✅ **Provider-Specific Services**: NLP Service (SpaCy, Gensim, HeidelTime), LLM Service (Ollama)
- ✅ **Provider-Specific Plugins**: 8 working plugins (ConceptNet, LLM, Gensim, SpaCy, HeidelTime, Merge, Q&A, Temporal Intelligence)
- ✅ **Standardized REST Endpoints**: `/providers/{id}/action` pattern everywhere
- ✅ **Resource Management**: Queue system with resource-aware task distribution
- ✅ **Monitoring GUI**: PyQt6 app for real-time queue and resource monitoring
- ✅ **Docker Stack**: Fully containerized with GPU support for Ollama
- ✅ **CLI Testing Tools**: Individual test scripts for each plugin endpoint
- ✅ **Code Cleanup**: Obsolete directories removed, tests organized in /tests/
- ✅ **Architecture Audit**: Services and providers audited for consistency
- ✅ **YAML Configuration**: New media type configuration system implemented

### What Needs Work:
- Prometheus/Grafana metrics integration
- Service splitting for minimal dependencies
- User-facing API implementation (ingestion manager using YAML config)
- Model Manager as dedicated service

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

## Stage 3: Monitoring and Metrics

### 3.1 Prometheus Integration
- [ ] Add Prometheus metrics to all plugins:
  - Execution time
  - Success/failure rates
  - Cache hit rates
  - Resource usage
- [ ] Add metrics to all services:
  - Request latency
  - Throughput
  - Error rates
  - Provider-specific metrics
- [ ] Add metrics to queue system:
  - Queue depth
  - Task wait time
  - Processing time
  - Resource utilization

### 3.2 Grafana Dashboards
- [ ] Create overview dashboard showing system health
- [ ] Create plugin performance dashboard
- [ ] Create service performance dashboard
- [ ] Create resource utilization dashboard
- [ ] Create alert rules for critical metrics

## Stage 4: Service Architecture Evolution

### 4.1 Model Manager Service
- [ ] Extract model management into dedicated service
- [ ] Centralize model downloads and caching
- [ ] Implement model versioning
- [ ] Add model performance tracking
- [ ] Enable shared model access across services

### 4.2 Split NLP Service
- [ ] Create minimal service images:
  - `conceptnet-service` (API-only, no models)
  - `gensim-service` (with word vectors)
  - `spacy-service` (with NLP models)
  - `heideltime-service` (with temporal models)
- [ ] Ensure each service has minimal dependencies
- [ ] Optimize Docker images for size
- [ ] Test service isolation and independence

### 4.3 Service Discovery Enhancement
- [ ] Implement proper service registry
- [ ] Add dynamic service discovery
- [ ] Enable service health monitoring
- [ ] Support service scaling

## Stage 5: User-Facing API Implementation

### 5.1 YAML Configuration Implementation ✅ COMPLETED
- ✅ Implemented the new movie.yaml format
- ✅ Created configuration parser for field mappings
- ✅ Enabled dynamic plugin selection per field
- ✅ Support media type detection
- ✅ Tested with real movie data
- ✅ Standardized field extraction patterns in config/media_types/

### 5.2 Main API Design
The main API (port 8000) currently only has basic health endpoints. Need to add business logic:

- [ ] Design RESTful API for media operations:
  - `POST /api/v1/ingest/media` - Ingest media with enrichment
  - `GET /api/v1/media/{id}` - Retrieve enriched media
  - `POST /api/v1/analyze/media` - Analyze media on demand
  - `GET /api/v1/search` - Search with intelligent query expansion
  - `POST /api/v1/enrich/field` - Enrich specific fields

### 5.3 Ingestion Manager
- [ ] Create ingestion manager that:
  - Reads media data from JSON or Jellyfin DB
  - Uses the YAML configuration from 5.1 to determine enrichment
  - Applies configured plugins based on media type and field mappings
  - Stores enriched data in MongoDB
  - Tracks ingestion progress and errors
- [ ] Support batch ingestion with progress tracking
- [ ] Implement rate limiting and resource management

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

### 10.2 Docker Stack Validation
- [ ] Full docker purge and rebuild test
- [ ] Verify all services start correctly
- [ ] Test inter-service communication
- [ ] Validate resource constraints
- [ ] Ensure GPU support works

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

## Next Immediate Steps

1. **Stage 5.2**: Design user-facing API endpoints for media operations
2. **Stage 5.3**: Create ingestion manager using existing YAML configuration
3. **Stage 3.1**: Add Prometheus metrics to plugins and services
4. **Stage 4.1**: Design and implement Model Manager Service

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