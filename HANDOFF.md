# CLAUDE.md - Comprehensive Claude Code Handover Document

## 🎯 Current Project Status

**Repository**: https://github.com/NTaylor71/Jellyfin.Ollama.Chat/tree/production-rag-system  
**Branch**: `production-rag-system`  
**Last Stage**: Adding Prometheus/Grafana monitoring - **YAML configuration issues need fixing**
**Thread Status**: Hit max length cutoff during monitoring implementation

## 🚧 Immediate Issues to Fix

### 1. Prometheus Configuration Errors (CRITICAL)
Location: `docker/monitoring/prometheus.yml`  
Problem: First code errors reported in conversation  
**Next Step**: Fix YAML syntax errors in prometheus.yml before proceeding

### 2. Directory Structure Creation
Need to create monitoring directories:
```bash
New-Item -ItemType Directory -Path "docker/monitoring" -Force
New-Item -ItemType Directory -Path "docker/monitoring/grafana/provisioning/datasources" -Force
```

## 📂 Complete Project Architecture

### ✅ Core Services (All Working)
```
├── API Service (FastAPI)           # Port 8000, Redis queue integration
├── FAISS Service                   # Vector database for RAG
├── Redis Queue Worker              # Background job processing  
├── Redis                          # Queue & caching
├── Prometheus (NEEDS FIXING)      # Metrics collection
└── Grafana (NEEDS FIXING)         # Dashboards
```

### 🏗️ Plugin Architecture (Comprehensive Design Discussed)

#### Plugin Types & Execution Contexts

| Plugin Type              | Execution Context         | Purpose                                    | Trigger Point                                   |
|--------------------------|---------------------------|--------------------------------------------|------------------------------------------------|
| 1. Query Embellishers    | API Container             | Modify or enhance the user's query.        | Called before enqueuing to Redis queue.        |
| 2. Embed Data Embellishers| FAISS Ingestor Container  | Modify or enhance embedding documents.     | Called during ingestion, before sending to FAISS service. |
| 3. FAISS CRUD Plugins    | FAISS Service Container   | Inject custom logic into FAISS CRUD ops.   | Called on add, search, or delete.              |

#### Hardware-Aware Plugin System Design
**User's Hardware Profile**: 48 CPU cores, limited GPU resources

```python
@dataclass
class SystemCapabilities:
    cpu_cores: int = 48
    gpu_memory_gb: float = 0.0  # User's case
    ram_gb: float = 32.0
    gpu_devices: List[str] = field(default_factory=list)

# CPU-optimized plugin for user's system
@plugin_registry.register(
    plugin_type="query_embellisher",
    weight=10,
    resource_requirements=PluginResourceRequirements(
        cpu_intensive=True,
        min_cpu_cores=8,  # Utilize user's 48 cores
        gpu_memory_required=0.0
    )
)
class ParallelQueryExpander:
    async def process(self, query: str, context: dict) -> str:
        # CPU-parallel processing using 48 cores
        pass
```

#### Plugin Management via MongoDB
```python
# Plugin registry in MongoDB
{
    "_id": "cpu_query_expander_v1.2.0",
    "name": "CPU Query Expander", 
    "version": "1.2.0",
    "plugin_type": "query_embellisher",
    "author": "your_team",
    "file_hash": "sha256:abc123...",
    "resource_requirements": {
        "cpu_intensive": True,
        "min_cpu_cores": 4,
        "gpu_memory_required": 0.0
    },
    "status": "published",  # draft, published, deprecated
    "release_date": "2025-01-15T10:00:00Z",
    "config_schema": {
        "max_synonyms": {"type": "int", "default": 10},
        "enable_fuzzy": {"type": "bool", "default": True}
    },
    "compatibility": ["rag-system>=2.0.0"],
    "file_url": "plugins/cpu_query_expander_v1.2.0.py"
}
```

#### Plugin Release Flow
- Admin releases plugin via MongoDB status update
- Triggers broadcast reload signal to all workers
- Workers hot-reload the plugin
- Health check confirms successful load

### 🔍 Literal vs Semantic Search Strategy

#### Current Literal Search Coverage Assessment: 60%
**Implemented**:
- ✅ Basic literal matching, fuzzy matching, lemmatization
- ✅ Synonym expansion (WordNet, Gensim)
- ✅ Acronym handling, numeric normalization

**Critical Gaps Identified**:
1. **Field-Specific Weighting** - Different weights per field (title, description, tags, cast)
2. **Positional Weighting** - Position-based scoring (first word, title start)  
3. **Match Quality Scoring** - Granular match quality (exact=1.0, fuzzy=0.6, etc.)
4. **Context-Aware Boosting** - Cross-field consistency boosting

#### Recommended Plugin Implementation
```python
@literal_search_plugin(weight=5)
class FieldSpecificMatcher:
    """Weight matches based on field importance"""
    
@literal_search_plugin(weight=10) 
class PositionalScorer:
    """Boost based on match position"""
    
@literal_search_plugin(weight=15)
class ContextualBooster:
    """Cross-field consistency boosting"""
```

#### Advanced Search Technologies Integration
- **NLTK**: Text processing, lemmatization, WordNet synonyms
- **Gensim**: Similarity modeling, topic modeling
- **MongoDB**: Document storage, metadata management
- **FAISS**: Vector similarity search

## 🛠️ Development Environment & Dependencies

### Dependency Management (STRICT RULE)
**ONLY** via `pyproject.toml` + `dev_setup.ps1`
- ✅ Service-specific dependencies: `[api]`, `[worker]`, `[faiss]`, `[local]`
- ✅ Clean rebuild every time: deletes .venv, fresh install
- ✅ No manual pip installs allowed

### Current pyproject.toml Structure
```toml
# Core dependencies shared across services
dependencies = [
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0", 
    "python-dotenv>=1.0.0",
    "redis>=5.0.0"
]

[project.optional-dependencies]
# API service - only FastAPI deps
api = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "prometheus-client>=0.19.0",
    "prometheus-fastapi-instrumentator>=6.1.0"
]

# Worker service - only queue processing deps  
worker = [
    "celery[redis]>=5.3.0",
    "rich>=13.7.0"
]

# FAISS service - only vector processing deps
faiss = [
    "faiss-cpu>=1.7.4",
    "numpy>=1.24.0"
]

# Monitoring dependencies
monitoring = [
    "prometheus-client>=0.19.0",
    "prometheus-fastapi-instrumentator>=6.1.0"
]

# Local development - everything
local = [
    "production-rag-system[api,worker,faiss,monitoring]",
    "pytest>=7.4.0",
    "httpx>=0.26.0"
]
```

### Environment Configuration
```bash
# Core Settings
ENV=docker                          # Auto-switches container/local config
PROJECT_NAME=production-rag-system

# Redis Configuration  
REDIS_HOST=redis                    # Docker internal name
REDIS_PORT=6379
REDIS_QUEUE=chat:queue

# FAISS/Vector Database
VECTORDB_URL=http://faiss_service:6333
FAISS_INDEX_PATH=/app/data/faiss.index

# Ollama Integration (Dual endpoints for chat vs embedding)
OLLAMA_CHAT_BASE_URL=http://localhost:12434
OLLAMA_CHAT_MODEL=llama3.2:3b
OLLAMA_EMBED_BASE_URL=http://localhost:12435  
OLLAMA_EMBED_MODEL=nomic-embed-text

# Jellyfin Integration
JELLYFIN_URL=redacted
JELLYFIN_API_KEY=redacted

# Monitoring (BROKEN - needs fixing)
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

## 🐳 Docker Architecture

### Service-Specific Containers (No Bloat)
```dockerfile
# API Container - ONLY API dependencies
FROM python:3.11-slim
COPY pyproject.toml src/ ./
RUN pip install -e ".[api]"  # Only gets FastAPI + uvicorn + shared

# FAISS Container - ONLY FAISS dependencies  
FROM python:3.11-slim
RUN pip install -e ".[faiss]"  # Only gets FAISS + numpy + shared

# Worker Container - ONLY worker dependencies
FROM python:3.11-slim
RUN pip install -e ".[worker]"  # Only gets Celery + shared
```

### Image Size Optimization Results
| Container Type | Bloated Approach | Efficient Approach | Savings |
|---------------|------------------|-------------------|---------|
| API Container | 2.1GB | 800MB | 1.3GB |
| FAISS Container | 2.1GB | 1.2GB | 900MB |
| Worker Container | 2.1GB | 900MB | 1.2GB |

## 🚀 Queue vs Async Execution Strategy

```python
# LIGHTWEIGHT/FAST: Direct async (sub-second)
@query_embellisher(execution="async")
class FastQueryCleaner:
    async def process(self, query: str) -> str:
        # Quick string processing
        return cleaned_query

# HEAVY/CPU-INTENSIVE: Queue processing  
@query_embellisher(execution="queued", cpu_intensive=True)
class HeavyQueryAnalyzer:
    async def process(self, query: str) -> str:
        # Complex NLP using 48 cores
        # Gets queued for background processing
        pass
```

## 🧪 Testing Strategy

### Current Working Test Files (production-rag-system branch)
```
├── test_api.py                     # ✅ End-to-end API testing
├── src/tests/test_config.py        # ✅ Configuration testing  
├── test_redis_queue.py             # ✅ Queue system testing
```

### Test Scripts from Previous Branch (DISCARDED)
*These were mentioned in conversation but are NOT in current branch:*
- `src/tests/test_llm_search.py` ❌ (previous attempt)
- `src/tests/jellyfin_export.py` ❌ (previous attempt)  
- `src/tests/get_jellyfin_movie_data_direct.py` ❌ (previous attempt)
- `src/tests/synonym_generator.py` ❌ (previous attempt)

### Template System (PLANNED, NOT YET IMPLEMENTED)
- **Jinja2 Templates**: Discussed for formatting metadata into LLM-friendly text
- **Location**: TBD (likely `src/rag/` when implemented)
- **Purpose**: Clean document formatting for RAG processing

### Test Command Pattern (Current Branch)
```bash
# Configuration tests (pytest)
python -m pytest src/tests/test_config.py -v

# End-to-end API test (standalone)
python test_api.py

# Queue system test  
python test_redis_queue.py
```

**Note**: Full pytest suite (`python -m pytest src/tests/`) will be available when more test files are added to current branch.

## 🔧 Development Workflow

### Seamless Local Development Rules
1. **Update dependencies**: Add to `pyproject.toml`
2. **Run setup**: `./dev_setup.ps1` (deletes .venv, rebuilds everything)
3. **Test immediately**: System always works at every step
4. **No manual commands**: Everything automated via scripts

### Build Commands
```bash
# Development setup (rebuilds everything)
./dev_setup.ps1

# Docker stack (all services)
docker compose -f docker-compose.dev.yml up -d --build

# Individual service logs
docker compose -f docker-compose.dev.yml logs -f [service_name]
```

## 📈 Monitoring Strategy (NEEDS COMPLETION)

### Metrics to Track
- Query processing latency (p50, p95, p99)
- FAISS search performance  
- Redis queue depth and processing time
- Plugin execution time per type
- Error rates and failure patterns

### Infrastructure
- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Dashboards (port 3000, admin/admin)
- **API Metrics**: Available at `/metrics` endpoint

## 🗄️ MongoDB Integration Design

### Purpose
- **Plugin Management**: Store plugin metadata, versions, release status
- **Data Storage**: Document storage, metadata management  
- **Configuration**: Per-deployment plugin configs
- **Release Control**: Plugin publishing and hot-reload triggers

### Document Schema
```javascript
// Plugin documents
{
  plugin_id: "cpu_query_expander_v1.2.0",
  status: "published",
  resource_requirements: {...},
  config_schema: {...}
}

// Data documents  
{
  content: "movie metadata...",
  metadata: {...},
  embeddings: [...],
  tags: [...]
}
```

## 🎯 Next Development Phases

### Phase 1: Fix Monitoring (CURRENT PRIORITY)
- [ ] Fix Prometheus YAML syntax errors
- [ ] Create monitoring directories  
- [ ] Verify Grafana datasource connection
- [ ] Add custom API metrics
- [ ] Create basic dashboards

### Phase 2: Plugin System Implementation
- [ ] Design plugin registry system
- [ ] Implement hot-reload mechanism
- [ ] Create plugin base classes
- [ ] Add plugin discovery and loading
- [ ] Implement weighted execution system

### Phase 3: Enhanced Search Capabilities
- [ ] Implement literal search plugins (field weighting, positional scoring)
- [ ] Add MongoDB document storage
- [ ] Integrate NLTK/Gensim processing
- [ ] Create hybrid semantic + literal search
- [ ] Add advanced query analysis

### Phase 4: Production Hardening
- [ ] Add comprehensive logging
- [ ] Implement health checks for all services
- [ ] Add retry logic and error handling
- [ ] Container resource optimization
- [ ] Security enhancements

## 💡 Key Design Principles Established

1. **Dependency Management**: ONLY via pyproject.toml + dev_setup.ps1
2. **Container Efficiency**: Service-specific dependencies, no bloat
3. **Hardware Optimization**: CPU-intensive plugins for 48-core system
4. **Plugin Architecture**: Hot-reloadable, weighted execution, MongoDB-managed
5. **Search Strategy**: Hybrid literal + semantic with field-specific weighting
6. **Development Experience**: One command setup, always working system
7. **Testing First**: Every component tested before moving forward
8. **Monitoring**: Comprehensive observability from the start

## 🚨 Critical Reminders for Claude Code

- **Conversation was cut off** during monitoring implementation
- **Fix Prometheus YAML** before proceeding with any other features
- **Respect dependency rules**: Only update via pyproject.toml + dev_setup.ps1  
- **Plugin system is core**: Heavy emphasis on extensible, hardware-aware plugins
- **Search is hybrid**: Both literal (NLTK/Gensim) and semantic (FAISS) components
- **MongoDB integration**: Essential for plugin management and data storage
- **Test everything**: Use existing comprehensive test suite
- **48 CPU cores**: Optimize for CPU-intensive parallel processing

## 🎯 Success Criteria

### Immediate (Fix monitoring)
- [ ] Prometheus starts without YAML errors
- [ ] Grafana connects to Prometheus datasource  
- [ ] API metrics visible at `/metrics` endpoint
- [ ] Basic dashboard shows API request rates

### Plugin System (Core Feature)
- [ ] Hot-reloadable plugins with weighted execution
- [ ] MongoDB-managed plugin registry
- [ ] Hardware-aware resource requirements
- [ ] CPU-optimized plugins for user's 48-core system

### Search System (Hybrid Approach)
- [ ] Literal search with field-specific weighting
- [ ] NLTK/Gensim integration for text processing
- [ ] FAISS semantic search integration
- [ ] MongoDB document storage and retrieval

### Production Ready
- [ ] All services pass health checks
- [ ] Comprehensive monitoring and alerting
- [ ] Plugin hot-reload without service restart
- [ ] Complete documentation and onboarding

---

**IMPORTANT**: I have a 48-core CPU setup with 4090  GPU resources. Others devs might have different. Hence the queue. It should be possible to register hardware (num CPUs, 1+ GPU) - All plugin recommendations and optimizations should be manageable