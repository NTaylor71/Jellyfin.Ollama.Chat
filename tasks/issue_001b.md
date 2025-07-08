# Issue 001b: Simplified Reorganization - HTTP-Only Plugins

## Problem Statement

Building on issue_001.md insights, we've identified additional architectural issues:
- **Duplicate plugin implementations**: Both `ConceptExpansionPlugin` and `RemoteConceptExpansionPlugin` do the same thing
- **Unnecessary complexity**: Plugins manage providers directly AND make HTTP calls
- **Confusing architecture**: "Remote" vs "regular" plugins distinction makes no sense in a microservices world
- **Resource management duplication**: Both plugins and services manage provider lifecycles
- **Lack of granular control**: Current plugins call ALL providers (conceptnet + gensim + llm) with no way to choose specific ones per field

## Key Insight

Since we've already split the monolithic worker into microservices, **ALL plugins should be HTTP-based**. There's no reason to have plugins that load providers directly - that's what services are for!

## Objective

Simplify the architecture by:
1. **Making all plugins HTTP-only** - they orchestrate calls to services
   - If both remote and local plugins exist already, favour functionality from the local version (more mature)
2. **Services own all providers** - they manage models, APIs, and resources
3. **Clear separation of concerns**:
   - Worker/Plugins: Orchestration, caching, queuing, field processing
   - Services: Provider management, model execution, API calls
4. **Remove all duplicate code** - one plugin per function, not two
5. **Provider-specific plugins** - separate plugins for each provider (conceptnet, llm, gensim) for fine-grained control
6. **Explicit merging** - when multiple providers are needed, use explicit merge plugins

## Simplified Reorganization Plan

### Phase 1: Clean Directory Structure (Simplified)

```
src/
├── providers/                         # Used ONLY by services, not plugins
│   ├── __init__.py
│   ├── base_provider.py              
│   │
│   ├── keyword_expansion/            
│   │   ├── __init__.py
│   │   ├── conceptnet_keyword_provider.py
│   │   ├── gensim_similarity_provider.py
│   │   └── llm_keyword_expansion_provider.py
│   │
│   ├── temporal_parsing/             
│   │   ├── __init__.py
│   │   ├── heideltime_parser_provider.py
│   │   ├── sutime_parser_provider.py
│   │   └── spacy_temporal_parser_provider.py
│   │
│   ├── question_answering/           
│   │   ├── __init__.py
│   │   └── llm_question_answer_provider.py
│   │
│   └── temporal_intelligence/        
│       ├── __init__.py
│       └── llm_temporal_concept_provider.py
│
├── plugins/                          # ALL plugins are HTTP-based
│   ├── __init__.py
│   ├── base.py                      # Common plugin base
│   ├── http_base.py                 # HTTP functionality for all plugins
│   │
│   ├── field_enrichment/            # Plugins that enrich data fields
│   │   ├── __init__.py
│   │   # Provider-specific keyword plugins
│   │   ├── conceptnet_keyword_plugin.py     # Calls ConceptNet only
│   │   ├── llm_keyword_plugin.py            # Calls LLM only
│   │   ├── gensim_similarity_plugin.py      # Calls Gensim only
│   │   # Provider-specific temporal plugins
│   │   ├── spacy_temporal_plugin.py         # Calls SpaCy only
│   │   ├── heideltime_temporal_plugin.py    # Calls HeidelTime only
│   │   ├── sutime_temporal_plugin.py         # Calls SUTime only
│   │   # Other enrichment plugins
│   │   ├── llm_question_answer_plugin.py    # Q&A via LLM
│   │   ├── llm_temporal_intelligence_plugin.py  # Temporal concepts via LLM
│   │   # Merge/utility plugins
│   │   └── merge_keywords_plugin.py         # Merges results from multiple keyword plugins
│   │
│   ├── query_processing/            # Plugins that process search queries
│   │   ├── __init__.py
│   │   ├── query_expansion_plugin.py        
│   │   └── question_understanding_plugin.py 
│   │
│   └── monitoring/                  # System monitoring plugins
│       ├── __init__.py
│       └── service_health_monitor_plugin.py
│
├── services/                        # HTTP services that OWN providers
│   ├── __init__.py
│   ├── base_service.py
│   ├── keyword_expansion_service.py    # Manages keyword providers
│   ├── temporal_parsing_service.py     # Manages temporal providers
│   ├── question_answer_service.py      # Manages Q&A providers
│   └── ingestion_service.py           # Orchestrates ingestion
│
├── ingestion/                       # YAML-driven field processing
│   ├── __init__.py
│   ├── ingestion_orchestrator.py   
│   ├── field_processor.py          
│   ├── config_loader.py            
│   └── enrichment_merger.py        
│
└── shared/
    ├── http_client/                 # Shared HTTP client for plugins
    │   ├── __init__.py
    │   ├── base_http_client.py
    │   └── circuit_breaker.py
    │
    └── llm_clients/                # Used by services, NOT plugins
        ├── __init__.py
        ├── base_llm_client.py
        └── ollama_client.py
```

### Phase 2: Provider-Specific Plugin Architecture

#### Problem with Current Architecture:
```python
# Current: One plugin calls ALL providers with no control
class ConceptExpansionPlugin(BaseConceptPlugin):
    async def expand_concept(self, concept):
        # Always calls conceptnet + llm + gensim
        results = {}
        results["conceptnet"] = await self.providers["conceptnet"].expand(concept)
        results["llm"] = await self.providers["llm"].expand(concept)
        results["gensim"] = await self.providers["gensim"].expand(concept)
        return self._merge_all_results(results)  # No choice!
```

#### After (Provider-Specific Plugins):
```python
# Individual plugins for each provider
class ConceptNetKeywordPlugin(HTTPBasePlugin):
    """Expands keywords using ONLY ConceptNet."""
    
    async def enrich_field(self, field_value: str, config: Dict) -> Dict:
        response = await self.http_post(
            f"{self.service_url}/keywords/conceptnet",
            {"text": field_value, "max_concepts": config.get("max_concepts", 10)}
        )
        return {"conceptnet_keywords": response["concepts"]}

class LLMKeywordPlugin(HTTPBasePlugin):
    """Expands keywords using ONLY LLM."""
    
    async def enrich_field(self, field_value: str, config: Dict) -> Dict:
        response = await self.http_post(
            f"{self.service_url}/keywords/llm",
            {"text": field_value, "max_concepts": config.get("max_concepts", 15)}
        )
        return {"llm_keywords": response["concepts"]}

class MergeKeywordsPlugin(HTTPBasePlugin):
    """Merges results from multiple keyword plugins."""
    
    async def merge_enrichments(self, enrichments: List[Dict], config: Dict) -> Dict:
        # Merge strategy can be configured
        strategy = config.get("strategy", "union")
        all_keywords = []
        
        for enrichment in enrichments:
            all_keywords.extend(enrichment.get("keywords", []))
            
        if strategy == "union":
            merged = list(set(all_keywords))
        elif strategy == "intersection":
            # Find common keywords across sources
            merged = self._find_common_keywords(enrichments)
            
        return {"merged_keywords": merged}
```

### Phase 3: Safe Migration Steps

#### 3.1: Create New Structure First (Keep Everything Working!)
```bash
# Create new directories - don't delete anything yet!
mkdir -p src/plugins/field_enrichment
mkdir -p src/plugins/query_processing
mkdir -p src/plugins/monitoring

# Create new HTTP base plugin
# src/plugins/http_base.py
```

```python
# src/plugins/http_base.py
class HTTPBasePlugin(BasePlugin):
    """Base class for all plugins - they all make HTTP calls."""
    
    def __init__(self):
        super().__init__()
        self.http_client = HTTPClient()  # With circuit breaker, retries, etc.
        
    async def http_post(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP POST with standard error handling."""
        return await self.http_client.post(url, json=data)
        
    async def http_get(self, url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make HTTP GET with standard error handling."""
        return await self.http_client.get(url, params=params)
```

#### 3.2: Implement Provider-Specific Plugins (Alongside Existing Ones)
```python
# NEW: src/plugins/field_enrichment/conceptnet_keyword_plugin.py
# Extract ConceptNet logic from existing ConceptExpansionPlugin
class ConceptNetKeywordPlugin(HTTPBasePlugin):
    """Calls ONLY ConceptNet for keyword expansion."""
    # Implementation based on existing ConceptExpansionPlugin logic

# NEW: src/plugins/field_enrichment/llm_keyword_plugin.py  
# Extract LLM logic from existing ConceptExpansionPlugin
class LLMKeywordPlugin(HTTPBasePlugin):
    """Calls ONLY LLM for keyword expansion."""
    # Implementation based on existing ConceptExpansionPlugin logic

# Continue for all providers...
```

**Key: Keep existing plugins running while building new ones!**

#### 3.3: Gradual Switchover Strategy
```yaml
# Start with one field in test environment
fields:
  Name:
    enrichments:
      # Switch from old to new
      # - plugin: concept_expansion  # OLD
      - plugin: conceptnet_keywords  # NEW
      
  Overview:
    enrichments:
      - plugin: concept_expansion  # Keep old for now
```

**Rollout Plan:**
1. Test new plugins in development
2. Switch one field at a time in staging
3. Monitor metrics and logs
4. Gradual production rollout
5. Keep old plugins as fallback

#### 3.4: Update Service Endpoints (Non-Breaking)
```python
# Add new specific endpoints alongside existing ones
@app.post("/keywords/expand")  # Keep existing
async def expand_keywords_legacy(...)

@app.post("/keywords/conceptnet")  # Add new
async def expand_conceptnet(...)

@app.post("/keywords/llm")  # Add new
async def expand_llm(...)
```

#### 3.5: Cleanup (Only After Full Validation)
```bash
# ONLY after all systems are using new plugins
# ONLY after production has been stable for X days
# ONLY after rollback is no longer needed

# Archive old plugins first (don't delete!)
mkdir -p src/plugins/_archived_2024
mv src/plugins/concept_expansion_plugin.py src/plugins/_archived_2024/
mv src/plugins/temporal_analysis_plugin.py src/plugins/_archived_2024/

# Later, once confident, remove archived files
rm -rf src/plugins/_archived_2024
```

### Phase 4: Update Plugin Implementations

**Before**: Complex plugin with provider management
```python
class ConceptExpansionPlugin(BaseConceptPlugin):
    # 500+ lines of provider management, resource strategies, queuing, etc.
```

**After**: Simple HTTP orchestrator
```python
class KeywordExpansionPlugin(HTTPBasePlugin):
    """Expands keywords for field enrichment."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="KeywordExpansionPlugin",
            version="2.0.0",
            description="Expands keywords via keyword service",
            plugin_type=PluginType.FIELD_ENRICHMENT,
            tags=["keyword", "expansion", "enrichment"]
        )
    
    async def enrich_field(
        self, 
        field_name: str, 
        field_value: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enrich a field with expanded keywords."""
        # Extract keywords from field
        keywords = self._extract_keywords(field_value)
        
        # Call service for expansion
        expanded = await self.http_post(
            f"{settings.keyword_service_url}/keywords/expand",
            {
                "keywords": keywords,
                "providers": config.get("providers", ["conceptnet"]),
                "max_concepts": config.get("max_concepts", 10)
            }
        )
        
        return {
            "original": field_value,
            "keywords": keywords,
            "expanded_keywords": expanded["concepts"],
            "metadata": expanded.get("metadata", {})
        }
```

### Phase 5: Service Ownership of Providers

Services now own ALL provider management:

```python
# src/services/keyword_expansion_service.py
class KeywordExpansionService:
    def __init__(self):
        # Service owns and manages providers
        self.providers = {
            "conceptnet": ConceptNetProvider(),
            "llm": LLMKeywordProvider(),
            "gensim": GensimProvider()
        }
        
    @app.post("/keywords/expand")
    async def expand_keywords(request: KeywordExpansionRequest):
        """Service handles all provider orchestration."""
        results = {}
        
        for provider_name in request.providers:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                results[provider_name] = await provider.expand(request.keywords)
                
        # Merge results intelligently
        merged = self._merge_expansions(results)
        return {"concepts": merged, "metadata": {...}}
```

### Phase 6: Fine-Grained YAML Configuration

With provider-specific plugins, you get precise control over field enrichment:

```yaml
# config/ingestion/movie.yaml
media_type: movie
version: 1.0

fields:
  Name:
    enrichments:
      # Just use ConceptNet for movie names
      - plugin: conceptnet_keywords
        config:
          max_concepts: 5
          
  Overview:
    enrichments:
      # Use multiple providers and merge results
      - plugin: conceptnet_keywords
        id: conceptnet_overview
        config:
          max_concepts: 10
      - plugin: llm_keywords
        id: llm_overview
        config:
          max_concepts: 15
      - plugin: merge_keywords
        inputs: [conceptnet_overview, llm_overview]
        config:
          strategy: union
          max_results: 20
      # Also extract temporal info
      - plugin: spacy_temporal
        config:
          extract_dates: true
      - plugin: heideltime_temporal
        config:
          reference_date: document_date
          
  Genres:
    enrichments:
      # Only LLM for genre expansion (it understands nuance better)
      - plugin: llm_keywords
        config:
          max_concepts: 10
          prompt: "Related movie genres and subgenres for: {value}"
          smart_retry_until: "list"  # Ensure LLM returns a list, not a string
          
  Tags:
    enrichments:
      # Use Gensim for finding similar tags
      - plugin: gensim_similarity
        config:
          threshold: 0.7
          max_similar: 5
```

This approach provides:
- **Precise control**: Choose exactly which provider for each field
- **Flexibility**: Mix and match providers as needed
- **Clarity**: Explicit about what's happening
- **Efficiency**: Only call the providers you need

### Phase 7: Benefits of Simplified Architecture

1. **50% Less Code**: No duplicate plugin implementations
2. **Clear Responsibilities**: 
   - Plugins: Orchestration only
   - Services: Provider management
3. **Easier Testing**: Mock HTTP calls instead of complex providers
4. **Better Scalability**: Services can scale independently
5. **Simpler Onboarding**: One way to do things, not two
6. **Resource Isolation**: Heavy models only in service containers
7. **Consistent Pattern**: Every plugin works the same way
8. **Fine-grained Control**: Choose specific providers per field
9. **Explicit Merging**: Clear control over how results are combined
10. **Better Performance**: Only call the providers you actually need

### Phase 8: Service Splitting to Match Plugin Architecture

Since we're creating provider-specific plugins, the monolithic NLP service should be split to match:

#### Current Monolithic NLP Service
```
nlp-service (huge container, 8GB+)
├── All NLP models loaded at once
├── ConceptNet, Gensim, SpaCy, HeidelTime, SUTime
└── Complex resource management
```

#### New Provider-Specific Services

```yaml
# docker-compose.yml
services:
  # Model initialization runs first
  model-init:
    image: model-manager
    volumes:
      - model-cache:/models
    environment:
      - MODELS_TO_DOWNLOAD=spacy_en,gensim_word2vec,heideltime_models
    command: python -m model_manager.download_all
    
  # Lightweight API-based service (no models)
  conceptnet-service:
    image: conceptnet-service
    ports:
      - "8001:8000"
    environment:
      - CONCEPTNET_API_URL=http://api.conceptnet.io
      
  # Medium-weight service with word vectors
  gensim-service:
    image: gensim-service
    ports:
      - "8002:8000"
    volumes:
      - model-cache:/models:ro
    depends_on:
      model-init:
        condition: service_completed_successfully
        
  # SpaCy service with NLP models
  spacy-service:
    image: spacy-service
    ports:
      - "8003:8000"
    volumes:
      - model-cache:/models:ro
    depends_on:
      model-init:
        condition: service_completed_successfully
        
  # Temporal parsing services
  temporal-service:
    image: temporal-service
    ports:
      - "8004:8000"
    volumes:
      - model-cache:/models:ro
    depends_on:
      model-init:
        condition: service_completed_successfully

volumes:
  model-cache:
```

#### Model Manager Preservation

The critical `model_manager` functionality is preserved:

```python
# model_manager/download_all.py
class ModelManager:
    """Ensures all models exist before services start."""
    
    def download_all(self):
        """Called on container startup."""
        self.ensure_spacy_models()
        self.ensure_gensim_models()
        self.ensure_temporal_models()
        self.warm_up_models()  # Pre-load into memory if needed
        
    def ensure_spacy_models(self):
        """Download SpaCy models if not present."""
        if not self.model_exists("en_core_web_lg"):
            self.download_spacy_model("en_core_web_lg")
            
    # ... similar for other model types
```

#### Benefits of Service Splitting

1. **Independent Scaling**: Scale ConceptNet differently than Gensim
2. **Smaller Containers**: Each ~1-2GB instead of one 8GB+ container
3. **Faster Startup**: Services start in parallel after model-init
4. **Better Resource Usage**: Only run what you need
5. **Cleaner Dependencies**: Each service has minimal requirements
6. **Model Sharing**: All services use same model cache
7. **Preserved Startup**: model_manager still ensures models exist

### Phase 9: Implementation Priority

1. **Day 1-2**: Create new directory structure
2. **Day 3-4**: Migrate and simplify plugins (delete duplicates)
3. **Day 5-6**: Split monolithic NLP service into provider-specific services
4. **Day 7**: Update docker-compose with model-init pattern
5. **Day 8-9**: Testing with simplified architecture
6. **Day 10**: Documentation updates

### Success Criteria

- [ ] All duplicate plugins removed
- [ ] No "remote" prefix anywhere in codebase
- [ ] All plugins inherit from HTTPBasePlugin
- [ ] Services own all provider management
- [ ] Worker container has minimal resource requirements
- [ ] All tests passing with new structure
- [ ] Plugin code reduced by >50%
- [ ] Clear separation: orchestration (plugins) vs execution (services)

### Migration Checklist

Before:
- [ ] 8 plugin files (regular + remote versions)
- [ ] Plugins load providers directly
- [ ] Complex resource management in plugins
- [ ] Confusing "remote" vs "regular" distinction
- [ ] One plugin calls ALL providers (no control)

After:
- [ ] ~15 focused plugin files (one per provider + utilities)
- [ ] Plugins only make HTTP calls
- [ ] Simple, consistent plugin structure
- [ ] Services handle all complexity
- [ ] Fine-grained control over which providers process which fields
- [ ] Explicit merging when multiple providers are needed

### Example Migration: ConceptExpansionPlugin

**Before**: One monolithic plugin
```
ConceptExpansionPlugin (500+ lines)
  - Loads conceptnet, llm, gensim providers
  - Complex resource management
  - Always calls all providers
  - Internal merging logic
```

**After**: Multiple focused plugins
```
ConceptNetKeywordPlugin (50 lines) - Just calls ConceptNet endpoint
LLMKeywordPlugin (50 lines) - Just calls LLM endpoint  
GensimSimilarityPlugin (50 lines) - Just calls Gensim endpoint
MergeKeywordsPlugin (50 lines) - Merges results when needed
```

This simplified architecture makes the codebase much easier to understand and maintain while providing better separation of concerns and fine-grained control for the microservices architecture.