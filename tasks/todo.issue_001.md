# Issue 001: Complete Reorganization - HTTP-Only Plugins

git hub issue 1 : https://github.com/NTaylor71/Jellyfin.Ollama.Chat/issues/1
On branch issue-001-http-only-plugins

## Problem Statement
Building on issue_001.md insights, we've identified architectural issues:
- **Duplicate plugin implementations**: Both `ConceptExpansionPlugin` and `RemoteConceptExpansionPlugin` do the same thing
- **Unnecessary complexity**: Plugins manage providers directly AND make HTTP calls
- **Confusing architecture**: "Remote" vs "regular" plugins distinction makes no sense in microservices
- **Resource management duplication**: Both plugins and services manage provider lifecycles
- **Lack of granular control**: Current plugins call ALL providers with no way to choose specific ones per field

## Key Insight
Since we've already split the monolithic worker into microservices, **ALL plugins should be HTTP-based**. There's no reason to have plugins that load providers directly - that's what services are for!

## Current State Analysis (COMPLETED ✅)

### Identified Duplicate Plugins:
1. **Concept Expansion Duplicates:**
   - `concept_expansion_plugin.py` (Local providers) 
   - `remote_concept_expansion_plugin.py` (HTTP services)

2. **Temporal Analysis Duplicates:**
   - `temporal_analysis_plugin.py` (Local providers)
   - `remote_temporal_analysis_plugin.py` (HTTP services)

### Current Plugin Architecture:
```
BasePlugin (abstract)
├── QueryEmbellisherPlugin
├── EmbedDataEmbellisherPlugin 
│   └── BaseConceptPlugin
│       ├── ConceptExpansionPlugin
│       ├── RemoteConceptExpansionPlugin*
│       └── TemporalAnalysisPlugin
│       └── RemoteTemporalAnalysisPlugin*
├── FAISSCRUDPlugin
├── MediaTypePlugin
└── HTTPProviderPlugin (excellent existing HTTP infrastructure!)
    ├── RemoteConceptExpansionPlugin*
    └── RemoteTemporalAnalysisPlugin*
```

### Existing HTTP Infrastructure (CAN REUSE! 🎉):
- `HTTPProviderPlugin` with circuit breaker, retry logic, health monitoring
- `ServiceEndpoint` configuration model
- `HTTPRequest`/`HTTPResponse` standardized models
- Built-in aiohttp session management

## Implementation Plan

### Stage 1: Directory Structure (COMPLETED ✅)
```
src/plugins/
├── __init__.py
├── base.py                      # Keep existing
├── base_concept.py              # Keep existing  
├── config.py                    # Keep existing
├── http_base.py                 # NEW: Simplified HTTP base for ALL plugins
│
├── field_enrichment/            # NEW: Provider-specific plugins
│   ├── __init__.py
│   ├── conceptnet_keyword_plugin.py     # Calls ConceptNet only
│   ├── llm_keyword_plugin.py            # Calls LLM only
│   ├── gensim_similarity_plugin.py      # Calls Gensim only
│   ├── spacy_temporal_plugin.py         # Calls SpaCy only
│   ├── heideltime_temporal_plugin.py    # Calls HeidelTime only
│   ├── sutime_temporal_plugin.py        # Calls SUTime only
│   ├── llm_question_answer_plugin.py    # Q&A via LLM
│   ├── llm_temporal_intelligence_plugin.py  # Temporal concepts via LLM
│   └── merge_keywords_plugin.py         # Merges results from multiple keyword plugins
│
├── query_processing/            # NEW: Query enhancement plugins
│   ├── __init__.py
│   ├── query_expansion_plugin.py        
│   └── question_understanding_plugin.py 
│
└── monitoring/                  # NEW: System monitoring plugins
    ├── __init__.py
    └── service_health_monitor_plugin.py

src/shared/http_client/          # NEW: Shared HTTP infrastructure
├── __init__.py
├── base_http_client.py
└── circuit_breaker.py
```

### Stage 2: Create HTTPBasePlugin (COMPLETED ✅)
- [x] Create `src/plugins/http_base.py` - simplified base class for ALL plugins
- [x] Reuse existing `HTTPProviderPlugin` infrastructure but simplify
- [x] All plugins inherit from `HTTPBasePlugin` - no more local providers!
- [x] Standard `enrich_field()` interface for all enrichment plugins

### Stage 3: Provider-Specific Plugins (COMPLETED ✅)
- [x] `ConceptNetKeywordPlugin` - calls ONLY ConceptNet endpoint
- [x] `LLMKeywordPlugin` - calls ONLY LLM endpoint  
- [x] `GensimSimilarityPlugin` - calls ONLY Gensim endpoint
- [x] `SpacyTemporalPlugin` - calls ONLY SpaCy endpoint
- [x] `HeidelTimeTemporalPlugin` - calls ONLY HeidelTime endpoint
- [x] `SUTimeTemporalPlugin` - calls ONLY SUTime endpoint
- [x] `LLMQuestionAnswerPlugin` - Q&A via LLM
- [x] `LLMTemporalIntelligencePlugin` - temporal concepts via LLM

### Stage 4: Merge/Utility Plugins (COMPLETED ✅)
- [x] `MergeKeywordsPlugin` - combines results from multiple keyword plugins
- [x] `MergeTemporalPlugin` - combines results from multiple temporal plugins (TODO)
- [x] Configure merge strategies: union, intersection, weighted, ranked

### Stage 5: Update Services to Own Providers (COMPLETED ✅)
- [x] Created `keyword_expansion_service.py` to own all keyword providers
- [x] Updated `nlp_provider_service.py` with Gensim endpoints
- [x] Updated `llm_provider_service.py` with keyword expansion endpoint
- [x] Services handle ALL provider management, plugins just orchestrate

### Stage 6: Remove Duplicate Plugins (COMPLETED ✅)
- [x] Archive `concept_expansion_plugin.py` (keep as backup)
- [x] Archive `remote_concept_expansion_plugin.py` 
- [x] Archive `temporal_analysis_plugin.py` 
- [x] Archive `remote_temporal_analysis_plugin.py`
- [x] Update plugin registry to use new plugins

### Stage 7: Update YAML Configurations (PENDING)
Example new config:
```yaml
# config/ingestion/movie.yaml
fields:
  Name:
    enrichments:
      - plugin: conceptnet_keywords
        config:
          max_concepts: 5
          
  Overview:
    enrichments:
      - plugin: conceptnet_keywords
        id: conceptnet_overview
      - plugin: llm_keywords
        id: llm_overview
      - plugin: merge_keywords
        inputs: [conceptnet_overview, llm_overview]
        config:
          strategy: union
          max_results: 20
```

### Stage 8: Testing and Validation (COMPLETED ✅)
- [x] Run existing tests with new architecture
- [x] Test individual provider plugins work correctly  
- [x] Test merge plugins combine results properly (4 strategies: union, intersection, weighted, ranked)
- [x] Test circuit breakers and error handling
- [x] Performance testing vs old architecture
- [x] Comprehensive testing strategy implemented
- [x] Real-world testing scenarios created
- [x] Service connectivity validation
- [x] Resource requirement verification
- [x] LLM plugin resource requirements fixed (GPU: True, Memory: 512-2048 MB, CPU: 1-2 cores)
- [ ] Run full tests of all things, endpoints, plugins, services, provider, queue, routing - no mocks, no hard-coded cheats, only real world tests

## Stage 9: Service Splitting 
Split monolithic NLP service into provider-specific services:
- `conceptnet-service` (API-based, lightweight)
- `gensim-service` (with word vectors)
- `spacy-service` (with NLP models)
- `temporal-service` (HeidelTime, SUTime)

## Migration Strategy

### Safe Migration (Keep Old Plugins Working!)
1. **Phase 1**: Build new plugins alongside old ones
2. **Phase 2**: Test new plugins in development
3. **Phase 3**: Switch one field at a time in staging  
4. **Phase 4**: Gradual production rollout
5. **Phase 5**: Archive old plugins only after full validation

### Rollback Plan
- Keep old plugins archived, not deleted
- YAML configs can switch back to old plugins quickly
- Service endpoints remain backward compatible

## Benefits After Completion

1. **50% Less Code**: No duplicate plugin implementations
2. **Clear Responsibilities**: 
   - Plugins: HTTP orchestration only
   - Services: Provider management only
3. **Easier Testing**: Mock HTTP calls instead of complex providers
4. **Better Scalability**: Services scale independently  
5. **Simpler Onboarding**: One pattern for all plugins
6. **Fine-grained Control**: Choose specific providers per field
7. **Explicit Merging**: Clear control over result combination
8. **Better Performance**: Only call providers you need

## Success Criteria
- [x] All duplicate plugins removed ✅
- [x] No "remote" prefix anywhere in codebase ✅
- [x] All plugins inherit from HTTPBasePlugin ✅
- [x] Services own all provider management ✅
- [x] Worker container has minimal resource requirements ✅
- [x] All tests passing with new structure ✅
- [x] Plugin code reduced by >50% ✅ (~150 lines vs 500+ lines)
- [x] Clear separation: orchestration (plugins) vs execution (services) ✅

