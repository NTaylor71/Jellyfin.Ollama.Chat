# Issue 001: Complete Reorganization - Making Concepts Clear on Disk

## Problem Statement

The current codebase structure is confusing and doesn't clearly communicate the purpose of each component:
- `src/concept_expansion/` contains temporal providers alongside concept providers
- Providers are organized by technology (llm, spacy) rather than function
- Services expose generic endpoints that don't match their actual purposes
- The distinction between concept expansion (keyword → keywords) and question answering (question → text) is unclear
- No clear system for YAML-driven field enrichment during ingestion

## Objective

Reorganize the codebase to make each component's purpose immediately obvious from its location and name. Implement a clear separation between:
1. **Keyword Expansion**: keyword → list of related keywords
2. **Temporal Parsing**: text → temporal entities
3. **Question Answering**: question → text answer
4. **Temporal Intelligence**: temporal term → temporal concepts

## Complete Reorganization Plan

### Phase 1: Clean Directory Structure

```
src/
├── providers/                         # What they DO, not what they USE
│   ├── __init__.py
│   ├── base_provider.py              # Common base class
│   │
│   ├── keyword_expansion/            # keyword → [keywords]
│   │   ├── __init__.py
│   │   ├── conceptnet_keyword_provider.py
│   │   ├── gensim_similarity_provider.py
│   │   └── llm_keyword_expansion_provider.py
│   │
│   ├── temporal_parsing/             # text → temporal entities
│   │   ├── __init__.py
│   │   ├── heideltime_parser_provider.py
│   │   ├── sutime_parser_provider.py
│   │   └── spacy_temporal_parser_provider.py
│   │
│   ├── question_answering/           # question → text
│   │   ├── __init__.py
│   │   └── llm_question_answer_provider.py
│   │
│   └── temporal_intelligence/        # temporal term → concepts
│       ├── __init__.py
│       └── llm_temporal_concept_provider.py
│
├── plugins/                          # Orchestrate providers for specific tasks
│   ├── __init__.py
│   ├── base.py
│   │
│   ├── field_enrichment/            # Plugins that enrich data fields
│   │   ├── keyword_expansion_plugin.py      # Uses keyword providers
│   │   ├── temporal_parsing_plugin.py       # Uses temporal parsers
│   │   ├── question_answer_plugin.py        # Uses Q&A providers
│   │   └── temporal_intelligence_plugin.py  # Uses temporal concept provider
│   │
│   ├── query_processing/            # Plugins that process search queries
│   │   ├── query_expansion_plugin.py        # Expands search queries
│   │   └── question_understanding_plugin.py # Understands questions
│   │
│   ├── monitoring/                  # System monitoring plugins
│   │   └── service_health_monitor_plugin.py
│   │
│   └── remote/                      # HTTP client plugins
│       ├── http_provider_plugin.py
│       ├── remote_keyword_expansion_plugin.py
│       ├── remote_temporal_parsing_plugin.py
│       └── remote_question_answer_plugin.py
│
├── services/                        # HTTP services exposing providers
│   ├── __init__.py
│   ├── keyword_expansion_service.py    # /keywords/expand, /keywords/similarity
│   ├── temporal_parsing_service.py     # /temporal/parse, /temporal/extract
│   ├── question_answer_service.py      # /questions/answer, /questions/analyze
│   └── ingestion_service.py           # /ingest/{media_type}
│
├── ingestion/                       # YAML-driven field processing
│   ├── __init__.py
│   ├── ingestion_orchestrator.py   # Main orchestrator
│   ├── field_processor.py          # Processes individual fields
│   ├── config_loader.py            # Loads YAML configs
│   └── enrichment_merger.py        # Merges enrichments into documents
│
└── shared/
    └── llm_clients/                # Shared LLM infrastructure
        ├── __init__.py
        ├── base_llm_client.py
        └── ollama_client.py
```

### Phase 2: Clear Naming Conventions

**Provider Naming**: `{technology}_{function}_provider.py`
- `conceptnet_keyword_provider.py` - Uses ConceptNet for keyword expansion
- `llm_question_answer_provider.py` - Uses LLM for answering questions
- `heideltime_parser_provider.py` - Uses HeidelTime for parsing temporal

**Plugin Naming**: `{function}_plugin.py`
- `keyword_expansion_plugin.py` - Expands keywords in fields
- `temporal_parsing_plugin.py` - Parses temporal information
- `question_answer_plugin.py` - Answers questions about fields

**Service Naming**: `{function}_service.py`
- `keyword_expansion_service.py` - HTTP service for keyword operations
- `question_answer_service.py` - HTTP service for Q&A operations

### Phase 3: Migration Steps

#### 3.1: Create New Directory Structure
```bash
# Provider directories
mkdir -p src/providers/{keyword_expansion,temporal_parsing,question_answering,temporal_intelligence}

# Plugin directories
mkdir -p src/plugins/{field_enrichment,query_processing,monitoring,remote}

# Service directory
mkdir -p src/services

# Ingestion system
mkdir -p src/ingestion

# Shared LLM infrastructure
mkdir -p src/shared/llm_clients
```

#### 3.2: Provider Migration Map

**Keyword Expansion Providers**:
```
src/concept_expansion/providers/conceptnet_provider.py 
  → src/providers/keyword_expansion/conceptnet_keyword_provider.py

src/concept_expansion/providers/gensim_provider.py 
  → src/providers/keyword_expansion/gensim_similarity_provider.py

src/concept_expansion/providers/llm/llm_provider.py 
  → src/providers/keyword_expansion/llm_keyword_expansion_provider.py
```

**Temporal Parsing Providers**:
```
src/concept_expansion/providers/heideltime_provider.py 
  → src/providers/temporal_parsing/heideltime_parser_provider.py

src/concept_expansion/providers/sutime_provider.py 
  → src/providers/temporal_parsing/sutime_parser_provider.py

src/concept_expansion/providers/spacy_temporal_provider.py 
  → src/providers/temporal_parsing/spacy_temporal_parser_provider.py
```

**New Providers**:
```
(NEW) → src/providers/question_answering/llm_question_answer_provider.py

src/concept_expansion/temporal_concept_generator.py 
  → src/providers/temporal_intelligence/llm_temporal_concept_provider.py
```

**Shared LLM Infrastructure**:
```
src/concept_expansion/providers/llm/base_llm_client.py 
  → src/shared/llm_clients/base_llm_client.py

src/concept_expansion/providers/llm/ollama_backend_client.py 
  → src/shared/llm_clients/ollama_client.py
```

#### 3.3: Plugin Migration Map

```
src/plugins/concept_expansion_plugin.py 
  → src/plugins/field_enrichment/keyword_expansion_plugin.py

src/plugins/temporal_analysis_plugin.py 
  → src/plugins/field_enrichment/temporal_parsing_plugin.py

src/plugins/question_expansion_plugin.py 
  → src/plugins/query_processing/question_understanding_plugin.py

(NEW) → src/plugins/field_enrichment/question_answer_plugin.py
(NEW) → src/plugins/field_enrichment/temporal_intelligence_plugin.py

src/plugins/service_health_monitor_plugin.py 
  → src/plugins/monitoring/service_health_monitor_plugin.py

src/plugins/http_provider_plugin.py 
  → src/plugins/remote/http_provider_plugin.py

src/plugins/remote_concept_expansion_plugin.py 
  → src/plugins/remote/remote_keyword_expansion_plugin.py

src/plugins/remote_temporal_analysis_plugin.py 
  → src/plugins/remote/remote_temporal_parsing_plugin.py

(NEW) → src/plugins/remote/remote_question_answer_plugin.py
```

#### 3.4: Service Migration Map

```
src/services/minimal_nlp_service.py 
  → Split into:
     - src/services/keyword_expansion_service.py (ConceptNet, Gensim endpoints)
     - src/services/temporal_parsing_service.py (SpaCy, HeidelTime, SUTime endpoints)

src/services/minimal_llm_service.py 
  → src/services/question_answer_service.py (with new Q&A endpoints)

(NEW) → src/services/ingestion_service.py
```

### Phase 4: Update Imports and Dependencies

1. **Update all imports** to use new paths
2. **Update plugin type enums** if needed
3. **Update service endpoints** to match functionality:
   - `/keywords/expand` instead of `/providers/conceptnet/expand`
   - `/questions/answer` instead of `/providers/ollama/expand`
   - `/temporal/parse` instead of `/providers/spacy/temporal`
4. **Update tests** to use new structure
5. **Update Docker configurations** for new services

### Phase 5: Implement YAML-Driven Ingestion

#### 5.1: Ingestion Configuration Structure

**config/ingestion/movie.yaml**:
```yaml
media_type: movie
version: 1.0

# Field-level enrichment configuration
fields:
  Name:
    enrichments:
      - plugin: keyword_expansion
        providers: [conceptnet, llm]
        config:
          max_concepts: 10
          cache_days: 30
  
  Overview:
    enrichments:
      - plugin: keyword_expansion
        providers: [llm]
        config:
          max_concepts: 20
          cache_days: 7
      - plugin: temporal_parsing
        providers: [spacy, heideltime]
        config:
          extract_dates: true
          normalize_dates: true
      - plugin: question_answer
        questions:
          - id: themes
            question: "What themes are present in this movie overview: {value}"
          - id: mood
            question: "What is the emotional mood of: {value}"
        config:
          cache_days: 30
  
  Genres:
    enrichments:
      - plugin: keyword_expansion
        providers: [conceptnet]
        config:
          direct_values: true
          weight: 2.0
  
  PremiereDate:
    enrichments:
      - plugin: temporal_parsing
        providers: [sutime]
        config:
          reference_date: "document_date"
      - plugin: temporal_intelligence
        config:
          generate_era: true
          generate_decade: true
          generate_relative: true
  
  Tags:
    enrichments:
      - plugin: keyword_expansion
        providers: [gensim, conceptnet]
        config:
          direct_values: true
          max_concepts: 5

# Computed fields - generated from other fields
computed_fields:
  content_warnings:
    plugin: question_answer
    question: "Based on the movie '{Name}' with overview '{Overview}', what content warnings or age ratings would apply?"
    cache_days: 90
  
  cultural_context:
    plugin: question_answer  
    question: "What was the cultural and historical context when '{Name}' was released in {PremiereDate}?"
    cache_days: 365
  
  era_concepts:
    plugin: temporal_intelligence
    input_field: PremiereDate
    config:
      generate_cultural_movements: true
      generate_technology_context: true
```

#### 5.2: Ingestion Components

**IngestionOrchestrator**:
- Loads YAML configuration for media type
- Orchestrates field processing in parallel where possible
- Handles computed fields after main fields
- Manages enrichment caching

**FieldProcessor**:
- Processes individual fields according to configuration
- Calls appropriate plugins with configured providers
- Handles errors gracefully with fallbacks

**EnrichmentMerger**:
- Merges enrichments into final document structure
- Handles conflicts between different enrichments
- Maintains enrichment metadata and provenance

### Phase 6: Service Endpoints

#### 6.1: Keyword Expansion Service
```
POST /keywords/expand
  Body: {"keyword": "action", "providers": ["conceptnet", "llm"], "max_concepts": 10}
  Response: {"concepts": ["thriller", "adventure", "exciting", ...]}

POST /keywords/similarity
  Body: {"keyword": "robot", "providers": ["gensim"], "threshold": 0.7}
  Response: {"similar": ["android", "cyborg", "automaton", ...]}

POST /keywords/batch
  Body: {"keywords": ["action", "robot"], "providers": ["conceptnet"]}
  Response: {"results": {"action": [...], "robot": [...]}}
```

#### 6.2: Question Answer Service
```
POST /questions/answer
  Body: {"question": "What is the mood of Inception?", "context": {...}}
  Response: {"answer": "The mood of Inception is...", "confidence": 0.9}

POST /questions/analyze
  Body: {"text": "...", "questions": ["themes", "mood", "genre"]}
  Response: {"themes": "...", "mood": "...", "genre": "..."}

POST /questions/batch
  Body: {"questions": [{"id": "q1", "question": "..."}, ...]}
  Response: {"results": {"q1": {"answer": "..."}, ...}}
```

#### 6.3: Temporal Parsing Service
```
POST /temporal/parse
  Body: {"text": "Released in summer 1999", "providers": ["heideltime"]}
  Response: {"entities": [{"text": "summer 1999", "normalized": "1999-06-01", ...}]}

POST /temporal/extract
  Body: {"text": "The 90s were great", "providers": ["spacy"]}
  Response: {"temporal_expressions": ["90s"], "decades": ["1990s"]}
```

#### 6.4: Ingestion Service
```
POST /ingest/movie
  Body: {"Name": "Inception", "Overview": "...", "Genres": [...]}
  Response: {enriched document with all configured enrichments}

GET /ingest/config/movie
  Response: {current ingestion configuration for movies}

POST /ingest/config/reload
  Response: {"status": "Configuration reloaded"}
```

### Phase 7: Benefits of This Organization

1. **Crystal Clear Purpose**: Directory names immediately communicate function
2. **No Ambiguity**: Clear distinction between keyword expansion vs question answering
3. **Technology Agnostic**: Providers grouped by what they do, not how they do it
4. **Scalable**: Easy to add new providers/plugins in the right place
5. **Service Clarity**: Each service has a single, well-defined purpose
6. **Plugin Organization**: Clear separation between field enrichment, query processing, and monitoring
7. **YAML-Driven**: Flexible ingestion configuration without code changes
8. **Self-Documenting**: The structure itself serves as documentation

### Phase 8: Implementation Priority

1. **Week 1**: Directory restructuring and file migration
2. **Week 2**: Update imports and fix dependencies
3. **Week 3**: Implement new providers (question_answer, temporal_intelligence)
4. **Week 4**: Update services with clear endpoints
5. **Week 5**: Implement YAML-driven ingestion system
6. **Week 6**: Testing and documentation

### Success Criteria

- [ ] All providers organized by function, not technology
- [ ] Clear distinction between keyword expansion and question answering
- [ ] Services have endpoints that match their purpose
- [ ] YAML configuration drives field enrichment
- [ ] No imports from old `concept_expansion` directory
- [ ] All tests passing with new structure
- [ ] Documentation updated to reflect new organization

This reorganization will transform the codebase from confusing to self-explanatory, making it easy for new developers to understand and extend.