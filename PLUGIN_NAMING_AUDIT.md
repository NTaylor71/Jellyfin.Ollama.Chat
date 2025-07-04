# PLUGIN NAMING AUDIT & REFACTORING PLAN

## Overview
Rename all plugin files and classes to clearly indicate their intended usage patterns following the successful pattern of `spacy_with_fallback_ingestion_and_query.py`.

## Current Plugin Categories

### 🎯 **USAGE PATTERNS IDENTIFIED:**

1. **INGESTION-ONLY**: Plugins that enhance data during ingestion for storage in MongoDB
2. **QUERY-ONLY**: Plugins that enhance user queries in real-time for search
3. **DUAL-USE**: Plugins that work on both content (ingestion) and queries (search)

## Current Plugin Inventory & Proposed Renames

### 📁 **src/plugins/examples/**

#### ✅ **INGESTION-ONLY PLUGINS** (Enhance content for storage)
```
CURRENT → PROPOSED RENAME
├── movie_summary_enhancer.py → content_summary_ingestion.py
│   └── MovieSummaryEnhancerPlugin → ContentSummaryIngestionPlugin  
│   └── PURPOSE: Enhances movie descriptions during ingestion with LLM-generated summaries
│
├── advanced_embed_enhancer.py → media_metadata_ingestion.py  
│   └── AdvancedEmbedDataEnhancerPlugin → MediaMetadataIngestionPlugin
│   └── PURPOSE: Enriches media content with metadata during ingestion
```

#### 🔍 **QUERY-ONLY PLUGINS** (Enhance user queries for search)
```
CURRENT → PROPOSED RENAME  
├── adaptive_query_expander.py → search_query_enhancement.py
│   └── AdaptiveQueryExpanderPlugin → SearchQueryEnhancementPlugin
│   └── PURPOSE: Expands user search queries with synonyms and related terms
```

#### 🔄 **DUAL-USE PLUGINS** (Work on both content and queries)
```
CURRENT → PROPOSED RENAME
├── faiss_crud_logger.py → vector_operations_dual_use.py
│   └── FAISSCRUDLoggerPlugin → VectorOperationsDualUsePlugin  
│   └── PURPOSE: Logs and optimizes vector operations for both ingestion and search
```

### 📁 **src/plugins/starters/** (Template Examples)

#### ✅ **INGESTION EXAMPLES**
```
CURRENT → PROPOSED RENAME
├── embed_enhancer.py → basic_content_ingestion_starter.py
│   └── EmbedEnhancerPlugin → BasicContentIngestionStarterPlugin
│
├── faiss_logger.py → vector_logging_ingestion_starter.py  
│   └── FAISSLoggerPlugin → VectorLoggingIngestionStarterPlugin
```

#### 🔍 **QUERY EXAMPLES**
```
CURRENT → PROPOSED RENAME
├── query_expander.py → basic_query_enhancement_starter.py
│   └── QueryExpanderPlugin → BasicQueryEnhancementStarterPlugin
```

### 📁 **src/plugins/temporal/**

#### ✅ **INGESTION-ONLY TEMPORAL**
```
CURRENT → PROPOSED RENAME
├── heideltime_ingestion.py → temporal_cultural_ingestion.py
│   └── HeidelTimeIngestionPlugin → TemporalCulturalIngestionPlugin
│   └── PURPOSE: Rich historical/cultural temporal analysis during ingestion
```

#### 🔍 **QUERY-ONLY TEMPORAL**  
```
CURRENT → PROPOSED RENAME
├── duckling_query.py → temporal_expression_query.py
│   └── DucklingQueryPlugin → TemporalExpressionQueryPlugin
│   └── PURPOSE: Fast user query temporal understanding
│
├── sutime_query.py → temporal_reasoning_query.py
│   └── SUTimeQueryPlugin → TemporalReasoningQueryPlugin  
│   └── PURPOSE: Complex temporal reasoning for queries
```

#### 🔄 **DUAL-USE TEMPORAL** (✅ Already correctly named)
```
CURRENT (CORRECT) 
├── spacy_with_fallback_ingestion_and_query.py ✅
│   └── SpacyWithFallbackIngestionAndQueryPlugin ✅
│   └── PURPOSE: spaCy temporal analysis for both content and queries
```

### 📁 **src/plugins/linguistic/**

#### 🔄 **DUAL-USE LINGUISTIC** (Analyze both content and queries)
```
CURRENT → PROPOSED RENAME
├── conceptnet.py → concept_expansion_dual_use.py
│   └── ConceptNetExpansionPlugin → ConceptExpansionDualUsePlugin
│   └── PURPOSE: Expands concepts in both content (ingestion) and queries
│
├── semantic_roles.py → semantic_analysis_dual_use.py
│   └── SemanticRoleLabelerPlugin → SemanticAnalysisDualUsePlugin
│   └── PURPOSE: Extracts semantic roles from both content and queries
```

## Directory Structure Reorganization

### **PROPOSED NEW STRUCTURE:**
```
src/plugins/
├── ingestion/          # Content enhancement during ingestion
│   ├── content_summary_ingestion.py
│   ├── media_metadata_ingestion.py  
│   ├── temporal_cultural_ingestion.py
│   └── starters/
│       ├── basic_content_ingestion_starter.py
│       └── vector_logging_ingestion_starter.py
│
├── query/              # Query enhancement for search
│   ├── search_query_enhancement.py
│   ├── temporal_expression_query.py
│   ├── temporal_reasoning_query.py
│   └── starters/ 
│       └── basic_query_enhancement_starter.py
│
├── dual_use/           # Both ingestion and query processing
│   ├── concept_expansion_dual_use.py
│   ├── semantic_analysis_dual_use.py  
│   ├── spacy_temporal_dual_use.py
│   └── vector_operations_dual_use.py
│
└── base.py            # Base plugin classes
```

## Implementation Plan

### **Phase 1: File & Class Renames** (1 day)
1. Rename all plugin files with clear usage indicators
2. Rename all plugin classes to match new naming convention  
3. Update all import statements throughout codebase
4. Update plugin registry and discovery mechanisms

### **Phase 2: Directory Reorganization** (1 day)
1. Create new directory structure (ingestion/, query/, dual_use/)
2. Move plugins to appropriate directories
3. Update all import paths
4. Update configuration files and documentation

### **Phase 3: Update Plugin Base Classes** (0.5 day)
1. Enhance base class naming for clarity:
   - `IngestionPlugin` (for content enhancement)
   - `QueryPlugin` (for search enhancement)  
   - `DualUsePlugin` (for both)
2. Update inheritance chains
3. Add usage pattern validation

### **Phase 4: Testing & Validation** (0.5 day)
1. Update all test files with new names
2. Validate plugin discovery works
3. Test plugin registration and execution
4. Ensure backward compatibility via aliases

## Benefits After Refactoring

### **🎯 IMMEDIATE CLARITY:**
- **File names** instantly reveal plugin purpose
- **Directory structure** organizes by usage pattern
- **Class names** clearly indicate intended use

### **🔧 DEVELOPER EXPERIENCE:**
- New developers immediately understand plugin categories
- Easy to find relevant plugins for specific needs
- Clear separation of concerns

### **🚀 FUTURE EXTENSIBILITY:**
- Easy to add new ingestion-only plugins for content enhancement
- Simple to create query-only plugins for search improvement  
- Clear pattern for dual-use plugins

### **📋 MAINTENANCE:**
- Easier to identify which plugins need updates for schema changes
- Clear understanding of impact when modifying ingestion vs query logic
- Better organization for documentation and examples

## Naming Convention Rules

### **FILE NAMING:**
- **Ingestion**: `{purpose}_ingestion.py`
- **Query**: `{purpose}_query.py`  
- **Dual-Use**: `{purpose}_dual_use.py` or `{purpose}_ingestion_and_query.py`

### **CLASS NAMING:**
- **Ingestion**: `{Purpose}IngestionPlugin`
- **Query**: `{Purpose}QueryPlugin`
- **Dual-Use**: `{Purpose}DualUsePlugin` or `{Purpose}IngestionAndQueryPlugin`

### **DIRECTORY NAMING:**
- `ingestion/` - Content enhancement plugins
- `query/` - Search enhancement plugins  
- `dual_use/` - Plugins that work on both

This refactoring will make the plugin system **crystal clear** and **highly organized** for the upcoming media-agnostic architecture!