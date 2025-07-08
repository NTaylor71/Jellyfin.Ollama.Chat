# Movie Media Type Configuration Documentation

## Overview

This document describes the new YAML configuration format for the movie media type in the Jellyfin.Ollama.Chat project. The configuration leverages the HTTP-only plugin architecture implemented as part of Issue #1, providing a powerful and flexible system for enriching movie metadata through various AI and NLP services.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Concepts](#core-concepts)
3. [Configuration Structure](#configuration-structure)
4. [Plugin System](#plugin-system)
5. [Weighting System](#weighting-system)
6. [Advanced Features](#advanced-features)
7. [Usage Examples](#usage-examples)
8. [Best Practices](#best-practices)

## Architecture Overview

The new configuration system is built on several key principles:

1. **HTTP-Only Plugins**: All plugins make HTTP calls to microservices - no local provider management
2. **Grouped Enrichments**: Plugins can be grouped for logical operations and scoped merging
3. **Plugin Chaining**: Results from one plugin can be referenced by subsequent plugins
4. **Weighted Merging**: Multiple strategies for combining results from different sources
5. **Synthetic Fields**: Generate new fields that don't exist in source data
6. **Field Templating**: Use Jellyfin field values in plugin prompts and configurations

## Core Concepts

### Fields

Fields represent data points about a movie. They can be:
- **Source fields**: Data from Jellyfin (e.g., `Name`, `Overview`, `Genres`)
- **Synthetic fields**: Generated through enrichment (e.g., `Reviews`, `CulturalImpact`)

### Enrichments

Enrichments are plugin operations that enhance field data. They support:
- Single plugin execution
- Grouped plugin execution with merging
- Chained operations using previous results

### Groups

Groups are collections of plugins that operate together:
- Defined implicitly through indentation
- Merge plugins only see siblings within their group
- Groups can execute in parallel
- Each group has independent scope

### Weighting

The system supports multiple levels of weighting:
- **Field weights**: Overall importance of fields
- **Group weights**: Importance of enrichment groups
- **Plugin weights**: Individual plugin importance within groups
- **Ranking factors**: For sophisticated merge strategies

## Configuration Structure

### Top-Level Structure

```yaml
media_type: movie
name: Movie
description: Movie media type with HTTP-only plugin enrichment

# Global configurations
field_weights: {}      # Field importance for ranking
fields: {}            # Field definitions with enrichments
weighting: {}         # Weight calculation configuration
execution: {}         # Execution parameters
priority_order: []    # Field processing order
output: {}           # Output formatting options
validation: {}       # Data validation rules
```

### Field Definition Structure

```yaml
fields:
  FieldName:
    source_field: Name           # Jellyfin field name (null for synthetic)
    type: string                 # Data type
    field_weight: 1.5           # Override global weight
    enrichments:                # List of enrichment operations
      - plugin: plugin_name
        config: {}
```

### Enrichment Types

#### 1. Simple Plugin Enrichment

```yaml
enrichments:
  - plugin: conceptnet_keywords
    config:
      max_concepts: 5
      min_confidence: 0.7
```

#### 2. Grouped Enrichment with Merging

```yaml
enrichments:
  - group:
      group_weight: 2.0
      plugins:
        - plugin: conceptnet_keywords
          id: cn_keywords
          plugin_weight: 1.0
          config: {}
          
        - plugin: llm_keywords
          id: llm_keywords
          plugin_weight: 2.0
          config: {}
          
        - plugin: merge_keywords
          inputs: [cn_keywords, llm_keywords]
          config:
            strategy: weighted
```

#### 3. Chained Enrichments

```yaml
enrichments:
  # First enrichment
  - plugin: conceptnet_keywords
    id: concepts
    
  # Second enrichment using first's results
  - plugin: llm_question_answer
    config:
      prompt: |
        Analyze these concepts: {@concepts.keywords}
```

## Plugin System

### Available Plugins

1. **Keyword Extraction**
   - `conceptnet_keywords`: Semantic concept extraction
   - `llm_keywords`: LLM-based keyword extraction
   - `gensim_similarity`: Word similarity expansion

2. **Temporal Analysis**
   - `heideltime_temporal`: Time expression extraction
   - `spacy_temporal`: SpaCy-based temporal analysis
   - `sutime_temporal`: Stanford SUTime
   - `llm_temporal_intelligence`: LLM temporal understanding

3. **Question Answering**
   - `llm_question_answer`: Flexible Q&A for any purpose

4. **Utility**
   - `merge_keywords`: Combine results from multiple sources

### Plugin Configuration

Each plugin supports specific configuration options:

```yaml
- plugin: llm_keywords
  id: unique_identifier        # For referencing results
  plugin_weight: 2.0          # Weight within group
  field_override: Name        # Use different source field
  config:
    prompt: "Template {Field}" # Field value substitution
    max_keywords: 10
    temperature: 0.3
    expected_format: list
```

### Reference Syntax

- `{FieldName}`: Substitute field value in prompts
- `{@plugin_id}`: Reference entire plugin output
- `{@plugin_id.field}`: Reference specific field from output
- `{@plugin_id.0}`: Array index access

## Weighting System

### Weight Types

1. **Field Weight** (`field_weight`)
   - Overall importance of the field
   - Used for search ranking and processing priority
   - Range: 0.1 to 10.0

2. **Group Weight** (`group_weight`)
   - Importance of an enrichment group within a field
   - Multiplies with plugin weights
   - Range: 0.1 to 10.0

3. **Plugin Weight** (`plugin_weight`)
   - Individual plugin importance within its group
   - Used by merge strategies
   - Range: 0.1 to 10.0

### Merge Strategies

#### 1. Union
Combines all results, removing duplicates:
```yaml
strategy: union
max_results: 20
```

#### 2. Intersection
Only keeps results that appear in all sources:
```yaml
strategy: intersection
min_sources: 2
```

#### 3. Weighted
Uses plugin weights to prioritize results:
```yaml
strategy: weighted
normalize_weights: true
```

#### 4. Ranked
Complex ranking using multiple factors:
```yaml
strategy: ranked
ranking_factors:
  plugin_weight: 0.4   # 40% based on source weight
  frequency: 0.3       # 30% based on occurrence count
  confidence: 0.3      # 30% based on confidence scores
```

## Advanced Features

### Synthetic Fields

Create fields that don't exist in source data:

```yaml
Reviews:
  source_field: null  # No source field
  type: dict
  enrichments:
    - plugin: llm_question_answer
      config:
        prompt: |
          Write a review of "{Name}" ({ProductionYear})
          Genre: {Genres}
          Plot: {Overview}
```

### Multi-Stage Processing

Build complex analyses through stages:

```yaml
Themes:
  enrichments:
    # Stage 1: Extract concepts
    - group:
        plugins:
          - plugin: conceptnet_keywords
            id: concepts
          - plugin: llm_keywords
            id: themes
            
    # Stage 2: Analyze concepts
    - plugin: llm_question_answer
      config:
        prompt: |
          Concepts: {@concepts.keywords}
          Themes: {@themes.keywords}
          
          Provide thematic analysis...
```

### Conditional Logic

While not explicitly supported in YAML, prompts can include conditional instructions:

```yaml
prompt: |
  Movie: {Name}
  Rating: {OfficialRating}
  
  If rated R or above, analyze mature themes.
  If family-friendly, focus on educational value.
```

### Field Arrays and Complex Types

Access array elements and nested structures:

```yaml
# Access specific person types
Director:
  source_field: People[?Type=='Director'].Name
  
# Access first element
LeadActor:
  source_field: People[0].Name
  
# Access nested fields
StudioName:
  source_field: Studios[0].Name
```

## Usage Examples

### Example 1: Simple Keyword Extraction

```yaml
Name:
  source_field: Name
  enrichments:
    - plugin: conceptnet_keywords
      config:
        max_concepts: 5
```

### Example 2: Multi-Source Merging

```yaml
Overview:
  enrichments:
    - group:
        plugins:
          - plugin: conceptnet_keywords
            id: cn
            plugin_weight: 1.0
          - plugin: llm_keywords
            id: llm
            plugin_weight: 2.0
          - plugin: merge_keywords
            inputs: [cn, llm]
            config:
              strategy: weighted
```

### Example 3: Review Generation with Analysis

```yaml
Reviews:
  enrichments:
    # Generate multiple reviews
    - group:
        plugins:
          - plugin: llm_question_answer
            id: critic_review
            config:
              prompt: "Write a critic's review..."
          - plugin: llm_question_answer
            id: audience_review
            config:
              prompt: "Write an audience review..."
              
    # Analyze the reviews
    - plugin: llm_question_answer
      config:
        prompt: |
          Critic: {@critic_review}
          Audience: {@audience_review}
          
          What's the consensus?
```

### Example 4: Chained Cultural Analysis

```yaml
CulturalImpact:
  enrichments:
    # Extract cultural concepts
    - plugin: llm_keywords
      id: cultural_themes
      config:
        prompt: "Extract cultural themes from {Name}"
        
    # Analyze impact using themes
    - plugin: llm_question_answer
      config:
        prompt: |
          Themes: {@cultural_themes.keywords}
          Year: {ProductionYear}
          
          Analyze cultural impact...
```

## Best Practices

### 1. Group Organization

- Keep related plugins in the same group
- Only use merge within groups that need it
- Separate concerns into different groups

### 2. Weight Assignment

- Use higher weights for more reliable sources
- Normalize weights within groups for clarity
- Consider computational cost when weighting

### 3. Prompt Engineering

- Be specific in LLM prompts
- Include context from multiple fields
- Specify expected output format
- Use examples when helpful

### 4. Performance Optimization

- Limit concurrent LLM calls
- Use appropriate timeouts
- Cache results when possible
- Order fields by priority

### 5. Error Handling

- Set `allow_null: true` for optional synthetic fields
- Validate required fields
- Handle missing source data gracefully
- Use appropriate retry policies

### 6. Maintainability

- Use descriptive plugin IDs
- Document complex prompt logic
- Keep enrichment chains readable
- Test with various input data

## Migration from Old Format

### Old Format
```yaml
field_extraction_rules:
  - field_name: Overview
    extraction_method: key_concepts
    max_concepts: 10

plugins_to_run:
  - ConceptExpansionPlugin
  - TemporalAnalysisPlugin
```

### New Format
```yaml
fields:
  Overview:
    source_field: Overview
    enrichments:
      - group:
          plugins:
            - plugin: conceptnet_keywords
              config:
                max_concepts: 10
            - plugin: heideltime_temporal
```

## Troubleshooting

### Common Issues

1. **Plugin ID not found**: Ensure IDs are unique within the configuration
2. **Field reference errors**: Check field names match exactly
3. **Merge without inputs**: Verify the merge plugin has valid input IDs
4. **Timeout errors**: Increase timeout or reduce concurrent operations

### Debugging Tips

1. Enable `include_plugin_timings: true` to identify slow operations
2. Start with simple enrichments and build complexity
3. Test synthetic fields with minimal dependencies first
4. Validate YAML syntax before deployment

## Future Enhancements

Potential improvements to the configuration system:

1. **Conditional Groups**: Execute groups based on field values
2. **Dynamic Weights**: Adjust weights based on confidence scores
3. **Result Caching**: Define cache strategies in configuration
4. **Custom Merge Strategies**: Plugin-based merge algorithms
5. **Field Dependencies**: Explicit ordering based on dependencies
6. **A/B Testing**: Multiple enrichment strategies with comparison

## Conclusion

The new movie configuration format provides a powerful, flexible system for enriching movie metadata. By leveraging HTTP-only plugins, grouped enrichments, and sophisticated merging strategies, it enables complex data enrichment pipelines while maintaining clarity and maintainability. The weighting system ensures high-quality results by prioritizing reliable sources, while synthetic fields and chaining enable creative applications beyond simple metadata extraction.