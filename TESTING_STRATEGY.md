# Testing Strategy for HTTP-Only Plugin Architecture

## Current Testing Status

### ✅ COMPLETED: Mock/Unit Testing
All basic functionality tested with mocked HTTP calls and hardcoded responses:
- Plugin imports and creation
- Interface compatibility  
- HTTP endpoint calls
- Merge functionality
- Error handling
- System compatibility

**Limitations**: Mock testing doesn't verify actual service integration or real-world performance.

### 🔄 IN PROGRESS: Real-World Testing  
Created `test_real_world_plugins.py` to test with realistic data and scenarios:
- Resource requirement validation
- Service connectivity testing
- Plugin initialization
- Performance characteristics
- Error scenario handling
- Configuration validation

## Testing Pyramid for HTTP-Only Plugins

### 1. Unit Tests (Mocked) ✅
**Purpose**: Verify plugin logic and interfaces
**Scope**: Individual plugin methods
**Data**: Mock HTTP responses, hardcoded test data
**Coverage**: 
- Plugin creation and metadata
- Method signatures and return types
- Error handling logic
- Configuration parsing

```python
# Example: Mock testing
mock_response = {'concepts': ['action', 'adventure']}
with patch.object(plugin, 'http_post', return_value=mock_response):
    result = await plugin.enrich_field('name', 'Mission Impossible', config)
```

### 2. Integration Tests (Semi-Real) 🔄
**Purpose**: Test plugin-service integration
**Scope**: Plugin → HTTP → Service workflow
**Data**: Real plugin configs, mocked service responses
**Coverage**:
- HTTP endpoint connectivity
- Request/response format validation
- Circuit breaker functionality
- Service health monitoring

```python
# Example: Semi-real testing
async def test_with_running_services():
    # Test actual HTTP calls to running services
    result = await plugin.enrich_field('name', real_movie_name, real_config)
```

### 3. End-to-End Tests (Full Real-World) ⏳
**Purpose**: Complete workflow validation
**Scope**: Full movie processing pipeline
**Data**: Real movie datasets, live services
**Coverage**:
- Complete ingestion workflows
- Performance under load
- Resource usage monitoring
- Production error scenarios

## Testing Approaches by Component

### Plugin-Specific Testing

#### Keyword Plugins
```python
# ConceptNet Plugin
test_data = {
    "simple": "Mission Impossible",
    "complex": "A thriller about espionage and covert operations",
    "multilingual": "Action movie 映画 película",
    "edge_cases": ["", "123", "🎬", "very"*1000]
}

# LLM Plugin  
test_configs = {
    "conservative": {"temperature": 0.1, "max_concepts": 5},
    "creative": {"temperature": 0.8, "max_concepts": 20},
    "custom_prompt": {"prompt_template": "Find {value} related themes"}
}
```

#### Temporal Plugins
```python
# SpaCy, HeidelTime, SUTime testing
temporal_test_cases = {
    "explicit_dates": "Released on March 15, 2023",
    "relative_dates": "Coming out next month",
    "duration": "A 2-hour epic adventure",
    "historical": "Set during World War II",
    "future": "In the year 2050"
}
```

#### Merge Plugin
```python
# Test all merge strategies
strategies = ["union", "intersection", "weighted", "ranked"]
test_scenarios = {
    "overlapping": [conceptnet_result, llm_result],  # Some common keywords
    "disjoint": [conceptnet_result, gensim_result],  # No overlap
    "identical": [same_result, same_result]          # Duplicate handling
}
```

## Real-World Testing Requirements

### 1. Service Dependencies
**Required Services**:
- Keyword Expansion Service (port 8001) - ConceptNet
- LLM Provider Service (port 8002) - Ollama/LLM
- NLP Provider Service (port 8003) - Gensim, SpaCy
- Temporal Service (port 8004) - HeidelTime, SUTime

**Setup**:
```bash
# Start all services
python -m src.services.keyword_expansion_service &
python -m src.services.llm_provider_service &
python -m src.services.nlp_provider_service &
# python -m src.services.temporal_service &  # TODO: Create this
```

### 2. Test Data Requirements

#### Movie Test Dataset
```python
realistic_movies = [
    {
        "name": "The Matrix",
        "overview": "A computer hacker learns about reality...",
        "genres": ["Action", "Sci-Fi"],
        "release_date": "1999-03-31",
        "runtime": 136
    },
    # ... more realistic entries
]
```

#### Performance Test Cases
```python
performance_scenarios = {
    "light_load": 10,      # 10 movies
    "medium_load": 100,    # 100 movies  
    "heavy_load": 1000,    # 1000 movies
    "stress_test": 10000   # 10k movies
}
```

### 3. Production-Like Testing

#### Resource Monitoring
```python
async def test_resource_usage():
    # Monitor actual CPU, memory, GPU usage
    # Validate resource requirements are accurate
    # Test under different load conditions
```

#### Failure Scenarios
```python
failure_tests = {
    "service_down": "Test when LLM service is offline",
    "network_timeout": "Test slow network responses", 
    "malformed_response": "Test invalid JSON responses",
    "rate_limiting": "Test service rate limits",
    "partial_failure": "Test when some providers fail"
}
```

## Testing Checklist

### ✅ Completed
- [x] Plugin imports and creation
- [x] Interface consistency
- [x] Mock HTTP endpoint testing
- [x] Merge strategy testing
- [x] Error handling with mocks
- [x] Resource requirement validation
- [x] LLM plugins require GPU and higher resources

### 🔄 In Progress  
- [ ] Service connectivity testing
- [ ] Real HTTP call validation
- [ ] Performance characteristics
- [ ] Configuration edge cases

### ⏳ TODO: Full Real-World Testing
- [ ] Live service integration
- [ ] Large dataset processing
- [ ] Performance benchmarking
- [ ] Production error scenarios
- [ ] Load testing
- [ ] Memory leak detection
- [ ] GPU utilization monitoring

## Next Steps

1. **Start Services**: Launch all microservices for integration testing
2. **Real Integration Test**: Test with actual HTTP calls to running services
3. **Performance Baseline**: Establish performance metrics for each plugin
4. **Load Testing**: Test with large movie datasets
5. **Production Simulation**: Test with realistic failure scenarios
6. **Documentation**: Document performance characteristics and operational requirements

## Conclusion

The current **mock-based testing** validates the plugin architecture and interfaces work correctly. However, **real-world testing** is essential to verify:

- Actual service integration
- Performance characteristics  
- Resource usage accuracy
- Production failure handling
- Scalability under load

The testing strategy progresses from **fast unit tests** (mocked) to **comprehensive integration tests** (real services) to **production validation** (full datasets).