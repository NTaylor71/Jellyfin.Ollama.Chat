# CPU/GPU Queue Management Enhancement

## Problem Statement

The current plugin system lacks intelligent resource management for CPU/GPU workloads. While we have hardware detection (`src/shared/hardware_config.py`) and basic resource requirements in plugins, we need a sophisticated queue management system that optimally distributes tasks based on available resources.

## Current State

### What We Have ✅
- Hardware detection system detecting 24 CPU cores, RTX 4090 GPU
- Plugin resource requirements (`PluginResourceRequirements`)
- Basic resource checking in `BasePlugin._check_resource_availability()`
- Redis queue system for task distribution
- Containerized Ollama with GPU support

### What's Missing ❌
- Intelligent CPU vs GPU task routing
- Dynamic resource allocation based on current system load
- Queue prioritization based on resource requirements
- GPU memory management for concurrent LLM operations
- CPU-intensive task batching for NLP operations

## Proposed Solution

### 1. Resource-Aware Queue Router
Create a smart queue router that:
- Routes GPU-intensive tasks (LLM operations) to GPU-optimized workers
- Routes CPU-intensive tasks (ConceptNet, Gensim, Heideltime) to CPU-optimized workers
- Monitors real-time resource utilization
- Implements backpressure when resources are saturated

### 2. Queue Types by Resource Profile
```
gpu_queue:          # Ollama LLM operations, limited concurrency
cpu_intensive:      # Gensim, SpaCy processing, high parallelism  
cpu_light:          # ConceptNet API calls, network I/O
mixed_workload:     # Tasks that can adapt based on availability
```

### 3. Dynamic Resource Allocation
- Monitor GPU memory usage for LLM operations
- Track CPU core utilization for NLP tasks
- Implement intelligent batching for similar operations
- Queue throttling based on resource pressure

## Implementation Plan

### Phase 1: Queue Classification System
- [ ] Extend `PluginResourceRequirements` with resource profiles
- [ ] Create `ResourceProfile` enum (GPU_INTENSIVE, CPU_INTENSIVE, IO_BOUND, MIXED)
- [ ] Update existing plugins with appropriate resource profiles
- [ ] Implement queue routing logic in `src/redis_worker/queue_manager.py`

### Phase 2: Resource Monitoring
- [ ] Real-time GPU memory monitoring via nvidia-ml-py
- [ ] CPU utilization tracking per worker process
- [ ] Memory pressure detection
- [ ] Queue depth monitoring and alerting

### Phase 3: Intelligent Scheduling
- [ ] Priority-based task scheduling
- [ ] Resource-aware task batching
- [ ] Dynamic worker scaling based on queue depth - custom Vs Kubernates (to discuss)
- [ ] Fallback mechanisms when preferred resources unavailable

### Phase 4: Performance Optimization
- [ ] GPU memory pooling for LLM operations
- [ ] CPU affinity for intensive NLP tasks
- [ ] Concurrent processing optimization
- [ ] Resource utilization metrics and dashboards

## Technical Details

### Resource Profiles for Current Plugins
```python
# GPU-intensive (limited concurrency)
RemoteConceptExpansionPlugin (LLM calls): GPU_INTENSIVE
TemporalAnalysisPlugin (LLM concepts): GPU_INTENSIVE

# CPU-intensive (high parallelism)  
ConceptExpansionPlugin (NLP): CPU_INTENSIVE
SpacyTemporalProvider: CPU_INTENSIVE
GensimProvider: CPU_INTENSIVE
HeideltimeProvider: CPU_INTENSIVE

# I/O bound (network limited)
ConceptNetProvider: IO_BOUND
ServiceHealthMonitorPlugin: IO_BOUND
```

### Queue Management Strategy
1. **GPU Queue**: Max 2-3 concurrent LLM operations (memory limited)
2. **CPU Queue**: Max N concurrent operations (N = CPU cores)
3. **I/O Queue**: High concurrency for network operations
4. **Overflow Handling**: CPU fallback for GPU tasks when GPU saturated

## Success Metrics

### Performance Targets
- **GPU Utilization**: >80% for LLM-heavy workloads
- **CPU Utilization**: >70% for NLP-heavy workloads  
- **Queue Latency**: <500ms for task routing decisions
- **Throughput**: 2x improvement in mixed workload scenarios

### Monitoring Requirements
- Real-time resource utilization dashboards
- Queue depth and processing time metrics
- Task completion rates by resource type
- Resource contention alerts

## Related Files

### Core Components
- `src/shared/hardware_config.py` - Hardware detection
- `src/plugins/base.py` - Resource requirements
- `src/redis_worker/queue_manager.py` - Queue management
- `src/redis_worker/main.py` - Worker coordination

### Plugin Integration
- `src/plugins/concept_expansion_plugin.py` - CPU-intensive NLP
- `src/plugins/remote_concept_expansion_plugin.py` - GPU LLM calls
- `src/plugins/temporal_analysis_plugin.py` - Mixed CPU/GPU workload

## Dependencies

### New Packages Required
- `nvidia-ml-py` for GPU monitoring
- `psutil` enhancements for CPU tracking  
- Enhanced Redis queue patterns

### Docker Configuration
- GPU device access for monitoring
- CPU affinity configuration
- Memory limits per service type

## Implementation Priority

**High Priority**: This directly impacts system scalability and resource efficiency. Critical for production deployment where optimal resource utilization determines cost-effectiveness and user experience.

## Notes

This builds on our existing service-oriented architecture (Stage 4.3.4) and sets the foundation for Stage 4.3.6 microservices testing. The queue management system should integrate seamlessly with our current plugin architecture while providing the intelligence needed for production-scale deployments.