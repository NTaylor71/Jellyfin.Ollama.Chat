"""
Custom Prometheus metrics for the Universal Media Ingestion Framework.
Provides application-specific metrics beyond the standard FastAPI metrics.
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Dict, Any, Optional
import time
from functools import wraps
import logging

logger = logging.getLogger(__name__)






media_ingestion_total = Counter(
    'media_ingestion_total',
    'Total number of media items ingested',
    ['media_type', 'source', 'status']
)

media_ingestion_duration = Histogram(
    'media_ingestion_duration_seconds',
    'Time spent ingesting media items',
    ['media_type', 'source'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
)

media_enrichment_total = Counter(
    'media_enrichment_total',
    'Total number of media items enriched',
    ['media_type', 'status']
)

media_enrichment_duration = Histogram(
    'media_enrichment_duration_seconds',
    'Time spent enriching media items',
    ['media_type'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)


search_queries_total = Counter(
    'search_queries_total',
    'Total number of search queries',
    ['media_type', 'strategy', 'status']
)

search_duration = Histogram(
    'search_duration_seconds',
    'Time spent processing search queries',
    ['media_type', 'strategy'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

search_results_count = Histogram(
    'search_results_count',
    'Number of results returned by search queries',
    ['media_type', 'strategy'],
    buckets=[0, 1, 5, 10, 20, 50, 100, 200]
)


media_retrieval_total = Counter(
    'media_retrieval_total',
    'Total number of media item retrievals',
    ['media_type', 'status']
)

media_retrieval_duration = Histogram(
    'media_retrieval_duration_seconds',
    'Time spent retrieving media items',
    ['media_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)






config_loading_total = Counter(
    'config_loading_total',
    'Total number of configuration loads',
    ['media_type', 'status']
)

config_loading_duration = Histogram(
    'config_loading_duration_seconds',
    'Time spent loading configurations',
    ['media_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)


data_source_items_total = Counter(
    'data_source_items_total',
    'Total number of items loaded from data sources',
    ['media_type', 'source', 'status']
)

data_source_duration = Histogram(
    'data_source_duration_seconds',
    'Time spent loading from data sources',
    ['media_type', 'source'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)


validation_errors_total = Counter(
    'validation_errors_total',
    'Total number of validation errors',
    ['media_type', 'error_type']
)





plugin_execution_total = Counter(
    'plugin_execution_total',
    'Total number of plugin executions',
    ['plugin_name', 'media_type', 'status']
)

plugin_execution_duration = Histogram(
    'plugin_execution_duration_seconds',
    'Time spent executing plugins',
    ['plugin_name', 'media_type'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)





mongodb_operations_total = Counter(
    'mongodb_operations_total',
    'Total number of MongoDB operations',
    ['operation', 'collection', 'status']
)

mongodb_operation_duration = Histogram(
    'mongodb_operation_duration_seconds',
    'Time spent on MongoDB operations',
    ['operation', 'collection'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)





active_managers = Gauge(
    'active_managers_total',
    'Number of active ingestion managers',
    ['media_type']
)

media_items_stored = Gauge(
    'media_items_stored_total',
    'Total number of media items in storage',
    ['media_type']
)





def track_ingestion_metrics(func):
    """Decorator to track ingestion metrics."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        media_type = getattr(self, 'media_type', 'unknown')
        start_time = time.time()
        
        try:
            result = await func(self, *args, **kwargs)
            

            if hasattr(result, '__len__'):
                count = len(result)
                media_ingestion_total.labels(
                    media_type=media_type,
                    source='unknown',
                    status='success'
                ).inc(count)
            

            duration = time.time() - start_time
            media_ingestion_duration.labels(
                media_type=media_type,
                source='unknown'
            ).observe(duration)
            
            return result
            
        except Exception as e:

            media_ingestion_total.labels(
                media_type=media_type,
                source='unknown',
                status='error'
            ).inc()
            

            duration = time.time() - start_time
            media_ingestion_duration.labels(
                media_type=media_type,
                source='unknown'
            ).observe(duration)
            
            raise
    
    return wrapper


def track_search_metrics(func):
    """Decorator to track search metrics."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request = args[0] if args else kwargs.get('request')
        media_type = getattr(request, 'media_type', 'unknown')
        strategy = getattr(request, 'search_strategy', 'unknown')
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            

            search_queries_total.labels(
                media_type=media_type,
                strategy=strategy,
                status='success'
            ).inc()
            

            duration = time.time() - start_time
            search_duration.labels(
                media_type=media_type,
                strategy=strategy
            ).observe(duration)
            

            if hasattr(result, 'results'):
                search_results_count.labels(
                    media_type=media_type,
                    strategy=strategy
                ).observe(len(result.results))
            
            return result
            
        except Exception as e:

            search_queries_total.labels(
                media_type=media_type,
                strategy=strategy,
                status='error'
            ).inc()
            

            duration = time.time() - start_time
            search_duration.labels(
                media_type=media_type,
                strategy=strategy
            ).observe(duration)
            
            raise
    
    return wrapper


def track_enrichment_metrics(func):
    """Decorator to track enrichment metrics."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        media_type = getattr(self, 'media_type', 'unknown')
        start_time = time.time()
        
        try:
            result = await func(self, *args, **kwargs)
            

            media_enrichment_total.labels(
                media_type=media_type,
                status='success'
            ).inc()
            

            duration = time.time() - start_time
            media_enrichment_duration.labels(
                media_type=media_type
            ).observe(duration)
            
            return result
            
        except Exception as e:

            media_enrichment_total.labels(
                media_type=media_type,
                status='error'
            ).inc()
            

            duration = time.time() - start_time
            media_enrichment_duration.labels(
                media_type=media_type
            ).observe(duration)
            
            raise
    
    return wrapper


def track_mongodb_metrics(operation: str, collection: str):
    """Decorator to track MongoDB operation metrics."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                

                mongodb_operations_total.labels(
                    operation=operation,
                    collection=collection,
                    status='success'
                ).inc()
                

                duration = time.time() - start_time
                mongodb_operation_duration.labels(
                    operation=operation,
                    collection=collection
                ).observe(duration)
                
                return result
                
            except Exception as e:

                mongodb_operations_total.labels(
                    operation=operation,
                    collection=collection,
                    status='error'
                ).inc()
                

                duration = time.time() - start_time
                mongodb_operation_duration.labels(
                    operation=operation,
                    collection=collection
                ).observe(duration)
                
                raise
        
        return wrapper
    return decorator


def update_manager_count(media_type: str, delta: int = 1):
    """Update the count of active managers."""
    active_managers.labels(media_type=media_type).inc(delta)


def update_stored_items_count(media_type: str, count: int):
    """Update the count of stored media items."""
    media_items_stored.labels(media_type=media_type).set(count)