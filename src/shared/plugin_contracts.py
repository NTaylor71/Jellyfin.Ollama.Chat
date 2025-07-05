"""
Plugin system data contracts for the procedural intelligence pipeline.
Standard formats for plugin outputs, caching, and enhancement results.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class PluginType(Enum):
    """Types of plugins in the system."""
    CONCEPT_EXPANSION = "concept_expansion"      # Stage 3: ConceptNet, LLM expansion
    CONTENT_ANALYSIS = "content_analysis"       # Stage 4: Movie content analysis
    QUERY_PROCESSING = "query_processing"       # Stage 6: Query intent understanding
    ENHANCEMENT = "enhancement"                 # General enhancement plugins


class CacheType(Enum):
    """Types of cached intelligence results."""
    CONCEPTNET = "conceptnet"
    LLM = "llm"
    GENSIM = "gensim"
    NLTK = "nltk"
    CUSTOM = "custom"


@dataclass
class PluginMetadata:
    """Metadata about the plugin that generated the result."""
    plugin_name: str
    plugin_version: str
    plugin_type: PluginType
    execution_time_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Resource usage information
    memory_used_mb: Optional[float] = None
    cpu_time_ms: Optional[float] = None
    
    # Plugin-specific metadata
    model_used: Optional[str] = None
    api_endpoint: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfidenceScore:
    """
    Confidence scoring for plugin results.
    
    Supports both overall confidence and per-item confidence for
    concept expansion and analysis results.
    """
    overall: float  # 0.0 to 1.0
    per_item: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate confidence scores."""
        if not 0.0 <= self.overall <= 1.0:
            raise ValueError(f"Overall confidence must be 0.0-1.0, got {self.overall}")
        
        for item, score in self.per_item.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Per-item confidence for {item} must be 0.0-1.0, got {score}")


@dataclass
class CacheKey:
    """
    Consistent cache key generation for Stage 2 ConceptCache.
    
    Format: "cache_type:input_term:media_context"
    Example: "conceptnet:action:movie"
    """
    cache_type: CacheType
    input_term: str
    media_context: str = "movie"  # Default to movie for now
    
    def generate_key(self) -> str:
        """
        Generate normalized cache key string.
        
        Uses shared to_ascii function for consistent normalization.
        """
        from src.shared.text_utils import clean_for_cache_key
        
        # Use optimized cache key cleaning
        normalized_type = clean_for_cache_key(self.cache_type.value)
        normalized_term = clean_for_cache_key(self.input_term)
        normalized_context = clean_for_cache_key(self.media_context)
        
        return f"{normalized_type}:{normalized_term}:{normalized_context}"
    
    @classmethod
    def from_string(cls, key_string: str) -> 'CacheKey':
        """Parse cache key from string format."""
        parts = key_string.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid cache key format: {key_string}")
        
        cache_type_str, input_term, media_context = parts
        cache_type = CacheType(cache_type_str)
        
        return cls(
            cache_type=cache_type,
            input_term=input_term,
            media_context=media_context
        )


@dataclass
class PluginResult:
    """
    Standard format for all plugin enhancement outputs.
    
    Compatible with MongoDB ConceptCache collection design and supports
    all plugin types across the procedural intelligence pipeline.
    """
    
    # Core result data
    enhanced_data: Dict[str, Any]
    confidence_score: ConfidenceScore
    plugin_metadata: PluginMetadata
    cache_key: CacheKey
    
    # Processing information
    input_data: Dict[str, Any] = field(default_factory=dict)
    processing_notes: List[str] = field(default_factory=list)
    
    # Error handling
    success: bool = True
    error_message: Optional[str] = None
    partial_result: bool = False
    
    # Cache management
    cache_ttl_seconds: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_cache_document(self) -> Dict[str, Any]:
        """
        Convert to MongoDB document format for Stage 2 ConceptCache.
        
        Matches the cache structure defined in Stage 2.1:
        {
          "_id": ObjectId,
          "cache_key": "concept:action:movie",
          "input_term": "action",
          "media_type": "movie", 
          "expansion_type": "conceptnet",
          "expanded_terms": ["fight", "combat", "battle", "intense"],
          "confidence_scores": {"fight": 0.9, "combat": 0.85},
          "source_metadata": {"api": "conceptnet", "endpoint": "/related"},
          "created_at": ISODate,
          "expires_at": ISODate
        }
        """
        # Calculate expiration date
        expires_at = None
        if self.cache_ttl_seconds:
            expires_at = datetime.utcnow().timestamp() + self.cache_ttl_seconds
        
        # Extract expanded terms for concept expansion results
        expanded_terms = []
        if "expanded_concepts" in self.enhanced_data:
            expanded_terms = list(self.enhanced_data["expanded_concepts"])
        elif "concepts" in self.enhanced_data:
            expanded_terms = list(self.enhanced_data["concepts"])
        
        return {
            "cache_key": self.cache_key.generate_key(),
            "input_term": self.cache_key.input_term,
            "media_type": self.cache_key.media_context,
            "expansion_type": self.cache_key.cache_type.value,
            "expanded_terms": expanded_terms,
            "confidence_scores": self.confidence_score.per_item,
            "overall_confidence": self.confidence_score.overall,
            "enhanced_data": self.enhanced_data,
            "source_metadata": {
                "plugin_name": self.plugin_metadata.plugin_name,
                "plugin_version": self.plugin_metadata.plugin_version,
                "plugin_type": self.plugin_metadata.plugin_type.value,
                "execution_time_ms": self.plugin_metadata.execution_time_ms,
                "model_used": self.plugin_metadata.model_used,
                "api_endpoint": self.plugin_metadata.api_endpoint,
                "parameters": self.plugin_metadata.parameters
            },
            "success": self.success,
            "error_message": self.error_message,
            "partial_result": self.partial_result,
            "processing_notes": self.processing_notes,
            "created_at": self.created_at,
            "expires_at": expires_at
        }
    
    @classmethod
    def from_cache_document(cls, doc: Dict[str, Any]) -> 'PluginResult':
        """Reconstruct PluginResult from MongoDB cache document."""
        cache_key = CacheKey.from_string(doc["cache_key"])
        
        confidence_score = ConfidenceScore(
            overall=doc.get("overall_confidence", 0.0),
            per_item=doc.get("confidence_scores", {})
        )
        
        source_metadata = doc.get("source_metadata", {})
        plugin_metadata = PluginMetadata(
            plugin_name=source_metadata.get("plugin_name", "unknown"),
            plugin_version=source_metadata.get("plugin_version", "unknown"),
            plugin_type=PluginType(source_metadata.get("plugin_type", "enhancement")),
            execution_time_ms=source_metadata.get("execution_time_ms", 0.0),
            timestamp=doc.get("created_at", datetime.utcnow()),
            model_used=source_metadata.get("model_used"),
            api_endpoint=source_metadata.get("api_endpoint"),
            parameters=source_metadata.get("parameters", {})
        )
        
        return cls(
            enhanced_data=doc.get("enhanced_data", {}),
            confidence_score=confidence_score,
            plugin_metadata=plugin_metadata,
            cache_key=cache_key,
            success=doc.get("success", True),
            error_message=doc.get("error_message"),
            partial_result=doc.get("partial_result", False),
            processing_notes=doc.get("processing_notes", []),
            created_at=doc.get("created_at", datetime.utcnow())
        )


# Helper functions for plugin developers

def create_concept_expansion_result(
    input_term: str,
    expanded_concepts: List[str],
    confidence_scores: Dict[str, float],
    plugin_name: str,
    plugin_version: str,
    cache_type: CacheType,
    execution_time_ms: float,
    media_context: str = "movie",
    **kwargs
) -> PluginResult:
    """
    Convenience function for creating concept expansion results (Stage 3).
    
    Used by ConceptNet, LLM, and other concept expansion plugins.
    """
    cache_key = CacheKey(
        cache_type=cache_type,
        input_term=input_term,
        media_context=media_context
    )
    
    overall_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.0
    
    confidence_score = ConfidenceScore(
        overall=overall_confidence,
        per_item=confidence_scores
    )
    
    plugin_metadata = PluginMetadata(
        plugin_name=plugin_name,
        plugin_version=plugin_version,
        plugin_type=PluginType.CONCEPT_EXPANSION,
        execution_time_ms=execution_time_ms,
        **kwargs
    )
    
    enhanced_data = {
        "expanded_concepts": expanded_concepts,
        "original_term": input_term
    }
    
    return PluginResult(
        enhanced_data=enhanced_data,
        confidence_score=confidence_score,
        plugin_metadata=plugin_metadata,
        cache_key=cache_key,
        cache_ttl_seconds=3600  # 1 hour default for concept expansion
    )


def create_content_analysis_result(
    media_id: str,
    analysis_results: Dict[str, Any],
    confidence_score: float,
    plugin_name: str,
    plugin_version: str,
    execution_time_ms: float,
    **kwargs
) -> PluginResult:
    """
    Convenience function for creating content analysis results (Stage 4).
    
    Used by movie content analyzers and pattern learning plugins.
    """
    cache_key = CacheKey(
        cache_type=CacheType.CUSTOM,
        input_term=f"content_analysis_{media_id}",
        media_context="movie"
    )
    
    confidence = ConfidenceScore(overall=confidence_score)
    
    plugin_metadata = PluginMetadata(
        plugin_name=plugin_name,
        plugin_version=plugin_version,
        plugin_type=PluginType.CONTENT_ANALYSIS,
        execution_time_ms=execution_time_ms,
        **kwargs
    )
    
    return PluginResult(
        enhanced_data=analysis_results,
        confidence_score=confidence,
        plugin_metadata=plugin_metadata,
        cache_key=cache_key,
        cache_ttl_seconds=86400  # 24 hours for content analysis
    )