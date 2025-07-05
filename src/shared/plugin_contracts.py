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
    """Types of cached field expansion results for data ingestion."""
    # Concept expansion
    CONCEPTNET = "conceptnet"
    LLM_CONCEPT = "llm_concept"
    
    # Synonym/similarity expansion  
    GENSIM_SIMILARITY = "gensim_similarity"
    WORDNET_SYNONYMS = "wordnet_synonyms"
    
    # Temporal extraction/parsing
    DUCKLING_TIME = "duckling_time"
    HEIDELTIME = "heideltime"
    SUTIME = "sutime"
    
    # NLP processing
    SPACY_NER = "spacy_ner"
    NLTK_TOKENIZE = "nltk_tokenize"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    
    # Field-specific expansions
    TAG_EXPANSION = "tag_expansion"
    GENRE_EXPANSION = "genre_expansion"
    PEOPLE_EXPANSION = "people_expansion"
    
    # Custom plugin types
    CUSTOM = "custom"
    
    # Backward compatibility aliases
    LLM = "llm_concept"  # Alias for LLM_CONCEPT
    GENSIM = "gensim_similarity"  # Alias for GENSIM_SIMILARITY
    NLTK = "nltk_tokenize"  # Alias for NLTK_TOKENIZE


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
    Consistent cache key generation for any field expansion during data ingestion.
    
    Format: "cache_type:field_name:input_value:media_context"
    Examples:
    - "conceptnet:tags:action:movie" (ConceptNet expansion of "action" tag)
    - "gensim_similarity:genres:thriller:movie" (Gensim similarity for "thriller" genre)
    - "duckling_time:release_date:next friday:movie" (Duckling parsing of "next friday")
    - "tag_expansion:tags:sci-fi:movie" (Tag similarity expansion)
    """
    cache_type: CacheType
    field_name: str  # Which field is being expanded (tags, genres, people, etc.)
    input_value: str  # The actual value being processed
    media_context: str = "movie"  # Media type context
    
    def generate_key(self) -> str:
        """
        Generate normalized cache key string.
        
        Uses shared to_ascii function for consistent normalization.
        Cache type is preserved exactly as enum value.
        """
        from src.shared.text_utils import clean_for_cache_key
        
        # Cache type preserved exactly (it's already lowercase and clean)
        cache_type_clean = self.cache_type.value
        normalized_field = clean_for_cache_key(self.field_name)
        normalized_value = clean_for_cache_key(self.input_value)
        normalized_context = clean_for_cache_key(self.media_context)
        
        return f"{cache_type_clean}:{normalized_field}:{normalized_value}:{normalized_context}"
    
    @classmethod
    def from_string(cls, key_string: str) -> 'CacheKey':
        """Parse cache key from string format."""
        parts = key_string.split(":")
        if len(parts) != 4:
            raise ValueError(f"Invalid cache key format: {key_string}. Expected format: cache_type:field_name:input_value:media_context")
        
        cache_type_str, field_name, input_value, media_context = parts
        cache_type = CacheType(cache_type_str)
        
        return cls(
            cache_type=cache_type,
            field_name=field_name,
            input_value=input_value,
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
        Convert to MongoDB document format for generic field expansion cache.
        
        Supports any type of field expansion during data ingestion:
        {
          "_id": ObjectId,
          "cache_key": "conceptnet:tags:action:movie",
          "field_name": "tags",
          "input_value": "action",
          "media_type": "movie", 
          "expansion_type": "conceptnet",
          "expansion_result": {...},  // Generic result structure
          "confidence_scores": {...},
          "source_metadata": {...},
          "created_at": ISODate,
          "expires_at": ISODate
        }
        """
        # Calculate expiration date
        expires_at = None
        if self.cache_ttl_seconds:
            expires_at = datetime.utcnow().timestamp() + self.cache_ttl_seconds
        
        return {
            "cache_key": self.cache_key.generate_key(),
            "field_name": self.cache_key.field_name,
            "input_value": self.cache_key.input_value,
            "media_type": self.cache_key.media_context,
            "expansion_type": self.cache_key.cache_type.value,
            "expansion_result": self.enhanced_data,  # Generic result structure
            "confidence_scores": self.confidence_score.per_item,
            "overall_confidence": self.confidence_score.overall,
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
            enhanced_data=doc.get("expansion_result", {}),  # Use generic expansion_result
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

def create_field_expansion_result(
    field_name: str,
    input_value: str,
    expansion_result: Dict[str, Any],
    confidence_scores: Dict[str, float],
    plugin_name: str,
    plugin_version: str,
    cache_type: CacheType,
    execution_time_ms: float,
    media_context: str = "movie",
    plugin_type: PluginType = PluginType.ENHANCEMENT,
    **kwargs
) -> PluginResult:
    """
    Generic function for creating any field expansion result.
    
    Examples:
    - ConceptNet expansion of "action" tag
    - Gensim similarity for "thriller" genre  
    - Duckling time parsing of "next friday"
    - Tag expansion for "sci-fi"
    """
    cache_key = CacheKey(
        cache_type=cache_type,
        field_name=field_name,
        input_value=input_value,
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
        plugin_type=plugin_type,
        execution_time_ms=execution_time_ms,
        **kwargs
    )
    
    return PluginResult(
        enhanced_data=expansion_result,
        confidence_score=confidence_score,
        plugin_metadata=plugin_metadata,
        cache_key=cache_key,
        cache_ttl_seconds=3600  # 1 hour default
    )


# Backward compatibility helper
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
    Backward compatibility wrapper for concept expansion.
    Use create_field_expansion_result() for new code.
    """
    expansion_result = {
        "expanded_concepts": expanded_concepts,
        "original_term": input_term
    }
    
    return create_field_expansion_result(
        field_name="concept",  # Generic field name for concepts
        input_value=input_term,
        expansion_result=expansion_result,
        confidence_scores=confidence_scores,
        plugin_name=plugin_name,
        plugin_version=plugin_version,
        cache_type=cache_type,
        execution_time_ms=execution_time_ms,
        media_context=media_context,
        plugin_type=PluginType.CONCEPT_EXPANSION,
        **kwargs
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