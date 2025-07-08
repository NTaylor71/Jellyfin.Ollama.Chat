"""
Field-Class architecture for media entities.
Eliminates hard-coded field assumptions through intelligent field classification.
"""

from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import re

if TYPE_CHECKING:
    from src.shared.field_analysis_plugins import FieldAnalysisPlugin


class FieldType(Enum):
    """Types of fields in media entities for intelligent processing."""
    TEXT_CONTENT = "text_content"       # Descriptive text suitable for NLP analysis
    METADATA = "metadata"               # Factual information (year, rating, etc.)
    PEOPLE = "people"                   # Person/organization information
    IDENTIFIERS = "identifiers"         # IDs, URLs, provider references
    TECHNICAL = "technical"             # File/streaming technical details
    STRUCTURAL = "structural"           # Lists, nested objects, complex data
    UNKNOWN = "unknown"                 # Unclassified fields


class AnalysisWeight(Enum):
    """Importance levels for NLP analysis and concept expansion."""
    CRITICAL = 1.0      # Primary content (overview, description)
    HIGH = 0.8          # Important content (taglines, themes)
    MEDIUM = 0.6        # Useful content (tags, genres)
    LOW = 0.3           # Supplementary content (technical details)
    IGNORE = 0.0        # Not useful for intelligence (internal IDs)


@dataclass
class MediaField:
    """
    Individual field in a media entity with intelligent classification.
    
    Replaces hard-coded field assumptions with dynamic field analysis.
    Each field knows its type, importance, and how it should be processed.
    """
    name: str
    value: Any
    field_type: FieldType
    analysis_weight: AnalysisWeight
    
    # Processing flags
    cache_key_eligible: bool = True     # Can be used for cache key generation
    nlp_ready: bool = False             # Text is ready for NLP processing
    concept_expandable: bool = False    # Suitable for concept expansion
    
    # Metadata
    original_type: type = field(default=str)
    processing_notes: List[str] = field(default_factory=list)
    
    def get_text_value(self) -> Optional[str]:
        """
        Extract text content suitable for NLP analysis.
        
        Handles various data types and converts to analyzable text.
        """
        if self.field_type != FieldType.TEXT_CONTENT:
            return None
        
        if isinstance(self.value, str):
            return self.value.strip()
        elif isinstance(self.value, list):
            # Handle lists of strings (tags, genres, etc.)
            text_items = [str(item) for item in self.value if item]
            return " ".join(text_items) if text_items else None
        elif isinstance(self.value, dict):
            # Extract text from dictionary values
            text_values = []
            for v in self.value.values():
                if isinstance(v, str) and v.strip():
                    text_values.append(v.strip())
            return " ".join(text_values) if text_values else None
        else:
            return str(self.value) if self.value else None
    
    def get_cache_key_component(self) -> Optional[str]:
        """
        Generate cache key component from field value.
        
        Used for Stage 2 ConceptCache key generation.
        """
        if not self.cache_key_eligible:
            return None
        
        text = self.get_text_value()
        if not text:
            return None
        
        # Use shared text normalization
        from src.shared.text_utils import clean_for_cache_key
        return clean_for_cache_key(text)


class FieldAnalyzer:
    """
    Plugin-based field classification system.
    
    Uses a list of analysis plugins to intelligently classify fields without
    hard-coded patterns. Each plugin specializes in different field types.
    """
    
    def __init__(self, plugins: Optional[List['FieldAnalysisPlugin']] = None):
        """
        Initialize with field analysis plugins.
        
        Args:
            plugins: List of field analysis plugins. If None, uses defaults.
        """
        if plugins is None:
            from src.shared.field_analysis_plugins import get_default_field_analysis_plugins
            plugins = get_default_field_analysis_plugins()
        
        # Sort plugins by priority (highest first)
        self.plugins = sorted(plugins, key=lambda p: p.priority, reverse=True)
    
    def analyze_field(self, name: str, value: Any, media_type: str = "Unknown") -> MediaField:
        """
        Analyze field using plugins to determine classification.
        
        Tries each plugin in priority order until one can classify the field.
        
        Args:
            name: Field name
            value: Field value
            media_type: Type of media (Movie, Book, etc.)
            
        Returns:
            MediaField with intelligent classification
        """
        # Try each plugin until one can handle the field
        for plugin in self.plugins:
            if plugin.can_analyze(name, value, media_type):
                try:
                    field = plugin.analyze_field(name, value, media_type)
                    # Add plugin info to processing notes
                    field.processing_notes.append(f"Classified by {plugin.plugin_name}")
                    return field
                except Exception as e:
                    # Plugin failed, try next one
                    continue
        
        # Fallback if no plugin worked (shouldn't happen with FallbackPlugin)
        return MediaField(
            name=name,
            value=value,
            field_type=FieldType.UNKNOWN,
            analysis_weight=AnalysisWeight.LOW,
            cache_key_eligible=True,
            original_type=type(value),
            processing_notes=["No plugin could classify this field"]
        )


@dataclass
class MediaEntity:
    """
    Generic media entity using Field-Class architecture.
    
    Eliminates hard-coded field assumptions. Works for movies, TV, books,
    music, or any future media type through intelligent field classification.
    """
    
    # Only truly universal fields
    entity_id: str
    entity_name: str
    media_type: str  # "Movie", "Series", "Book", "Album", etc.
    
    # All fields stored as MediaField objects
    fields: Dict[str, MediaField] = field(default_factory=dict)
    
    # Processing metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    processing_notes: List[str] = field(default_factory=list)
    
    @classmethod
    def from_raw_data(cls, raw_data: Dict[str, Any], media_type: str = "Unknown") -> 'MediaEntity':
        """
        Create MediaEntity from raw data using intelligent field analysis.
        
        This replaces the hard-coded from_jellyfin_data approach with
        a generic analyzer that works for any media type.
        """
        # Extract universal fields
        entity_id = raw_data.get('Id') or raw_data.get('id') or raw_data.get('ID') or 'unknown'
        entity_name = raw_data.get('Name') or raw_data.get('name') or raw_data.get('title') or 'Unknown'
        detected_type = raw_data.get('Type') or raw_data.get('type') or media_type
        
        # Initialize field analyzer with default plugins
        field_analyzer = FieldAnalyzer()
        
        # Analyze all other fields
        fields = {}
        processing_notes = []
        
        for field_name, field_value in raw_data.items():
            # Skip universal fields already processed
            if field_name.lower() in ['id', 'name', 'type']:
                continue
            
            # Skip empty/null values
            if field_value is None or field_value == '' or field_value == []:
                continue
            
            # Analyze field using plugins
            media_field = field_analyzer.analyze_field(field_name, field_value, detected_type)
            
            # Set processing flags for text fields
            if media_field.field_type == FieldType.TEXT_CONTENT:
                media_field.nlp_ready = bool(media_field.get_text_value())
            
            fields[field_name] = media_field
            
            if media_field.field_type == FieldType.UNKNOWN:
                processing_notes.append(f"Unknown field type: {field_name}")
        
        # Combine processing notes from analyzer and fields
        all_processing_notes = processing_notes.copy()
        for field in fields.values():
            all_processing_notes.extend(field.processing_notes)
        
        return cls(
            entity_id=entity_id,
            entity_name=entity_name,
            media_type=detected_type,
            fields=fields,
            processing_notes=all_processing_notes
        )
    
    def get_text_fields(self) -> Dict[str, str]:
        """
        Get all text fields suitable for NLP analysis.
        
        Replaces the hard-coded get_text_for_analysis() with dynamic discovery.
        """
        text_fields = {}
        
        for field_name, field in self.fields.items():
            if field.field_type == FieldType.TEXT_CONTENT and field.nlp_ready:
                text_content = field.get_text_value()
                if text_content:
                    text_fields[field_name.lower()] = text_content
        
        return text_fields
    
    def get_weighted_text_fields(self) -> Dict[str, tuple[str, float]]:
        """
        Get text fields with their analysis weights.
        
        Used by plugins to prioritize which fields to analyze first.
        """
        weighted_fields = {}
        
        for field_name, field in self.fields.items():
            if field.field_type == FieldType.TEXT_CONTENT and field.nlp_ready:
                text_content = field.get_text_value()
                if text_content:
                    weighted_fields[field_name.lower()] = (text_content, field.analysis_weight.value)
        
        return weighted_fields
    
    def get_concept_expandable_fields(self) -> Dict[str, str]:
        """
        Get fields suitable for concept expansion (Stage 3).
        
        These fields contain terms that can be expanded via ConceptNet, LLM, etc.
        """
        expandable_fields = {}
        
        for field_name, field in self.fields.items():
            if field.concept_expandable and field.nlp_ready:
                text_content = field.get_text_value()
                if text_content:
                    expandable_fields[field_name.lower()] = text_content
        
        return expandable_fields
    
    def get_cache_context(self) -> str:
        """Generate context string for cache key generation."""
        return self.media_type.lower()
    
    def get_field_summary(self) -> Dict[str, Any]:
        """
        Get summary of field analysis for debugging/monitoring.
        
        Useful for understanding how the field analyzer classified the data.
        """
        summary = {
            'total_fields': len(self.fields),
            'field_types': {},
            'text_fields': 0,
            'concept_expandable': 0,
            'nlp_ready': 0,
            'unknown_fields': []
        }
        
        for field_name, field in self.fields.items():
            # Count by field type
            field_type_name = field.field_type.value
            summary['field_types'][field_type_name] = summary['field_types'].get(field_type_name, 0) + 1
            
            # Count special categories
            if field.field_type == FieldType.TEXT_CONTENT:
                summary['text_fields'] += 1
            if field.concept_expandable:
                summary['concept_expandable'] += 1
            if field.nlp_ready:
                summary['nlp_ready'] += 1
            if field.field_type == FieldType.UNKNOWN:
                summary['unknown_fields'].append(field_name)
        
        return summary