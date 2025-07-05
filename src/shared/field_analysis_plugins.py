"""
Field analysis plugins for intelligent field classification.
Replaces hard-coded patterns with plugin-based field understanding.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re

from src.shared.media_fields import MediaField, FieldType, AnalysisWeight
from src.shared.text_utils import to_ascii, safe_string_conversion


class FieldAnalysisPlugin(ABC):
    """
    Abstract base class for field analysis plugins.
    
    Each plugin specializes in analyzing specific types of fields or media types.
    No hard-coded assumptions - plugins learn from actual data patterns.
    """
    
    @abstractmethod
    def can_analyze(self, name: str, value: Any, media_type: str) -> bool:
        """
        Determine if this plugin can analyze the given field.
        
        Args:
            name: Field name
            value: Field value
            media_type: Type of media (Movie, Book, etc.)
            
        Returns:
            True if this plugin can handle the field
        """
        pass
    
    @abstractmethod
    def analyze_field(self, name: str, value: Any, media_type: str) -> MediaField:
        """
        Analyze field and return classified MediaField.
        
        Args:
            name: Field name
            value: Field value  
            media_type: Type of media
            
        Returns:
            MediaField with intelligent classification
        """
        pass
    
    @property
    @abstractmethod
    def plugin_name(self) -> str:
        """Plugin name for debugging/logging."""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """Plugin priority (higher = checked first)."""
        pass


class GenericTextAnalysisPlugin(FieldAnalysisPlugin):
    """
    Analyzes text content using linguistic features rather than field names.
    
    Uses actual content analysis to determine if field contains descriptive text
    suitable for NLP processing. No hard-coded field name assumptions.
    """
    
    @property
    def plugin_name(self) -> str:
        return "GenericTextAnalysis"
    
    @property
    def priority(self) -> int:
        return 100  # High priority for text detection
    
    def can_analyze(self, name: str, value: Any, media_type: str) -> bool:
        """Check if value contains substantial text content."""
        if not value:
            return False
        
        # Convert to string safely
        text = safe_string_conversion(value)
        if not text:
            return False
        
        # Check for substantial text content (not just IDs or short metadata)
        if len(text) < 10:  # Too short to be descriptive
            return False
        
        # Check if it's mostly alphabetic (not technical data)
        ascii_text = to_ascii(text)
        alpha_chars = sum(1 for c in ascii_text if c.isalpha())
        total_chars = len(ascii_text.replace(' ', ''))
        
        if total_chars == 0:
            return False
        
        alpha_ratio = alpha_chars / total_chars
        return alpha_ratio > 0.5  # More than 50% alphabetic content
    
    def analyze_field(self, name: str, value: Any, media_type: str) -> MediaField:
        """Analyze text content to determine type and importance."""
        text = safe_string_conversion(value)
        
        # Determine analysis weight based on content characteristics
        weight = self._determine_text_weight(name.lower(), text)
        
        # All substantial text is concept-expandable
        return MediaField(
            name=name,
            value=value,
            field_type=FieldType.TEXT_CONTENT,
            analysis_weight=weight,
            cache_key_eligible=True,
            nlp_ready=True,
            concept_expandable=True,
            original_type=type(value),
            processing_notes=[f"Analyzed by {self.plugin_name}"]
        )
    
    def _determine_text_weight(self, field_name: str, text: str) -> AnalysisWeight:
        """
        Determine importance weight based on content analysis.
        
        Uses content characteristics rather than hard-coded field names.
        """
        # Longer text generally more important for analysis
        text_length = len(text)
        
        if text_length > 200:  # Long descriptive content
            return AnalysisWeight.CRITICAL
        elif text_length > 100:  # Medium descriptive content  
            return AnalysisWeight.HIGH
        elif text_length > 30:   # Short but useful content
            return AnalysisWeight.MEDIUM
        else:
            return AnalysisWeight.LOW


class StructuredDataAnalysisPlugin(FieldAnalysisPlugin):
    """
    Analyzes structured data (lists, objects) to classify as PEOPLE, METADATA, etc.
    
    Uses data structure patterns rather than field names to determine type.
    """
    
    @property
    def plugin_name(self) -> str:
        return "StructuredDataAnalysis"
    
    @property
    def priority(self) -> int:
        return 90  # High priority for structured data
    
    def can_analyze(self, name: str, value: Any, media_type: str) -> bool:
        """Check if value is structured data (list, dict)."""
        return isinstance(value, (list, dict))
    
    def analyze_field(self, name: str, value: Any, media_type: str) -> MediaField:
        """Analyze structured data to determine type."""
        if isinstance(value, list) and value:
            return self._analyze_list_field(name, value)
        elif isinstance(value, dict):
            return self._analyze_dict_field(name, value)
        else:
            # Empty or unknown structure
            return MediaField(
                name=name,
                value=value,
                field_type=FieldType.STRUCTURAL,
                analysis_weight=AnalysisWeight.LOW,
                cache_key_eligible=False,
                original_type=type(value),
                processing_notes=[f"Empty structure analyzed by {self.plugin_name}"]
            )
    
    def _analyze_list_field(self, name: str, value: List[Any]) -> MediaField:
        """Analyze list to determine if it's people, text items, or metadata."""
        if not value:
            return self._create_unknown_field(name, value)
        
        first_item = value[0]
        
        # Check if list contains person-like objects
        if isinstance(first_item, dict) and self._looks_like_person_object(first_item):
            return MediaField(
                name=name,
                value=value,
                field_type=FieldType.PEOPLE,
                analysis_weight=AnalysisWeight.MEDIUM,
                cache_key_eligible=True,
                original_type=type(value),
                processing_notes=[f"Person objects detected by {self.plugin_name}"]
            )
        
        # Check if list contains text items (genres, tags, etc.)
        if isinstance(first_item, str):
            # Analyze if these are concept-expandable terms
            combined_text = " ".join(str(item) for item in value if item)
            if len(combined_text) > 5:  # Has meaningful content
                return MediaField(
                    name=name,
                    value=value,
                    field_type=FieldType.TEXT_CONTENT,
                    analysis_weight=AnalysisWeight.MEDIUM,
                    cache_key_eligible=True,
                    nlp_ready=True,
                    concept_expandable=True,
                    original_type=type(value),
                    processing_notes=[f"Text list analyzed by {self.plugin_name}"]
                )
        
        # Default to metadata for other lists
        return MediaField(
            name=name,
            value=value,
            field_type=FieldType.METADATA,
            analysis_weight=AnalysisWeight.LOW,
            cache_key_eligible=True,
            original_type=type(value),
            processing_notes=[f"Metadata list analyzed by {self.plugin_name}"]
        )
    
    def _analyze_dict_field(self, name: str, value: Dict[str, Any]) -> MediaField:
        """Analyze dictionary structure."""
        # Look for identifier-like patterns
        if self._looks_like_identifier_dict(value):
            return MediaField(
                name=name,
                value=value,
                field_type=FieldType.IDENTIFIERS,
                analysis_weight=AnalysisWeight.IGNORE,
                cache_key_eligible=False,
                original_type=type(value),
                processing_notes=[f"Identifier dict analyzed by {self.plugin_name}"]
            )
        
        # Default to structural for complex objects
        return MediaField(
            name=name,
            value=value,
            field_type=FieldType.STRUCTURAL,
            analysis_weight=AnalysisWeight.LOW,
            cache_key_eligible=False,
            original_type=type(value),
            processing_notes=[f"Complex structure analyzed by {self.plugin_name}"]
        )
    
    def _looks_like_person_object(self, obj: Dict[str, Any]) -> bool:
        """Check if object has person-like structure."""
        if not isinstance(obj, dict):
            return False
        
        keys = [k.lower() for k in obj.keys()]
        person_indicators = {'name', 'role', 'type', 'id'}
        
        # If it has name + role/type, likely a person
        has_name = any('name' in k for k in keys)
        has_role_or_type = any(k in ['role', 'type'] for k in keys)
        
        return has_name and has_role_or_type
    
    def _looks_like_identifier_dict(self, obj: Dict[str, Any]) -> bool:
        """Check if dict contains mostly identifiers/URLs."""
        if not isinstance(obj, dict):
            return False
        
        keys = [k.lower() for k in obj.keys()]
        id_patterns = ['id', 'url', 'uri', 'tmdb', 'imdb', 'isbn', 'doi']
        
        id_matches = sum(1 for key in keys if any(pattern in key for pattern in id_patterns))
        return id_matches > len(keys) * 0.5  # More than 50% ID-like keys
    
    def _create_unknown_field(self, name: str, value: Any) -> MediaField:
        """Create field with unknown classification."""
        return MediaField(
            name=name,
            value=value,
            field_type=FieldType.UNKNOWN,
            analysis_weight=AnalysisWeight.LOW,
            cache_key_eligible=True,
            original_type=type(value),
            processing_notes=[f"Unknown structure by {self.plugin_name}"]
        )


class NumericMetadataAnalysisPlugin(FieldAnalysisPlugin):
    """
    Analyzes numeric and date-like fields as metadata.
    
    Uses data type and value patterns rather than field names.
    """
    
    @property
    def plugin_name(self) -> str:
        return "NumericMetadataAnalysis"
    
    @property
    def priority(self) -> int:
        return 80  # Medium priority
    
    def can_analyze(self, name: str, value: Any, media_type: str) -> bool:
        """Check if value is numeric or date-like."""
        if isinstance(value, (int, float)):
            return True
        
        if isinstance(value, str):
            # Check for year patterns (4 digits)
            if re.match(r'^\d{4}$', value.strip()):
                return True
            
            # Check for other numeric patterns
            if re.match(r'^[\d.,]+$', value.strip()):
                return True
        
        return False
    
    def analyze_field(self, name: str, value: Any, media_type: str) -> MediaField:
        """Classify numeric field as metadata."""
        return MediaField(
            name=name,
            value=value,
            field_type=FieldType.METADATA,
            analysis_weight=AnalysisWeight.LOW,
            cache_key_eligible=True,
            original_type=type(value),
            processing_notes=[f"Numeric metadata analyzed by {self.plugin_name}"]
        )


class IdentifierAnalysisPlugin(FieldAnalysisPlugin):
    """
    Analyzes fields that look like identifiers, URLs, or technical data.
    
    Uses content patterns rather than field names to identify technical fields.
    """
    
    @property
    def plugin_name(self) -> str:
        return "IdentifierAnalysis"
    
    @property
    def priority(self) -> int:
        return 70  # Medium priority
    
    def can_analyze(self, name: str, value: Any, media_type: str) -> bool:
        """Check if value looks like an identifier or URL."""
        if not isinstance(value, str):
            return False
        
        text = value.strip().lower()
        
        # Check for URL patterns
        if any(text.startswith(prefix) for prefix in ['http://', 'https://', 'ftp://', 'mailto:']):
            return True
        
        # Check for UUID/GUID patterns
        if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', text):
            return True
        
        # Check for long hex strings (like ETags)
        if len(text) > 16 and re.match(r'^[0-9a-f]+$', text):
            return True
        
        return False
    
    def analyze_field(self, name: str, value: Any, media_type: str) -> MediaField:
        """Classify as identifier field."""
        return MediaField(
            name=name,
            value=value,
            field_type=FieldType.IDENTIFIERS,
            analysis_weight=AnalysisWeight.IGNORE,
            cache_key_eligible=False,
            original_type=type(value),
            processing_notes=[f"Identifier pattern detected by {self.plugin_name}"]
        )


class FallbackAnalysisPlugin(FieldAnalysisPlugin):
    """
    Fallback plugin that handles any field that other plugins couldn't classify.
    
    Always returns UNKNOWN classification for unhandled fields.
    """
    
    @property
    def plugin_name(self) -> str:
        return "FallbackAnalysis"
    
    @property
    def priority(self) -> int:
        return 0  # Lowest priority (fallback)
    
    def can_analyze(self, name: str, value: Any, media_type: str) -> bool:
        """Can always analyze (fallback)."""
        return True
    
    def analyze_field(self, name: str, value: Any, media_type: str) -> MediaField:
        """Create unknown field classification."""
        return MediaField(
            name=name,
            value=value,
            field_type=FieldType.UNKNOWN,
            analysis_weight=AnalysisWeight.LOW,
            cache_key_eligible=True,
            original_type=type(value),
            processing_notes=[f"Unclassified field handled by {self.plugin_name}"]
        )


def get_default_field_analysis_plugins() -> List[FieldAnalysisPlugin]:
    """
    Get default field analysis plugins in priority order.
    
    Returns list of plugins that provide basic field classification
    without hard-coded patterns. Plugins are ordered by priority.
    """
    plugins = [
        GenericTextAnalysisPlugin(),
        StructuredDataAnalysisPlugin(),
        NumericMetadataAnalysisPlugin(),
        IdentifierAnalysisPlugin(),
        FallbackAnalysisPlugin()  # Always last
    ]
    
    # Sort by priority (highest first)
    return sorted(plugins, key=lambda p: p.priority, reverse=True)