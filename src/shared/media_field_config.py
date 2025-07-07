"""
Media Field Configuration System
Defines field extraction rules for different media types via YAML configuration.
"""

import logging
import yaml
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from src.shared.media_types import MediaType

logger = logging.getLogger(__name__)


class FieldType(str, Enum):
    """Types of fields for extraction."""
    TEXT = "text"           # String fields containing text
    LIST = "list"           # Array fields containing items
    DATE = "date"           # Date/time fields
    NUMERIC = "numeric"     # Numeric fields
    METADATA = "metadata"   # Metadata fields


class ExtractionMethod(str, Enum):
    """Methods for extracting concepts from fields."""
    KEY_CONCEPTS = "key_concepts"       # Extract key concepts using NLP
    DIRECT_VALUES = "direct_values"     # Use field values directly
    LLM_EXTRACTION = "llm_extraction"   # Use LLM for intelligent extraction
    CUSTOM_PLUGIN = "custom_plugin"     # Use custom plugin for extraction


@dataclass
class FieldExtractionRule:
    """Configuration for extracting concepts from a specific field."""
    field_name: str
    field_type: FieldType
    extraction_method: ExtractionMethod
    max_concepts: int = 10
    weight: float = 1.0
    llm_prompt: Optional[str] = None
    expected_return_type: Optional[str] = None
    plugin_name: Optional[str] = None
    preprocessing: Optional[List[str]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldExtractionRule':
        """Create from dictionary configuration."""
        return cls(
            field_name=data["field_name"],
            field_type=FieldType(data["field_type"]),
            extraction_method=ExtractionMethod(data["extraction_method"]),
            max_concepts=data.get("max_concepts", 10),
            weight=data.get("weight", 1.0),
            llm_prompt=data.get("llm_prompt"),
            expected_return_type=data.get("expected_return_type"),
            plugin_name=data.get("plugin_name"),
            preprocessing=data.get("preprocessing", [])
        )


@dataclass
class MediaTypeConfig:
    """Configuration for a specific media type."""
    media_type: MediaType
    name: str
    description: str
    field_extraction_rules: List[FieldExtractionRule]
    plugins_to_run: List[str]
    priority_order: List[str]
    
    @classmethod
    def from_dict(cls, media_type: MediaType, data: Dict[str, Any]) -> 'MediaTypeConfig':
        """Create from dictionary configuration."""
        rules = [
            FieldExtractionRule.from_dict(rule_data) 
            for rule_data in data.get("field_extraction_rules", [])
        ]
        
        return cls(
            media_type=media_type,
            name=data["name"],
            description=data["description"],
            field_extraction_rules=rules,
            plugins_to_run=data.get("plugins_to_run", []),
            priority_order=data.get("priority_order", [])
        )


class MediaFieldConfigManager:
    """Manages media type field configurations."""
    
    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or Path("config/media_types")
        self.media_configs: Dict[MediaType, MediaTypeConfig] = {}
        self._loaded = False
    
    async def load_configurations(self) -> bool:
        """Load all media type configurations."""
        try:
            if not self.config_dir.exists():
                logger.warning(f"Media config directory not found: {self.config_dir}")
                self._create_default_configs()
            
            # Load all YAML files in config directory
            for config_file in self.config_dir.glob("*.yaml"):
                await self._load_config_file(config_file)
            
            self._loaded = True
            logger.info(f"Loaded configurations for {len(self.media_configs)} media types")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load media configurations: {e}")
            return False
    
    async def _load_config_file(self, config_file: Path):
        """Load a single configuration file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Determine media type from filename or config
            media_type_name = config_data.get("media_type") or config_file.stem
            
            try:
                media_type = MediaType(media_type_name)
            except ValueError:
                logger.warning(f"Unknown media type in {config_file}: {media_type_name}")
                return
            
            # Create media type configuration
            media_config = MediaTypeConfig.from_dict(media_type, config_data)
            self.media_configs[media_type] = media_config
            
            logger.debug(f"Loaded config for {media_type.value} from {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
    
    def get_config(self, media_type: MediaType) -> Optional[MediaTypeConfig]:
        """Get configuration for a media type."""
        if not self._loaded:
            raise RuntimeError("Configurations not loaded. Call load_configurations() first.")
        
        return self.media_configs.get(media_type)
    
    def extract_concepts_from_data(
        self, 
        data: Dict[str, Any], 
        media_type: MediaType
    ) -> List[str]:
        """Extract concepts from data using media-type-specific rules."""
        config = self.get_config(media_type)
        if not config:
            logger.warning(f"No configuration found for media type: {media_type}")
            return []
        
        concepts = set()
        
        for rule in config.field_extraction_rules:
            field_concepts = self._extract_concepts_from_field(data, rule)
            
            # Apply weight and limit
            weighted_concepts = field_concepts[:rule.max_concepts]
            concepts.update(weighted_concepts)
        
        # Apply media type priority order
        if config.priority_order:
            prioritized_concepts = []
            remaining_concepts = list(concepts)
            
            for priority_field in config.priority_order:
                # Find concepts from this field and add them first
                for rule in config.field_extraction_rules:
                    if rule.field_name == priority_field:
                        field_concepts = self._extract_concepts_from_field(data, rule)
                        for concept in field_concepts:
                            if concept in remaining_concepts:
                                prioritized_concepts.append(concept)
                                remaining_concepts.remove(concept)
            
            # Add remaining concepts
            prioritized_concepts.extend(remaining_concepts)
            return prioritized_concepts[:15]  # Global limit
        
        return list(concepts)[:15]
    
    def _extract_concepts_from_field(
        self, 
        data: Dict[str, Any], 
        rule: FieldExtractionRule
    ) -> List[str]:
        """Extract concepts from a specific field using the configured rule."""
        field_value = data.get(rule.field_name)
        if field_value is None:
            return []
        
        try:
            if rule.extraction_method == ExtractionMethod.DIRECT_VALUES:
                return self._extract_direct_values(field_value, rule)
            elif rule.extraction_method == ExtractionMethod.KEY_CONCEPTS:
                return self._extract_key_concepts(field_value, rule)
            elif rule.extraction_method == ExtractionMethod.LLM_EXTRACTION:
                # This would be handled by plugins in real implementation
                logger.debug(f"LLM extraction for {rule.field_name} - would use plugin")
                return []
            elif rule.extraction_method == ExtractionMethod.CUSTOM_PLUGIN:
                # This would be handled by plugins in real implementation
                logger.debug(f"Custom plugin extraction for {rule.field_name} - would use {rule.plugin_name}")
                return []
            else:
                logger.warning(f"Unknown extraction method: {rule.extraction_method}")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting concepts from {rule.field_name}: {e}")
            return []
    
    def _extract_direct_values(
        self, 
        field_value: Any, 
        rule: FieldExtractionRule
    ) -> List[str]:
        """Extract values directly from field."""
        concepts = []
        
        if rule.field_type == FieldType.TEXT and isinstance(field_value, str):
            # For text fields, split and clean
            concepts.append(field_value.lower().strip())
        elif rule.field_type == FieldType.LIST and isinstance(field_value, list):
            for item in field_value[:rule.max_concepts]:
                if isinstance(item, str):
                    concepts.append(item.lower().strip())
                elif isinstance(item, dict) and "Name" in item:
                    concepts.append(item["Name"].lower().strip())
                elif isinstance(item, dict) and "name" in item:
                    concepts.append(item["name"].lower().strip())
        
        return [c for c in concepts if c]
    
    def _extract_key_concepts(
        self, 
        field_value: Any, 
        rule: FieldExtractionRule
    ) -> List[str]:
        """Extract key concepts using NLP."""
        if rule.field_type == FieldType.TEXT and isinstance(field_value, str):
            # Use the existing extract_key_concepts function
            from src.shared.text_utils import extract_key_concepts
            return extract_key_concepts(field_value)[:rule.max_concepts]
        
        # For non-text fields, fall back to direct extraction
        return self._extract_direct_values(field_value, rule)
    
    def _create_default_configs(self):
        """Create default configuration files if they don't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create default movie configuration
        movie_config = {
            "media_type": "movie",
            "name": "Movie",
            "description": "Configuration for movie media type",
            "field_extraction_rules": [
                {
                    "field_name": "Name",
                    "field_type": "text",
                    "extraction_method": "key_concepts",
                    "max_concepts": 5,
                    "weight": 2.0
                },
                {
                    "field_name": "OriginalTitle",
                    "field_type": "text", 
                    "extraction_method": "key_concepts",
                    "max_concepts": 5,
                    "weight": 2.0
                },
                {
                    "field_name": "Overview",
                    "field_type": "text",
                    "extraction_method": "key_concepts", 
                    "max_concepts": 10,
                    "weight": 1.5
                },
                {
                    "field_name": "Taglines",
                    "field_type": "text",
                    "extraction_method": "key_concepts",
                    "max_concepts": 5,
                    "weight": 1.0
                },
                {
                    "field_name": "Genres",
                    "field_type": "list",
                    "extraction_method": "direct_values",
                    "max_concepts": 10,
                    "weight": 2.0
                },
                {
                    "field_name": "Tags", 
                    "field_type": "list",
                    "extraction_method": "direct_values",
                    "max_concepts": 10,
                    "weight": 1.5
                }
            ],
            "plugins_to_run": [
                "ConceptExpansionPlugin",
                "TemporalAnalysisPlugin"
            ],
            "priority_order": ["Genres", "Name", "OriginalTitle", "Overview"]
        }
        
        with open(self.config_dir / "movie.yaml", 'w') as f:
            yaml.dump(movie_config, f, default_flow_style=False)
        
        logger.info("Created default movie configuration")


# Global instance
_config_manager: Optional[MediaFieldConfigManager] = None


async def get_media_field_config() -> MediaFieldConfigManager:
    """Get the global media field configuration manager."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = MediaFieldConfigManager()
        await _config_manager.load_configurations()
    
    return _config_manager


def detect_media_type_from_data(data: Dict[str, Any]) -> MediaType:
    """Detect media type from data fields using pattern matching."""
    try:
        import re
        
        # Load detection rules
        detection_config_path = Path("config/media_types/media_detection.yaml")
        if not detection_config_path.exists():
            logger.warning("Media detection config not found, using fallback detection")
            return _fallback_media_detection(data)
        
        with open(detection_config_path, 'r') as f:
            detection_config = yaml.safe_load(f)
        
        media_scores = {}
        detection_rules = detection_config.get("media_detection_rules", {})
        
        # Score each media type based on field patterns
        for media_type_name, rules in detection_rules.items():
            score = 0.0
            field_patterns = rules.get("field_patterns", [])
            weight = rules.get("weight", 1.0)
            
            for pattern_config in field_patterns:
                pattern = pattern_config["pattern"]
                case_insensitive = pattern_config.get("case_insensitive", True)
                
                flags = re.IGNORECASE if case_insensitive else 0
                compiled_pattern = re.compile(pattern, flags)
                
                # Check if any data field matches this pattern
                for field_name in data.keys():
                    if compiled_pattern.match(field_name):
                        score += weight
                        break
            
            if score > 0:
                media_scores[media_type_name] = score
        
        # Return the media type with highest score
        if media_scores:
            best_match = max(media_scores, key=media_scores.get)
            try:
                return MediaType(best_match)
            except ValueError:
                logger.warning(f"Unknown media type from detection: {best_match}")
        
        # Fallback
        return _fallback_media_detection(data)
        
    except Exception as e:
        logger.error(f"Media type detection failed: {e}")
        return _fallback_media_detection(data)


def _fallback_media_detection(data: Dict[str, Any]) -> MediaType:
    """Fallback media type detection using simple heuristics."""
    import re
    
    # Count field patterns that suggest different media types
    movie_indicators = 0
    tv_indicators = 0
    book_indicators = 0
    music_indicators = 0
    
    for field_name in data.keys():
        field_lower = field_name.lower()
        
        # Movie patterns
        if re.search(r'(premiere|production|runtime|original.*title)', field_lower):
            movie_indicators += 1
        
        # TV patterns
        if re.search(r'(series|season|episode)', field_lower):
            tv_indicators += 1
        
        # Book patterns  
        if re.search(r'(author|publisher|isbn|page)', field_lower):
            book_indicators += 1
        
        # Music patterns
        if re.search(r'(artist|album|track|duration)', field_lower):
            music_indicators += 1
    
    # Return type with most indicators
    max_score = max(movie_indicators, tv_indicators, book_indicators, music_indicators)
    
    if max_score == 0:
        return MediaType.MOVIE  # Default fallback
    elif tv_indicators == max_score:
        return MediaType.TV_SHOW
    elif book_indicators == max_score:
        return MediaType.BOOK
    elif music_indicators == max_score:
        return MediaType.MUSIC
    else:
        return MediaType.MOVIE