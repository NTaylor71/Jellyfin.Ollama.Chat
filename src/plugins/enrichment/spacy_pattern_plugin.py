"""
SpaCy Pattern Matching Plugin
HTTP-only plugin that uses spaCy's pattern matching capabilities for domain-specific extraction.
"""

import logging
from typing import Dict, Any, List

from src.plugins.http_base import HTTPBasePlugin
from src.plugins.base import PluginMetadata, PluginType, ExecutionPriority

logger = logging.getLogger(__name__)


class SpacyPatternPlugin(HTTPBasePlugin):
    """
    HTTP-only plugin that extracts domain-specific patterns using SpaCy.
    
    Features:
    - Award mentions ("winner of", "nominated for")
    - Critical reception ("critically acclaimed", "box office hit") 
    - Technical specifications ("shot in IMAX", "Dolby Atmos")
    - Creator patterns ("directed by", "written by", "produced by")
    - Rating patterns ("rated R", "PG-13")
    - Genre indicators and descriptors
    """
    
    # Predefined patterns for media content
    MEDIA_PATTERNS = {
        "awards": [
            "winner of", "won the", "nominated for", "academy award", "oscar", "golden globe",
            "emmy", "grammy", "bafta", "cannes", "sundance", "venice film festival",
            "best picture", "best actor", "best actress", "best director"
        ],
        "critical_reception": [
            "critically acclaimed", "box office hit", "commercial success", "critical success",
            "cult classic", "sleeper hit", "breakthrough performance", "tour de force",
            "masterpiece", "landmark film", "groundbreaking", "influential"
        ],
        "technical_specs": [
            "shot in imax", "dolby atmos", "surround sound", "3d", "4k", "hd", "blu-ray",
            "digital", "film", "35mm", "70mm", "black and white", "color", "widescreen",
            "aspect ratio", "runtime", "minutes long"
        ],
        "creators": [
            "directed by", "written by", "produced by", "starring", "featuring",
            "music by", "cinematography by", "edited by", "executive producer",
            "screenplay by", "story by", "created by"
        ],
        "ratings": [
            "rated r", "rated pg", "rated pg-13", "rated g", "rated nc-17",
            "unrated", "tv-ma", "tv-14", "tv-pg", "tv-g", "tv-y", "tv-y7"
        ],
        "genres": [
            "action", "adventure", "comedy", "drama", "horror", "thriller", "sci-fi",
            "fantasy", "romance", "mystery", "crime", "documentary", "animation",
            "musical", "western", "war", "biography", "historical", "sports"
        ],
        "time_periods": [
            "set in", "takes place", "period piece", "historical", "contemporary",
            "modern day", "future", "post-apocalyptic", "medieval", "victorian",
            "1920s", "1930s", "1940s", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s"
        ],
        "themes": [
            "coming of age", "love story", "revenge", "redemption", "survival",
            "good vs evil", "family drama", "political thriller", "social commentary",
            "character study", "ensemble cast", "based on true events", "inspired by"
        ]
    }
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="SpacyPatternPlugin",
            version="1.0.0",
            description="Domain-specific pattern extraction for media content using SpaCy",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["patterns", "matching", "spacy", "domain-specific", "media"],
            execution_priority=ExecutionPriority.HIGH
        )
    
    async def enrich_field(
        self, 
        field_name: str, 
        field_value: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich a field with domain-specific pattern matching.
        
        Args:
            field_name: Name of the field being enriched
            field_value: Value of the field
            config: Plugin configuration
            
        Returns:
            Dict containing pattern matching results
        """
        try:
            if isinstance(field_value, str):
                text = field_value
            elif isinstance(field_value, list):
                text = " ".join(str(item) for item in field_value)
            else:
                text = str(field_value)
            
            if not text.strip():
                self._logger.debug(f"No text found in field {field_name}")
                return self._create_empty_result(field_name)
            
            self._logger.debug(f"Extracting patterns from {len(text)} characters using SpaCy pattern matching")
            
            # For now, use local pattern matching before implementing service call
            patterns = self._extract_local_patterns(text.lower(), config)
            
            result = {
                "spacy_patterns": patterns,
                "pattern_summary": self._create_pattern_summary(patterns),
                "original_text": text[:200] + "..." if len(text) > 200 else text,
                "field_name": field_name,
                "metadata": {
                    "provider": "spacy_pattern",
                    "total_patterns": sum(len(p) for p in patterns.values()),
                    "text_length": len(text),
                    "pattern_categories": list(patterns.keys()),
                    "extraction_method": "rule_based"
                }
            }
            
            self._logger.info(
                f"SpaCy pattern matching extracted {result['metadata']['total_patterns']} patterns for field {field_name}"
            )
            
            return self.normalize_text(result)
            
        except Exception as e:
            self._logger.error(f"SpaCy pattern extraction failed for field {field_name}: {e}")
            return self.normalize_text(self._create_error_result(field_name, str(e)))
    
    def _extract_local_patterns(self, text: str, config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract patterns using local rule-based matching."""
        patterns = {}
        
        # Get pattern categories to extract
        categories = config.get("pattern_categories", list(self.MEDIA_PATTERNS.keys()))
        case_sensitive = config.get("case_sensitive", False)
        
        search_text = text if case_sensitive else text.lower()
        
        for category in categories:
            if category not in self.MEDIA_PATTERNS:
                continue
                
            category_patterns = []
            
            for pattern in self.MEDIA_PATTERNS[category]:
                search_pattern = pattern if case_sensitive else pattern.lower()
                
                if search_pattern in search_text:
                    # Find all occurrences
                    start = 0
                    while True:
                        pos = search_text.find(search_pattern, start)
                        if pos == -1:
                            break
                        
                        # Extract context around the match
                        context_start = max(0, pos - 20)
                        context_end = min(len(text), pos + len(pattern) + 20)
                        context = text[context_start:context_end].strip()
                        
                        category_patterns.append({
                            "pattern": pattern,
                            "matched_text": text[pos:pos + len(pattern)],
                            "position": pos,
                            "context": context,
                            "confidence": self._calculate_pattern_confidence(pattern, context),
                            "category": category
                        })
                        
                        start = pos + 1
            
            if category_patterns:
                patterns[category] = category_patterns
        
        return patterns
    
    def _calculate_pattern_confidence(self, pattern: str, context: str) -> float:
        """Calculate confidence score for a pattern match."""
        base_confidence = 0.8
        
        # Boost confidence for longer patterns
        if len(pattern) > 10:
            base_confidence += 0.1
        
        # Boost confidence if pattern appears at sentence boundaries
        context_lower = context.lower()
        if pattern in context_lower and (
            context_lower.startswith(pattern) or 
            context_lower.endswith(pattern) or
            f" {pattern} " in context_lower or
            f". {pattern}" in context_lower
        ):
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _create_pattern_summary(self, patterns: Dict[str, List]) -> Dict[str, Any]:
        """Create a summary of extracted patterns."""
        summary = {}
        for category, pattern_list in patterns.items():
            if pattern_list:
                summary[category] = {
                    "count": len(pattern_list),
                    "top_patterns": list(set(p["pattern"] for p in pattern_list[:5])),
                    "confidence_avg": sum(p["confidence"] for p in pattern_list) / len(pattern_list)
                }
        return summary
    
    def _create_empty_result(self, field_name: str) -> Dict[str, Any]:
        """Create empty result structure."""
        return {
            "spacy_patterns": {},
            "pattern_summary": {},
            "original_text": "",
            "field_name": field_name,
            "metadata": {
                "provider": "spacy_pattern",
                "total_patterns": 0,
                "success": True
            }
        }
    
    def _create_error_result(self, field_name: str, error: str, text: str = "") -> Dict[str, Any]:
        """Create error result structure."""
        return {
            "spacy_patterns": {},
            "pattern_summary": {},
            "original_text": text[:200] + "..." if len(text) > 200 else text,
            "field_name": field_name,
            "error": error,
            "metadata": {
                "provider": "spacy_pattern",
                "success": False,
                "total_patterns": 0
            }
        }
    
    async def extract_awards(
        self, 
        text: str, 
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract award mentions from text.
        
        Args:
            text: Text to extract awards from
            config: Extraction configuration
            
        Returns:
            Award extraction results
        """
        if config is None:
            config = {}
        
        config["pattern_categories"] = ["awards"]
        result = await self.enrich_field("text", text, config)
        
        awards = result.get("spacy_patterns", {}).get("awards", [])
        
        return self.normalize_text({
            "awards": awards,
            "count": len(awards),
            "metadata": result.get("metadata", {})
        })
    
    async def extract_technical_specs(
        self, 
        text: str, 
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract technical specifications from text.
        
        Args:
            text: Text to extract technical specs from
            config: Extraction configuration
            
        Returns:
            Technical specs extraction results
        """
        if config is None:
            config = {}
        
        config["pattern_categories"] = ["technical_specs"]
        result = await self.enrich_field("text", text, config)
        
        specs = result.get("spacy_patterns", {}).get("technical_specs", [])
        
        return self.normalize_text({
            "technical_specs": specs,
            "count": len(specs),
            "metadata": result.get("metadata", {})
        })
    
    async def extract_creators(
        self, 
        text: str, 
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract creator mentions from text.
        
        Args:
            text: Text to extract creators from
            config: Extraction configuration
            
        Returns:
            Creator extraction results
        """
        if config is None:
            config = {}
        
        config["pattern_categories"] = ["creators"]
        result = await self.enrich_field("text", text, config)
        
        creators = result.get("spacy_patterns", {}).get("creators", [])
        
        return self.normalize_text({
            "creators": creators,
            "count": len(creators),
            "metadata": result.get("metadata", {})
        })
    
    async def extract_critical_reception(
        self, 
        text: str, 
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract critical reception indicators from text.
        
        Args:
            text: Text to extract reception info from
            config: Extraction configuration
            
        Returns:
            Critical reception extraction results
        """
        if config is None:
            config = {}
        
        config["pattern_categories"] = ["critical_reception"]
        result = await self.enrich_field("text", text, config)
        
        reception = result.get("spacy_patterns", {}).get("critical_reception", [])
        
        return self.normalize_text({
            "critical_reception": reception,
            "count": len(reception),
            "metadata": result.get("metadata", {})
        })
    
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin health."""
        base_health = await super().health_check()
        
        # Test pattern matching capability
        try:
            test_text = "This critically acclaimed film won the Academy Award for Best Picture."
            test_result = self._extract_local_patterns(test_text.lower(), {"pattern_categories": ["awards", "critical_reception"]})
            
            base_health["pattern_test"] = {
                "status": "healthy",
                "test_patterns_found": sum(len(p) for p in test_result.values()),
                "categories_tested": list(test_result.keys())
            }
            
        except Exception as e:
            base_health["pattern_test"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        return base_health