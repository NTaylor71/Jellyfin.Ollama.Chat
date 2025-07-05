"""
Universal Genre Classifier Plugin
Implements the MediaTypePlugin interface as specified in todo.md Phase 1.
"""

from typing import List, Dict, Any
from src.plugins.base import MediaTypePlugin, PluginMetadata, PluginResourceRequirements, PluginType, PluginExecutionContext, PluginExecutionResult
from src.shared.media_types import MediaType, load_config


class UniversalGenreClassifier(MediaTypePlugin):
    """Universal genre classifier that adapts behavior based on media type."""
    
    def __init__(self):
        super().__init__()
        self._patterns_cache: Dict[MediaType, Dict[str, Any]] = {}
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="UniversalGenreClassifier",
            version="1.0.0",
            description="Universal genre classifier that adapts to different media types",
            author="System",
            plugin_type=PluginType.GENERAL
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        return PluginResourceRequirements(
            min_cpu_cores=1.0,
            preferred_cpu_cores=2.0,
            min_memory_mb=100.0,
            preferred_memory_mb=200.0
        )
    
    def get_supported_media_types(self) -> List[MediaType]:
        return [MediaType.MOVIE, MediaType.BOOK, MediaType.MUSIC]
    
    def analyze_for_media_type(self, text: str, media_type: MediaType) -> Dict[str, Any]:
        """Analyze text for a specific media type using configuration patterns."""
        patterns = self._get_patterns_for_media(media_type)
        return self._classify_with_patterns(text, patterns)
    
    def _get_patterns_for_media(self, media_type: MediaType) -> Dict[str, Any]:
        """Get patterns for specific media type, with caching."""
        if media_type not in self._patterns_cache:
            self._patterns_cache[media_type] = load_config(f"patterns/{media_type.value}.yml")
        return self._patterns_cache[media_type]
    
    def _classify_with_patterns(self, text: str, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Classify text using media-specific patterns."""
        text_lower = text.lower()
        results = {
            "detected_genres": [],
            "detected_people": [],
            "confidence_scores": {}
        }
        
        # Genre detection
        genres = patterns.get('genres', {})
        for genre, keywords in genres.items():
            for keyword in keywords:
                if keyword in text_lower:
                    results["detected_genres"].append(genre)
                    results["confidence_scores"][genre] = results["confidence_scores"].get(genre, 0) + 1
                    break
        
        # People detection
        people_indicators = patterns.get('people_indicators', {})
        for role, indicators in people_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    results["detected_people"].append(role)
                    break
        
        return results
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        try:
            # Pre-load patterns for supported media types
            for media_type in self.get_supported_media_types():
                self._get_patterns_for_media(media_type)
            return True
        except Exception as e:
            self._logger.error(f"Failed to initialize UniversalGenreClassifier: {e}")
            return False
    
    async def execute(self, data: Any, context: PluginExecutionContext) -> PluginExecutionResult:
        """Execute the plugin analysis."""
        try:
            if not isinstance(data, dict) or 'text' not in data or 'media_type' not in data:
                return PluginExecutionResult(
                    success=False,
                    error_message="Expected dict with 'text' and 'media_type' keys"
                )
            
            text = data['text']
            media_type_str = data['media_type']
            
            # Convert string to MediaType enum
            try:
                media_type = MediaType(media_type_str)
            except ValueError:
                return PluginExecutionResult(
                    success=False,
                    error_message=f"Unsupported media type: {media_type_str}"
                )
            
            # Check if media type is supported
            if media_type not in self.get_supported_media_types():
                return PluginExecutionResult(
                    success=False,
                    error_message=f"Media type {media_type.value} not supported by this plugin"
                )
            
            # Perform analysis
            analysis_result = self.analyze_for_media_type(text, media_type)
            
            return PluginExecutionResult(
                success=True,
                data=analysis_result,
                metadata={
                    "media_type": media_type.value,
                    "text_length": len(text),
                    "patterns_used": f"patterns/{media_type.value}.yml"
                }
            )
            
        except Exception as e:
            return PluginExecutionResult(
                success=False,
                error_message=f"UniversalGenreClassifier analysis failed: {str(e)}"
            )