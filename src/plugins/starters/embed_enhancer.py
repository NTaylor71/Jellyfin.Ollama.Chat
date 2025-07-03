"""
Embed Data Enhancer Plugin - Starter Example
Demonstrates data preprocessing and enrichment before embedding.
"""

import asyncio
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..base import (
    EmbedDataEmbellisherPlugin, PluginResourceRequirements, PluginExecutionContext,
    plugin_decorator, PluginType, ExecutionPriority
)


@plugin_decorator(
    name="EmbedDataEnhancer",
    version="1.0.0", 
    description="Enhances data before embedding with metadata extraction and content preprocessing",
    author="System",
    plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
    resource_requirements=PluginResourceRequirements(
        min_cpu_cores=1.0,
        preferred_cpu_cores=2.0,
        min_memory_mb=100.0,
        preferred_memory_mb=300.0,
        max_execution_time_seconds=10.0,
        can_use_distributed_resources=False
    ),
    execution_priority=ExecutionPriority.NORMAL,
    tags=["embedding", "preprocessing", "metadata"]
)
class EmbedDataEnhancerPlugin(EmbedDataEmbellisherPlugin):
    """Example plugin that enhances data before embedding with metadata and preprocessing."""
    
    def __init__(self):
        super().__init__()
        self.text_processors = []
        self.metadata_extractors = []
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the embed data enhancer plugin."""
        try:
            self._logger.info("Initializing EmbedDataEnhancer plugin")
            
            # Initialize text processors
            self.text_processors = [
                self._clean_text,
                self._normalize_whitespace,
                self._extract_entities,
                self._enhance_readability
            ]
            
            # Initialize metadata extractors
            self.metadata_extractors = [
                self._extract_text_stats,
                self._extract_content_type,
                self._extract_language_hints,
                self._extract_temporal_info
            ]
            
            self._is_initialized = True
            self._logger.info("EmbedDataEnhancer plugin initialized successfully")
            return True
            
        except Exception as e:
            self._initialization_error = str(e)
            self._logger.error(f"Failed to initialize EmbedDataEnhancer: {e}")
            return False
    
    async def embellish_embed_data(self, data: Dict[str, Any], context: PluginExecutionContext) -> Dict[str, Any]:
        """Enhance embed data with preprocessing and metadata extraction."""
        try:
            enhanced_data = data.copy()
            
            # Extract and process text content
            text_content = self._extract_text_content(data)
            if text_content:
                # Apply text processors
                processed_text = await self._process_text(text_content, context)
                enhanced_data['processed_text'] = processed_text
                
                # Extract metadata
                metadata = await self._extract_metadata(text_content, data, context)
                enhanced_data['enhancement_metadata'] = metadata
                
                # Add content enrichments
                enrichments = await self._generate_enrichments(processed_text, metadata, context)
                enhanced_data['content_enrichments'] = enrichments
            
            # Add processing timestamp
            enhanced_data['enhancement_timestamp'] = datetime.utcnow().isoformat()
            enhanced_data['enhancement_version'] = self.metadata.version
            
            self._logger.info(f"Enhanced embed data with {len(enhanced_data) - len(data)} new fields")
            return enhanced_data
            
        except Exception as e:
            self._logger.error(f"Error enhancing embed data: {e}")
            return data  # Return original data on error
    
    def _extract_text_content(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract text content from various data fields."""
        # Common text fields to check
        text_fields = ['text', 'content', 'description', 'summary', 'title', 'body']
        
        for field in text_fields:
            if field in data and isinstance(data[field], str):
                return data[field]
        
        # Try to concatenate multiple text fields
        text_parts = []
        for field in text_fields:
            if field in data and isinstance(data[field], str):
                text_parts.append(data[field])
        
        return ' '.join(text_parts) if text_parts else None
    
    async def _process_text(self, text: str, context: PluginExecutionContext) -> str:
        """Apply text processors to clean and normalize text."""
        processed_text = text
        
        # Apply processors sequentially
        for processor in self.text_processors:
            try:
                processed_text = await processor(processed_text, context)
            except Exception as e:
                self._logger.warning(f"Text processor failed: {e}")
                continue
        
        return processed_text
    
    async def _clean_text(self, text: str, context: PluginExecutionContext) -> str:
        """Clean text by removing unwanted characters and formatting."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        return text
    
    async def _normalize_whitespace(self, text: str, context: PluginExecutionContext) -> str:
        """Normalize whitespace and line breaks."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n+', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    async def _extract_entities(self, text: str, context: PluginExecutionContext) -> str:
        """Extract and mark entities in text (simplified version)."""
        # Simple entity patterns (in production, use proper NER)
        
        # Mark potential movie/show titles (words in quotes)
        text = re.sub(r'"([^"]+)"', r'[TITLE:\1]', text)
        
        # Mark potential person names (capitalized words)
        text = re.sub(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', r'[PERSON:\1]', text)
        
        # Mark years
        text = re.sub(r'\b(19\d\d|20\d\d)\b', r'[YEAR:\1]', text)
        
        return text
    
    async def _enhance_readability(self, text: str, context: PluginExecutionContext) -> str:
        """Enhance text readability."""
        # Add spacing around punctuation for better tokenization
        text = re.sub(r'([.!?])', r' \1 ', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Final cleanup
        text = re.sub(r' +', ' ', text).strip()
        
        return text
    
    async def _extract_metadata(self, text: str, original_data: Dict[str, Any], 
                              context: PluginExecutionContext) -> Dict[str, Any]:
        """Extract metadata from text and original data."""
        metadata = {}
        
        # Apply metadata extractors
        for extractor in self.metadata_extractors:
            try:
                extractor_metadata = await extractor(text, original_data, context)
                metadata.update(extractor_metadata)
            except Exception as e:
                self._logger.warning(f"Metadata extractor failed: {e}")
                continue
        
        return metadata
    
    async def _extract_text_stats(self, text: str, original_data: Dict[str, Any], 
                                context: PluginExecutionContext) -> Dict[str, Any]:
        """Extract basic text statistics."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len([s for s in sentences if s.strip()]) if sentences else 0
        }
    
    async def _extract_content_type(self, text: str, original_data: Dict[str, Any], 
                                  context: PluginExecutionContext) -> Dict[str, Any]:
        """Detect content type based on patterns."""
        content_indicators = {
            'review': ['review', 'rating', 'stars', 'recommend'],
            'synopsis': ['plot', 'story', 'about', 'synopsis'],
            'technical': ['specs', 'technical', 'format', 'resolution'],
            'biographical': ['born', 'career', 'biography', 'life'],
            'news': ['reported', 'according', 'sources', 'breaking']
        }
        
        detected_types = []
        text_lower = text.lower()
        
        for content_type, indicators in content_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                detected_types.append(content_type)
        
        return {
            'detected_content_types': detected_types,
            'primary_content_type': detected_types[0] if detected_types else 'general'
        }
    
    async def _extract_language_hints(self, text: str, original_data: Dict[str, Any], 
                                    context: PluginExecutionContext) -> Dict[str, Any]:
        """Extract language and locale hints."""
        # Simple language detection based on common words
        language_indicators = {
            'english': ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that'],
            'spanish': ['el', 'la', 'es', 'en', 'de', 'un', 'una', 'que'],
            'french': ['le', 'de', 'et', 'est', 'une', 'dans', 'que', 'pour']
        }
        
        text_words = set(text.lower().split())
        language_scores = {}
        
        for language, indicators in language_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_words)
            language_scores[language] = score
        
        detected_language = max(language_scores, key=language_scores.get) if language_scores else 'unknown'
        
        return {
            'detected_language': detected_language,
            'language_confidence': language_scores.get(detected_language, 0) / len(text.split()) if text.split() else 0
        }
    
    async def _extract_temporal_info(self, text: str, original_data: Dict[str, Any], 
                                   context: PluginExecutionContext) -> Dict[str, Any]:
        """Extract temporal information from text."""
        # Extract years
        years = re.findall(r'\b(19\d\d|20\d\d)\b', text)
        
        # Extract temporal keywords
        temporal_keywords = ['recent', 'new', 'old', 'classic', 'modern', 'vintage', 'current', 'latest']
        found_temporal = [word for word in temporal_keywords if word.lower() in text.lower()]
        
        return {
            'mentioned_years': list(set(years)),
            'temporal_keywords': found_temporal,
            'estimated_time_period': self._estimate_time_period(years, found_temporal)
        }
    
    def _estimate_time_period(self, years: List[str], temporal_keywords: List[str]) -> str:
        """Estimate time period based on years and keywords."""
        if not years and not temporal_keywords:
            return 'unknown'
        
        if 'classic' in [kw.lower() for kw in temporal_keywords]:
            return 'classic'
        
        if 'recent' in [kw.lower() for kw in temporal_keywords] or 'new' in [kw.lower() for kw in temporal_keywords]:
            return 'recent'
        
        if years:
            latest_year = max(int(year) for year in years)
            current_year = datetime.now().year
            
            if latest_year >= current_year - 2:
                return 'recent'
            elif latest_year >= current_year - 10:
                return 'modern'
            elif latest_year >= 1990:
                return 'contemporary'
            else:
                return 'classic'
        
        return 'unknown'
    
    async def _generate_enrichments(self, processed_text: str, metadata: Dict[str, Any], 
                                  context: PluginExecutionContext) -> Dict[str, Any]:
        """Generate content enrichments based on processed text and metadata."""
        enrichments = {}
        
        # Generate content tags
        tags = await self._generate_content_tags(processed_text, metadata)
        enrichments['content_tags'] = tags
        
        # Generate summary keywords
        keywords = await self._extract_keywords(processed_text)
        enrichments['keywords'] = keywords
        
        # Generate content quality score
        quality_score = await self._calculate_quality_score(processed_text, metadata)
        enrichments['quality_score'] = quality_score
        
        return enrichments
    
    async def _generate_content_tags(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate content tags based on text and metadata."""
        tags = []
        
        # Add content type tags
        content_types = metadata.get('detected_content_types', [])
        tags.extend(content_types)
        
        # Add time period tags
        time_period = metadata.get('estimated_time_period', 'unknown')
        if time_period != 'unknown':
            tags.append(time_period)
        
        # Add length-based tags
        word_count = metadata.get('word_count', 0)
        if word_count > 500:
            tags.append('long-form')
        elif word_count > 100:
            tags.append('medium-form')
        else:
            tags.append('short-form')
        
        return list(set(tags))
    
    async def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Simple keyword extraction (in production, use TF-IDF or other methods)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Common stop words to filter out
        stop_words = {'that', 'this', 'with', 'from', 'they', 'been', 'have', 'their', 'would', 'there', 'could', 'which'}
        
        # Filter and count words
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [word for word, freq in keywords]
    
    async def _calculate_quality_score(self, text: str, metadata: Dict[str, Any]) -> float:
        """Calculate a content quality score."""
        score = 0.0
        
        # Length score (optimal range)
        word_count = metadata.get('word_count', 0)
        if 50 <= word_count <= 500:
            score += 0.3
        elif word_count > 10:
            score += 0.1
        
        # Readability score
        avg_sentence_length = metadata.get('avg_sentence_length', 0)
        if 10 <= avg_sentence_length <= 25:
            score += 0.2
        
        # Content completeness
        if metadata.get('detected_content_types'):
            score += 0.2
        
        # Language confidence
        lang_confidence = metadata.get('language_confidence', 0)
        score += min(lang_confidence * 0.3, 0.3)
        
        return min(score, 1.0)
    
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        self.text_processors.clear()
        self.metadata_extractors.clear()
        self._logger.info("EmbedDataEnhancer plugin cleaned up")