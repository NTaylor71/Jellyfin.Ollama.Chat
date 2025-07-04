"""
Base classes for linguistic analysis plugins.
Provides dual-use functionality for both ingestion and query processing.
"""

from typing import Dict, Any, List, Optional
from abc import abstractmethod
import logging

from ..base import QueryEmbellisherPlugin, EmbedDataEmbellisherPlugin

logger = logging.getLogger(__name__)


class LinguisticPlugin:
    """Base class for all linguistic analysis plugins."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialize_models()
    
    @abstractmethod
    def _initialize_models(self):
        """Initialize any NLP models or resources needed."""
        pass
    
    @abstractmethod
    async def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform linguistic analysis on text.
        
        Args:
            text: The text to analyze
            context: Optional context information
            
        Returns:
            Dictionary containing analysis results
        """
        pass


class DualUsePlugin(QueryEmbellisherPlugin, EmbedDataEmbellisherPlugin, LinguisticPlugin):
    """
    Base class for plugins that work on both queries and content.
    Implements symmetric processing for ingestion and search.
    """
    
    def __init__(self):
        QueryEmbellisherPlugin.__init__(self)
        EmbedDataEmbellisherPlugin.__init__(self)
        LinguisticPlugin.__init__(self)
    
    async def embellish_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process query using linguistic analysis."""
        try:
            analysis = await self.analyze(query, context)
            
            # Add query-specific processing
            result = {
                "original_query": query,
                "analysis": analysis
            }
            
            # Extract expansions for query
            if "expansions" in analysis:
                result["expanded_terms"] = analysis["expansions"]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in query embellishment: {e}")
            return {"original_query": query, "error": str(e)}
    
    async def enhance_data(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance content data using linguistic analysis."""
        try:
            # Extract text from data based on media type
            text = self._extract_text(data, context)
            
            # Perform analysis
            analysis = await self.analyze(text, context)
            
            # Structure for storage
            return {
                "linguistic_analysis": {
                    self.__class__.__name__: analysis
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in data enhancement: {e}")
            return {"error": str(e)}
    
    def _extract_text(self, data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Extract searchable text from data based on media type."""
        media_type = context.get("media_type", "generic")
        
        if media_type == "movie":
            # Combine relevant movie fields
            parts = []
            if "name" in data:
                parts.append(data["name"])
            if "overview" in data:
                parts.append(data["overview"])
            if "taglines" in data:
                parts.extend(data["taglines"])
            return " ".join(parts)
        
        elif media_type == "book":
            # Book-specific extraction
            parts = []
            if "title" in data:
                parts.append(data["title"])
            if "description" in data:
                parts.append(data["description"])
            if "synopsis" in data:
                parts.append(data["synopsis"])
            return " ".join(parts)
        
        else:
            # Generic text extraction
            return str(data.get("text", "") or data.get("content", "") or data)