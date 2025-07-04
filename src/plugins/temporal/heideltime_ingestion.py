"""
HeidelTime Ingestion Plugin for rich temporal and cultural analysis during data ingestion.
Extracts historical periods, cultural eras, and temporal context from movie content.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from heideltime import HeidelTime
    HEIDELTIME_AVAILABLE = True
except ImportError as e:
    HEIDELTIME_AVAILABLE = False
    logging.getLogger(__name__).debug(f"HeidelTime not available: {e}. Install with: pip install heideltime")

from ..base import EmbedDataEmbellisherPlugin, PluginExecutionContext


class HeidelTimeIngestionPlugin(EmbedDataEmbellisherPlugin):
    """
    Sophisticated temporal analysis for content ingestion using HeidelTime.
    
    Extracts rich temporal context including:
    - Historical periods and eras
    - Cultural time markers
    - Temporal expressions in plot/description
    - Time-based content categorization
    """
    
    def __init__(self):
        super().__init__()
        self.heideltime = None
        self.cultural_eras = self._load_cultural_eras()
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize HeidelTime processor."""
        try:
            if HEIDELTIME_AVAILABLE:
                self.heideltime = HeidelTime()
                self.logger.info("✅ HeidelTime initialized for ingestion temporal analysis")
                return True
            else:
                self.logger.warning("⚠️ HeidelTime not available - temporal ingestion analysis disabled")
                return True  # Don't fail if HeidelTime not available
                
        except Exception as e:
            self.logger.error(f"💥 Failed to initialize HeidelTime: {e}")
            return False
    
    async def embellish_embed_data(self, data: Dict[str, Any], context: PluginExecutionContext) -> Dict[str, Any]:
        """Embellish movie data with rich temporal analysis during ingestion."""
        try:
            if not self.heideltime:
                return data
            
            # Extract text content for temporal analysis
            content_text = self._extract_content_text(data)
            if not content_text.strip():
                return data
            
            # Perform rich temporal analysis
            temporal_analysis = await self._analyze_content_temporally(content_text, data)
            
            # Add to enhanced_fields
            if 'enhanced_fields' not in data:
                data['enhanced_fields'] = {}
            
            data['enhanced_fields']['heideltime_analysis'] = temporal_analysis
            
            # Add searchable temporal metadata
            if temporal_analysis.get('periods') or temporal_analysis.get('eras'):
                data['enhanced_fields']['temporal_search_tags'] = self._create_search_tags(temporal_analysis)
            
            self.logger.debug(f"HeidelTime analysis completed for content: {len(temporal_analysis.get('periods', []))} periods detected")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in HeidelTime ingestion analysis: {e}")
            return data
    
    def _extract_content_text(self, data: Dict[str, Any]) -> str:
        """Extract relevant text content for temporal analysis."""
        content_parts = []
        
        # Movie-specific fields for temporal analysis
        temporal_fields = [
            'overview', 'plot', 'synopsis', 'description',
            'taglines', 'name', 'title', 'storyline'
        ]
        
        for field in temporal_fields:
            if field in data and data[field]:
                if isinstance(data[field], list):
                    content_parts.extend(data[field])
                else:
                    content_parts.append(str(data[field]))
        
        # Add genre context for temporal understanding
        if 'genres' in data and data['genres']:
            genre_context = f"This is a {', '.join(data['genres'])} film."
            content_parts.append(genre_context)
        
        return " ".join(content_parts)
    
    async def _analyze_content_temporally(self, text: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive temporal analysis using HeidelTime."""
        analysis = {
            "periods": [],
            "eras": [],
            "historical_events": [],
            "temporal_expressions": [],
            "cultural_context": [],
            "confidence": "medium"
        }
        
        try:
            # HeidelTime temporal extraction
            heideltime_results = self.heideltime.parse(text)
            
            # Process HeidelTime results
            for result in heideltime_results:
                temporal_info = {
                    "text": result.get("text", ""),
                    "value": result.get("value", ""),
                    "type": result.get("type", "unknown"),
                    "confidence": result.get("confidence", 0.7)
                }
                analysis["temporal_expressions"].append(temporal_info)
                
                # Categorize by type
                if result.get("type") == "DATE":
                    analysis["periods"].append(temporal_info)
                elif result.get("type") == "DURATION":
                    analysis["eras"].append(temporal_info)
            
            # Add cultural era analysis
            cultural_analysis = self._analyze_cultural_context(text, data)
            analysis["cultural_context"] = cultural_analysis
            
            # Add historical event detection
            historical_events = self._detect_historical_events(text)
            analysis["historical_events"] = historical_events
            
            # Determine overall confidence
            if len(analysis["temporal_expressions"]) >= 3:
                analysis["confidence"] = "high"
            elif len(analysis["temporal_expressions"]) >= 1:
                analysis["confidence"] = "medium"
            else:
                analysis["confidence"] = "low"
            
        except Exception as e:
            self.logger.debug(f"Error in HeidelTime analysis: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    def _analyze_cultural_context(self, text: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze cultural and historical context."""
        cultural_markers = []
        text_lower = text.lower()
        
        # Check against known cultural eras
        for era_name, era_info in self.cultural_eras.items():
            # Check keywords
            if any(keyword in text_lower for keyword in era_info["keywords"]):
                cultural_markers.append({
                    "era": era_name,
                    "period": era_info["period"],
                    "characteristics": era_info["characteristics"],
                    "confidence": 0.8
                })
        
        # Add genre-specific temporal context
        if 'genres' in data:
            for genre in data.get('genres', []):
                genre_context = self._get_genre_temporal_context(genre.lower(), text_lower)
                if genre_context:
                    cultural_markers.extend(genre_context)
        
        return cultural_markers
    
    def _detect_historical_events(self, text: str) -> List[Dict[str, Any]]:
        """Detect references to historical events."""
        events = []
        text_lower = text.lower()
        
        # Major historical events relevant to cinema
        historical_events = {
            "world_war_ii": {
                "keywords": ["world war", "wwii", "ww2", "nazi", "holocaust", "pearl harbor", "d-day"],
                "period": "1939-1945",
                "impact": "major_historical_event"
            },
            "cold_war": {
                "keywords": ["cold war", "soviet union", "ussr", "berlin wall", "nuclear", "eisenhower", "kennedy"],
                "period": "1947-1991", 
                "impact": "geopolitical_era"
            },
            "vietnam_war": {
                "keywords": ["vietnam", "saigon", "napalm", "draft", "protest"],
                "period": "1955-1975",
                "impact": "social_upheaval"
            },
            "moon_landing": {
                "keywords": ["moon landing", "apollo", "nasa", "space race", "neil armstrong"],
                "period": "1969",
                "impact": "technological_milestone"
            },
            "digital_revolution": {
                "keywords": ["computer", "internet", "digital", "technology", "silicon valley"],
                "period": "1980-2000",
                "impact": "technological_era"
            }
        }
        
        for event_name, event_info in historical_events.items():
            if any(keyword in text_lower for keyword in event_info["keywords"]):
                events.append({
                    "event": event_name,
                    "period": event_info["period"],
                    "impact": event_info["impact"],
                    "confidence": 0.7
                })
        
        return events
    
    def _get_genre_temporal_context(self, genre: str, text: str) -> List[Dict[str, Any]]:
        """Get temporal context specific to movie genres."""
        context = []
        
        genre_temporal_maps = {
            "western": {
                "default_era": "old_west",
                "period": "1850-1890",
                "keywords": ["frontier", "cowboys", "saloon", "sheriff"]
            },
            "film noir": {
                "default_era": "post_war",
                "period": "1940-1960", 
                "keywords": ["detective", "crime", "city", "shadows"]
            },
            "sci-fi": {
                "default_era": "future_speculation",
                "period": "varies",
                "keywords": ["future", "space", "technology", "alien"]
            },
            "historical": {
                "default_era": "period_piece",
                "period": "varies",
                "keywords": ["based on", "true story", "historical"]
            }
        }
        
        if genre in genre_temporal_maps:
            genre_info = genre_temporal_maps[genre]
            # Check if genre-specific keywords are present
            if any(keyword in text for keyword in genre_info["keywords"]):
                context.append({
                    "genre_era": genre_info["default_era"],
                    "period": genre_info["period"],
                    "genre": genre,
                    "confidence": 0.6
                })
        
        return context
    
    def _create_search_tags(self, analysis: Dict[str, Any]) -> List[str]:
        """Create searchable temporal tags from analysis."""
        tags = []
        
        # Add period tags
        for period in analysis.get("periods", []):
            if period.get("value"):
                tags.append(f"period_{period['value']}")
        
        # Add era tags
        for era in analysis.get("eras", []):
            if era.get("text"):
                clean_era = era["text"].lower().replace(" ", "_")
                tags.append(f"era_{clean_era}")
        
        # Add cultural context tags
        for context in analysis.get("cultural_context", []):
            if context.get("era"):
                tags.append(f"cultural_{context['era']}")
        
        # Add historical event tags
        for event in analysis.get("historical_events", []):
            if event.get("event"):
                tags.append(f"event_{event['event']}")
        
        return list(set(tags))  # Remove duplicates
    
    def _load_cultural_eras(self) -> Dict[str, Dict[str, Any]]:
        """Load cultural era definitions for temporal analysis."""
        return {
            "golden_age_hollywood": {
                "period": "1930-1960",
                "keywords": ["studio system", "golden age", "classic hollywood", "star system"],
                "characteristics": ["glamour", "studio_control", "censorship_code"]
            },
            "new_hollywood": {
                "period": "1967-1980", 
                "keywords": ["new hollywood", "auteur", "independent", "countercultural"],
                "characteristics": ["director_driven", "anti_establishment", "experimental"]
            },
            "blockbuster_era": {
                "period": "1975-present",
                "keywords": ["blockbuster", "franchise", "sequel", "high concept"],
                "characteristics": ["mass_appeal", "merchandising", "special_effects"]
            },
            "reagan_era": {
                "period": "1980-1988",
                "keywords": ["reagan", "conservative", "wall street", "materialism"],
                "characteristics": ["excess", "capitalism", "cold_war_tension"]
            },
            "post_9_11": {
                "period": "2001-2010",
                "keywords": ["terrorism", "security", "paranoia", "surveillance"],
                "characteristics": ["anxiety", "patriotism", "war_on_terror"]
            },
            "digital_age": {
                "period": "1990-present", 
                "keywords": ["digital", "internet", "cyber", "virtual"],
                "characteristics": ["connectivity", "information_age", "social_media"]
            }
        }