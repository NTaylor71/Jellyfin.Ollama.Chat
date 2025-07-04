"""
Duckling Query Plugin for fast temporal understanding of user search queries.
Optimized for real-time query processing with Facebook's Duckling parser.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

try:
    from duckling import DucklingWrapper
    DUCKLING_AVAILABLE = True
except ImportError as e:
    DUCKLING_AVAILABLE = False
    logging.getLogger(__name__).debug(f"Duckling not available: {e}. Install with: pip install duckling")

from ..base import QueryEmbellisherPlugin, PluginExecutionContext


class DucklingQueryPlugin(QueryEmbellisherPlugin):
    """
    Fast temporal analysis for user queries using Facebook's Duckling.
    
    Handles sophisticated user expressions like:
    - "movies from two decades before the millennium"
    - "recent superhero films from this decade"  
    - "sci-fi movies around the turn of the century"
    - "films from approximately 2010"
    """
    
    def __init__(self):
        super().__init__()
        self.duckling = None
        self.query_patterns = self._load_query_patterns()
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize Duckling wrapper for query processing."""
        try:
            if DUCKLING_AVAILABLE:
                self.duckling = DucklingWrapper()
                self.logger.info("✅ Duckling initialized for query temporal analysis")
                return True
            else:
                self.logger.warning("⚠️ Duckling not available - using fallback query temporal analysis")
                return True  # Don't fail if Duckling not available
                
        except Exception as e:
            self.logger.error(f"💥 Failed to initialize Duckling: {e}")
            return False
    
    async def embellish_query(self, query: str, context: PluginExecutionContext) -> str:
        """Embellish user query with temporal understanding."""
        try:
            # Analyze temporal aspects of the query
            temporal_analysis = await self._analyze_query_temporally(query)
            
            if not temporal_analysis.get("temporal_entities"):
                return query  # No temporal content found
            
            # Create enhanced query with temporal context
            enhanced_query = self._enhance_query_with_temporal_context(query, temporal_analysis)
            
            # Log the enhancement for debugging
            if enhanced_query != query:
                self.logger.debug(f"Query enhanced: '{query}' → '{enhanced_query}'")
            
            return enhanced_query
            
        except Exception as e:
            self.logger.error(f"Error in Duckling query analysis: {e}")
            return query  # Return original query on error
    
    async def _analyze_query_temporally(self, query: str) -> Dict[str, Any]:
        """Analyze temporal aspects of user query using Duckling."""
        analysis = {
            "temporal_entities": [],
            "query_intent": "unknown",
            "time_ranges": [],
            "relative_references": [],
            "confidence": "low"
        }
        
        try:
            if self.duckling:
                # Use Duckling for sophisticated temporal parsing
                duckling_results = self.duckling.parse(query, dim_filter=["time"])
                
                for result in duckling_results:
                    temporal_entity = {
                        "text": result.get("body", ""),
                        "value": result.get("value", {}),
                        "start": result.get("start", 0),
                        "end": result.get("end", 0),
                        "confidence": result.get("confidence", 0.7),
                        "grain": result.get("value", {}).get("grain", "unknown")
                    }
                    analysis["temporal_entities"].append(temporal_entity)
                
                # Convert Duckling results to usable time ranges
                analysis["time_ranges"] = self._convert_to_time_ranges(duckling_results)
                
            else:
                # Fallback analysis without Duckling
                analysis = self._fallback_temporal_analysis(query)
            
            # Analyze query intent
            analysis["query_intent"] = self._determine_query_intent(query, analysis)
            
            # Determine confidence
            if len(analysis["temporal_entities"]) >= 2:
                analysis["confidence"] = "high"
            elif len(analysis["temporal_entities"]) >= 1:
                analysis["confidence"] = "medium"
            
        except Exception as e:
            self.logger.debug(f"Error in Duckling analysis: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    def _convert_to_time_ranges(self, duckling_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Duckling temporal results to searchable time ranges."""
        ranges = []
        
        for result in duckling_results:
            value = result.get("value", {})
            
            if "value" in value:
                # Absolute time reference
                time_range = self._parse_absolute_time(value)
                if time_range:
                    ranges.append(time_range)
            
            elif "from" in value and "to" in value:
                # Time interval
                time_range = self._parse_time_interval(value)
                if time_range:
                    ranges.append(time_range)
            
            elif "grain" in value:
                # Relative time with grain (decade, year, etc.)
                time_range = self._parse_relative_time(value)
                if time_range:
                    ranges.append(time_range)
        
        return ranges
    
    def _parse_absolute_time(self, value: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse absolute time reference from Duckling."""
        try:
            time_value = value.get("value")
            grain = value.get("grain", "year")
            
            if time_value:
                # Convert to datetime
                if isinstance(time_value, str):
                    from dateutil.parser import parse
                    dt = parse(time_value)
                else:
                    dt = datetime.fromisoformat(str(time_value))
                
                # Create range based on grain
                if grain == "decade":
                    start_year = (dt.year // 10) * 10
                    return {
                        "start": start_year,
                        "end": start_year + 9,
                        "precision": "decade",
                        "type": "absolute"
                    }
                elif grain == "year":
                    return {
                        "start": dt.year,
                        "end": dt.year,
                        "precision": "year", 
                        "type": "absolute"
                    }
        
        except Exception as e:
            self.logger.debug(f"Error parsing absolute time: {e}")
        
        return None
    
    def _parse_time_interval(self, value: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse time interval from Duckling."""
        try:
            from_time = value.get("from", {}).get("value")
            to_time = value.get("to", {}).get("value")
            
            if from_time and to_time:
                from dateutil.parser import parse
                start_dt = parse(str(from_time))
                end_dt = parse(str(to_time))
                
                return {
                    "start": start_dt.year,
                    "end": end_dt.year,
                    "precision": "range",
                    "type": "interval"
                }
        
        except Exception as e:
            self.logger.debug(f"Error parsing time interval: {e}")
        
        return None
    
    def _parse_relative_time(self, value: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse relative time reference from Duckling."""
        try:
            grain = value.get("grain")
            direction = value.get("direction", 0)  # -1 = past, 1 = future, 0 = present
            
            current_year = datetime.now().year
            
            if grain == "decade":
                if direction == -1:  # Past decade
                    return {
                        "start": current_year - 10,
                        "end": current_year - 1,
                        "precision": "decade",
                        "type": "relative_past"
                    }
                elif direction == 0:  # This decade  
                    decade_start = (current_year // 10) * 10
                    return {
                        "start": decade_start,
                        "end": current_year,
                        "precision": "current_decade",
                        "type": "relative_current"
                    }
            
        except Exception as e:
            self.logger.debug(f"Error parsing relative time: {e}")
        
        return None
    
    def _fallback_temporal_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback temporal analysis when Duckling is not available."""
        analysis = {
            "temporal_entities": [],
            "query_intent": "unknown",
            "time_ranges": [],
            "relative_references": [],
            "confidence": "low"
        }
        
        query_lower = query.lower()
        
        # Check for common temporal patterns
        for pattern_name, pattern_info in self.query_patterns.items():
            for keyword in pattern_info["keywords"]:
                if keyword in query_lower:
                    analysis["temporal_entities"].append({
                        "text": keyword,
                        "pattern": pattern_name,
                        "confidence": 0.6
                    })
                    
                    # Add corresponding time range
                    if "time_range" in pattern_info:
                        analysis["time_ranges"].append(pattern_info["time_range"])
        
        return analysis
    
    def _determine_query_intent(self, query: str, analysis: Dict[str, Any]) -> str:
        """Determine the temporal intent of the user query."""
        query_lower = query.lower()
        
        # Intent patterns
        if any(word in query_lower for word in ["recent", "latest", "new", "current"]):
            return "recent_content"
        elif any(word in query_lower for word in ["classic", "old", "vintage", "retro"]):
            return "historical_content"
        elif any(word in query_lower for word in ["before", "prior", "earlier"]):
            return "content_before"
        elif any(word in query_lower for word in ["after", "since", "following"]):
            return "content_after"
        elif any(word in query_lower for word in ["around", "circa", "approximately"]):
            return "content_around"
        elif any(word in query_lower for word in ["between", "from", "to"]):
            return "content_range"
        else:
            return "general_temporal"
    
    def _enhance_query_with_temporal_context(self, query: str, analysis: Dict[str, Any]) -> str:
        """Enhance the original query with temporal context for better search."""
        enhanced_parts = [query]
        
        # Add explicit temporal terms based on analysis
        for time_range in analysis.get("time_ranges", []):
            if time_range.get("precision") == "decade":
                start_year = time_range["start"]
                decade_term = f"{start_year}s"
                if decade_term not in query:
                    enhanced_parts.append(decade_term)
            
            elif time_range.get("precision") == "year":
                year_term = str(time_range["start"])
                if year_term not in query:
                    enhanced_parts.append(year_term)
        
        # Add intent-based temporal terms
        intent = analysis.get("query_intent")
        if intent == "recent_content":
            current_decade = (datetime.now().year // 10) * 10
            enhanced_parts.append(f"{current_decade}s")
        elif intent == "historical_content":
            enhanced_parts.extend(["classic", "vintage"])
        
        # Join enhanced parts
        enhanced_query = " ".join(enhanced_parts)
        
        # Clean up duplicates and return
        words = enhanced_query.split()
        unique_words = []
        seen = set()
        for word in words:
            word_lower = word.lower()
            if word_lower not in seen:
                unique_words.append(word)
                seen.add(word_lower)
        
        return " ".join(unique_words)
    
    def _load_query_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load common temporal query patterns for fallback analysis."""
        current_year = datetime.now().year
        current_decade = (current_year // 10) * 10
        
        return {
            "recent": {
                "keywords": ["recent", "latest", "new", "current"],
                "time_range": {
                    "start": current_year - 5,
                    "end": current_year,
                    "precision": "recent"
                }
            },
            "this_decade": {
                "keywords": ["this decade", "current decade"],
                "time_range": {
                    "start": current_decade,
                    "end": current_year,
                    "precision": "current_decade"
                }
            },
            "last_decade": {
                "keywords": ["last decade", "past decade", "previous decade"],
                "time_range": {
                    "start": current_decade - 10,
                    "end": current_decade - 1,
                    "precision": "decade"
                }
            },
            "turn_of_century": {
                "keywords": ["turn of the century", "millennium", "y2k"],
                "time_range": {
                    "start": 1995,
                    "end": 2005,
                    "precision": "era"
                }
            },
            "classic": {
                "keywords": ["classic", "vintage", "old", "golden age"],
                "time_range": {
                    "start": 1930,
                    "end": 1970,
                    "precision": "era"
                }
            }
        }