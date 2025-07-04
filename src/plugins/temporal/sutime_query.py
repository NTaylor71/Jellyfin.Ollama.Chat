"""
SUTime Query Plugin for complex temporal reasoning in user queries.
Alternative/complement to Duckling using Stanford's SUTime parser.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from sutime import SUTime
    SUTIME_AVAILABLE = True
except ImportError as e:
    SUTIME_AVAILABLE = False
    logging.getLogger(__name__).debug(f"SUTime not available: {e}. Install with: pip install sutime")

from ..base import QueryEmbellisherPlugin, PluginExecutionContext


class SUTimeQueryPlugin(QueryEmbellisherPlugin):
    """
    Complex temporal reasoning for user queries using Stanford's SUTime.
    
    Complements Duckling with academic-grade temporal understanding:
    - Complex relative expressions ("two decades before the millennium")
    - Temporal reasoning and inference
    - Cultural and historical time references
    - Multi-step temporal calculations
    """
    
    def __init__(self):
        super().__init__()
        self.sutime = None
        self.temporal_reasoning_rules = self._load_reasoning_rules()
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize SUTime for complex temporal reasoning."""
        try:
            if SUTIME_AVAILABLE:
                self.sutime = SUTime(mark_time_ranges=True, include_range=True)
                self.logger.info("✅ SUTime initialized for complex temporal reasoning")
                return True
            else:
                self.logger.warning("⚠️ SUTime not available - complex temporal reasoning disabled")
                return True  # Don't fail if SUTime not available
                
        except Exception as e:
            self.logger.error(f"💥 Failed to initialize SUTime: {e}")
            return False
    
    async def embellish_query(self, query: str, context: PluginExecutionContext) -> str:
        """Embellish query with complex temporal reasoning."""
        try:
            if not self.sutime:
                return query
            
            # Perform complex temporal analysis
            temporal_analysis = await self._analyze_complex_temporal(query)
            
            if not temporal_analysis.get("complex_expressions"):
                return query
            
            # Apply temporal reasoning to resolve complex expressions
            resolved_query = self._apply_temporal_reasoning(query, temporal_analysis)
            
            if resolved_query != query:
                self.logger.debug(f"SUTime reasoning: '{query}' → '{resolved_query}'")
            
            return resolved_query
            
        except Exception as e:
            self.logger.error(f"Error in SUTime query reasoning: {e}")
            return query
    
    async def _analyze_complex_temporal(self, query: str) -> Dict[str, Any]:
        """Analyze complex temporal expressions using SUTime."""
        analysis = {
            "complex_expressions": [],
            "temporal_reasoning": [],
            "resolved_dates": [],
            "confidence": "medium"
        }
        
        try:
            # Parse with SUTime
            sutime_results = self.sutime.parse(query)
            
            for result in sutime_results:
                # Extract temporal information
                temporal_expr = {
                    "text": result.get("text", ""),
                    "start_char": result.get("start", 0),
                    "end_char": result.get("end", 0),
                    "type": result.get("type", "unknown"),
                    "value": result.get("value", ""),
                    "temporal_function": result.get("temporalFunction", ""),
                    "confidence": 0.8
                }
                
                analysis["complex_expressions"].append(temporal_expr)
                
                # Apply reasoning for complex expressions
                if self._is_complex_expression(temporal_expr):
                    reasoning = self._reason_about_expression(temporal_expr, query)
                    if reasoning:
                        analysis["temporal_reasoning"].append(reasoning)
            
            # Resolve complex temporal references
            analysis["resolved_dates"] = self._resolve_complex_dates(analysis["complex_expressions"])
            
            # Determine confidence based on complexity handled
            if len(analysis["temporal_reasoning"]) >= 2:
                analysis["confidence"] = "high"
            elif len(analysis["temporal_reasoning"]) >= 1:
                analysis["confidence"] = "medium"
            
        except Exception as e:
            self.logger.debug(f"Error in SUTime analysis: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    def _is_complex_expression(self, expr: Dict[str, Any]) -> bool:
        """Determine if temporal expression requires complex reasoning."""
        text = expr.get("text", "").lower()
        
        # Complex patterns that require reasoning
        complex_patterns = [
            "before", "after", "prior to", "following",
            "decades before", "years before", "centuries before",
            "around the time", "during the period", "in the era of",
            "millennium", "turn of", "dawn of", "end of",
            "post-", "pre-", "early", "late", "mid"
        ]
        
        return any(pattern in text for pattern in complex_patterns)
    
    def _reason_about_expression(self, expr: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
        """Apply temporal reasoning to complex expressions."""
        text = expr.get("text", "").lower()
        
        # Apply reasoning rules
        for rule_name, rule in self.temporal_reasoning_rules.items():
            if rule["pattern"](text):
                reasoning_result = rule["resolver"](text, expr, query)
                if reasoning_result:
                    return {
                        "rule": rule_name,
                        "original": expr["text"],
                        "reasoning": reasoning_result,
                        "confidence": rule.get("confidence", 0.7)
                    }
        
        return None
    
    def _resolve_complex_dates(self, expressions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve complex temporal expressions to concrete date ranges."""
        resolved = []
        
        for expr in expressions:
            text = expr.get("text", "").lower()
            
            # Handle "before millennium" type expressions
            if "millennium" in text:
                if "before" in text:
                    resolved.append({
                        "original": expr["text"],
                        "resolved": "1000-1999",
                        "start": 1000,
                        "end": 1999,
                        "reasoning": "before_millennium"
                    })
                elif "after" in text or "post" in text:
                    resolved.append({
                        "original": expr["text"],
                        "resolved": "2000-present",
                        "start": 2000,
                        "end": datetime.now().year,
                        "reasoning": "after_millennium"
                    })
            
            # Handle "turn of century" expressions
            elif "turn of" in text and "century" in text:
                resolved.append({
                    "original": expr["text"],
                    "resolved": "1995-2005",
                    "start": 1995,
                    "end": 2005,
                    "reasoning": "turn_of_century"
                })
            
            # Handle "decades before/after" with reference points
            elif "decades" in text and ("before" in text or "after" in text):
                reference_year = self._extract_reference_year(text)
                if reference_year:
                    if "before" in text:
                        num_decades = self._extract_number_word(text)
                        start_year = reference_year - (num_decades * 10)
                        resolved.append({
                            "original": expr["text"], 
                            "resolved": f"{start_year}-{start_year + 9}",
                            "start": start_year,
                            "end": start_year + 9,
                            "reasoning": f"{num_decades}_decades_before_{reference_year}"
                        })
        
        return resolved
    
    def _extract_reference_year(self, text: str) -> Optional[int]:
        """Extract reference year from complex temporal expression."""
        import re
        
        # Look for explicit years
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            return int(year_match.group(0))
        
        # Special cases
        if "millennium" in text:
            return 2000
        elif "century" in text:
            return 2000  # Assume 21st century context
        
        return None
    
    def _extract_number_word(self, text: str) -> int:
        """Extract number from word form in text."""
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "a": 1, "an": 1, "couple": 2, "few": 3, "several": 4
        }
        
        for word, num in number_words.items():
            if word in text:
                return num
        
        # Look for digits
        import re
        digit_match = re.search(r'\b(\d+)\b', text)
        if digit_match:
            return int(digit_match.group(1))
        
        return 2  # Default assumption
    
    def _apply_temporal_reasoning(self, query: str, analysis: Dict[str, Any]) -> str:
        """Apply temporal reasoning to enhance query."""
        enhanced_query = query
        
        # Replace complex expressions with resolved dates
        for reasoning in analysis.get("temporal_reasoning", []):
            original = reasoning.get("original", "")
            resolved = reasoning.get("reasoning", {})
            
            if "resolved_terms" in resolved:
                # Replace the complex expression with resolved terms
                for term in resolved["resolved_terms"]:
                    if term not in enhanced_query:
                        enhanced_query += f" {term}"
        
        # Add resolved date ranges as search terms
        for resolved in analysis.get("resolved_dates", []):
            start = resolved.get("start")
            end = resolved.get("end")
            
            if start and end:
                if end - start <= 1:
                    # Single year or short range
                    enhanced_query += f" {start}"
                elif end - start == 9:
                    # Decade
                    decade = f"{start}s"
                    if decade not in enhanced_query:
                        enhanced_query += f" {decade}"
                else:
                    # Multi-year range
                    enhanced_query += f" {start}-{end}"
        
        return enhanced_query.strip()
    
    def _load_reasoning_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load temporal reasoning rules for complex expressions."""
        return {
            "before_reference": {
                "pattern": lambda text: "before" in text and any(ref in text for ref in ["millennium", "century", "war", "event"]),
                "resolver": self._resolve_before_reference,
                "confidence": 0.8
            },
            "after_reference": {
                "pattern": lambda text: ("after" in text or "post" in text) and any(ref in text for ref in ["millennium", "century", "war", "event"]),
                "resolver": self._resolve_after_reference,
                "confidence": 0.8
            },
            "era_reference": {
                "pattern": lambda text: any(era in text for era in ["era", "age", "period", "time"]),
                "resolver": self._resolve_era_reference,
                "confidence": 0.7
            },
            "relative_decades": {
                "pattern": lambda text: "decades" in text and any(rel in text for rel in ["before", "after", "ago"]),
                "resolver": self._resolve_relative_decades,
                "confidence": 0.9
            }
        }
    
    def _resolve_before_reference(self, text: str, expr: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Resolve 'before X' temporal references."""
        if "millennium" in text:
            return {
                "resolved_terms": ["1990s", "pre-2000", "20th century"],
                "date_range": {"start": 1000, "end": 1999},
                "type": "before_millennium"
            }
        return {}
    
    def _resolve_after_reference(self, text: str, expr: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Resolve 'after X' temporal references."""
        if "millennium" in text:
            return {
                "resolved_terms": ["2000s", "post-2000", "21st century"],
                "date_range": {"start": 2000, "end": datetime.now().year},
                "type": "after_millennium"
            }
        return {}
    
    def _resolve_era_reference(self, text: str, expr: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Resolve era-based temporal references."""
        era_mappings = {
            "golden age": {"start": 1930, "end": 1960, "terms": ["classic", "1930s", "1940s", "1950s"]},
            "new hollywood": {"start": 1967, "end": 1980, "terms": ["1970s", "auteur", "independent"]},
            "digital age": {"start": 1990, "end": 2020, "terms": ["1990s", "2000s", "digital", "CGI"]}
        }
        
        for era, info in era_mappings.items():
            if era in text:
                return {
                    "resolved_terms": info["terms"],
                    "date_range": {"start": info["start"], "end": info["end"]},
                    "type": f"era_{era.replace(' ', '_')}"
                }
        
        return {}
    
    def _resolve_relative_decades(self, text: str, expr: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Resolve relative decade references."""
        current_year = datetime.now().year
        num_decades = self._extract_number_word(text)
        
        if "ago" in text or "before" in text:
            target_decade = current_year - (num_decades * 10)
            decade_start = (target_decade // 10) * 10
            return {
                "resolved_terms": [f"{decade_start}s"],
                "date_range": {"start": decade_start, "end": decade_start + 9},
                "type": "relative_decades_past"
            }
        
        return {}