"""
Temporal expression plugin for parsing time-related queries.
Handles expressions like "90s movies", "last decade", "summer of 2020".
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    DATEPARSER_AVAILABLE = False

from .base import DualUsePlugin


class TemporalExpressionPlugin(DualUsePlugin):
    """Parse and normalize temporal expressions in text."""
    
    def __init__(self):
        super().__init__()
        self.current_year = datetime.now().year
        
    def _initialize_models(self):
        """Initialize temporal processing patterns."""
        # Decade patterns
        self.decade_patterns = [
            (r'\b(\d{2})s\b', lambda m: (int(m.group(1)) * 10 + 1900 if int(m.group(1)) >= 20 else int(m.group(1)) * 10 + 2000, 10)),
            (r'\b(\d{4})s\b', lambda m: (int(m.group(1)), 10)),
            (r'\bearly (\d{2})s\b', lambda m: (int(m.group(1)) * 10 + 1900 if int(m.group(1)) >= 20 else int(m.group(1)) * 10 + 2000, 3)),
            (r'\bmid[- ]?(\d{2})s\b', lambda m: (int(m.group(1)) * 10 + 1900 + 3 if int(m.group(1)) >= 20 else int(m.group(1)) * 10 + 2000 + 3, 4)),
            (r'\blate (\d{2})s\b', lambda m: (int(m.group(1)) * 10 + 1900 + 7 if int(m.group(1)) >= 20 else int(m.group(1)) * 10 + 2000 + 7, 3)),
        ]
        
        # Relative patterns
        self.relative_patterns = [
            (r'\blast decade\b', lambda: (self.current_year - 10, 10)),
            (r'\blast (\d+) years?\b', lambda m: (self.current_year - int(m.group(1)), int(m.group(1)))),
            (r'\brecent years?\b', lambda: (self.current_year - 5, 5)),
            (r'\bthis decade\b', lambda: (self.current_year // 10 * 10, self.current_year % 10 + 1)),
            (r'\bprevious decade\b', lambda: (self.current_year // 10 * 10 - 10, 10)),
        ]
        
        # Season patterns
        self.season_patterns = [
            (r'\bspring (?:of )?(\d{4})\b', lambda m: self._season_to_dates(int(m.group(1)), "spring")),
            (r'\bsummer (?:of )?(\d{4})\b', lambda m: self._season_to_dates(int(m.group(1)), "summer")),
            (r'\bfall (?:of )?(\d{4})\b', lambda m: self._season_to_dates(int(m.group(1)), "fall")),
            (r'\bautumn (?:of )?(\d{4})\b', lambda m: self._season_to_dates(int(m.group(1)), "fall")),
            (r'\bwinter (?:of )?(\d{4})\b', lambda m: self._season_to_dates(int(m.group(1)), "winter")),
        ]
        
        # Year range patterns
        self.range_patterns = [
            (r'\b(\d{4})[- ](?:to|through) (\d{4})\b', lambda m: (int(m.group(1)), int(m.group(2)) - int(m.group(1)) + 1)),
            (r'\bbetween (\d{4}) and (\d{4})\b', lambda m: (int(m.group(1)), int(m.group(2)) - int(m.group(1)) + 1)),
            (r'\bfrom (\d{4}) to (\d{4})\b', lambda m: (int(m.group(1)), int(m.group(2)) - int(m.group(1)) + 1)),
        ]
        
        # Single year patterns
        self.year_patterns = [
            (r'\bin (\d{4})\b', lambda m: (int(m.group(1)), 1)),
            (r'\b(\d{4}) (?:movies?|films?|releases?)\b', lambda m: (int(m.group(1)), 1)),
        ]
        
    def _season_to_dates(self, year: int, season: str) -> Tuple[str, str]:
        """Convert season to start/end dates."""
        seasons = {
            "spring": ("03-01", "05-31"),
            "summer": ("06-01", "08-31"),
            "fall": ("09-01", "11-30"),
            "winter": ("12-01", "02-28")  # Simplified winter
        }
        
        start_month_day, end_month_day = seasons.get(season, ("01-01", "12-31"))
        
        # Handle winter spanning years
        if season == "winter":
            start_date = f"{year}-{start_month_day}"
            end_date = f"{year + 1}-{end_month_day}"
        else:
            start_date = f"{year}-{start_month_day}"
            end_date = f"{year}-{end_month_day}"
            
        return start_date, end_date
    
    async def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract and normalize temporal expressions from text."""
        try:
            expressions = []
            normalized = []
            
            text_lower = text.lower()
            
            # Process different pattern types
            expressions.extend(self._extract_decade_expressions(text_lower))
            expressions.extend(self._extract_relative_expressions(text_lower))
            expressions.extend(self._extract_season_expressions(text_lower))
            expressions.extend(self._extract_range_expressions(text_lower))
            expressions.extend(self._extract_year_expressions(text_lower))
            
            # Try dateparser if available
            if DATEPARSER_AVAILABLE:
                expressions.extend(self._extract_with_dateparser(text))
            
            # Normalize all expressions
            for expr in expressions:
                norm = self._normalize_expression(expr)
                if norm:
                    normalized.append(norm)
            
            # Remove duplicates
            unique_normalized = []
            seen = set()
            for norm in normalized:
                key = (norm.get("start"), norm.get("end"), norm.get("precision"))
                if key not in seen:
                    seen.add(key)
                    unique_normalized.append(norm)
            
            return {
                "expressions": [expr["text"] for expr in expressions],
                "normalized": unique_normalized,
                "decade_detected": any(norm.get("precision") == "decade" for norm in unique_normalized),
                "year_ranges": [norm for norm in unique_normalized if norm.get("precision") in ["year", "range"]],
                "seasons": [norm for norm in unique_normalized if norm.get("precision") == "season"]
            }
            
        except Exception as e:
            self.logger.error(f"Error in temporal analysis: {e}")
            return {"error": str(e), "expressions": [], "normalized": []}
    
    def _extract_decade_expressions(self, text: str) -> List[Dict[str, Any]]:
        """Extract decade expressions like '90s', 'early 2000s'."""
        expressions = []
        
        for pattern, handler in self.decade_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    start_year, duration = handler(match)
                    expressions.append({
                        "text": match.group(0),
                        "type": "decade",
                        "start_year": start_year,
                        "duration": duration,
                        "match_span": match.span()
                    })
                except Exception as e:
                    self.logger.debug(f"Error processing decade match: {e}")
        
        return expressions
    
    def _extract_relative_expressions(self, text: str) -> List[Dict[str, Any]]:
        """Extract relative expressions like 'last decade', 'recent years'."""
        expressions = []
        
        for pattern, handler in self.relative_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if callable(handler):
                        result = handler()
                    else:
                        result = handler(match)
                    
                    start_year, duration = result
                    expressions.append({
                        "text": match.group(0),
                        "type": "relative",
                        "start_year": start_year,
                        "duration": duration,
                        "match_span": match.span()
                    })
                except Exception as e:
                    self.logger.debug(f"Error processing relative match: {e}")
        
        return expressions
    
    def _extract_season_expressions(self, text: str) -> List[Dict[str, Any]]:
        """Extract season expressions like 'summer of 2020'."""
        expressions = []
        
        for pattern, handler in self.season_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    start_date, end_date = handler(match)
                    expressions.append({
                        "text": match.group(0),
                        "type": "season",
                        "start_date": start_date,
                        "end_date": end_date,
                        "match_span": match.span()
                    })
                except Exception as e:
                    self.logger.debug(f"Error processing season match: {e}")
        
        return expressions
    
    def _extract_range_expressions(self, text: str) -> List[Dict[str, Any]]:
        """Extract year range expressions like '1990-1995', 'between 2000 and 2010'."""
        expressions = []
        
        for pattern, handler in self.range_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    start_year, duration = handler(match)
                    expressions.append({
                        "text": match.group(0),
                        "type": "range",
                        "start_year": start_year,
                        "duration": duration,
                        "match_span": match.span()
                    })
                except Exception as e:
                    self.logger.debug(f"Error processing range match: {e}")
        
        return expressions
    
    def _extract_year_expressions(self, text: str) -> List[Dict[str, Any]]:
        """Extract single year expressions like 'in 1995', '2020 movies'."""
        expressions = []
        
        for pattern, handler in self.year_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    start_year, duration = handler(match)
                    # Basic validation
                    if 1900 <= start_year <= self.current_year + 5:
                        expressions.append({
                            "text": match.group(0),
                            "type": "year",
                            "start_year": start_year,
                            "duration": duration,
                            "match_span": match.span()
                        })
                except Exception as e:
                    self.logger.debug(f"Error processing year match: {e}")
        
        return expressions
    
    def _extract_with_dateparser(self, text: str) -> List[Dict[str, Any]]:
        """Extract dates using dateparser library if available."""
        expressions = []
        
        try:
            # Look for date-like phrases
            date_phrases = re.findall(r'\b(?:(?:last|next|this)\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b[^.]*?(?:\d{4}|\b(?:year|decade|century)\b)', text, re.IGNORECASE)
            
            for phrase in date_phrases:
                parsed = dateparser.parse(phrase)
                if parsed:
                    expressions.append({
                        "text": phrase,
                        "type": "dateparser",
                        "parsed_date": parsed.isoformat(),
                        "year": parsed.year
                    })
        
        except Exception as e:
            self.logger.debug(f"Dateparser error: {e}")
        
        return expressions
    
    def _normalize_expression(self, expr: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a temporal expression to standard format."""
        try:
            expr_type = expr["type"]
            
            if expr_type in ["decade", "relative", "range", "year"]:
                start_year = expr["start_year"]
                duration = expr["duration"]
                end_year = start_year + duration - 1
                
                return {
                    "text": expr["text"],
                    "start": start_year,
                    "end": end_year,
                    "precision": "decade" if duration >= 10 else "year" if duration == 1 else "range",
                    "type": expr_type
                }
            
            elif expr_type == "season":
                return {
                    "text": expr["text"],
                    "start": expr["start_date"],
                    "end": expr["end_date"],
                    "precision": "season",
                    "type": expr_type
                }
            
            elif expr_type == "dateparser" and "year" in expr:
                return {
                    "text": expr["text"],
                    "start": expr["year"],
                    "end": expr["year"],
                    "precision": "year",
                    "type": expr_type
                }
        
        except Exception as e:
            self.logger.debug(f"Error normalizing expression: {e}")
        
        return None